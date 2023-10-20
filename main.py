# Standard library imports
import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Optional, Union

# Third-party library imports
import datasets
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Local imports
sys.path.append('./ADL_HW1/main_code/')
from utils_qa import postprocess_qa_predictions

# Transformers library imports
import transformers
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    CONFIG_MAPPING,
    DataCollatorWithPadding,
    EvalPrediction,
    MODEL_MAPPING,
    PreTrainedTokenizerBase,
    SchedulerType,
    default_data_collator,
    get_scheduler
)
from transformers.utils import PaddingStrategy, check_min_version, send_example_telemetry


parser = argparse.ArgumentParser()
parser.add_argument("context_path", help="Path to context.json")
parser.add_argument("test_path", help="Path to test.json")
parser.add_argument("output_path", help="Path to the output prediction file")
args = parser.parse_args()


import pandas as pd
import datasets


test_df = pd.read_json(args.test_path)
context_df = pd.read_json(args.context_path)

def reformat(df, have_label=0):
    # Create new columns based on the existing ones
    df['sent1'] = df['question']
    df['sent2'] = ''

    # Extract elements from the 'paragraphs' list into new columns
    df['ending0'] = df['paragraphs'].apply(lambda x: context_df.iloc[x[0], 0])
    df['ending1'] = df['paragraphs'].apply(lambda x: context_df.iloc[x[1], 0])
    df['ending2'] = df['paragraphs'].apply(lambda x: context_df.iloc[x[2], 0])
    df['ending3'] = df['paragraphs'].apply(lambda x: context_df.iloc[x[3], 0])

    # Find the index of 'relevant' in 'paragraphs'
    if have_label:
        df['label'] = df.apply(lambda row: row['paragraphs'].index(row['relevant']), axis=1)
        df.drop(['relevant'], axis=1, inplace=True)

    df.drop(['question', 'paragraphs'], axis=1, inplace=True)
    
    return df

test_df = reformat(test_df)
test_dataset = datasets.Dataset.from_pandas(test_df)
raw_dataset = datasets.DatasetDict({"test":test_dataset})

column_names = raw_dataset["test"].column_names
print('\n', raw_dataset)

# Model Setup
model_dir = './ADL_HW1/set_up/PS_model'
config = AutoConfig.from_pretrained(model_dir)
model = AutoModelForMultipleChoice.from_pretrained(model_dir, from_tf=False, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

accelerator_log_kwargs = {}
accelerator = Accelerator(gradient_accumulation_steps=2, **accelerator_log_kwargs)
accelerator.wait_for_everyone()
device = accelerator.device
model.to(device)


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
#         labels = [feature.pop("labels") for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
#         batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch
    

dataset_name = 'test'

ending_names = [f"ending{i}" for i in range(4)]
context_name = "sent1"
question_header_name = "sent2"

# Pre-process
def preprocess_function(examples):
    first_sentences = [[context] * 4 for context in examples[context_name]]
    question_headers = examples[question_header_name]
    second_sentences = [
        [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
    ]

    # Flatten out
    first_sentences = list(chain(*first_sentences))
    second_sentences = list(chain(*second_sentences))

    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        max_length=512,
        padding="max_length",
        truncation=True
    )
    tokenized_inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
#     tokenized_inputs["labels"] = examples['label']
    return tokenized_inputs
with accelerator.main_process_first():
    processed_datasets = raw_dataset.map(preprocess_function, batched=True, remove_columns=raw_dataset[dataset_name].column_names)
eval_dataset = processed_datasets[dataset_name]


data_collator = DataCollatorForMultipleChoice(tokenizer, pad_to_multiple_of=None)
eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=1)
model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

all_predictions = []

model.eval()
for step, batch in enumerate(eval_dataloader):
    with torch.no_grad():
        outputs = model(**batch)
    predictions = outputs.logits.argmax(dim=-1).item()
#     label = batch['labels'].item()
    
    all_predictions.append(predictions)


prediction_result = pd.DataFrame.from_dict({
    'id': test_df['id'],
    'prediction': all_predictions
})

prediction_result.to_csv('prediction_result.csv', index=False)
prediction_result.head()


raw_test_df = pd.read_json(args.test_path)
context_df = pd.read_json(args.context_path)
prediction_df = prediction_result


test_df = pd.merge(prediction_df, raw_test_df, on='id')
test_df['context'] = test_df.apply(lambda row: context_df.loc[row['paragraphs'][row['prediction']],0], axis=1)
test_df.drop(['prediction', 'paragraphs'], axis=1, inplace=True)


test_dataset = datasets.Dataset.from_pandas(test_df)
raw_dataset = datasets.DatasetDict({"test":test_dataset})

print('\n', raw_dataset)


# Model Setup
model_dir = './ADL_HW1/set_up/SS_model'
config = AutoConfig.from_pretrained(model_dir)
model = AutoModelForQuestionAnswering.from_pretrained(model_dir, from_tf=False, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

column_names = raw_dataset["test"].column_names
question_column_name = "question"
context_column_name = "context"
answer_column_name = "answers"

pad_on_right = tokenizer.padding_side == "right"
max_seq_length = min(512, tokenizer.model_max_length)

# Validation preprocessing
def prepare_validation_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding= False,
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def post_processing_function(examples, features, predictions, stage="eval"):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        version_2_with_negative=False,
        n_best_size=20,
        max_answer_length=30,
        null_score_diff_threshold=0.0,
        output_dir='./',
        prefix=stage,
    )

    # Format the result to the format the metric expects.
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    return formatted_predictions
#     references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
#     return EvalPrediction(predictions=formatted_predictions, label_ids=references)

def create_and_fill_np_array(start_or_end_logits, dataset, max_len):

    step = 0
    # create a numpy array and fill it with -100.
    logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
    # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
    for i, output_logit in enumerate(start_or_end_logits):  # populate columns
        # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
        # And after every iteration we have to change the step

        batch_size = output_logit.shape[0]
        cols = output_logit.shape[1]

        if step + batch_size < len(dataset):
            logits_concat[step : step + batch_size, :cols] = output_logit
        else:
            logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

        step += batch_size

    return logits_concat


eval_examples = raw_dataset["test"]
with accelerator.main_process_first():
    eval_dataset = eval_examples.map(
        prepare_validation_features,
        batched=True,
        num_proc=1,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on validation dataset",
    )


data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)
eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])
eval_dataloader = DataLoader(eval_dataset_for_model, collate_fn=data_collator, batch_size=1)

# Prepare everything with our `accelerator`.
model, eval_dataloader = accelerator.prepare(model, eval_dataloader)


all_start_logits = []
all_end_logits = []

model.eval()
for step, batch in enumerate(eval_dataloader):
#     print(step)
    with torch.no_grad():
        outputs = model(**batch)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
        end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

        all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
        all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())

max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor

# concatenate the numpy array
start_logits_concat = create_and_fill_np_array(all_start_logits, eval_dataset, max_len)
end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len)

# delete the list of numpy arrays
del all_start_logits
del all_end_logits

outputs_numpy = (start_logits_concat, end_logits_concat)
prediction = post_processing_function(eval_examples, eval_dataset, outputs_numpy)


final_submission_df = pd.DataFrame(prediction)

# Rename the 'prediction_text' column to 'answer'
final_submission_df.rename(columns={'prediction_text': 'answer'}, inplace=True)

# Save as CSV
final_submission_df.to_csv(args.output_path, index=False)
