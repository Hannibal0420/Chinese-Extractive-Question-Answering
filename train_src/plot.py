import pandas as pd
import matplotlib.pyplot as plt

# Manual collect printed "Loss" during training
df = pd.DataFrame({'Epoch': [1,2,3,4,5], 'Loss': [1.22,0.77,0.57,0.44,0.36]})
plt.figure(figsize=(10, 6))

# Plot the line
plt.plot(df['Epoch'], df['Loss'], label='Training Loss')

# Show the data points
plt.scatter(df['Epoch'], df['Loss'], color='red')

# Set x-axis ticks to show only 1-5
plt.xticks([1, 2, 3, 4, 5])

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.savefig('../images/training_loss.png')
plt.show()
