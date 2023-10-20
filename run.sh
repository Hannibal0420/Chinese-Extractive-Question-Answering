if [ "$#" -ne 3 ]; then
    echo "Usage: ./run.sh /path/to/context.json /path/to/test.json /path/to/pred/prediction.csv"
    exit 1
fi

python3 main.py "$1" "$2" "$3"