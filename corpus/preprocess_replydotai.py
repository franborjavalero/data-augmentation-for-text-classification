import argparse
from data_utils import reverse_dict, export_dict, write_file
import langid
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main():    
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    parser = argparse.ArgumentParser()     
    parser.add_argument("INPUT_FILE", type=str)
    parser.add_argument("OUTPUT_DIR", type=str)
    parser.add_argument("--threshold", type=int, default=400)
    args = parser.parse_args()
    labels = {
        "category",
        "topic",
    }
    df = pd.read_csv(args.INPUT_FILE)
    # Ignores japaness queries
    df['lang'] = df['query'].apply(lambda query: langid.classify(query)[0])
    df = df.loc[df['lang'] != 'ja']
    # Remove queries whose length will be greather than 400 tokens
    df["mlen"] = df['query'].apply(lambda text: len(str(text).split(" ")))
    df = df[df["mlen"] < args.threshold]
    queries = df['query'].to_list()
    for label in labels:
        current_dir = os.path.join(args.OUTPUT_DIR, label)
        array_label = df[label]
        unique_label = array_label.unique()
        label_id = {unique_label[id_] : id_ for id_ in range(len(unique_label))}
        ground_truth = array_label.to_list()
        ground_truth = list(map(lambda key: label_id[key], ground_truth))
        id_label = reverse_dict(label_id)
        if not os.path.exists(current_dir):
            os.mkdir(current_dir)
        X_train, X_valid, y_train, y_valid = train_test_split(queries, ground_truth, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        write_file(os.path.join(current_dir, "train"), X_train, y_train, "query", "label")
        write_file(os.path.join(current_dir, "validate"), X_valid, y_valid, "query", "label")
        export_dict(os.path.join(current_dir, "labels"), id_label)


if __name__ == "__main__":
    main()