import argparse
from data_utils import reverse_dict, export_dict, write_file
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def main():    
    
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    parser = argparse.ArgumentParser()     
    parser.add_argument("INPUT_FILE", type=str)
    parser.add_argument("OUTPUT_DIR", type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.INPUT_FILE)
    text = df['question_text'].to_list()
    topic = df['question_topic'].to_list()
    
    unique_topic = sorted(list(set(topic)))
    topic_id = {unique_topic[id_] : id_ for id_ in range(len(unique_topic))}
    ground_truth = list(map(lambda key: topic_id[key], topic))
    id_topic = reverse_dict(topic_id)
    
    X_train, X_valid, y_train, y_valid = train_test_split(text, ground_truth, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    write_file(os.path.join(args.OUTPUT_DIR, "train"), X_train, y_train, "text", "topic")
    write_file(os.path.join(args.OUTPUT_DIR, "validate"), X_valid, y_valid, "text", "topic")
    export_dict(os.path.join(args.OUTPUT_DIR, "labels"), id_topic)


if __name__ == "__main__":
    main()