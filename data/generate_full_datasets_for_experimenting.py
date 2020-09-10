import argparse
from data_utils import get_n_examples_per_class, load_dict, reverse_dict, remove_classes
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def main():    
    
    parser = argparse.ArgumentParser()     
    parser.add_argument("CORPUS", type=str, choices=["SST-2", "TREC", "SNIPS", "RAIC", "RAIT", "Question-Topic"])
    parser.add_argument("TRAIN_FILE", type=str)
    parser.add_argument("TEST_FILE", type=str)
    parser.add_argument("--CLASSES_FILE", type=str, default=None)
    parser.add_argument("--VALIDATION_FILE", type=str, default=None)
    parser.add_argument("--SEED", type=int, default=None)
    parser.add_argument("--VALIDATION_SPLIT_SIZE", type=float, default=None)
    parser.add_argument("--TEST_SPLIT_SIZE", type=float, default=None)
    parser.add_argument("--N_VALIDATION_CLASS", type=int, default=None)
    parser.add_argument("--FILE_ID_CLASSES_RAW", type=str, default=None)
    parser.add_argument("--FILE_ID_CLASSES_FILTERED", type=str, default=None)
    args = parser.parse_args()

    corpus_split_training_set = frozenset({
        "TREC",
        "RAIC",
        "RAIT",
        "Question-Topic",
    })

    corpus_rai = frozenset({
        "RAIC",
        "RAIT",
    })

    OUPUT_DIR = os.path.join(args.CORPUS, "full")
    if not os.path.exists(OUPUT_DIR):
        os.makedirs(OUPUT_DIR)
    
    df_train = pd.read_csv(args.TRAIN_FILE, sep='\t')
    df_test = pd.read_csv(args.TEST_FILE, sep='\t')
    _, label_y = df_train.columns
    
    if args.CORPUS == "SST-2":    
        df_val = pd.read_csv(args.VALIDATION_FILE, sep='\t')
    
    elif args.CORPUS in corpus_split_training_set:
        if args.CORPUS in corpus_rai:
            raw_id_classes = load_dict(args.FILE_ID_CLASSES_RAW)
            filtered_id_classes = load_dict(args.FILE_ID_CLASSES_FILTERED)
            filtered_classes_id = reverse_dict(filtered_id_classes)
            df_train = remove_classes(df_train, label_y, raw_id_classes, filtered_classes_id)
            df_test = remove_classes(df_test, label_y, raw_id_classes, filtered_classes_id)
        df_train, df_val = train_test_split(df_train, stratify=df_train[label_y], test_size=args.VALIDATION_SPLIT_SIZE, random_state=args.SEED)
        
    elif args.CORPUS == "SNIPS":
        df_train, df_val = get_n_examples_per_class(df_train, args.N_VALIDATION_CLASS, label_y) 
    
    df_train["is_valid"] = False
    df_val["is_valid"] = True    
    df_train_val = pd.concat([df_train, df_val], ignore_index=True)
    df_train_val = df_train_val
    df_train_val.to_csv(os.path.join(OUPUT_DIR, "train.tsv"), sep="\t", index=False)
    df_test = df_test
    df_test.to_csv(os.path.join(OUPUT_DIR, "test.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()