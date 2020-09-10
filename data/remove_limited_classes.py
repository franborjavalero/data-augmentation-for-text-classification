import argparse
from data_utils import load_dict, export_dict
import os

def main():    

    parser = argparse.ArgumentParser()
    parser.add_argument("FILE_STATISTICS", type=str)
    parser.add_argument("FILE_LABELS", type=str)
    parser.add_argument("THRESHOLD", type=int)
    parser.add_argument("--VALIDATION_SPLIT_SIZE", type=float, default=1)
    args = parser.parse_args()
    
    kept_classes = []
    new_id_class = {}
    classes_examples = load_dict(args.FILE_STATISTICS)["classes"]
    TRAIN_SPLIT_SIZE = 1 - args.VALIDATION_SPLIT_SIZE
    for (class_, dict_examples) in classes_examples.items():
        try :
            num_examples = int(dict_examples["train"] * TRAIN_SPLIT_SIZE)
            if num_examples > args.THRESHOLD:
                kept_classes.append(f"{class_} ({num_examples})")
                new_id_class[len(new_id_class)] = class_
        except:
            continue
    
    export_dict(args.FILE_LABELS, new_id_class)


if __name__ == "__main__":
    main()