import argparse
from data_utils import reverse_dict, export_dict, read_file, write_file
import os
import pandas as pd


def main():    
    parser = argparse.ArgumentParser()     
    parser.add_argument("INPUT_DIR", type=str)
    parser.add_argument("OUTPUT_DIR", type=str)
    args = parser.parse_args()
    file_section = {
        "train_5500.label" : "train",
        "TREC_10.label" : "test",
    }
    if not os.path.exists(args.OUTPUT_DIR):
        os.mkdir(args.OUTPUT_DIR)
    for (base_name, section) in file_section.items():
        corpus_path = os.path.join(args.INPUT_DIR, base_name)
        labels_full, examples = read_file(corpus_path, encoding="iso-8859-1")
        labels = [ l.split(":")[0] for l in labels_full]    
        if section == "train":
            unique_label = sorted(list(set(labels)))
            label_id = {unique_label[id_] : id_ for id_ in range(len(unique_label))}
            id_label = reverse_dict(label_id)
            export_dict(os.path.join(args.OUTPUT_DIR, "labels"), id_label)
        labels = list(map(lambda key: label_id[key], labels))
        write_file(os.path.join(args.OUTPUT_DIR, section), examples, labels, "question", "label")


if __name__ == "__main__":
    main()