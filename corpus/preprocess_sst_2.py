import argparse
from data_utils import read_file, write_file, export_dict
import os
import pandas as pd

def main():

    parser = argparse.ArgumentParser()     
    parser.add_argument("INPUT_DIR", type=str)
    parser.add_argument("OUTPUT_DIR", type=str)
    args = parser.parse_args()
    sections = [
        "train",
        "dev",
        "test",
    ]
    for section in sections:
        file_name = os.path.join(args.INPUT_DIR, f"{section}.txt")
        labels, sentences = read_file(file_name)
        write_file(os.path.join(args.OUTPUT_DIR, f"{section}"), sentences, labels, "sentence", "label")
    
    id_label = {
        0: "negative",
        1: "positive", 
    }
    
    export_dict(os.path.join(args.OUTPUT_DIR, "labels"), id_label)
        

if __name__ == "__main__":
    main()