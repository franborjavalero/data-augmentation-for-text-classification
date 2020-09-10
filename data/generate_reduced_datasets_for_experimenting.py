import argparse
from data_utils import get_n_examples_per_class
import os
import pandas as pd
from shutil import copyfile


def main():    
    
    parser = argparse.ArgumentParser()     
    parser.add_argument("DATA_DIRECTORY", type=str)
    parser.add_argument("--N_train", type=int, nargs='+')
    parser.add_argument("--N_valid", type=int, nargs='+')
    parser.add_argument("--SEED", type=int, default=42)
    args = parser.parse_args()

    assert len(args.N_train) == len(args.N_valid)

    df_full_train = pd.read_csv(os.path.join(args.DATA_DIRECTORY, "full", "train.tsv"), sep='\t')
    df_full_test_name = os.path.join(args.DATA_DIRECTORY, "full", "test.tsv")
    df_valid = df_full_train.loc[df_full_train["is_valid"] == True]
    df_train = df_full_train.loc[df_full_train["is_valid"] == False]
    label_y = df_train.columns[1]

    for (n_train, n_valid) in zip(args.N_train, args.N_valid):
        directory_section = os.path.join(args.DATA_DIRECTORY, str(n_train))    
        if not os.path.exists(directory_section):
            os.makedirs(directory_section)
        df_section_test_name = os.path.join(directory_section, "test.tsv")
        copyfile(df_full_test_name, df_section_test_name)
        _, df_reduced_train = get_n_examples_per_class(df_train, n_train, label_y) 
        _, df_reduced_valid = get_n_examples_per_class(df_valid, n_valid, label_y)
        # FULL VALIDATION SET
        df_ = pd.concat([df_reduced_train, df_valid]).sample(frac=1, random_state=args.SEED).reset_index(drop=True)
        df_.to_csv(os.path.join(directory_section, "train_full_valid.tsv"), sep="\t", index=False)
        # REDUCED VALIDATION SET
        df_ = pd.concat([df_reduced_train, df_reduced_valid]).sample(frac=1, random_state=args.SEED).reset_index(drop=True)
        df_.to_csv(os.path.join(directory_section, "train_red_valid.tsv"), sep="\t", index=False)
        
        
if __name__ == "__main__":
    main()