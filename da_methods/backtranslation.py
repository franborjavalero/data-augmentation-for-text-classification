import argparse
from collections import defaultdict
from nltk import word_tokenize
import os
import pandas as pd
import torch
from transformers import (
    MarianTokenizer, 
    MarianMTModel,
)

def load_data(args):
    """ Load data from tsv into pd.DataFrame """
    df = pd.read_csv(args.input_file, delimiter="\t")
    df_original = df[~df[args.validation_col]]
    df_valid = df[df[args.validation_col]]
    return df_original, df_valid

def generate_output_df(output_dir, df_original, df_valid, columns, language_translation, list_languages):
    print(len(df_original))
    df = pd.concat(
        [
            df_original, 
            df_valid
        ], 
        ignore_index=True
    )
    df[columns[1]].astype("int")
    str_list_languages = "_".join(list_languages)
    for language_ in language_translation:
        df = pd.concat(
        [
            df, 
            language_translation[language_],
        ], 
        ignore_index=True
    )
    df[columns[1]].astype("int")
    column_lower = "lower"
    df[column_lower] = df[columns[0]].apply(lambda sample: sample.lower())
    # sorting by first name 
    df.sort_values(column_lower, inplace=True) 
    # dropping ALL duplicte values 
    df.drop_duplicates(subset=column_lower, keep=False, inplace=True) 
    # remove lower column
    df = df.drop([column_lower], axis=1)
    df.to_csv(os.path.join(output_dir, f"train_en_{str_list_languages}.tsv"), sep="\t", index=False, columns=columns)

def translate(sample_text, src_lang, tgt_lang, device="cuda", batch_size=16):
    num_samples = len(sample_text)
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    model = MarianMTModel.from_pretrained(model_name).to(device)
    tok = MarianTokenizer.from_pretrained(model_name)
    translated_samples = []
    for i in range(0, num_samples, batch_size):
        j = min(i+batch_size, num_samples)
        batch = tok.prepare_translation_batch(src_texts=sample_text[i:j]).to(device)
        gen_forward = model.generate(**batch)
        translated_samples_ = tok.batch_decode(gen_forward, skip_special_tokens=True)
        translated_samples += translated_samples_
    model.to("cpu")
    del model, tok
    return translated_samples

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--input_file",
        default=None,
        help="The input .tsv files for the task",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="The output dir for saving the .tsv files for the task.",
    )
    parser.add_argument(
        "--batch_size", default=16, type=int, help="batch size for the translation",
    )
    parser.add_argument(
        "--text_col",
        default=None,
        type=str,
        help="The name of the column in the tsv containing the text to be classified",
    )
    parser.add_argument(
        "--label_col",
        default="label",
        type=str,
        help="The name of the column in the tsv containing the label to be predicted",
    )
    parser.add_argument(
        "--validation_col",
        default="is_valid",
        type=str,
        help="The name of the column containing a bool to delect validation data",
    )
    parser.add_argument(
        "--anonymized_tokens",
        action="store_true",
        help="Set this flag to consider the anonymized tokens",
    )
    parser.add_argument(
        "--random_seed",
        default=42,
        type=int,
        help="fix random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    pivot_languages = [
        "it",
        "fr",
        "de",
        "ca",
        "ru",
        "cs",
        "et",
        "nl",
        "id",
        "bg",
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # make sure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load data
    df_original, df_valid = load_data(args)
    columns = df_original.columns.to_list()
    sample_text = df_original[args.text_col].to_list()
    sample_label = df_original[args.label_col].to_list()
    sample_validation = df_original[args.validation_col].to_list()
    lang_bt_samples = defaultdict(list)

    for pivot_language in pivot_languages:
        # Forward: EN -> PIVOT
        print(f"lang={pivot_language}, Forward")
        pivot_samples = translate(sample_text, "en", pivot_language, device=device, batch_size=args.batch_size)
        # Backward: PIVOT -> EN
        print(f"lang={pivot_language}, Backward")
        backtranslated_samples = translate(pivot_samples, pivot_language, "en", device=device, batch_size=args.batch_size)
        # Refine output format
        tmp = [" ".join(word_tokenize(sample)) for sample in backtranslated_samples]
        lang_bt_samples[pivot_language] = tmp
    
    # Invidual language
    lang_df = {}
    for (lang, bt_sentences) in lang_bt_samples.items():
        data = {
            columns[0]: bt_sentences,
            columns[1]: sample_label,
            columns[2]: sample_validation,
        }
        df_ind = pd.DataFrame(data)
        df_ind[columns[1]].astype("int")
        lang_df[lang] = df_ind
        generate_output_df(args.output_dir, df_original, df_valid, columns, lang_df, [lang])
    
    # ca+fr
    generate_output_df(args.output_dir, df_original, df_valid, columns, lang_df, ["ca", "fr"])

    # ca+fr+de+it
    generate_output_df(args.output_dir, df_original, df_valid, columns, lang_df, ["ca", "fr", "de", "it"])

    # common languages
    generate_output_df(args.output_dir, df_original, df_valid, columns, lang_df, pivot_languages[:5])

    # uncommon languages
    generate_output_df(args.output_dir, df_original, df_valid, columns, lang_df, pivot_languages[5:])

    # all languages
    generate_output_df(args.output_dir, df_original, df_valid, columns, lang_df, pivot_languages)


if __name__ == "__main__":
    main()
