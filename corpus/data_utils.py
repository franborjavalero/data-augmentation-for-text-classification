import json
import pandas as pd

def read_file(file_name, encoding="utf-8"):
    labels = []
    examples = []
    with open(file_name, 'r', encoding=encoding) as file:
        for line in file:
            line = line.strip()
            line_tokens = line.split(" ")
            label_ = line_tokens[0]
            example_ = " ".join(line_tokens[1:])
            labels.append(label_)
            examples.append(example_)
    return labels, examples

def write_file(file_name, list_examples, list_labels, example_name, label_name, delimiter="\t", ext="tsv", index=False):
    d = {example_name: list_examples, label_name: list_labels}
    columns = [example_name, label_name]
    df = pd.DataFrame(d, columns=columns)
    df.to_csv(f"{file_name}.{ext}", sep=delimiter, index=index, columns=columns)

def reverse_dict(dict_):
    return { value: key for (key, value) in dict_.items()}

def export_dict(file_name, dict_):
    with open(f"{file_name}.json", 'w') as f_json:
        json.dump(dict_, f_json, indent=4, sort_keys=True)

def load_dict(file_name):
    with open(file_name) as json_file:
        key_value = json.load(json_file)
    return key_value