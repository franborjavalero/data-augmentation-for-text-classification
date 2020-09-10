import json
import pandas as pd

def get_n_examples_per_class(df, n, col_label, seed=42):
    labels = df[col_label].unique()
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    for label in labels:
        df_tmp = df.loc[df[col_label] == label].sample(frac=1, random_state=seed).reset_index(drop=True)
        tmp_val, tmp_train = df_tmp[:n], df_tmp[n:]
        df_train = df_train.append(tmp_train)
        df_val = df_val.append(tmp_val)
    return df_train, df_val

def remove_classes(df, col_label, raw_id_classes, filtered_classes_id):
    df[col_label] = df[col_label].apply(lambda id_: raw_id_classes[str(id_)])
    selected_classes = frozenset(filtered_classes_id.keys())
    df["kept"] = df[col_label].apply(lambda class_: class_ in selected_classes)
    df = df.loc[df['kept'] == True].iloc[:, 0:2]
    df[col_label] = df[col_label].apply(lambda class_: int(filtered_classes_id[class_]))
    return df

def reverse_dict(dict_):
    return {value: key for (key, value) in dict_.items()}

def export_dict(file_name, dict_):
    with open(file_name, 'w') as f_json:
        json.dump(dict_, f_json, indent=4, sort_keys=True)

def load_dict(file_name):
    with open(file_name) as json_file:
        key_value = json.load(json_file)
    return key_value