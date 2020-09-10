import collections
import random
from typing import Collection, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import torch
from fastai.text.transform import BaseTokenizer, Vocab
from matplotlib import pyplot
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import PreTrainedModel, PreTrainedTokenizer
import json


def seed_all(seed_value: int) -> None:
    """fix all random seeds for reproducible experiments"""
    random.seed(seed_value)  # Python
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


# Custom Glue
class CustomTransformerModel(torch.nn.Module):
    """Wrapper around pretrained Transformer model to be compatible with fast.ai"""

    def __init__(self, transformer_model: PreTrainedModel):
        super(CustomTransformerModel, self).__init__()
        self.transformer = transformer_model

    def forward(self, input_ids):
        # Return only the logits from the transfomer
        logits = self.transformer(input_ids)[0]
        return logits


class TransformersBaseTokenizer(BaseTokenizer):
    """Wrapper around PreTrainedTokenizer to be compatible with fast.ai"""

    def __init__(
        self, pretrained_tokenizer: PreTrainedTokenizer, model_type="bert", **kwargs
    ):
        self._pretrained_tokenizer = pretrained_tokenizer
        self.max_seq_len = pretrained_tokenizer.max_len
        self.model_type = model_type

    def __call__(self, *args, **kwargs):
        return self

    def tokenizer(self, t: str) -> List[str]:
        """Limits the maximum sequence length and add the spesial tokens"""
        CLS = self._pretrained_tokenizer.cls_token
        SEP = self._pretrained_tokenizer.sep_token
        if self.model_type in ["roberta"]:
            tokens = self._pretrained_tokenizer.tokenize(t, add_prefix_space=True)[
                : self.max_seq_len - 2
            ]
        else:
            tokens = self._pretrained_tokenizer.tokenize(t)[: self.max_seq_len - 2]
        return [CLS] + tokens + [SEP]


class TransformersVocab(Vocab):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super(TransformersVocab, self).__init__(itos=[])
        self.tokenizer = tokenizer

    def numericalize(self, t: Collection[str]) -> List[int]:
        "Convert a list of tokens `t` to their ids."
        return self.tokenizer.convert_tokens_to_ids(t)
        # return self.tokenizer.encode(t)

    def textify(self, nums: Collection[int], sep=" ") -> List[str]:
        "Convert a list of `nums` to their tokens."
        nums = np.array(nums).tolist()
        return (
            sep.join(self.tokenizer.convert_ids_to_tokens(nums))
            if sep is not None
            else self.tokenizer.convert_ids_to_tokens(nums)
        )

    def __getstate__(self):
        return {"itos": self.itos, "tokenizer": self.tokenizer}

    def __setstate__(self, state: dict):
        self.itos = state["itos"]
        self.tokenizer = state["tokenizer"]
        self.stoi = collections.defaultdict(
            int, {v: k for k, v in enumerate(self.itos)}
        )


class ClassificationEval:
    """A CalssificationEva object takes in a dataframe with evaluated pedictions and analyzes them.

       Args:
               x_col(str): name of x column
               y_col(str): name of y / true label column
               pred_col(str): name of column with model predictions
               score_col(str): name of column with model confidence scores
               average(str): type of averaging performed on the data to calculate metrics
    """

    def __init__(
        self,
        df,
        x_col="text",
        y_col="label",
        pred_col="prediction",
        score_col="confidence",
        average="macro",
    ):
        self.df = df
        self.x_col = x_col
        self.y_col = y_col
        self.pred_col = pred_col
        self.score_col = score_col
        self.average = average
        self.categories = self.get_categories()
        # metrics
        (
            self.accuracy,
            self.confusion_matrix,
            self.precision,
            self.recall,
            self.f1,
        ) = self.calculate_metrics()
        # error cases / correct
        self.errors, self.correct = self.get_errors()

    def get_categories(self):
        pred_cats = self.df[self.pred_col].unique()
        y_cats = self.df[self.y_col].unique()
        # get set of pred_cats, y_cats
        categories = np.concatenate([pred_cats, y_cats[~np.isin(y_cats, pred_cats)]])
        # sort alphabetically
        categories = sorted(categories, key=str.lower)
        return categories

    def calculate_metrics(self):
        y_true = self.df[self.y_col].values
        y_pred = self.df[self.pred_col].values

        accuracy = (y_true == y_pred).mean()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=self.average
        )

        return accuracy, confusion_matrix(y_pred, y_true), precision, recall, f1

    def get_classification_report(self, order="recall"):
        """get classification report for each category

        Args:

            order: order report by precision, recall, f1-score, or support
        """
        y_true = self.df[self.y_col].values
        y_pred = self.df[self.pred_col].values

        report_dict = classification_report(
            y_true, y_pred, labels=self.categories, output_dict=True
        )
        # create
        report_dict = {k: report_dict[k] for k in self.categories}
        report_df = pd.DataFrame.from_dict(report_dict, orient="index")
        report_df = report_df.sort_values(order)
        return report_df

    def get_errors(self):
        errors = self.df.loc[self.df[self.y_col] != self.df[self.pred_col]]
        correct = self.df.loc[self.df[self.y_col] == self.df[self.pred_col]]
        return errors, correct

    def plot_confusion_matrix(self, normalize=False, file_name=None):
        """This function prints and plots the confusion matrix.

        Args:

            normalize(bool): Apply normalization

            file_name(str): optionally save plot to disk as html
        """

        if normalize:
            cnf_matrix = self.confusion_matrix / len(self.df)
        else:
            cnf_matrix = self.confusion_matrix

        trace1 = {
            "x": self.categories,
            "y": self.categories[::-1],
            "z": cnf_matrix.tolist()[::-1],
        }

        fig = go.Figure(data=go.Heatmap(trace1))
        fig.update_layout(template="simple_white")
        if file_name:
            pio.write_html(fig, file_name)
        else:
            pio.show(fig)

    def plot_confidence(self, use_cats: List = None):
        """plot confidence scores to for threshold analysis

        Args:

            use_cats: optionally plot specific ore single category; by default use all
        """

        # use all categories by default
        if use_cats is None:
            use_cats = self.categories

        _df = self.df.loc[self.df[self.y_col].isin(use_cats)]

        errors = _df[_df[self.pred_col] != _df[self.y_col]]
        correct = _df[_df[self.pred_col] == _df[self.y_col]]

        print(f"categories: {use_cats}")
        pyplot.hist(correct[self.score_col].values, alpha=0.5, label="correct")
        pyplot.hist(errors[self.score_col].values, alpha=0.5, label="errors")
        pyplot.legend(loc="upper right")
        pyplot.show()

def export_dict(file_name, dict_):
    """Serializes a dictionary into a JSON file."""
    with open(file_name, 'w') as f_json:
        json.dump(dict_, f_json, indent=4, sort_keys=True)

def load_dict(file_name):
    """Loads a dictionary saved in a JSON file."""
    with open(file_name) as json_file:
        key_value = json.load(json_file)
    return key_value