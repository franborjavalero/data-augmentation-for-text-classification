import argparse
import logging
from collections import namedtuple
from functools import partial
from pathlib import Path
import os

import pandas as pd
import torch
from fastai.basic_train import Learner
from fastai.callbacks.tracker import EarlyStoppingCallback, SaveModelCallback
from fastai.metrics import accuracy
from fastai.text.data import NumericalizeProcessor, TextList, TokenizeProcessor
from fastai.text.transform import Tokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import wandb
from wandb.fastai import WandbCallback

from utils import (
    ClassificationEval,
    CustomTransformerModel,
    TransformersBaseTokenizer,
    TransformersVocab,
    seed_all,
    export_dict,
    load_dict,
)

# fix pandas warnings
pd.set_option("mode.chained_assignment", None)

logger = logging.getLogger(__name__)

MODEL_TYPES = ["roberta", "bert"]
MODEL_NAMES = [
    "distilroberta-base",
    "roberta-base",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-cased",
]

Prediction = namedtuple(
    "Prediction", ["label", "confidence", "logit_scores"], rename=False,
)

Metrics = namedtuple(
    "Metrics", ["accuracy", "f1", "precision", "recall"], rename=False,
)


def load_data(args) -> pd.DataFrame:
    """ Load data from tsv into pd.DataFrame """

    df = pd.read_csv(os.path.join(args.data_dir, args.training_file), delimiter="\t").sample(frac=1, random_state=args.random_seed).reset_index(drop=True)
    df_test = pd.read_csv(os.path.join(args.data_dir, args.testing_file), delimiter="\t")

    # startified validation split
    if not args.use_custom_split:
        train_df, valid_df = train_test_split(
            df, stratify=df[args.label_col], test_size=args.split_size
        )
        # add is_valid column
        train_df[args.validation_col] = False
        valid_df[args.validation_col] = True
        df = pd.concat([train_df, valid_df]).reset_index(drop=True)
        # free up memory
        del train_df, valid_df

    return df, df_test


def create_learner(args, df: pd.DataFrame) -> Learner:
    """ create fastai Learner object """

    # Load tokenizer, model and config from Hugging Face's Transformers
    # Tokenizer
    transformer_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
    # Config
    transformer_config = AutoConfig.from_pretrained(args.pretrained_model_name)
    transformer_config.num_labels = df[args.label_col].nunique()
    # Model
    transformer_model = AutoModelForSequenceClassification.from_pretrained(
        args.pretrained_model_name, config=transformer_config
    )
    if args.anonymized_tokens:
        special_tokens_dict = {
            'additional_special_tokens': ['xxnum','xxphone', 'xxname', 'xxemail', 'xxurl'],
        }
        transformer_tokenizer.add_special_tokens(special_tokens_dict)
        transformer_model.resize_token_embeddings(len(transformer_tokenizer)) 
    # save tokenizer for later
    transformer_tokenizer.save_pretrained(args.export_dir)

    # custom glue for transformers/fastai tokenizer
    # this is a hack to use Hugging Face's Transformers inside Fast.ai
    transformer_base_tokenizer = TransformersBaseTokenizer(
        pretrained_tokenizer=transformer_tokenizer,
        model_type=args.pretrained_model_type,
    )
    fastai_tokenizer = Tokenizer(
        tok_func=transformer_base_tokenizer, pre_rules=[], post_rules=[]
    )
    transformer_vocab = TransformersVocab(tokenizer=transformer_tokenizer)
    numericalize_processor = NumericalizeProcessor(vocab=transformer_vocab)
    tokenize_processor = TokenizeProcessor(
        tokenizer=fastai_tokenizer, include_bos=False, include_eos=False
    )
    transformer_processor = [tokenize_processor, numericalize_processor]

    # Create Fastai Databunch
    databunch = (
        TextList.from_df(
            df, cols=args.text_col, processor=transformer_processor, path=args.cache_dir
        )
        .split_from_df(col=args.validation_col)
        .label_from_df(cols=args.label_col)
        .databunch(
            bs=args.batch_size,
            pad_first=args.pad_first,
            pad_idx=transformer_tokenizer.pad_token_id,
        )
    )

    # Wrap Transformer Model into Custom Glue
    custom_transformer_model = CustomTransformerModel(
        transformer_model=transformer_model
    )

    # wrap optimizer into partial to work with fastai training loop
    CustomAdamW = partial(AdamW, correct_bias=False)

    learner = Learner(
        databunch,
        custom_transformer_model,
        opt_func=CustomAdamW,
        callback_fns=[
            partial(
                EarlyStoppingCallback,
                monitor="valid_loss",
                min_delta=args.min_delta,
                patience=args.pacience,
            ),
        ],
        metrics=[accuracy],
    )

    if args.track_with_wb:
        # wrap callback into partial to work with fastai training loop
        CustomWandbCallback = partial(WandbCallback, seed=args.random_seed)
        learner.callback_fns.append(CustomWandbCallback)

    # Train in FP16 precision mode.
    # TODO test export and inference on CPU
    if args.use_fp16:
        learner = learner.to_fp16()

    return learner


def split_model(args, learner: Learner) -> (Learner, int):
    """" Define split points for discriminative fine-tuning
    and  gradual unfreezing.
    ULMFiT (https://arxiv.org/abs/1801.06146) """

    if args.pretrained_model_name in ["distilroberta-base", "roberta-base"]:
        model_layers = learner.model.transformer.roberta

    if args.pretrained_model_name in [
        "bert-base-multilingual-uncased",
        "bert-base-multilingual-cased",
        "bert-base-cased",
    ]:
        model_layers = learner.model.transformer.bert

    list_layers = (
        [model_layers.embeddings]
        + [layer for layer in model_layers.encoder.layer]
        + [model_layers.pooler]
    )

    learner.split(list_layers)
    num_groups = len(learner.layer_groups)
    # print("Learner split in", num_groups, "groups")
    # print(learner.layer_groups)
    return learner, num_groups


def train(args, learner: Learner) -> None:
    """ Train model """

    # Define split points
    learner, num_groups = split_model(args, learner)

    # first cycle
    learner.freeze_to(-1)
    learner.callback_fns.append(
        partial(SaveModelCallback, every="improvement", name="first_cycle")
    )
    lr = 1e-04
    learner.fit_one_cycle(10, max_lr=lr, moms=(0.8, 0.7))
    learner.load("first_cycle")

    # second cycle
    learner.freeze_to(-2)
    # update SaveModelCallback filename
    learner.callback_fns[-1] = partial(
        SaveModelCallback, every="improvement", name="second_cycle"
    )
    lr = 1e-5
    learner.fit_one_cycle(10, max_lr=slice(lr * 0.95 ** num_groups, lr), moms=(0.8, 0.9))
    learner.load("second_cycle")

    # third cycle
    learner.freeze_to(-3)
    # update SaveModelCallback filename
    learner.callback_fns[-1] = partial(
        SaveModelCallback, every="improvement", name="third_cycle"
    )
    lr = 1e-5
    learner.fit_one_cycle(10, max_lr=slice(lr * 0.95 ** num_groups, lr), moms=(0.8, 0.9))
    learner.load("third_cycle")

    # unfreeze
    learner.unfreeze()
    # update SaveModelCallback filename
    learner.callback_fns[-1] = partial(
        SaveModelCallback, every="improvement", name="fourth_cycle"
    )
    lr = 1e-5
    learner.fit_one_cycle(10, max_lr=slice(lr * 0.95 ** num_groups, lr), moms=(0.8, 0.9))
    learner.load("fourth_cycle")


def evaluate(
    args, learner: Learner, df: pd.DataFrame, save_preds: bool = True, section: str = "training"
) -> pd.DataFrame:
    """ Evaluate trained model """

    logger.info("Get predictions for evaluation ...")

    # get all predictions
    preds = []
    for text in tqdm(df[args.text_col].values):
        raw_pred = learner.predict(text)
        prediction = Prediction(
            label=raw_pred[0].obj,
            confidence=round(raw_pred[2][raw_pred[1]].item(), 3),
            logit_scores=raw_pred[2].numpy(),
        )
        preds.append(prediction)

    df["prediction"] = [x.label for x in preds]
    df["confidence"] = [x.confidence for x in preds]
    df["logit_scores"] = [x.logit_scores for x in preds]

    if save_preds:
        df.to_pickle(os.path.join(args.export_dir, f"data_evaluated_{section}.pkl"))

    return df


def get_metrics(args, df: pd.DataFrame, save_metrics: bool = True, section: str = "training") -> Metrics:
    """ Calculate metrics for section set  """

    evaluator = ClassificationEval(
        df, x_col=args.text_col, y_col=args.label_col,
    )

    metrics = Metrics(
        accuracy=evaluator.accuracy,
        f1=evaluator.f1,
        precision=evaluator.precision,
        recall=evaluator.recall,
    )
    
    if save_metrics:
        export_dict(os.path.join(args.export_dir, f"metrics_{section}.json"), metrics._asdict())

    return metrics


def export_model(args, learner: Learner) -> None:
    # extract model
    model = learner.model.transformer
    # add label / id mapping
    model.config.id2label = {i: x for i, x in enumerate(learner.data.classes)}
    model.config.label2id = {x: i for i, x in enumerate(learner.data.classes)}
    # save model
    model.save_pretrained(args.export_dir)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--pretrained_model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--pretrained_model_name",
        default=None,
        type=str,
        required=True,
        help="Model name selected in the list: " + ", ".join(MODEL_NAMES),
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=lambda p: Path(p).absolute(),
        help="The input data dir. Should contain the .csv files for the task.",
    )
    parser.add_argument(
        "--training_file",
        default="train.tsv",
        type=str,
        help="The name of the tsv file containing the training data",
    )

    parser.add_argument(
        "--testing_file",
        default="test.tsv",
        type=str,
        help="The name of the tsv file containing the testing data",
    )

    # Other parameters
    parser.add_argument(
        "--batch_size", default=12, type=int, help="batch size for training",
    )
    parser.add_argument(
        "--cache_dir",
        default="./cache",
        help="The cache dir, where to save intermediate output files",
    )
    parser.add_argument(
        "--export_dir",
        default="./export",
        help="The cache dir, where to save intermediate model output files",
    )
    parser.add_argument(
        "--text_col",
        default="text",
        type=str,
        help="The name of the column in the csv containing the text to be classified",
    )
    parser.add_argument(
        "--label_col",
        default="label",
        type=str,
        help="The name of the column in the csv containing the label to be predicted",
    )
    parser.add_argument(
        "--validation_col",
        default="is_valid",
        type=str,
        help="The name of the column containing a bool to delect validation data",
    )
    parser.add_argument(
        "--use_custom_split",
        action="store_true",
        help="Set this flag if a train/validation split already exists.",
    )
    parser.add_argument(
        "--split_size",
        default=0.2,
        type=float,
        help="Ratio of the validation set with respect to the training set",
    )
    parser.add_argument(
        "--random_seed",
        default=42,
        type=int,
        help="fix random seed for reproducibility",
    )
    parser.add_argument(
        "--pad_first",
        action="store_true",
        help="Set this flag to left pad sequence; set to true for xlnet",
    )
    parser.add_argument(
        "--use_fp16", action="store_true", help="Set this flag to use fp16 precision"
    )
    parser.add_argument(
        "--min_delta",
        default=0.01,
        type=float,
        help="minimum improvement to avoid early stopping",
    )
    parser.add_argument(
        "--pacience",
        default=3,
        type=int,
        help="number of epochs to keep going without improvement before early stopping",
    )

    parser.add_argument(
        "--track_with_wb",
        action="store_true",
        help="Set this flag to track the experiment with Weights & Biases",
    )

    parser.add_argument(
        "--project_name_wb",
        default="intent-classification",
        type=str,
        help="The name of the wanb project",
    )

    parser.add_argument(
        "--anonymized_tokens",
        action="store_true",
        help="Set this flag to consider the anonymized tokens",
    )

    parser.add_argument(
        "--export_model",
        action="store_true",
        help="Set this flag to export the model",
    )

    parser.add_argument(
        "--labels_file",
        default=None,
        type=str,
        help="file name with the mappign between identifier and label",
    )

    args = parser.parse_args()

    # make sure export directory exists
    if not os.path.exists(args.export_dir):
        os.makedirs(args.export_dir)
    
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    # Setup logging
    logging.basicConfig(
        filename=os.path.join(args.export_dir, f"{args.project_name_wb}.log"),
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # setup tracking with weights&biases
    if args.track_with_wb:
        wandb.init(project=args.project_name_wb)

    # Make sure GPU is available
    if not torch.cuda.is_available():
        logger.warning("No GPU Training Available")

    # Set seed
    seed_all(args.random_seed)

    # Load data
    df_train_val, df_test = load_data(args)
    id_label = load_dict(args.labels_file)
    df_train_val[args.label_col] = df_train_val[args.label_col].apply(lambda id_: id_label[str(id_)])
    df_test[args.label_col] = df_test[args.label_col].apply(lambda id_: id_label[str(id_)])

    # Create fastai learner
    learner = create_learner(args, df_train_val)

    # Train
    train(args, learner)

    # Evaluate
    df_train = df_train_val[~df_train_val[args.validation_col]]
    df_valid = df_train_val[df_train_val[args.validation_col]]

    df_train_eval = evaluate(args, learner, df_train)
    df_valid_eval = evaluate(args, learner, df_valid, section="validation")
    df_test_eval = evaluate(args, learner, df_test, section="test")

    # Get metrics
    train_metrics = get_metrics(args, df_train_eval)
    valid_metrics = get_metrics(args, df_valid_eval, section="validation")
    test_metrics = get_metrics(args, df_test_eval, section="test")
    
    logger.info(f"Results on training data:\n\taccuracy: {train_metrics.accuracy:.3f}\n\tf1: {train_metrics.f1:.3f}\n\tprecision: {train_metrics.precision:.3f}\n\trecall: {train_metrics.recall:.3f}")
    logger.info(f"Results on validation data:\n\taccuracy: {valid_metrics.accuracy:.3f}\n\tf1: {valid_metrics.f1:.3f}\n\tprecision: {valid_metrics.precision:.3f}\n\trecall: {valid_metrics.recall:.3f}")
    logger.info(f"Results on testing data:\n\taccuracy: {test_metrics.accuracy:.3f}\n\tf1: {test_metrics.f1:.3f}\n\tprecision: {test_metrics.precision:.3f}\n\trecall: {test_metrics.recall:.3f}")

    # Save and export model
    if args.export_model:
        export_model(args, learner)


if __name__ == "__main__":
    main()
