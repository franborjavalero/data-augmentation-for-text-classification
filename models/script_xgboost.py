import argparse
import fasttext.util
from hyperopt import STATUS_OK, hp, tpe, Trials, fmin
import os
from nltk import word_tokenize
from nltk.corpus import stopwords
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, precision_score, recall_score, f1_score, accuracy_score
import xgboost as xgb


def load_data(args):
    """ Load data from tsv into pd.DataFrame """
    df_train = pd.read_csv(os.path.join(args.data_dir, args.training_file), delimiter="\t")
    df_val = df_train[df_train[args.validation_col]]
    df_train = df_train[~df_train[args.validation_col]].sample(frac=1, random_state=args.random_seed).reset_index(drop=True)
    df_test = pd.read_csv(os.path.join(args.data_dir, args.testing_file), delimiter="\t")
    return df_train, df_val, df_test

def sent2vec(sentence, word_embedding, stop_words):
    sentence = sentence.lower()
    words = word_tokenize(sentence) # use another tokenizer for fastext embeddings?
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    num_words = len(words)
    d = int(word_embedding.get_dimension())
    M = np.zeros((num_words, d), dtype=np.float32)
    for i in range(num_words):
        try:
            M[i,:] = word_embedding.get_word_vector(words[i])
        except:
            continue
    v = M.sum(axis=0) # [d]
    if type(v) != np.ndarray:
        return np.zeros(d)
    return v / np.sqrt((v ** 2).sum())

def compute_metrics(clf, features, y):
    predictions = clf.predict(features)
    acs = accuracy_score(y, predictions)
    ps = precision_score(y, predictions, average='macro')
    rs = recall_score(y, predictions, average='macro')
    f1s = f1_score(y, predictions, average='macro')
    return acs, ps, rs, f1s


def main():

    stop_words_aux = frozenset({'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've',
    'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
    "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", "'s", "'re"})

    stop_words = frozenset(stopwords.words('english')) | stop_words_aux

    params = {
        'n_estimators': hp.choice('n_estimators', [50, 100, 200, 250, 500]),
        'colsample_bytree': hp.choice('colsample_bytree', [0.6, 0.8, 1.0]),
        'subsample': hp.choice('subsample', [0.6, 0.8, 1.0]),
        'max_depth': hp.choice('max_depth', [3, 4, 5, 7]),
        'gamma': hp.choice('gamma', [0.5, 1, 1.5, 2, 5]),
        'learning_rate': hp.choice('learning_rate', [0.1, 0.01, 0.05]),
    }

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default=None,
        help="The input data dir. Should contain the .tsv files for the task.",
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
        "--random_seed",
        default=42,
        type=int,
        help="fix random seed for reproducibility",
    )

    parser.add_argument(
        "--nthread",
        default=4,
        type=int,
        help="Number of parallel threads used to run XGBoost",
    )

    parser.add_argument(
        "--log_name",
        default="logging",
        type=str,
        help="The name of the log file",
    )

    args = parser.parse_args()

    # make sure export directory exists
    if not os.path.exists(args.export_dir):
        os.makedirs(args.export_dir)

    # Setup logging
    logging.basicConfig(
        filename=os.path.join(args.export_dir, f"{args.log_name}.log"),
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Load data
    logging.info("Loading data ...")
    df_train, df_val, df_test = load_data(args)

    # Compute sentence Word embeddings
    logging.info("Computing sentence word embeddings...")
    fasttext.util.download_model('en', if_exists='ignore')  # English
    word_embeddings = fasttext.load_model('cc.en.300.bin')
    train_features = np.array([sent2vec(example, word_embeddings, stop_words) for example in df_train[args.text_col].to_list()])
    train_y = df_train[args.label_col].to_numpy()
    val_features = np.array([sent2vec(example, word_embeddings, stop_words) for example in df_val[args.text_col].to_list()])
    val_y = df_val[args.label_col].to_numpy()
    test_features = np.array([sent2vec(example, word_embeddings, stop_words) for example in df_test[args.text_col].to_list()])
    test_y = df_test[args.label_col].to_numpy()

    # Fitting a simple XGBoost on sentence word embeddings
    logging.info("Training XGBoost...")
        
    def objective_xgboost(space):
        clf = xgb.XGBClassifier(
            max_depth=space['max_depth'],
            n_estimators=space['n_estimators'],
            colsample_bytree=space['colsample_bytree'],
            subsample=space['subsample'],
            gamma=space['gamma'],
            learning_rate=space['learning_rate'],
            nthread=args.nthread, 
            seed=args.random_seed,
        )
        clf.fit(train_features, train_y)
        prob_hipotesys = clf.predict_proba(val_features)
        loss = log_loss(val_y, prob_hipotesys)
        return {'loss':loss, 'status': STATUS_OK, 'clf': clf}
        
    xgboost_trials = Trials()
    
    fmin(fn=objective_xgboost, space=params, algo=tpe.suggest,verbose=False,
        max_evals=1, trials=xgboost_trials, rstate=np.random.RandomState(args.random_seed))
    
    data_evaluation = [
        ("Train", train_features, train_y),
        ("Validation", val_features, val_y),
        ("Test", test_features, test_y),
    ]
    
    for (section, X, y) in data_evaluation:
        acs, ps, rs, f1s = compute_metrics(xgboost_trials.results[0]['clf'], X, y)
        logging.info(f"{section} set: \n\taccuracy: {acs:.3f}\n\tprecision: {ps:.3f}\n\trecall: {rs:.3f}\n\tf1: {f1s:.3f}")

    
if __name__ == "__main__":
    main()