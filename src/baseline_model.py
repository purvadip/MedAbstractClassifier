"""
baseline_model.py
-----------------
TF-IDF pipeline construction and training for Logistic Regression
and Multinomial Naive Bayes classifiers.

Hyperparameters were chosen via 5-fold stratified GridSearchCV
optimising macro-F1:
    • Logistic Regression  →  C = 10.0
    • Multinomial NB       →  alpha = 0.05

Course  : SSIM916 — Problem Set #2: Using Text as Data
Student : 750091800
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.preprocess import balance_dataset, load_data, map_labels, reconstruct_abstracts

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = "data"
SPLIT_INDICES_PATH = "split_indices.json"
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.10  # fraction of the remaining training data used for validation

# TF-IDF shared settings (leakage-safe: heuristic keywords already stripped)
_TFIDF_KWARGS = dict(
    max_features=50_000,
    ngram_range=(1, 2),
    min_df=5,
    sublinear_tf=True,
    stop_words="english",
)


# ---------------------------------------------------------------------------
# Data split (serialised indices)
# ---------------------------------------------------------------------------

def load_split_data(
    data_dir: str = DATA_DIR,
    split_path: str = SPLIT_INDICES_PATH,
    target_size: int = 20_000,
    random_state: int = RANDOM_STATE,
):
    """
    Build (or reload) the preprocessed, balanced dataset and return
    deterministic train / test text arrays and label arrays.

    Split indices are serialised to *split_path* on first run so that
    subsequent runs use the identical held-out test set.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray (str), np.ndarray (str)
    """
    df_lines = load_data(data_dir)
    df_abs = reconstruct_abstracts(df_lines)
    df_labeled = map_labels(df_abs)
    df_balanced = balance_dataset(df_labeled, target_size=target_size,
                                  random_state=random_state)

    X = df_balanced["abstract_text"].values
    y = df_balanced["label"].values

    if os.path.exists(split_path):
        with open(split_path) as f:
            indices = json.load(f)
        train_idx = np.array(indices["train"])
        test_idx = np.array(indices["test"])
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=random_state,
        )
        all_idx = np.arange(len(X))
        train_idx = np.array([i for i in all_idx if X[i] in X_train])
        test_idx = np.array([i for i in all_idx if X[i] in X_test])
        with open(split_path, "w") as f:
            json.dump({"train": train_idx.tolist(), "test": test_idx.tolist()}, f)
        print(f"Split indices saved to {split_path}")

    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def train_logreg(
    X_train,
    y_train_enc,
    label_encoder: LabelEncoder,
    C: float = 10.0,
) -> Pipeline:
    """
    Build and fit a TF-IDF + Logistic Regression pipeline.

    Parameters
    ----------
    X_train : array-like of str
    y_train_enc : array-like of int
        Integer-encoded labels from a fitted LabelEncoder.
    label_encoder : LabelEncoder
        Fitted encoder (stored inside pipeline for inverse_transform later).
    C : float
        Inverse regularisation strength (optimal from GridSearchCV).

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(**_TFIDF_KWARGS)),
        ("clf", LogisticRegression(
            C=C,
            max_iter=2_000,
            solver="lbfgs",
            multi_class="multinomial",
            random_state=RANDOM_STATE,
        )),
    ])
    pipe.fit(X_train, y_train_enc)
    print(f"Logistic Regression trained  (C={C})")
    return pipe


def train_nb(
    X_train,
    y_train_enc,
    label_encoder: LabelEncoder,
    alpha: float = 0.05,
) -> Pipeline:
    """
    Build and fit a TF-IDF + Multinomial Naive Bayes pipeline.

    Parameters
    ----------
    X_train : array-like of str
    y_train_enc : array-like of int
    label_encoder : LabelEncoder
    alpha : float
        Laplace smoothing parameter (optimal from GridSearchCV).

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(**_TFIDF_KWARGS)),
        ("clf", MultinomialNB(alpha=alpha)),
    ])
    pipe.fit(X_train, y_train_enc)
    print(f"Multinomial NB trained  (alpha={alpha})")
    return pipe
