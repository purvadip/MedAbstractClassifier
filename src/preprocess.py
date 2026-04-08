"""
preprocess.py
-------------
Data loading, abstract reconstruction, weak-supervision labelling,
keyword stripping (leakage prevention), and stratified undersampling
for the PubMed 20k RCT Medical Abstract Classification project.

Course  : SSIM916 — Problem Set #2: Using Text as Data
Student : 750091800
"""

import os
import re
import json
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Keyword vocabularies (shared with biobert_colab.ipynb)
# ---------------------------------------------------------------------------
TREATMENT_KW = [
    "intervention", "trial", "therapy", "drug", "dose",
    "randomised", "placebo", "administered", "efficacy",
]
DIAGNOSIS_KW = [
    "screening", "diagnostic", "sensitivity", "specificity",
    "test", "accuracy", "imaging", "biopsy", "detection",
]
PREVENTION_KW = [
    "vaccine", "prevention", "prophylaxis", "risk reduction",
    "protective", "immunisation", "incidence",
]

# Build one compiled regex covering every heuristic keyword (+ class names)
_ALL_KWS = TREATMENT_KW + DIAGNOSIS_KW + PREVENTION_KW + [
    "diagnosis", "treatment", "prevention",
]
_LEAKAGE_PATTERN = re.compile(
    r"\b(" + "|".join(map(re.escape, _ALL_KWS)) + r")\b",
    flags=re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Load all available CSV splits (train / dev / test) from *data_dir*
    and return a single concatenated DataFrame.

    Parameters
    ----------
    data_dir : str
        Directory containing train.csv, dev.csv and/or test.csv.

    Returns
    -------
    pd.DataFrame
        Columns: abstract_id, line_number, abstract_text (at minimum).
    """
    dfs = []
    for split in ("train.csv", "dev.csv", "test.csv"):
        path = os.path.join(data_dir, split)
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
    if not dfs:
        raise FileNotFoundError(
            f"No CSV splits found in '{data_dir}'. "
            "Place train.csv / dev.csv / test.csv there before running."
        )
    df = pd.concat(dfs, ignore_index=True)
    return df


def reconstruct_abstracts(df_lines: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct per-abstract full texts by concatenating sentence lines
    ordered by line_number within each abstract_id.

    Parameters
    ----------
    df_lines : pd.DataFrame
        Row-per-sentence DataFrame with columns: abstract_id,
        line_number, abstract_text.

    Returns
    -------
    pd.DataFrame
        One row per abstract with columns: abstract_id, abstract_text.
    """
    df_sorted = df_lines.sort_values(["abstract_id", "line_number"])
    df_abs = (
        df_sorted
        .groupby("abstract_id")["abstract_text"]
        .apply(lambda x: " ".join(x.astype(str)))
        .reset_index()
    )
    return df_abs


def _assign_label(text_lower: str) -> str:
    """Return the weak-supervision label for a lower-cased abstract."""
    counts = {
        "Treatment": sum(text_lower.count(kw) for kw in TREATMENT_KW),
        "Diagnosis": sum(text_lower.count(kw) for kw in DIAGNOSIS_KW),
        "Prevention": sum(text_lower.count(kw) for kw in PREVENTION_KW),
    }
    max_count = max(counts.values())
    if max_count == 0:
        return "Treatment"  # default for no-keyword abstracts
    top = [k for k, v in counts.items() if v == max_count]
    return "Treatment" if len(top) > 1 else top[0]


def map_labels(df_abs: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the keyword-frequency heuristic to assign weak labels, then
    strip those keywords from every abstract to prevent data leakage.

    Parameters
    ----------
    df_abs : pd.DataFrame
        Must contain column 'abstract_text'.

    Returns
    -------
    pd.DataFrame
        Original columns + 'label' (str) and 'abstract_text' with
        heuristic keywords removed.
    """
    labels, cleaned = [], []
    for text in df_abs["abstract_text"]:
        labels.append(_assign_label(text.lower()))
        clean = _LEAKAGE_PATTERN.sub("", text)
        clean = re.sub(r"\s+", " ", clean).strip()
        cleaned.append(clean)
    df_out = df_abs.copy()
    df_out["label"] = labels
    df_out["abstract_text"] = cleaned
    return df_out


def balance_dataset(
    df_labeled: pd.DataFrame,
    target_size: int = 20_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Stratified undersampling (with replacement when a class is smaller
    than its quota) to produce a class-balanced corpus.

    Parameters
    ----------
    df_labeled : pd.DataFrame
        Must contain column 'label'.
    target_size : int
        Total number of rows in the balanced dataset.
    random_state : int
        NumPy seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Shuffled, balanced DataFrame.
    """
    classes = df_labeled["label"].unique()
    samples_per_class = target_size // len(classes)
    remainder = target_size % len(classes)

    balanced = []
    for i, cls in enumerate(classes):
        n = samples_per_class + (1 if i < remainder else 0)
        subset = df_labeled[df_labeled["label"] == cls]
        balanced.append(
            subset.sample(n=n, replace=(n > len(subset)), random_state=random_state)
        )

    df_balanced = (
        pd.concat(balanced)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )
    return df_balanced
