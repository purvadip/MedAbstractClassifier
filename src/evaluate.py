"""
evaluate.py
-----------
Model evaluation utilities: classification report, confusion matrix,
and one-vs-rest ROC curves. All figures are saved to *results_dir*.

Course  : SSIM916 — Problem Set #2: Using Text as Data
Student : 750091800
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

RESULTS_DIR = "results"


def evaluate_model(
    y_true,
    y_pred,
    y_prob,
    classes: list,
    model_name: str,
    results_dir: str = RESULTS_DIR,
) -> dict:
    """
    Compute and display evaluation metrics, save confusion matrix
    and ROC curve to *results_dir*, and return a summary dict.

    Parameters
    ----------
    y_true : array-like of str  (original string labels)
    y_pred : array-like of str  (predicted string labels)
    y_prob : np.ndarray shape (n_samples, n_classes)
    classes : list of str       (label names, e.g. ['Diagnosis', ...])
    model_name : str            (used in plot titles and file names)
    results_dir : str

    Returns
    -------
    dict with keys: model, accuracy, macro_f1, precision, recall, roc_auc
    """
    os.makedirs(results_dir, exist_ok=True)
    slug = model_name.lower().replace(" ", "_")

    # ── Classification report ───────────────────────────────────────────────
    report = classification_report(y_true, y_pred, target_names=classes,
                                   output_dict=True)
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print('='*60)
    print(classification_report(y_true, y_pred, target_names=classes))

    accuracy = report["accuracy"]
    macro_f1 = report["macro avg"]["f1-score"]
    precision = report["macro avg"]["precision"]
    recall = report["macro avg"]["recall"]

    # ── ROC-AUC (one-vs-rest, macro) ───────────────────────────────────────
    y_true_bin = label_binarize(y_true, classes=classes)
    try:
        roc_auc = roc_auc_score(y_true_bin, y_prob, multi_class="ovr",
                                average="macro")
    except ValueError:
        roc_auc = float("nan")

    # ── Confusion matrix ────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=14, pad=12)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    fig.tight_layout()
    cm_path = os.path.join(results_dir, f"confusion_matrix_{slug}.png")
    fig.savefig(cm_path, dpi=300)
    plt.close(fig)
    print(f"Confusion matrix saved → {cm_path}")

    # ── ROC curves (one-vs-rest) ────────────────────────────────────────────
    colours = ["aqua", "darkorange", "cornflowerblue"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (cls, colour) in enumerate(zip(classes, colours)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        area = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colour, lw=2,
                label=f"{cls} (AUC = {area:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"One-vs-Rest ROC Curve: {model_name}", fontsize=14, pad=12)
    ax.legend(loc="lower right")
    fig.tight_layout()
    roc_path = os.path.join(results_dir, f"roc_curve_{slug}.png")
    fig.savefig(roc_path, dpi=300)
    plt.close(fig)
    print(f"ROC curve saved      → {roc_path}")

    return {
        "model": model_name,
        "accuracy": round(accuracy * 100, 2),
        "macro_f1": round(macro_f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "roc_auc": round(roc_auc, 4),
    }
