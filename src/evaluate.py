import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import os
import numpy as np

def evaluate_model(y_true, y_pred, y_prob, classes, model_name, output_dir):
    """
    Evaluates a model and saves the confusion matrix and ROC curve.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities for each class.
        classes: List of class names.
        model_name: Name of the model (for titles and filenames).
        output_dir: Directory to save figures.
        
    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"\\n--- {model_name} Evaluation ---")
    print(classification_report(y_true, y_pred, target_names=classes))
    
    metrics = {
        'Model': model_name,
        'Accuracy': acc,
        'Precision (Macro)': prec_macro,
        'Recall (Macro)': rec_macro,
        'F1 (Macro)': f1_macro,
        'F1 (Weighted)': f1_weighted
    }
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. ROC-AUC One-vs-Rest
    y_true_bin = label_binarize(y_true, classes=classes)
    n_classes = len(classes)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.figure(figsize=(8, 6))
    colors = ['aqua', 'darkorange', 'cornflowerblue']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:.2f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'One-vs-Rest ROC Curve: {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(output_dir, f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Add mean ROC-AUC to metrics
    mean_auc = np.mean(list(roc_auc.values())[:-1]) # excluding micro
    metrics['ROC-AUC (OVR)'] = mean_auc
    
    return metrics
