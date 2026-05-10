"""
evaluation.py — Model evaluation with correctly labelled confusion matrix.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)

def evaluate(df, save_path="confusion_matrix.png"):
    y_true = df['label'].values
    y_pred = df['anomaly'].values

    print("\n[RESULT] Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Attack']))

    # AUC metrics if probability available
    if 'anomaly_prob' in df.columns:
        auc_roc = roc_auc_score(y_true, df['anomaly_prob'].values)
        auc_pr  = average_precision_score(y_true, df['anomaly_prob'].values)
        print(f"[RESULT] AUC-ROC: {auc_roc:.4f} | AUC-PR: {auc_pr:.4f}")

    # Confusion matrix with correct labels
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='viridis')
    plt.colorbar(im, ax=ax)

    classes = ['Normal', 'Attack']
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticklabels(classes, fontsize=12)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] < thresh else 'black',
                    fontsize=14, fontweight='bold')

    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved to {save_path}")
