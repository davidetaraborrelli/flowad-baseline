from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_roc(fpr, tpr, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC')
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR (recall)')
    plt.title('ROC curve')
    plt.legend()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def save_pr(precision, recall, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.plot(recall, precision, label='PR')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()