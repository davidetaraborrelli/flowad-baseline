from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

def compute_metrics(y_true, y_score) -> Dict[str, float]:
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    return {'roc_auc': float(roc_auc), 'pr_auc': float(pr_auc)}

def threshold_at_tpr(y_true, y_score, target_tpr=0.9) -> Tuple[float, Dict[str, float]]:
    fpr, tpr, thresh = roc_curve(y_true, y_score)
    # find the threshold with TPR closest to the target
    idx = np.argmin(np.abs(tpr - target_tpr))
    thr = thresh[idx]
    return float(thr), {'tpr': float(tpr[idx]), 'fpr': float(fpr[idx])}

def pr_curve_values(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    return precision, recall, thresholds

def roc_curve_values(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return fpr, tpr, thresholds