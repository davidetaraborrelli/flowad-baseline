from __future__ import annotations
from typing import Dict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM

def get_supervised_models(n_jobs: int = 1) -> Dict[str, object]:
    # Higher max_iter avoids ConvergenceWarning for the demo
    return {
        'logreg': LogisticRegression(max_iter=500),  # solver predefinito (lbfgs), va bene per la demo
        'rf': RandomForestClassifier(n_estimators=100, n_jobs=n_jobs, random_state=42),
    }

def get_unsupervised_models(n_jobs: int = 1) -> Dict[str, object]:
    return {
        'ocsvm': OneClassSVM(gamma='auto'),
        'iforest': IsolationForest(n_estimators=100, n_jobs=n_jobs, random_state=42),
    }
