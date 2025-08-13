from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline

from data import make_synthetic_flows, load_csv, prepare_features, train_test_split_df
from model import get_supervised_models, get_unsupervised_models
from metrics import compute_metrics, threshold_at_tpr, pr_curve_values, roc_curve_values
from plot import save_pr, save_roc

def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def binarize_labels(y, positive_label='ATTACK'):
    lb = LabelBinarizer(pos_label=1, neg_label=0)
    y_bin = lb.fit_transform((y == positive_label).astype(int)).ravel()
    return y_bin

def run_fast_demo(cfg):
    print('[fast] Genero dati sintetici...')
    df = make_synthetic_flows(n=5000, seed=cfg['random_seed'])
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--fast', action='store_true', help='Usa dati sintetici per una demo rapida')
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Load data
    if args.fast:
        df = run_fast_demo(cfg)
    else:
        csv_path = cfg['data']['csv_path']
        max_rows = cfg['data'].get('max_rows')
        print(f'Carico CSV: {csv_path} (max_rows={max_rows})')
        df = load_csv(csv_path, max_rows=max_rows)

    # Prepare features and split
    X, y_raw, pre = prepare_features(
        df,
        label_col=cfg['data']['label_col'],
        drop_cols=cfg['features'].get('drop_cols'),
        categorical=cfg['features'].get('categorical')
    )
    y = binarize_labels(y_raw, positive_label=cfg['data']['positive_label'])

    X_train, X_test, y_train, y_test = train_test_split_df(
        X, y, test_size=cfg['data']['test_size'], stratify=cfg['data']['stratify'], random_state=cfg['random_seed']
    )

    # Supervised
    metrics_all = {}
    thresholds_all = {}

    sup_models = get_supervised_models(n_jobs=cfg['training']['n_jobs'])
    for name, clf in sup_models.items():
        pipe = Pipeline(steps=[('pre', pre), ('clf', clf)])
        print(f'[supervised] Fitting {name}...')
        pipe.fit(X_train, y_train)
        # scoring: use predict_proba if available, else decision_function
        if hasattr(pipe.named_steps['clf'], 'predict_proba'):
            y_score = pipe.predict_proba(X_test)[:, 1]
        elif hasattr(pipe.named_steps['clf'], 'decision_function'):
            y_score = pipe.decision_function(X_test)
        else:
            y_score = pipe.predict(X_test)

        m = compute_metrics(y_test, y_score)
        thr, pt = threshold_at_tpr(y_test, y_score, target_tpr=0.9)
        metrics_all[f'supervised_{name}'] = m
        thresholds_all[f'supervised_{name}'] = {'threshold': thr, **pt}

        # curves once (for the last model is fine in the baseline)
        fpr, tpr, _ = roc_curve_values(y_test, y_score)
        save_roc(fpr, tpr, cfg['output']['roc_curve_png'])
        precision, recall, _ = pr_curve_values(y_test, y_score)
        save_pr(precision, recall, cfg['output']['pr_curve_png'])

    # Unsupervised: train on BENIGN-only samples from the training set
    benign_mask = (y_train == 0)
    X_train_benign = X_train[benign_mask]

    unsup_models = get_unsupervised_models(n_jobs=cfg['training']['n_jobs'])
    for name, model in unsup_models.items():
        pipe = Pipeline(steps=[('pre', pre), ('clf', model)])
        print(f'[unsupervised] Fitting {name} su soli BENIGN...')
        pipe.fit(X_train_benign)

        # scoring: IsolationForest -> score_samples (higher = more normal); invert to get anomaly score
        clf = pipe.named_steps['clf']
        if hasattr(clf, 'decision_function'):
            score = pipe.decision_function(X_test)
            y_score = -score  # alto = anomalo
        elif hasattr(clf, 'score_samples'):
            y_score = -pipe.score_samples(X_test)  # inverti: alto = anomalo
        else:
            # fallback
            y_pred = pipe.predict(X_test)  # 1 o -1
            y_score = (y_pred == -1).astype(int)

        m = compute_metrics(y_test, y_score)
        thr, pt = threshold_at_tpr(y_test, y_score, target_tpr=0.9)
        metrics_all[f'unsupervised_{name}'] = m
        thresholds_all[f'unsupervised_{name}'] = {'threshold': thr, **pt}

    os.makedirs(os.path.dirname(cfg['output']['metrics_json']), exist_ok=True)
    with open(cfg['output']['metrics_json'], 'w') as f:
        json.dump(metrics_all, f, indent=2)

    with open(cfg['output']['thresholds_json'], 'w') as f:
        json.dump(thresholds_all, f, indent=2)

    print('Done. Metrics saved to', cfg['output']['metrics_json'])

if __name__ == '__main__':
    main()