from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def make_synthetic_flows(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # fake features: duration, bytes, packets, protocol
    duration = rng.gamma(shape=2.0, scale=2.0, size=n)
    bytes_fwd = rng.lognormal(mean=8.5, sigma=1.0, size=n)
    bytes_bwd = rng.lognormal(mean=8.0, sigma=1.1, size=n)
    packets_fwd = rng.poisson(lam=30, size=n)
    packets_bwd = rng.poisson(lam=25, size=n)
    protocol = rng.choice(['TCP','UDP','ICMP'], size=n, p=[0.7,0.25,0.05])
    # label: "attacks" have slightly different patterns
    score = 0.002*bytes_fwd + 0.003*bytes_bwd + 0.2*(protocol=='ICMP') + 0.03*packets_fwd
    p_attack = 1 / (1 + np.exp(-(score - np.percentile(score, 80))/5))
    y = (rng.random(n) < p_attack).astype(int)
    df = pd.DataFrame({
        'Duration': duration,
        'BytesFwd': bytes_fwd,
        'BytesBwd': bytes_bwd,
        'PktsFwd': packets_fwd,
        'PktsBwd': packets_bwd,
        'Protocol': protocol,
        'Label': np.where(y==1, 'ATTACK', 'BENIGN')
    })
    return df

def load_csv(csv_path: str, max_rows: Optional[int]=None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)
    return df

def prepare_features(
    df: pd.DataFrame,
    label_col: str = 'Label',
    drop_cols: Optional[List[str]] = None,
    categorical: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    df = df.copy()
    if drop_cols:
        for c in drop_cols:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)

    y = df[label_col].copy()
    X = df.drop(columns=[label_col])

    cat_cols = [c for c in (categorical or []) if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('num', Pipeline([('scaler', StandardScaler())]), num_cols),
        ]
    )
    return X, y, pre

def train_test_split_df(
    X, y, test_size=0.3, stratify=True, random_state=42
):
    strat = y if stratify else None
    return train_test_split(X, y, test_size=test_size, stratify=strat, random_state=random_state)