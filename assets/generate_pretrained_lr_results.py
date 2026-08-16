"""Pre-compute the Model Training tab's default preview.

The Model Training tab shows a result immediately on load (before the user
clicks "Train model"), sourced from model/pretrained_lr_results.pkl. This
script (re)generates that file using the *exact* code path and default
hyperparameters the tab itself uses (Logistic Regression, C=1.0,
max_iter=1000, random_state=42, 80/20 split, random_state=42), trained on
the current test_data.csv — so the default preview matches what clicking
"Train model" with the default sliders would produce, and stays consistent
with the ~0.76 accuracy shown elsewhere (Model Comparison, Diagnostics).

Run this locally whenever test_data.csv changes, then commit the updated
model/pretrained_lr_results.pkl:

    python model/generate_pretrained_lr_results.py
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score,
)

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / 'model'


def normalize_col(col_name: str) -> str:
    return str(col_name).strip().replace('﻿', '').replace('\t', ' ').strip()


def compute_metrics(y_true, y_pred, y_proba, classes):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro', labels=classes)
    except Exception:
        auc = None
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'mcc': mcc, 'auc': auc}


def main():
    schema = json.loads((MODEL_DIR / 'schema.json').read_text(encoding='utf8'))

    df = pd.read_csv(ROOT / 'test_data.csv', sep=None, engine='python', encoding='utf-8-sig')
    df.columns = [normalize_col(c) for c in df.columns]

    raw_feature_columns = [normalize_col(c) for c in schema['feature_columns']]
    target_col = normalize_col(schema.get('target_column', 'Target'))
    nominal = [c for c in raw_feature_columns if normalize_col(c) in
               {normalize_col(x) for x in schema.get('nominal_columns', [])}]
    numeric = [c for c in raw_feature_columns if c not in nominal]

    missing = [c for c in raw_feature_columns + [target_col] if c not in df.columns]
    if missing:
        raise SystemExit(f'test_data.csv is missing required columns: {missing}')

    X_tr = df[raw_feature_columns].copy()
    y_tr = df[target_col]

    pre = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), numeric),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                           ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), nominal),
    ], remainder='drop')

    # Default slider values from the Model Training tab: C=1.0, max_iter=1000.
    est = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf_demo = Pipeline([('pre', pre), ('est', est)])

    min_cls = y_tr.value_counts().min()
    strat = y_tr if min_cls >= 2 else None
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42, stratify=strat)
    clf_demo.fit(X_tr2, y_tr2)
    y_p2 = clf_demo.predict(X_te2)
    try:
        y_pb2 = clf_demo.predict_proba(X_te2)
    except Exception:
        y_pb2 = None

    classes2 = clf_demo.classes_
    m2 = compute_metrics(
        y_te2, y_p2,
        y_pb2 if y_pb2 is not None else np.zeros((len(y_p2), len(classes2))),
        classes=classes2,
    )

    result = {
        'metrics': m2,
        'y_test': y_te2,
        'y_pred': y_p2,
        'classes': classes2,
        'estimator': clf_demo.named_steps['est'],
        'feature_names': numeric,
        'nominal': nominal,
        'model_name': 'Logistic Regression',
    }

    out_path = MODEL_DIR / 'pretrained_lr_results.pkl'
    joblib.dump(result, out_path)
    print(f"accuracy={m2['accuracy']:.4f} auc={m2['auc']:.4f} f1={m2['f1']:.4f}")
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
