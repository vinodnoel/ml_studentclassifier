"""Pre-compute the Model Comparison tab's results offline.

Runs the *exact* same training code the "Model Comparison" tab in app.py runs
live (same hyperparameters, same 80/20 split, same random_state=42), against
the current test_data.csv, and writes the result to model/comparison_metrics.json.

The app loads this file instantly instead of retraining every time someone
opens the tab — retraining live only happens if the user clicks "Re-run live"
or uploads their own CSV. Since the code path and settings are identical, a
live re-run reproduces these numbers exactly.

Run this locally whenever test_data.csv changes, then commit the updated
model/comparison_metrics.json:

    python model/generate_comparison_metrics.py
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
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
    nom = [c for c in raw_feature_columns if normalize_col(c) in
           {normalize_col(x) for x in schema.get('nominal_columns', [])}]
    num = [c for c in raw_feature_columns if c not in nom]

    missing = [c for c in raw_feature_columns + [target_col] if c not in df.columns]
    if missing:
        raise SystemExit(f'test_data.csv is missing required columns: {missing}')

    pre = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                           ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), nom),
    ], remainder='drop')

    estimators = {
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=15),
        'Gaussian Naive Bayes': GaussianNB(var_smoothing=1e-2),
        # n_jobs=1 matches the live app's Model Comparison tab exactly (both avoid
        # multiprocessing spawn overhead on Streamlit Community Cloud's single-core
        # free tier) so "identical code/settings" is strictly accurate.
        'Random Forest':       RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=1),
    }

    X = df[raw_feature_columns].copy()
    y = df[target_col]
    min_class = y.value_counts().min()
    stratify = y if min_class >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)

    rows = []
    for name, est in estimators.items():
        clf = Pipeline([('pre', pre), ('est', est)])
        clf.fit(Xtr, ytr)
        yp = clf.predict(Xte)
        try:
            ypb = clf.predict_proba(Xte)
        except Exception:
            ypb = None
        m = compute_metrics(yte, yp, ypb if ypb is not None else np.zeros((len(yp), len(clf.classes_))), classes=clf.classes_)
        rows.append({
            'Model': name,
            'Accuracy': round(m['accuracy'], 4),
            'Precision': round(m['precision'], 4),
            'Recall': round(m['recall'], 4),
            'F1 Score': round(m['f1'], 4),
            'MCC': round(m['mcc'], 4),
            'AUC': round(m['auc'], 4) if m['auc'] is not None else None,
        })
        print(f"{name:22s} accuracy={m['accuracy']:.4f}")

    out_path = MODEL_DIR / 'comparison_metrics.json'
    out_path.write_text(json.dumps(rows, indent=2), encoding='utf8')
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
