"""One-time script to pre-train a Logistic Regression model and save results to disk.
Run once: python pretrain_lr.py
Output: model/pretrained_lr_results.pkl
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / 'model'
DATA_PATH = ROOT / 'test_data.csv'

schema = json.loads((MODEL_DIR / 'schema.json').read_text(encoding='utf8'))
target_col = schema['target_column']
feature_cols = schema['feature_columns']
nominal_cols = schema['nominal_columns']

df = pd.read_csv(DATA_PATH, sep=None, engine='python', encoding='utf-8-sig', skipinitialspace=True)
df.columns = [str(c).strip().replace('﻿', '').replace('\t', ' ').strip() for c in df.columns]

feature_cols = [str(c).strip().replace('﻿', '').replace('\t', ' ').strip() for c in feature_cols]
nominal_cols = [str(c).strip().replace('﻿', '').replace('\t', ' ').strip() for c in nominal_cols]

feature_cols = [c for c in feature_cols if c in df.columns]
nominal = [c for c in nominal_cols if c in feature_cols]
numeric = [c for c in feature_cols if c not in nominal]

X = df[feature_cols].copy()
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pre = ColumnTransformer([
    ('num', SKPipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), numeric),
    ('cat', SKPipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), nominal),
], remainder='drop')

est = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
clf = SKPipeline([('pre', pre), ('est', est)])
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)
classes = clf.classes_

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)
try:
    auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro', labels=classes)
except Exception:
    auc = None

results = {
    'metrics': {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'mcc': mcc, 'auc': auc},
    'y_test': y_test,
    'y_pred': y_pred,
    'classes': classes,
    'estimator': clf.named_steps['est'],
    'feature_names': numeric,
    'nominal': nominal,
    'model_name': 'Logistic Regression',
}

out_path = MODEL_DIR / 'pretrained_lr_results.pkl'
joblib.dump(results, out_path)
print(f"Saved to {out_path}")
print(f"Accuracy: {acc*100:.3f}%  |  F1: {f1:.3f}  |  AUC: {auc:.3f if auc else 'n/a'}")
