"""Train required models and export artifacts for the Streamlit app.
Produces:
 - model/{logistic_regression,decision_tree,knn,naive_bayes,random_forest}.pkl
 - model/metrics.json
 - model/schema.json
 - ../test_data.csv
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
import joblib

RANDOM_STATE = 42
ROOT = Path(__file__).parent.parent
RAW = ROOT / 'data' / 'raw_dataset.csv'
# If raw_dataset.csv missing, fall back to data/students/data.csv
if not RAW.exists():
    fallback = ROOT / 'data' / 'students' / 'data.csv'
    if fallback.exists():
        RAW = fallback
    else:
        raise SystemExit('No dataset found at data/raw_dataset.csv or data/students/data.csv')

print('Using dataset:', RAW)

df = pd.read_csv(RAW, sep=None, engine='python')
print('shape:', df.shape)
if 'Target' not in df.columns:
    raise SystemExit('Target column missing from dataset')

FEATURE_COLUMNS = [c for c in df.columns if c != 'Target']
# detect numeric vs nominal
numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'Target']
nominal_candidates = []
for c in numeric_cols:
    try:
        if pd.api.types.is_integer_dtype(df[c]) and df[c].nunique(dropna=True) <= 30:
            nominal_candidates.append(c)
    except Exception:
        continue
NOMINAL_COLS = nominal_candidates
NUMERIC_COLS = [c for c in FEATURE_COLUMNS if c not in NOMINAL_COLS]
print('numeric count', len(NUMERIC_COLS), 'nominal count', len(NOMINAL_COLS))

# split
X = df[FEATURE_COLUMNS].copy()
y = df['Target'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
# export raw test set including Target
test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
test_df.to_csv(ROOT / 'test_data.csv', index=False)
print('Wrote test_data.csv', test_df.shape)

# build preprocessor
num_pipeline = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
# handle OneHotEncoder arg differences
ohe_kwargs = {'handle_unknown': 'ignore'}
try:
    # sklearn 1.4+ uses sparse_output
    ohe_kwargs['sparse_output'] = False
except Exception:
    ohe_kwargs['sparse'] = False
cat_pipeline = Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(**ohe_kwargs))])
transformers = []
if NUMERIC_COLS:
    transformers.append(('num', num_pipeline, NUMERIC_COLS))
if NOMINAL_COLS:
    transformers.append(('cat', cat_pipeline, NOMINAL_COLS))
from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer(transformers)

models = {
    'logistic_regression': LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    'decision_tree': DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=RANDOM_STATE),
    'knn': KNeighborsClassifier(n_neighbors=15),
    'naive_bayes': GaussianNB(),
    'random_forest': RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
}

out_model_dir = ROOT / 'model'
out_model_dir.mkdir(parents=True, exist_ok=True)

metrics = {}

def compute_metrics(y_true, y_pred, y_proba, classes):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro', labels=classes)
    return {'accuracy': float(acc), 'precision': float(prec), 'recall': float(rec), 'f1': float(f1), 'mcc': float(mcc), 'auc': float(auc)}

for slug, clf in models.items():
    print('Training', slug)
    pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    try:
        y_proba = pipe.predict_proba(X_test)
    except Exception as e:
        # Some classifiers may not support predict_proba
        # create dummy uniform probs
        print('predict_proba failed for', slug, e)
        n_classes = len(pipe.classes_)
        y_proba = np.full((len(X_test), n_classes), 1.0 / n_classes)
    m = compute_metrics(y_test, y_pred, y_proba, classes=pipe.classes_)
    metrics[slug] = {'display_name': slug.replace('_', ' ').title(), **m}
    joblib.dump(pipe, out_model_dir / f'{slug}.pkl')
    print('Saved', slug)

# write metrics.json and schema.json
with open(out_model_dir / 'metrics.json', 'w', encoding='utf8') as f:
    json.dump(metrics, f, indent=2)

schema = {
    'target_column': 'Target',
    'feature_columns': FEATURE_COLUMNS,
    'class_labels': sorted(df['Target'].unique().tolist()),
    'nominal_columns': NOMINAL_COLS,
    'sklearn_version': sklearn.__version__,
    'n_train': int(X_train.shape[0]),
    'n_test': int(X_test.shape[0]),
}
with open(out_model_dir / 'schema.json', 'w', encoding='utf8') as f:
    json.dump(schema, f, indent=2)

print('Wrote metrics.json and schema.json to', out_model_dir)
print('Done')
