"""Student Dropout Classifier Streamlit app.

Follows the assignment spec: loads serialized sklearn Pipelines from `model/` and
allows uploading a raw CSV to run predictions and diagnostics.
"""
from pathlib import Path
import json
import io

import pandas as pd
import numpy as np
import joblib
import streamlit as st

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / 'model'
DEFAULT_TEST_DATA = ROOT / 'data' / 'students' / 'data.csv'

def normalize_col(col_name: str) -> str:
    return str(col_name).strip().replace('\ufeff', '').replace('\t', ' ').strip()

st.set_page_config(page_title="StudentLens", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")


@st.cache_resource
def load_models():
    models = {}
    for p in MODEL_DIR.glob('*.pkl'):
        name = p.stem
        try:
            models[name] = joblib.load(p)
        except Exception as e:
            st.warning(f'Failed to load {p.name}: {e}')
    return models

@st.cache_data
def load_metrics():
    p = MODEL_DIR / 'metrics.json'
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding='utf8'))

@st.cache_data
def load_schema():
    p = MODEL_DIR / 'schema.json'
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding='utf8'))

def parse_upload(uploaded_file) -> pd.DataFrame:
    content = uploaded_file.read()
    if isinstance(content, str):
        content = content.encode('utf-8')
    # auto-detect delimiter and strip BOM from headers
    parse_options = {
        'engine': 'python',
        'encoding': 'utf-8-sig',
        'skipinitialspace': True,
    }

    # try the builtin sniffer first
    try:
        df = pd.read_csv(io.BytesIO(content), sep=None, **parse_options)
        # if parser returned a single-column dataframe, try common delimiters
        if df.shape[1] == 1:
            for sep in [';', ',', '\t']:
                try:
                    df2 = pd.read_csv(io.BytesIO(content), sep=sep, **parse_options)
                    if df2.shape[1] > 1:
                        return df2
                except Exception:
                    continue
        return df
    except Exception:
        # fallback attempts
        for sep in [';', ',', '\t']:
            try:
                df = pd.read_csv(io.BytesIO(content), sep=sep, **parse_options)
                if df.shape[1] > 0:
                    return df
            except Exception:
                continue
        raise

def compute_metrics(y_true, y_pred, y_proba, classes):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = None
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro', labels=classes)
    except Exception:
        auc = None
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'mcc': mcc, 'auc': auc}

models = load_models()
metrics_ref = load_metrics()
schema = load_schema()

if 'train_results' not in st.session_state:
    _pretrained_path = MODEL_DIR / 'pretrained_lr_results.pkl'
    if _pretrained_path.exists():
        st.session_state['train_results'] = joblib.load(_pretrained_path)

model_names = [metrics_ref[k]['display_name'] if k in metrics_ref else k for k in sorted(models.keys())]
model_map = {metrics_ref[k]['display_name'] if k in metrics_ref else k: k for k in sorted(models.keys())}
_key_to_display = {v: k for k, v in model_map.items()}

selected_model_key = st.session_state.get('diag_model_select') or (sorted(models.keys())[0] if models else None)
selected_display = _key_to_display.get(selected_model_key, selected_model_key)

import base64
_logo_b64 = base64.b64encode((ROOT / 'assets' / 'bits_logo.png').read_bytes()).decode()
st.sidebar.markdown(f'<div style="text-align:center;"><img src="data:image/png;base64,{_logo_b64}" style="width:240px;height:240px;object-fit:contain;"><br><strong style="font-size:2em;">Birla Institute of Technology and Science, Pilani</strong></div>', unsafe_allow_html=True)
st.sidebar.markdown('---')
st.sidebar.markdown('**🚀 What can you do here?**')
st.sidebar.markdown(
    """
- 📊 **Dataset** — Explore Student data
- 🧠 **Train Model** — Build prediction model
- ⚖️ **Compare Models** — Compare 5 algo
- 📋 **Model Report** — View performance
- 🎯 **Predict** — Predict an outcome
"""
)
st.sidebar.markdown('---')
st.sidebar.markdown('**Created by:** Joseph M Vinod Noel  \n**BITS ID:** 2025AC05003')
_mypic_b64 = base64.b64encode((ROOT / 'assets' / 'mypic.png').read_bytes()).decode()
st.sidebar.markdown(f'<div style="text-align:center;"><img src="data:image/png;base64,{_mypic_b64}" style="width:40%;object-fit:contain;"></div>', unsafe_allow_html=True)

st.markdown('# StudentLens')
st.markdown(
    'Explore how academic performance and student backgrounds influence enrolment outcomes. '
    'Compare five machine learning models to predict completion, dropout, or continued enrolment.'
)

tabs = st.tabs(['Dataset', 'Train a Model', 'Compare Models', 'Model Report', 'Predict', 'Upload Data'])

# Upload Data tab
with tabs[5]:
    st.markdown('### Upload Data')
    st.markdown(
        'Upload a CSV file with student data to analyse and predict outcomes. '
        'The file must contain the columns listed below.'
    )
    uploaded = st.file_uploader('Choose a CSV file', type=['csv'], label_visibility='collapsed')

    st.markdown('#### Required columns')
    _col_info = [
        ('Marital status', 'Integer code — 1=Single, 2=Married, 3=Widower, 4=Divorced, 5=Facto union, 6=Legally separated'),
        ('Application mode', 'Integer code for application pathway (e.g. 1=1st phase general, 17=2nd phase, 39=Over 23)'),
        ('Application order', 'Order of preference (0=first choice … 9=last choice)'),
        ('Course', 'Integer code for the enrolled course'),
        ('Daytime/evening attendance', '1=Daytime, 0=Evening'),
        ('Previous qualification', 'Integer code for highest prior qualification'),
        ('Previous qualification (grade)', 'Grade of previous qualification (0–200 scale)'),
        ('Nacionality', 'Integer code for nationality'),
        ("Mother's qualification", 'Integer code for mother\'s education level'),
        ("Father's qualification", 'Integer code for father\'s education level'),
        ("Mother's occupation", 'Integer code for mother\'s occupation'),
        ("Father's occupation", 'Integer code for father\'s occupation'),
        ('Admission grade', 'Admission grade at enrolment (0–200 scale)'),
        ('Displaced', '1=Yes, 0=No'),
        ('Educational special needs', '1=Yes, 0=No'),
        ('Debtor', '1=Yes, 0=No'),
        ('Tuition fees up to date', '1=Yes, 0=No'),
        ('Gender', '1=Male, 0=Female'),
        ('Scholarship holder', '1=Yes, 0=No'),
        ('Age at enrollment', 'Age in years at time of enrolment'),
        ('International', '1=Yes, 0=No'),
        ('Curricular units 1st sem (credited)', 'Number of credited units in 1st semester'),
        ('Curricular units 1st sem (enrolled)', 'Number of enrolled units in 1st semester'),
        ('Curricular units 1st sem (evaluations)', 'Number of evaluations in 1st semester'),
        ('Curricular units 1st sem (approved)', 'Number of approved units in 1st semester'),
        ('Curricular units 1st sem (grade)', 'Average grade of approved units in 1st semester (0–20)'),
        ('Curricular units 1st sem (without evaluations)', 'Units with no evaluation in 1st semester'),
        ('Curricular units 2nd sem (credited)', 'Number of credited units in 2nd semester'),
        ('Curricular units 2nd sem (enrolled)', 'Number of enrolled units in 2nd semester'),
        ('Curricular units 2nd sem (evaluations)', 'Number of evaluations in 2nd semester'),
        ('Curricular units 2nd sem (approved)', 'Number of approved units in 2nd semester'),
        ('Curricular units 2nd sem (grade)', 'Average grade of approved units in 2nd semester (0–20)'),
        ('Curricular units 2nd sem (without evaluations)', 'Units with no evaluation in 2nd semester'),
        ('Unemployment rate', 'Regional unemployment rate (%)'),
        ('Inflation rate', 'Regional inflation rate (%)'),
        ('GDP', 'GDP growth rate (%)'),
        ('Target', 'Optional — Dropout / Enrolled / Graduate. Required for metrics tabs.'),
    ]
    st.dataframe(
        pd.DataFrame(_col_info, columns=['Column', 'Description']),
        use_container_width=True,
        hide_index=True,
    )
    if uploaded is None and DEFAULT_TEST_DATA.exists():
        st.info('No file uploaded — using the preloaded test_data.csv (885 rows).')

if uploaded is not None:
    try:
        df = parse_upload(uploaded)
    except Exception as e:
        st.exception(e)
        st.stop()
elif DEFAULT_TEST_DATA.exists():
    try:
        with open(DEFAULT_TEST_DATA, 'rb') as f:
            df = parse_upload(io.BytesIO(f.read()))
    except Exception as e:
        st.exception(e)
        st.stop()
else:
    df = None

if df is None:
    with tabs[2]:
        st.info('Upload a dataset to compare models live.')
    with tabs[4]:
        st.markdown('### Held-out reference metrics (from training notebook)')
        if metrics_ref:
            dfm = pd.DataFrame(metrics_ref).T
            available_cols = [c for c in ['accuracy','auc','precision','recall','f1','mcc'] if c in dfm.columns]
            dfm = dfm[available_cols]
            dfm.index = [metrics_ref[k].get('display_name', k) for k in dfm.index]
            st.dataframe(dfm.style.highlight_max(axis=0, color='#c8e6c9').format('{:.4f}'), use_container_width=True)
        else:
            st.warning('metrics.json not found in model/')
    st.stop()

# Preserve the raw feature names for the trained sklearn pipeline,
# while using normalized labels only for display.
raw_feature_columns = schema.get('feature_columns', [])
rename_map = {}
normalized_schema = {normalize_col(name): name for name in raw_feature_columns}
for col in df.columns:
    if col in raw_feature_columns:
        continue
    normalized = normalize_col(col)
    if normalized in normalized_schema:
        rename_map[col] = normalized_schema[normalized]

if rename_map:
    df = df.rename(columns=rename_map)

required = set(raw_feature_columns)
missing = required - set(df.columns)
if missing:
    st.error(f'Uploaded CSV is missing {len(missing)} required column(s): {sorted(missing)}')
    st.write('Uploaded columns (first 50):', list(df.columns)[:50])
    st.write('First row preview to help diagnose delimiter/encoding issues:')
    st.write(df.head(1).T)
    st.stop()

# target handling
target_col = schema.get('target_column', 'Target')
has_target = target_col in df.columns

# Data Preview
preview_df = df.copy()
preview_df.columns = [normalize_col(c) for c in preview_df.columns]
with tabs[0]:
    st.header('Data Profile')
    c1, c2, c3 = st.columns(3)
    c1.metric('Rows', preview_df.shape[0])
    c2.metric('Features', preview_df.shape[1])
    if has_target:
        c3.metric('Classes', df[target_col].nunique())

    # Class distribution + Feature distributions side by side
    col_bar, col_kde = st.columns(2)

    with col_bar:
        st.markdown('### Class distribution')
        if has_target:
            import matplotlib.pyplot as plt
            class_counts = df[target_col].value_counts()
            _CLASS_COLORS = {
                'Dropout':  '#e53935',
                'Enrolled': '#fb8c00',
                'Graduate': '#43a047',
            }
            _bar_colors = [_CLASS_COLORS.get(str(c), '#1565C0') for c in class_counts.index]
            fig_cd, ax_cd = plt.subplots(figsize=(4, 2.5))
            _bars = ax_cd.bar(class_counts.index.astype(str), class_counts.values, color=_bar_colors, width=0.5)
            ax_cd.set_ylim(0, class_counts.max() * 1.18)
            ax_cd.set_ylabel('Count')
            ax_cd.set_xlabel('')
            for _bar, _val in zip(_bars, class_counts.values):
                ax_cd.text(
                    _bar.get_x() + _bar.get_width() / 2, _val + class_counts.max() * 0.02,
                    f'{_val:,}', ha='center', va='bottom', fontsize=11, fontweight='bold',
                )
            ax_cd.set_xticks(range(len(class_counts)))
            ax_cd.set_xticklabels([str(c) for c in class_counts.index],
                                   fontsize=12, fontweight='bold')
            for _i, _lbl in enumerate(ax_cd.get_xticklabels()):
                _lbl.set_color(_bar_colors[_i])
            fig_cd.tight_layout()
            st.pyplot(fig_cd)
            plt.close(fig_cd)
        else:
            st.info('Target column not present — upload data with ground-truth to view class distribution.')

    with col_kde:
        st.markdown('### Feature distributions by class')

        _VALUE_LABELS = {
            'Marital status': {1: 'Single', 2: 'Married', 3: 'Widower', 4: 'Divorced', 5: 'Facto union', 6: 'Legally separated'},
            'Application mode': {1: '1st phase general', 2: 'Ordinance 612/93', 5: '1st phase special (Azores)', 7: 'Other EU holders', 10: 'Ordinance 854-B/99', 15: 'International student', 16: '1st phase special (Madeira)', 17: '2nd phase general', 18: '3rd phase general', 26: 'Ordinance 533-A/99 (b2)', 27: 'Ordinance 533-A/99 (b3)', 39: 'Over 23 years old', 42: 'Transfer', 43: 'Change of major', 44: 'Tech diploma holders', 51: 'Change institution/major', 53: 'Short cycle diploma holders', 57: 'Change institution (intl)'},
            'Daytime/evening attendance\t': {1: 'Daytime', 0: 'Evening'},
            'Previous qualification': {1: 'Secondary', 2: 'Bachelor', 3: 'Degree', 4: 'Master', 5: 'Doctorate', 6: 'Freq higher ed', 9: '12th yr not completed', 10: '11th yr not completed', 12: 'Other 11th yr', 14: '10th yr', 15: '10th yr not completed', 19: 'Basic ed 3rd cycle', 38: 'Basic ed 2nd cycle', 39: 'Tech-vocational', 40: '12th yr (foreign)', 42: 'Freq higher ed (foreign)', 43: 'Freq higher ed 12th yr', 93: 'Higher ed 10th yr', 100: 'Not applicable'},
            'Nacionality': {1: 'Portuguese', 2: 'German', 6: 'Spanish', 11: 'Italian', 13: 'Dutch', 14: 'English', 17: 'Lithuanian', 21: 'Angolan', 22: 'Cape Verdean', 24: 'Guinean', 25: 'Mozambican', 26: 'Santomean', 32: 'Turkish', 41: 'Brazilian', 62: 'Romanian', 100: 'Moldovan', 101: 'Mexican', 103: 'Ukrainian', 105: 'Russian', 108: 'Cuban', 109: 'Colombian'},
            "Mother's qualification": {1: 'Secondary', 2: 'Bachelor', 3: 'Degree', 4: 'Master', 5: 'Doctorate', 6: 'Freq higher ed', 9: '12th yr not completed', 10: '11th yr not completed', 11: '7th yr (old)', 12: 'Other 11th yr', 14: '10th yr', 18: 'General commerce', 19: 'Basic ed 3rd cycle', 22: 'Tech-vocational', 26: 'Specialized course', 27: 'Freq higher ed', 29: 'Vocational training', 30: 'Youth program', 34: 'Basic ed 2nd cycle', 35: 'Unknown', 36: 'Cannot read/write', 37: 'Basic ed 1st cycle', 38: 'Basic ed 2nd cycle', 39: 'Basic ed 3rd cycle'},
            "Father's qualification": {1: 'Secondary', 2: 'Bachelor', 3: 'Degree', 4: 'Master', 5: 'Doctorate', 6: 'Freq higher ed', 9: '12th yr not completed', 10: '11th yr not completed', 11: '7th yr (old)', 12: 'Other 11th yr', 13: '2nd yr high school', 14: '10th yr', 18: 'General commerce', 19: 'Basic ed 3rd cycle', 20: 'Basic ed 2nd cycle', 22: 'Tech-vocational', 25: 'Complementary HS', 26: 'Specialized course', 27: 'Freq higher ed', 29: 'Vocational training', 30: 'Youth program', 31: 'Basic ed 1st cycle', 33: 'Basic ed 3rd cycle', 34: 'Basic ed 2nd cycle', 35: 'Unknown', 36: 'Cannot read/write', 37: 'Basic ed 1st cycle', 38: 'Basic ed 2nd cycle', 39: 'Basic ed 3rd cycle'},
            "Mother's occupation": {0: 'Student', 1: 'Legislative/exec', 2: 'Intellectual/sci', 3: 'Intermediate tech', 4: 'Admin staff', 5: 'Services/sales', 6: 'Agri/forestry', 7: 'Industry/construction', 8: 'Machine operators', 9: 'Unskilled workers', 10: 'Armed forces', 90: 'Other', 99: 'Blank', 122: 'Health prof', 123: 'Teachers', 125: 'ICT specialists', 131: 'Science/eng tech', 132: 'Health tech', 134: 'Legal/social/cultural', 141: 'Managers', 143: 'Accountants', 144: 'Admin support', 151: 'Personal care', 152: 'Sellers', 153: 'Personal services', 171: 'Construction', 173: 'Printing', 175: 'Food processing', 191: 'Cleaners', 192: 'Agri workers', 193: 'Industry workers', 194: 'Meal prep', 195: 'Street vendors'},
            "Father's occupation": {0: 'Student', 1: 'Legislative/exec', 2: 'Intellectual/sci', 3: 'Intermediate tech', 4: 'Admin staff', 5: 'Services/sales', 6: 'Agri/forestry', 7: 'Industry/construction', 8: 'Machine operators', 9: 'Unskilled workers', 10: 'Armed forces', 90: 'Other', 99: 'Blank', 101: 'Armed forces off', 102: 'Armed forces sgt', 103: 'Armed forces other', 112: 'Admin directors', 114: 'Hotel/trade/other', 121: 'Physical sci', 122: 'Health prof', 123: 'Teachers', 124: 'Finance/admin', 125: 'ICT specialists', 131: 'Science/eng tech', 132: 'Health tech', 134: 'Legal/social/cultural', 135: 'ICT tech', 141: 'Managers', 143: 'Accountants', 144: 'Admin support', 151: 'Personal care', 152: 'Sellers', 153: 'Personal services', 154: 'Security', 161: 'Market-oriented agri', 163: 'Subsistence agri', 171: 'Construction', 172: 'Metalworkers', 174: 'Electricians', 175: 'Food processing', 181: 'Plant operators', 182: 'Assembly workers', 183: 'Drivers', 192: 'Agri workers', 193: 'Industry workers', 194: 'Meal prep', 195: 'Street vendors'},
            'Displaced': {0: 'No', 1: 'Yes'},
            'Educational special needs': {0: 'No', 1: 'Yes'},
            'Debtor': {0: 'No', 1: 'Yes'},
            'Tuition fees up to date': {0: 'No', 1: 'Yes'},
            'Gender': {0: 'Female', 1: 'Male'},
            'Scholarship holder': {0: 'No', 1: 'Yes'},
            'International': {0: 'No', 1: 'Yes'},
        }

        norm_map = {normalize_col(r): r for r in raw_feature_columns}
        numeric_feats = []
        for nrm, rawc in norm_map.items():
            try:
                if pd.api.types.is_numeric_dtype(df[rawc]):
                    numeric_feats.append(nrm)
            except Exception:
                continue

        if not numeric_feats:
            st.info('No numeric features detected for distribution plots.')
        else:
            sel_feat = st.selectbox('Feature', options=numeric_feats, index=0, key='feature_dist')
            raw_sel = norm_map[sel_feat]
            # strip BOM/whitespace to match _VALUE_LABELS keys
            _raw_sel_clean = raw_sel.strip().lstrip('﻿')
            _label_map = _VALUE_LABELS.get(_raw_sel_clean)
            _is_nominal = _raw_sel_clean in _VALUE_LABELS or raw_sel in schema.get('nominal_columns', [])
            if has_target:
                try:
                    import seaborn as sns
                    import matplotlib.pyplot as plt
                    classes = schema.get('class_labels') or list(df[target_col].unique())
                    _CLASS_COLORS = {'Dropout': '#e53935', 'Enrolled': '#fb8c00', 'Graduate': '#43a047'}

                    if _is_nominal and _label_map:
                        # grouped bar chart with human-readable labels
                        fig, ax = plt.subplots(figsize=(6, 3))
                        _all_codes = sorted(df[raw_sel].dropna().unique())
                        _x_labels = [_label_map.get(int(v), str(int(v))) for v in _all_codes]
                        x = range(len(_all_codes))
                        bar_width = 0.8 / max(len(classes), 1)
                        for i, cls in enumerate(classes):
                            subset = df[df[target_col] == cls]
                            counts = [subset[raw_sel].value_counts().get(code, 0) for code in _all_codes]
                            offsets = [xi + i * bar_width - (len(classes) - 1) * bar_width / 2 for xi in x]
                            ax.bar(offsets, counts, width=bar_width * 0.9,
                                   label=str(cls), alpha=0.85,
                                   color=_CLASS_COLORS.get(str(cls), None))
                        ax.set_xticks(list(x))
                        ax.set_xticklabels(_x_labels, rotation=35, ha='right', fontsize=7)
                        ax.set_ylabel('Count')
                        ax.legend(title='Class', fontsize=7)
                        fig.tight_layout()
                        st.pyplot(fig)
                    else:
                        # KDE for continuous numeric features
                        fig, ax = plt.subplots(figsize=(6, 3))
                        for cls in classes:
                            subset = df[df[target_col] == cls]
                            vals = pd.to_numeric(subset[raw_sel], errors='coerce').dropna()
                            if len(vals) == 0:
                                continue
                            try:
                                sns.kdeplot(vals, label=str(cls), fill=True, alpha=0.4, ax=ax,
                                            color=_CLASS_COLORS.get(str(cls), None))
                            except Exception:
                                ax.hist(vals, bins=30, alpha=0.4, label=str(cls))
                        ax.set_xlabel(sel_feat)
                        ax.set_ylabel('Density')
                        ax.legend(title='Class', fontsize=7)
                        fig.tight_layout()
                        st.pyplot(fig)
                except Exception as e:
                    st.warning(f'Could not render feature distributions: {e}')
            else:
                st.info('Upload data with target to see feature distributions by class.')

    # Raw data sample
    st.markdown('### Raw data sample')
    st.dataframe(preview_df.head(10), use_container_width=True)

@st.cache_data
def run_predictions(_pipe, data_hash: str, feature_cols: list):
    X = df[feature_cols]
    y_pred = _pipe.predict(X)
    try:
        y_proba = _pipe.predict_proba(X)
    except Exception:
        y_proba = None
    return y_pred, y_proba

# Predictions and metrics
pipe = models.get(selected_model_key)
if pipe is None:
    st.error('Selected model not available')
else:
    X = df[raw_feature_columns]
    _data_hash = f"{selected_model_key}_{len(df)}"
    try:
        y_pred, _cached_proba = run_predictions(pipe, _data_hash, raw_feature_columns)
    except Exception as e:
        st.exception(e)
        st.stop()

    # Helper: get the final estimator from a pipeline-like object
    def _final_estimator(m):
        try:
            if hasattr(m, 'steps') and len(m.steps) > 0:
                return m.steps[-1][1]
        except Exception:
            pass
        return m

    final_est = _final_estimator(pipe)

    # Use cached proba; fallback to one-hot if unavailable
    y_proba = _cached_proba
    if y_proba is None:
        classes = getattr(final_est, 'classes_', np.unique(y_pred))
        y_proba = np.zeros((len(y_pred), len(classes)), dtype=float)
        for i, cls in enumerate(classes):
            y_proba[:, i] = (y_pred == cls).astype(float)
        try:
            final_est.classes_ = classes
        except Exception:
            pass

    if not hasattr(pipe, 'classes_'):
        try:
            pipe.classes_ = getattr(final_est, 'classes_', np.unique(y_pred))
        except Exception:
            pipe.classes_ = np.unique(y_pred)

    pred_df = df.copy()
    pred_df['predicted_class'] = y_pred
    for i, cls in enumerate(pipe.classes_):
        pred_df[f'prob_{cls}'] = y_proba[:, i]


    # Train a Model tab
    with tabs[1]:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline as SKPipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import confusion_matrix, classification_report

        _MODEL_NAMES = ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors', 'Gaussian Naive Bayes', 'Random Forest']

        if not has_target:
            st.info('Upload a dataset with the target column to enable training.')
        else:
            # Hyperparameters — model selector + controls in same row
            hp_col0, hp_col1, hp_col2 = st.columns(3)
            model_choice = hp_col0.selectbox(
                'Model',
                options=_MODEL_NAMES,
                key='train_model_choice',
            )
            hp = {}
            if model_choice == 'Logistic Regression':
                hp['C'] = hp_col1.slider('Regularization C', 0.01, 10.0, 1.0, key='lr_C')
                hp['max_iter'] = int(hp_col2.slider('Max iterations', 100, 3000, 1000, step=100, key='lr_iter'))
            elif model_choice == 'Decision Tree':
                hp['max_depth'] = int(hp_col1.slider('Max depth', 1, 20, 8, key='dt_depth'))
                hp['min_samples_leaf'] = int(hp_col2.slider('Min samples per leaf', 1, 50, 10, key='dt_leaf'))
            elif model_choice == 'K-Nearest Neighbors':
                hp['n_neighbors'] = int(hp_col1.slider('Neighbours (k)', 1, 50, 15, key='knn_k'))
            elif model_choice == 'Random Forest':
                hp['n_estimators'] = int(hp_col1.slider('Number of trees', 10, 500, 100, step=10, key='rf_trees'))
                hp['max_depth'] = int(hp_col2.slider('Max depth', 1, 20, 10, key='rf_depth'))
            # GaussianNB has no meaningful hyperparameters to expose

            if st.button('Train model', key='train_btn'):
                with st.spinner('Training…'):
                    try:
                        y_tr = df[target_col]
                        X_tr = df[raw_feature_columns].copy()
                        nominal = [c for c in schema.get('nominal_columns', []) if c in X_tr.columns]
                        numeric = [c for c in raw_feature_columns if c in X_tr.columns and c not in nominal]

                        pre = ColumnTransformer([
                            ('num', SKPipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), numeric),
                            ('cat', SKPipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), nominal),
                        ], remainder='drop')

                        if model_choice == 'Logistic Regression':
                            est = LogisticRegression(C=hp['C'], max_iter=hp['max_iter'], random_state=42)
                        elif model_choice == 'Decision Tree':
                            est = DecisionTreeClassifier(max_depth=hp['max_depth'], min_samples_leaf=hp['min_samples_leaf'], random_state=42)
                        elif model_choice == 'K-Nearest Neighbors':
                            est = KNeighborsClassifier(n_neighbors=hp['n_neighbors'])
                        elif model_choice == 'Gaussian Naive Bayes':
                            est = GaussianNB()
                        else:
                            est = RandomForestClassifier(n_estimators=hp['n_estimators'], max_depth=hp['max_depth'], random_state=42, n_jobs=-1)

                        clf_demo = SKPipeline([('pre', pre), ('est', est)])
                        X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)
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
                        st.session_state['train_results'] = {
                            'metrics': m2,
                            'y_test': y_te2,
                            'y_pred': y_p2,
                            'classes': classes2,
                            'estimator': clf_demo.named_steps['est'],
                            'feature_names': numeric,
                            'nominal': nominal,
                            'model_name': model_choice,
                        }
                    except Exception as e:
                        st.error(f'Training failed: {e}')

            # Render results if available
            if 'train_results' in st.session_state:
                res = st.session_state['train_results']
                m2 = res['metrics']
                st.success(f"Model trained: {res['model_name']}")

                # Metrics row — 6 chips
                mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
                mc1.metric('Accuracy', f"{m2['accuracy']*100:.3f}%")
                mc2.metric('AUC', f"{m2['auc']:.3f}" if m2['auc'] is not None else 'n/a')
                mc3.metric('Precision', f"{m2['precision']:.3f}")
                mc4.metric('Recall', f"{m2['recall']:.3f}")
                mc5.metric('F1 Score', f"{m2['f1']:.3f}")
                mc6.metric('MCC', f"{m2['mcc']:.3f}")

                st.markdown('---')

                # Confusion matrix + Feature importance side by side
                diag_left, diag_right = st.columns(2)

                with diag_left:
                    st.markdown('**Confusion matrix**')
                    cm2 = confusion_matrix(res['y_test'], res['y_pred'], labels=res['classes'])
                    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                    sns.heatmap(
                        cm2, annot=True, fmt='d', cmap='Blues',
                        xticklabels=res['classes'], yticklabels=res['classes'],
                        ax=ax_cm,
                    )
                    ax_cm.set_xlabel('Predicted')
                    ax_cm.set_ylabel('Actual')
                    st.pyplot(fig_cm)
                    plt.close(fig_cm)

                with diag_right:
                    est2 = res['estimator']
                    feat_names = res['feature_names']
                    if hasattr(est2, 'feature_importances_') and feat_names:
                        importances = est2.feature_importances_
                        n_feats = min(len(feat_names), len(importances))
                        fi_df = pd.DataFrame({
                            'feature': feat_names[:n_feats],
                            'importance': importances[:n_feats],
                        }).sort_values('importance', ascending=False).head(20)
                        st.markdown('**What drove the predictions**')
                        st.caption('Feature Importance (Gini-based)')
                        fig_fi, ax_fi = plt.subplots(figsize=(6, max(4, len(fi_df) * 0.3)))
                        ax_fi.barh(fi_df['feature'][::-1], fi_df['importance'][::-1], color='steelblue')
                        ax_fi.set_xlabel('Importance')
                        st.pyplot(fig_fi)
                        plt.close(fig_fi)
                    elif hasattr(est2, 'coef_') and feat_names:
                        coefs = np.abs(est2.coef_).mean(axis=0) if est2.coef_.ndim > 1 else np.abs(est2.coef_[0])
                        n_feats = min(len(feat_names), len(coefs))
                        fi_df = pd.DataFrame({
                            'feature': feat_names[:n_feats],
                            'importance': coefs[:n_feats],
                        }).sort_values('importance', ascending=False).head(20)
                        st.markdown('**What drove the predictions**')
                        st.caption('Coefficient magnitude (Logistic Regression)')
                        fig_fi, ax_fi = plt.subplots(figsize=(6, max(4, len(fi_df) * 0.3)))
                        ax_fi.barh(fi_df['feature'][::-1], fi_df['importance'][::-1], color='steelblue')
                        ax_fi.set_xlabel('|Coefficient|')
                        st.pyplot(fig_fi)
                        plt.close(fig_fi)
                    else:
                        st.info('Feature importance not available for this model type.')

                # Per-class report
                st.markdown('**Per-class report**')
                cr2 = classification_report(res['y_test'], res['y_pred'], output_dict=True, zero_division=0)
                cr_df = pd.DataFrame(cr2).transpose()
                st.dataframe(
                    cr_df.style.format('{:.3f}', subset=['precision', 'recall', 'f1-score'])
                         .format('{:.0f}', subset=['support']),
                    use_container_width=True,
                )

    # Compare Models tab
    with tabs[2]:
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split as _tts
        from sklearn.pipeline import Pipeline as _SKP
        from sklearn.compose import ColumnTransformer as _CT
        from sklearn.impute import SimpleImputer as _SI
        from sklearn.preprocessing import StandardScaler as _SS, OneHotEncoder as _OHE
        from sklearn.linear_model import LogisticRegression as _LR
        from sklearn.tree import DecisionTreeClassifier as _DT
        from sklearn.neighbors import KNeighborsClassifier as _KNN
        from sklearn.naive_bayes import GaussianNB as _GNB
        from sklearn.ensemble import RandomForestClassifier as _RF

        st.write('Train all five models with their **default** hyperparameters on the same split, and compare.')

        if not has_target:
            st.info('Upload a dataset with the target column to enable model comparison.')
        else:
            _run_compare = st.button('Re-run comparison', key='compare_btn') or 'compare_results' not in st.session_state
            if _run_compare:
                _nom = [c for c in schema.get('nominal_columns', []) if c in df.columns]
                _num = [c for c in raw_feature_columns if c in df.columns and c not in _nom]
                _pre = _CT([
                    ('num', _SKP([('imp', _SI(strategy='median')), ('sc', _SS())]), _num),
                    ('cat', _SKP([('imp', _SI(strategy='most_frequent')), ('ohe', _OHE(handle_unknown='ignore', sparse_output=False))]), _nom),
                ], remainder='drop')
                _estimators = {
                    'Logistic Regression': _LR(max_iter=2000, random_state=42),
                    'Decision Tree':       _DT(max_depth=8, min_samples_leaf=10, random_state=42),
                    'K-Nearest Neighbors': _KNN(n_neighbors=15),
                    'Gaussian Naive Bayes':_GNB(),
                    'Random Forest':       _RF(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
                }
                _X = df[raw_feature_columns].copy()
                _y = df[target_col]
                _Xtr, _Xte, _ytr, _yte = _tts(_X, _y, test_size=0.2, random_state=42, stratify=_y)
                _compare_rows = []
                _progress = st.progress(0, text='Training models…')
                for _idx, (_name, _est) in enumerate(_estimators.items()):
                    _progress.progress((_idx) / 5, text=f'Training {_name}…')
                    try:
                        _clf = _SKP([('pre', _pre), ('est', _est)])
                        _clf.fit(_Xtr, _ytr)
                        _yp = _clf.predict(_Xte)
                        try:
                            _ypb = _clf.predict_proba(_Xte)
                        except Exception:
                            _ypb = None
                        _m = compute_metrics(
                            _yte, _yp,
                            _ypb if _ypb is not None else np.zeros((len(_yp), len(_clf.classes_))),
                            classes=_clf.classes_,
                        )
                        _compare_rows.append({
                            'Model': _name,
                            'Accuracy': round(_m['accuracy'], 4),
                            'Precision': round(_m['precision'], 4),
                            'Recall': round(_m['recall'], 4),
                            'F1 Score': round(_m['f1'], 4),
                            'MCC': round(_m['mcc'], 4),
                            'AUC': round(_m['auc'], 4) if _m['auc'] is not None else None,
                        })
                    except Exception as _e:
                        _compare_rows.append({'Model': _name, 'error': str(_e)})
                _progress.progress(1.0, text='Done.')
                st.session_state['compare_results'] = _compare_rows

            if 'compare_results' in st.session_state:
                _rows = st.session_state['compare_results']
                _cdf = pd.DataFrame([r for r in _rows if 'error' not in r])
                if _cdf.empty:
                    st.error('All models failed to train.')
                else:
                    _metric_cols = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
                    _available = [c for c in _metric_cols if c in _cdf.columns and _cdf[c].notna().any()]

                    sel_metric = st.selectbox('Metric to chart', options=_available, key='compare_metric')

                    # Bar chart — one bar per model, sorted descending
                    _plot_df = _cdf[['Model', sel_metric]].dropna().sort_values(sel_metric, ascending=False)
                    fig_cmp, ax_cmp = plt.subplots(figsize=(10, 4))
                    _bars = ax_cmp.bar(_plot_df['Model'], _plot_df[sel_metric], color='#1565C0', width=0.5)
                    for _bar, _val in zip(_bars, _plot_df[sel_metric]):
                        ax_cmp.text(
                            _bar.get_x() + _bar.get_width() / 2,
                            _bar.get_height() - 0.03,
                            f'{_val:.4f}',
                            ha='center', va='top', color='white', fontsize=9, fontweight='bold',
                        )
                    ax_cmp.set_ylim(0, 1.05)
                    ax_cmp.set_ylabel(sel_metric)
                    ax_cmp.set_xlabel('Model')
                    ax_cmp.tick_params(axis='x', labelsize=9)
                    fig_cmp.tight_layout()
                    st.pyplot(fig_cmp)
                    plt.close(fig_cmp)

                    # Summary table
                    _display_cols = ['Model'] + _available
                    st.dataframe(
                        _cdf[_display_cols].style
                            .highlight_max(subset=_available, axis=0, color='#c8e6c9')
                            .format({c: '{:.4f}' for c in _available}),
                        use_container_width=True,
                        hide_index=True,
                    )

    # Diagnostics tab
    with tabs[3]:
        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc as _auc_score

        # Model selector — stays in sync with sidebar via shared session state key
        _sel_col, _ = st.columns([1, 3])
        with _sel_col:
            _diag_model_key = st.selectbox(
                'Model',
                options=list(model_map.values()),
                format_func=lambda k: metrics_ref[k]['display_name'] if k in metrics_ref else k,
                key='diag_model_select',
            )
        _diag_pipe = models.get(_diag_model_key)
        _diag_display = metrics_ref[_diag_model_key]['display_name'] if _diag_model_key in metrics_ref else _diag_model_key
        st.caption(f'Showing diagnostics for: **{_diag_display}**')

        if _diag_pipe is None:
            st.error('Model not loaded.')
        elif not has_target:
            st.info('Upload with ground-truth to see diagnostics (confusion matrix, classification report, ROC).')
        else:
            y_true = df[target_col]
            try:
                _diag_pred = _diag_pipe.predict(X)
                try:
                    _diag_proba = _diag_pipe.predict_proba(X)
                except Exception:
                    _diag_proba = None
            except Exception as _e:
                st.error(f'Prediction failed: {_e}')
                _diag_pred = None

            if _diag_pred is not None:
                _diag_classes = _diag_pipe.classes_ if hasattr(_diag_pipe, 'classes_') else np.unique(_diag_pred)
                _d_left, _d_right = st.columns(2)

                with _d_left:
                    st.markdown('**Confusion matrix**')
                    cm = confusion_matrix(y_true, _diag_pred, labels=_diag_classes)
                    fig_d, ax_d = plt.subplots(figsize=(2.5, 2))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=_diag_classes, yticklabels=_diag_classes, ax=ax_d)
                    ax_d.set_xlabel('Predicted')
                    ax_d.set_ylabel('Actual')
                    st.pyplot(fig_d, use_container_width=False)
                    plt.close(fig_d)

                with _d_right:
                    st.markdown('**Classification report**')
                    st.markdown('<style>[data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th { font-size: 20px !important; }</style>', unsafe_allow_html=True)
                    cr = classification_report(y_true, _diag_pred, output_dict=True, zero_division=0)
                    cr_df = pd.DataFrame(cr).transpose()
                    _cr_col, _ = st.columns([0.7, 0.3])
                    with _cr_col:
                        st.dataframe(
                            cr_df.style.format('{:.3f}', subset=['precision','recall','f1-score'])
                                       .format('{:.0f}', subset=['support']),
                            use_container_width=True,
                            height=300,
                        )

                if _diag_proba is not None:
                    st.markdown('**ROC curves (one-vs-rest)**')
                    try:
                        fig_r, ax_r = plt.subplots(figsize=(7, 2))
                        for i, cls in enumerate(_diag_classes):
                            fpr, tpr, _ = roc_curve((y_true == cls).astype(int), _diag_proba[:, i])
                            ax_r.plot(fpr, tpr, label=f'{cls} (AUC={_auc_score(fpr, tpr):.3f})')
                        ax_r.plot([0,1],[0,1],'k--')
                        ax_r.set_xlabel('FPR')
                        ax_r.set_ylabel('TPR')
                        ax_r.legend()
                        st.pyplot(fig_r, use_container_width=False)
                        plt.close(fig_r)
                    except Exception:
                        st.warning('ROC curves not available for this upload')

    # Predict tab — single-sample explorer
    with tabs[4]:
        import matplotlib.pyplot as plt

        # which model is active
        _active_display = selected_display
        st.caption(f"Using the last model you trained: **{_active_display}**")

        # Pre-fill on first load so something is always shown
        if 'predict_sample_idx' not in st.session_state:
            st.session_state['predict_sample_idx'] = int(np.random.randint(0, len(df)))

        if st.button('🎲 Pick a random test sample', key='predict_pick_btn'):
            st.session_state['predict_sample_idx'] = int(np.random.randint(0, len(df)))

        if True:
            _idx = st.session_state['predict_sample_idx']
            _sample_row = df.iloc[[_idx]]
            _sample_X = _sample_row[raw_feature_columns]

            # --- prediction for this sample ---
            try:
                _sample_pred = pipe.predict(_sample_X)[0]
                try:
                    _sample_proba = pipe.predict_proba(_sample_X)[0]
                    _sample_classes = pipe.classes_
                except Exception:
                    _sample_proba = None
                    _sample_classes = pipe.classes_ if hasattr(pipe, 'classes_') else np.array([_sample_pred])
            except Exception as _e:
                st.error(f'Prediction failed: {_e}')
                st.stop()

            # build importance vector once — needed by both sections below
            _final = _final_estimator(pipe)
            _nominal_cols = schema.get('nominal_columns', [])
            _numeric_cols = [c for c in raw_feature_columns if c not in _nominal_cols]
            if hasattr(_final, 'feature_importances_'):
                _imp_raw = _final.feature_importances_
                _n = min(len(_numeric_cols), len(_imp_raw))
                _global_imp = dict(zip(_numeric_cols[:_n], _imp_raw[:_n]))
            elif hasattr(_final, 'coef_'):
                _coef = np.abs(_final.coef_).mean(axis=0) if _final.coef_.ndim > 1 else np.abs(_final.coef_[0])
                _n = min(len(_numeric_cols), len(_coef))
                _global_imp = dict(zip(_numeric_cols[:_n], _coef[:_n]))
            else:
                _global_imp = {c: 1.0 for c in raw_feature_columns}
            _top10_feats = sorted(_global_imp, key=_global_imp.get, reverse=True)[:10]

            _actual = df[target_col].iloc[_idx] if has_target else None
            _correct = (_actual == _sample_pred) if _actual is not None else None

            # Two-column layout: left = prediction + probabilities, right = feature grid
            _panel_left, _panel_right = st.columns([1, 2], gap='large')

            with _panel_left:
                # --- 1. PREDICTION ---
                st.markdown('#### 🎯 Prediction')
                _p1, _p2 = st.columns(2)
                with _p1:
                    st.caption('🤖 Predicted')
                    st.markdown(f'<div style="font-size:1.6rem;font-weight:700;line-height:1.2">{_sample_pred}</div>', unsafe_allow_html=True)
                if _actual is not None:
                    with _p2:
                        st.caption('✅ Actual' if _correct else '❌ Actual')
                        st.markdown(f'<div style="font-size:1.6rem;font-weight:700;line-height:1.2">{_actual}</div>', unsafe_allow_html=True)
                        if _correct:
                            st.markdown('<span style="color:#2e7d32;font-weight:600">✓ correct</span>', unsafe_allow_html=True)
                        else:
                            st.markdown('<span style="color:#c62828;font-weight:600">✗ incorrect</span>', unsafe_allow_html=True)

                # --- 3. CLASS PROBABILITIES ---
                st.markdown('#### 📊 Class probabilities')
                _safe_classes = (
                    list(pipe.classes_) if hasattr(pipe, 'classes_') and len(pipe.classes_) > 0
                    else _sample_classes
                )
                if _sample_proba is not None and len(_safe_classes) > 0 and len(_sample_proba) == len(_safe_classes):
                    import matplotlib.pyplot as _plt2
                    _prob_df = pd.Series(_sample_proba, index=_safe_classes).sort_values(ascending=False)
                    fig_pred, ax_pred = _plt2.subplots(figsize=(4, max(1.8, len(_prob_df) * 0.55)))
                    _colors = ['#1565C0' if str(c) == str(_sample_pred) else '#90CAF9' for c in _prob_df.index]
                    ax_pred.barh([str(c) for c in _prob_df.index[::-1]], _prob_df.values[::-1], color=_colors[::-1])
                    ax_pred.set_xlim(0, 1.18)
                    ax_pred.set_xlabel('Probability')
                    for _i, (_cls, _pv) in enumerate(zip(_prob_df.index[::-1], _prob_df.values[::-1])):
                        ax_pred.text(_pv + 0.02, _i, f'{_pv:.3f}', va='center', fontsize=9)
                    fig_pred.tight_layout()
                    st.pyplot(fig_pred)
                    _plt2.close(fig_pred)
                else:
                    st.info('Probabilities not available for the selected model.')

            with _panel_right:
                # --- 2. TOP-10 FEATURE VALUES ---
                st.markdown('#### Feature values')
                st.caption('Top 10 most important features for this sample.')
                _cols_per_row = 2
                for _row_start in range(0, len(_top10_feats), _cols_per_row):
                    _row_feats = _top10_feats[_row_start:_row_start + _cols_per_row]
                    _grid = st.columns(_cols_per_row)
                    for _gi, _feat in enumerate(_row_feats):
                        _rank = _row_start + _gi + 1
                        _raw_val = _sample_row[_feat].iloc[0] if _feat in _sample_row.columns else 0.0
                        try:
                            _display_val = float(_raw_val)
                        except (TypeError, ValueError):
                            _display_val = 0.0
                        _grid[_gi].markdown(f'**{_rank}. {_feat}**')
                        _grid[_gi].markdown(f'<p style="font-size:2em; font-weight:bold; color:#1c83e1;">{_display_val:.1f}</p>', unsafe_allow_html=True)

        st.markdown('---')
        buf = pred_df.to_csv(index=False).encode('utf8')
        st.download_button('Download predictions.csv', data=buf, file_name='predictions.csv')

