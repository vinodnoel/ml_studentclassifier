# Student Dropout & Academic Success Classification

> **Created by Joseph Vinod · BITS ID: 2025AC05003**

---

## a. Problem Statement

Predicting whether a student will drop out, remain enrolled, or graduate is normally identified only after it is too late to intervene. This project builds and compares five classic supervised classification algorithms that predict a student's academic outcome (one of 3 classes) from 36 demographic, academic, and socioeconomic features recorded at enrolment time, and packages them in an interactive Streamlit app for training, evaluating, and testing the models.

---

## b. Dataset Description

- **Source:** [UCI ML Repository — Predict Students' Dropout and Academic Success (Dataset #697)](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- **Samples:** 4,424 student records
- **Classes (3):** Dropout, Enrolled, Graduate
- **Features (36):** demographic background (age, gender, nationality, marital status, international student flag), academic performance (previous qualification grade, admission grade, curricular units credited/enrolled/approved/graded across two semesters), and socioeconomic context (parents' education and occupation, scholarship holder, debtor, tuition fees up to date, unemployment rate, inflation rate, GDP).
- **Split:** 80% train (3,539) / 20% test (885), stratified by class, with `random_state=42`.
- **Preprocessing:** Median imputation + standard scaling for 12 continuous numeric features; mode imputation + one-hot encoding for 24 integer-coded nominal/categorical features.
- **Class balance:** Moderately imbalanced — Graduate (49.9%) dominates; Enrolled (17.9%) is the smallest class and hardest to predict, which depresses macro-averaged recall across all models.

*Citation: Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2022). Predict Students' Dropout and Academic Success. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89*

---

## c. GitHub Repository Link

[https://github.com/vinodnoel/student-classifier](https://github.com/vinodnoel/student-classifier)

---

## d. Models Used

Five classic classification algorithms were trained on the same train/test split using consistent preprocessing:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7627 | 0.8848 | 0.7043 | 0.6839 | 0.6910 | 0.6073 |
| Decision Tree | 0.7469 | 0.8248 | 0.6908 | 0.6735 | 0.6800 | 0.5818 |
| K-Nearest Neighbors | 0.6960 | 0.8251 | 0.6420 | 0.5850 | 0.5938 | 0.4913 |
| Naive Bayes | 0.2203 | 0.7125 | 0.5401 | 0.3636 | 0.1727 | 0.0987 |
| Random Forest (Ensemble) | 0.7548 | 0.8861 | 0.7058 | 0.6483 | 0.6558 | 0.5917 |

*Precision, Recall, and F1 are macro-averaged across all 3 classes; AUC is one-vs-rest macro-averaged. Evaluated on the held-out test set (885 samples, `test_size=0.2, random_state=42`).*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest F1 (0.6910) and accuracy (0.7627) of all five models. The linear decision boundary works well here because, after standardisation and one-hot expansion, the Dropout and Graduate classes are broadly separable in feature space — particularly along the curricular-unit grade and approval-rate axes. The Enrolled class, which sits ambiguously between the other two outcomes, is the main source of misclassification. Coefficient magnitudes also provide direct interpretability: tuition fees up to date and 2nd-semester approved units emerge as the strongest predictors. |
| Decision Tree | Second-lowest AUC (0.8248). The single tree's recursive splits can capture non-linear feature interactions (e.g. age × scholarship status), but depth-capping at `max_depth=8` to prevent overfitting leaves some boundary complexity unexplored. Without ensemble averaging, predictions near split thresholds are noisy, resulting in a meaningful accuracy gap (0.7469) vs. Logistic Regression — despite both models sharing identical preprocessing. |
| K-Nearest Neighbors | Weakest accuracy (0.6960) and F1 (0.5938). One-hot encoding the 24 nominal columns inflates the feature space, diluting Euclidean distances between samples — the classic curse of dimensionality for kNN. The Enrolled minority class suffers most, pulling macro recall down to 0.5850. With `k=15`, the model is also forced to average across many potentially dissimilar neighbours in this high-dimensional space. |
| Naive Bayes | Catastrophic accuracy (0.2203) and F1 (0.1727): the model collapses nearly all predictions onto a single class. The Gaussian independence assumption is violated on two fronts — the 1st and 2nd semester curricular-unit columns (credited, enrolled, approved, grade) are highly collinear, and the Gaussian distribution is inappropriate for the many binary indicator features (gender, scholarship, debtor, tuition fees). The relatively strong AUC (0.7125) shows the probability ranking still carries some signal, but the hard-threshold predictions are unreliable. |
| Random Forest (Ensemble) | Highest AUC (0.8861) — the best probability ranker of the five, which matters most in an early-warning context where ranking at-risk students is more useful than a hard label. Averaging across 300 trees eliminates the single Decision Tree's split-boundary noise, and random feature subsampling at each node handles the collinear semester columns better than a single greedy split. Trades interpretability for marginally superior discrimination. |
| Overall Winner for your dataset? | **Random Forest** — its AUC of 0.8861 is the highest of all five models, making it the most reliable ranker for identifying students at risk of dropping out. In a real deployment, probability scores (not just predicted labels) drive intervention priority lists, so AUC is the operationally relevant metric. Logistic Regression is the practical runner-up if a human-interpretable model is required, as it trails Random Forest by only 0.0013 in AUC while offering direct coefficient-level explanations. |

---

## Project Structure

```
bits-ml-assignment-2/
├── app.py                        # Streamlit app (entry point)
├── requirements.txt
├── README.md
├── test_data.csv                 # Held-out test split (raw features + Target column)
├── data/
│   └── students/data.csv        # UCI Student Dropout dataset (semicolon-delimited)
├── model/
│   ├── train_models.ipynb        # Training notebook — run once on BITS Virtual Lab
│   ├── logistic_regression.pkl   # Serialized sklearn Pipeline
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── metrics.json              # Pre-computed evaluation metrics for all 5 models
│   └── schema.json               # Feature schema (column names, nominal columns, class labels)
└── .streamlit/
    └── config.toml               # Custom theme
```

---

## The Streamlit App

The app has five tabs:

1. **📊 Dataset** — class balance bar chart and per-feature distributions by class (bar charts with human-readable labels for categorical features; KDE plots for continuous features).
2. **🧠 Train Model** — select one of the five models, tune its hyperparameters, train on the uploaded data, and view all 6 evaluation metrics, a confusion matrix, classification report, and feature importance / coefficients.
3. **⚖️ Compare Models** — automatically trains all five models with default hyperparameters on the same split and displays a side-by-side metrics table (best value highlighted) and a metric bar chart.
4. **📋 Model Report** — deep-dive diagnostics for any pre-trained model: confusion matrix heatmap, full classification report, and one-vs-rest ROC curves.
5. **🎯 Predict** — pick a random sample from the uploaded test data and see the model's predicted class, true label, and per-class probability breakdown.

---

## How to Run Locally

```bash
git clone https://github.com/vinodnoel/student-classifier.git
cd bits-ml-assignment-2
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Upload `test_data.csv` from the repository when prompted. The app opens at http://localhost:8501.

---

## Deploy to Streamlit Community Cloud (free)

1. Push this project to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.
3. Click **New app**, then select:
   - **Repository:** `Joseph-Vinod/bits-ml-assignment-2`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt` and starts the app automatically — first deploy takes a minute or two.
5. Any future `git push` to `main` auto-redeploys the app.

No paid tier or credit card is needed for a public app on Community Cloud.

---

## Dataset Citation

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2022). *Predict Students' Dropout and Academic Success.* UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89

## Note on Model Count

The assignment brief states "all the 6 ML models" but enumerates exactly 5. This submission implements all 5 enumerated models: Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, and Random Forest.
