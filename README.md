# Student Dropout & Academic Success Classification

> **Created by Joseph Mruthunjaya Vinod Noel · BITS ID: 2025AC05003**

---

## a. Problem Statement

Predicting whether a student will drop out, remain enrolled, or graduate is normally identified only after it is too late to intervene. This project builds and compares five classic supervised classification algorithms that predict a student's academic outcome (one of 3 classes) from 36 demographic, academic, and socioeconomic features recorded at enrolment time, and packages them in an interactive Streamlit app for training, evaluating, and testing the models.

---

## b. Dataset Description

- **Source:** [UCI ML Repository — Predict Students' Dropout and Academic Success (Dataset #697)](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- **Samples:** 4,424 student records
- **Classes (3):** Dropout, Enrolled, Graduate
- **Features (36):** demographic background (age, gender, nationality, marital status, international student flag), academic performance (previous qualification grade, admission grade, curricular units credited/enrolled/approved/graded across two semesters), and socioeconomic context (parents' education and occupation, scholarship holder, debtor, tuition fees up to date, unemployment rate, inflation rate, GDP).
- **Split:** 75% train (3,318) / 25% test (1,106), stratified by class, with `random_state=42`.
- **Preprocessing:** Median imputation + standard scaling for 12 continuous numeric features; mode imputation + one-hot encoding for 24 integer-coded nominal/categorical features.
- **Class balance:** Moderately imbalanced — Graduate (49.9%) dominates; Enrolled (17.9%) is the smallest class and hardest to predict, which depresses macro-averaged recall across all models.
- **Meets assignment minimums:** 36 features (≥ 12 required) and 4,424 instances (≥ 500 required).

*Citation: Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2022). Predict Students' Dropout and Academic Success. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89*

### Exploratory Data Analysis

A full EDA notebook is available at [`model/eda_student_dropout.ipynb`](model/eda_student_dropout.ipynb). Key findings:

- **Graduate students dominate (~49%)** with Enrolled the smallest class (~18%), creating the moderate imbalance that suppresses macro-averaged recall across all models.
- **Admission grade and 2nd-semester grade are the strongest numeric separators** — Graduate students score noticeably higher on both, explaining their prominence as top Logistic Regression features.
- **Tuition fees up to date and scholarship status are the most discriminating categorical features** — students current on fees and scholarship holders have markedly higher Graduate proportions and lower Dropout rates.
- **Exact class counts:** Dropout 1,421 (32.1%), Enrolled 794 (17.9%), Graduate 2,209 (49.9%).
- **Column-typing audit confirms the preprocessing split is semantically correct**, not just dtype-based: 24 integer-coded columns with ≤30 unique values are genuine nominal categories (e.g. Course, Marital status), and the 12 remaining columns are truly continuous (grades, rates, GDP).
- **Zero missing values and zero duplicate rows** in the raw dataframe — the pipeline's `SimpleImputer` never actually fires on this dataset and is retained purely as a safeguard for user-uploaded data in the Streamlit app.
- **Numeric feature ranges span up to 43×** (e.g. GDP ≈ [-4, 3] vs. Father's occupation ≈ [0, 195]), confirming `StandardScaler` is necessary rather than optional for distance- and gradient-based models.
- **IQR outlier audit flags two columns** for elevated outlier rates — Age at enrollment (10.0% of rows) and 1st-semester grade (16.4%) — both retained as domain-plausible (older re-entrant students, zero-grade non-completions) rather than data errors.
- **Correlation heatmap shows admission grade, 1st-semester grade, and 2nd-semester grade are moderately positively correlated**, introducing mild multicollinearity that Logistic Regression handles via standardisation but that contributes to the single Decision Tree's instability.
- A full **data dictionary** covering all 36 features + target (sourced from the UCI variable table, CC BY 4.0) is included in the notebook, since 24 of 36 features are integer-coded categoricals that are unreadable without a legend.

---

## c. GitHub Repository Link

[https://github.com/vinodnoel/ml_studentclassifier](https://github.com/vinodnoel/ml_studentclassifier)

## Live Streamlit App Link

[https://studentclassifier-josephvinod-bits.streamlit.app/](https://studentclassifier-josephvinod-bits.streamlit.app/)

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
| Logistic Regression | 0.7613 | 0.8812 | 0.7122 | 0.6856 | 0.6949 | 0.6046 |
| Decision Tree | 0.7306 | 0.8203 | 0.6734 | 0.6538 | 0.6606 | 0.5547 |
| K-Nearest Neighbors | 0.6899 | 0.8202 | 0.6415 | 0.5841 | 0.5943 | 0.4814 |
| Gaussian Naive Bayes | 0.6763 | 0.8340 | 0.6503 | 0.6554 | 0.6409 | 0.5025 |
| Random Forest (Ensemble) | 0.7459 | 0.8735 | 0.7129 | 0.6192 | 0.6146 | 0.5791 |

*Precision, Recall, and F1 are macro-averaged across all 3 classes; AUC is one-vs-rest macro-averaged. Evaluated on the held-out test set (1,106 samples, `test_size=0.25, random_state=42`).*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest accuracy (0.7613) and F1 (0.6950) of all five models. The linear decision boundary works well here because, after standardisation and one-hot expansion, the Dropout and Graduate classes are broadly separable in feature space — particularly along the curricular-unit grade and approval-rate axes. The Enrolled class, which sits ambiguously between the other two outcomes, is the main source of misclassification. Coefficient magnitudes also provide direct interpretability: tuition fees up to date and 2nd-semester approved units emerge as the strongest predictors. |
| Decision Tree | Second-lowest AUC (0.8203). The single tree's recursive splits can capture non-linear feature interactions (e.g. age × scholarship status), but depth-capping at `max_depth=8` to prevent overfitting leaves some boundary complexity unexplored. Without ensemble averaging, predictions near split thresholds are noisy, resulting in a meaningful accuracy gap (0.7306) vs. Logistic Regression — despite both models sharing identical preprocessing. |
| K-Nearest Neighbors | Weakest F1 (0.5943) of all five models, and second-weakest accuracy (0.6899) — Gaussian Naive Bayes has the single lowest accuracy at 0.6763. One-hot encoding the 24 nominal columns inflates the feature space, diluting Euclidean distances between samples — the classic curse of dimensionality for kNN. The Enrolled minority class suffers most, pulling macro recall down to 0.5841. With `k=15`, the model is also forced to average across many potentially dissimilar neighbours in this high-dimensional space. |
| Gaussian Naive Bayes | Third-best AUC (0.8340) and the lowest accuracy (0.6763) of all five models. Gaussian Naive Bayes assumes each feature follows a class-conditional Gaussian distribution. With `var_smoothing=1e-2` a fraction of the largest feature variance is added to all per-class variances, preventing near-zero variance on low-variance integer columns. The independence assumption is only approximately satisfied — several curricular-unit features are correlated — but the calibrated probability estimates still produce a useful AUC, and its F1 (0.6409) actually edges out both Random Forest (0.6146) and KNN (0.5943). |
| Random Forest (Ensemble) | Deliberately bounded to `n_estimators=150, max_depth=12, min_samples_leaf=5` (down from an initial unbounded 300-tree configuration) after the larger forest produced a 22.8 MB pickle that triggered `MemoryError` crashes on Streamlit Community Cloud's free-tier container. This trades some predictive performance for deployability: accuracy drops to 0.7459 and AUC to 0.8735 — both now behind Logistic Regression — though Random Forest keeps a narrow precision edge (0.7129 vs 0.7122). Random feature subsampling still handles the collinear semester-grade columns better than a single greedy split, but with fewer, shallower trees the ensemble's variance-reduction benefit is smaller than an unconstrained forest would give. |
| Overall Winner for your dataset? | **Logistic Regression** — it now leads on 5 of 6 metrics (accuracy, AUC, recall, F1, and MCC), including MCC (0.6046), the metric most robust to class imbalance and generally recommended for multi-class problems like this one. Random Forest's only remaining edge is precision, by a margin of 0.0007 — well within noise. This gap widened after Random Forest's tree count and depth were deliberately constrained to keep the deployed pickle small enough for Streamlit Community Cloud's memory limit (see observation above); an unconstrained forest would likely have stayed closer to Logistic Regression, but the smaller, deployable configuration is what's actually running in production, and that is the fair basis for comparison. |

---

## Project Structure

```
ml_studentclassifier/
├── app.py                       
├── requirements.txt
├── README.md
├── test_data.csv                
├── data
├── model/
│   ├── train_models.ipynb        
│   ├── eda_student_dropout.ipynb 
│   ├── logistic_regression.pkl  
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── metrics.json             
│   └── schema.json               
└── .streamlit/
    └── config.toml
```

---

## The Streamlit App

The app has the following sections:

1. **📊 Data Explorer** — examine the dataset: class balance bar chart and per-feature distributions by class (bar charts with human-readable labels for categorical features; KDE plots for continuous features).
2. **🧠 Model Training** — select one of the five models, tune its hyperparameters, train on the uploaded data, and view all 6 evaluation metrics, a confusion matrix, classification report, and feature importance / coefficients.
3. **⚖️ Model Comparison** — automatically trains all five models with default hyperparameters on the same split and displays a side-by-side metrics table (best value highlighted) and a metric bar chart.
4. **📋 Diagnostics** — deep-dive diagnostics for any pre-trained model: confusion matrix heatmap, full classification report, and one-vs-rest ROC curves.
5. **🎯 Predict Outcome** — pick a random sample from the uploaded test data and see the model's predicted class, true label, and per-class probability breakdown.
6. **📤 Data Setup** — upload your own test dataset (CSV), download the sample template, and review the required column reference.

---

## How to Run Locally

```bash
git clone https://github.com/vinodnoel/ml_studentclassifier.git
cd ml_studentclassifier
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Upload `test_data.csv` from the repository when prompted. The app opens at http://localhost:8501.

---

## Deployed on Streamlit Community Cloud

- **Repository:** `vinodnoel/ml_studentclassifier`
- **Branch:** `main`
- **Main file path:** `app.py`
- **Live app:** [https://studentclassifier-josephvinod-bits.streamlit.app/](https://studentclassifier-josephvinod-bits.streamlit.app/)

Any future `git push` to `main` auto-redeploys the app. No paid tier or credit card is needed for a public app on Community Cloud.

---

## Dataset Citation

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2022). *Predict Students' Dropout and Academic Success.* UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89

## Note on Model Count

The assignment brief states "all the 6 ML models" but enumerates exactly 5. This submission implements all 5 enumerated models: Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, and Random Forest.
