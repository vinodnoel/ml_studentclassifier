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
- **Split:** 75% train (3,318) / 25% test (1,106), stratified by class, with `random_state=42`.
- **Preprocessing:** Median imputation + standard scaling for 12 continuous numeric features; mode imputation + one-hot encoding for 24 integer-coded nominal/categorical features.
- **Class balance:** Moderately imbalanced — Graduate (49.9%) dominates; Enrolled (17.9%) is the smallest class and hardest to predict, which depresses macro-averaged recall across all models.
- **Meets assignment minimums:** 36 features (≥ 12 required) and 4,424 instances (≥ 500 required).

*Citation: Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2022). Predict Students' Dropout and Academic Success. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89*

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
4. Bernoulli Naive Bayes
5. Random Forest (Ensemble)

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7613 | 0.8812 | 0.7126 | 0.6856 | 0.6950 | 0.6046 |
| Decision Tree | 0.7306 | 0.8203 | 0.6734 | 0.6538 | 0.6606 | 0.5547 |
| K-Nearest Neighbors | 0.6899 | 0.8202 | 0.6415 | 0.5841 | 0.5943 | 0.4814 |
| Bernoulli Naive Bayes | 0.6890 | 0.8349 | 0.6500 | 0.6573 | 0.6490 | 0.5106 |
| Random Forest (Ensemble) | 0.7595 | 0.8823 | 0.7200 | 0.6511 | 0.6591 | 0.6005 |

*Precision, Recall, and F1 are macro-averaged across all 3 classes; AUC is one-vs-rest macro-averaged. Evaluated on the held-out test set (1,106 samples, `test_size=0.25, random_state=42`).*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest accuracy (0.7613) and F1 (0.6950) of all five models. The linear decision boundary works well here because, after standardisation and one-hot expansion, the Dropout and Graduate classes are broadly separable in feature space — particularly along the curricular-unit grade and approval-rate axes. The Enrolled class, which sits ambiguously between the other two outcomes, is the main source of misclassification. Coefficient magnitudes also provide direct interpretability: tuition fees up to date and 2nd-semester approved units emerge as the strongest predictors. |
| Decision Tree | Second-lowest AUC (0.8203). The single tree's recursive splits can capture non-linear feature interactions (e.g. age × scholarship status), but depth-capping at `max_depth=8` to prevent overfitting leaves some boundary complexity unexplored. Without ensemble averaging, predictions near split thresholds are noisy, resulting in a meaningful accuracy gap (0.7306) vs. Logistic Regression — despite both models sharing identical preprocessing. |
| K-Nearest Neighbors | Weakest accuracy (0.6899) and F1 (0.5943). One-hot encoding the 24 nominal columns inflates the feature space, diluting Euclidean distances between samples — the classic curse of dimensionality for kNN. The Enrolled minority class suffers most, pulling macro recall down to 0.5841. With `k=15`, the model is also forced to average across many potentially dissimilar neighbours in this high-dimensional space. |
| Bernoulli Naive Bayes | Third-best F1 (0.6490) and MCC (0.5106), close behind Decision Tree. Bernoulli Naive Bayes is the appropriate variant here since 24 of the 36 features are one-hot/binary indicators — it models each feature's presence/absence probability directly, rather than assuming a continuous Gaussian distribution that binary columns don't follow. AUC (0.8349) is comfortably ahead of Decision Tree and KNN, showing the class-conditional probability estimates rank students reasonably well despite the independence assumption being only approximately true (several curricular-unit features are correlated). |
| Random Forest (Ensemble) | Highest AUC (0.8823), but only by a 0.0011 margin over Logistic Regression — essentially a tie in ranking quality. Averaging across 300 trees eliminates the single Decision Tree's split-boundary noise, and random feature subsampling at each node handles the collinear semester columns better than a single greedy split. However, it trails Logistic Regression on accuracy (0.7595 vs 0.7613), recall, F1 (0.6591 vs 0.6950), and MCC (0.6005 vs 0.6046), so the AUC edge does not translate into better hard classifications. |
| Overall Winner for your dataset? | **Logistic Regression** — it leads on 4 of 6 metrics (accuracy, recall, F1, and MCC), including MCC (0.6046 vs 0.6005), the metric most robust to class imbalance and generally recommended for multi-class problems like this one. Random Forest's only advantages are AUC and precision, both by margins under 0.01, so they don't outweigh Logistic Regression's broader lead in actual classification performance. Random Forest remains the better choice specifically if the deployment goal is ranking students by risk probability (e.g. a triage/priority list) rather than producing hard labels, since AUC measures ranking quality independent of a decision threshold. |

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

1. **📊 Dataset** — examine the dataset: class balance bar chart and per-feature distributions by class (bar charts with human-readable labels for categorical features; KDE plots for continuous features).
2. **📤 Upload Own Data** — upload your own test dataset (CSV) to train, predict, and compare model performance on it.
3. **🧠 Train Model** — select one of the five models, tune its hyperparameters, train on the uploaded data, and view all 6 evaluation metrics, a confusion matrix, classification report, and feature importance / coefficients.
4. **📋 Model Report** — deep-dive diagnostics for any pre-trained model: confusion matrix heatmap, full classification report, and one-vs-rest ROC curves.
5. **⚖️ Compare Models** — automatically trains all five models with default hyperparameters on the same split and displays a side-by-side metrics table (best value highlighted) and a metric bar chart.
6. **🎯 Predict** — pick a random sample from the uploaded test data and see the model's predicted class, true label, and per-class probability breakdown.

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

The assignment brief states "all the 6 ML models" but enumerates exactly 5. This submission implements all 5 enumerated models: Logistic Regression, Decision Tree, K-Nearest Neighbors, Bernoulli Naive Bayes, and Random Forest.
