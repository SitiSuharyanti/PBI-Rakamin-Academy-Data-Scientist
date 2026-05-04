# Credit Risk Prediction

A machine learning project that predicts the probability of loan default using data from 2007-2014. The pipeline compares six models across two class-imbalance strategies (class weighting and SMOTE).

---

## Dataset

- **Source:** Loan Data 2007-2014
- **Size:** 466,285 loan records, 74 features (before cleaning)
- **Target variable:** `status_loan`, binary label derived from `loan_status`
  - `1` = Bad loan (Charged Off, Default, Does not meet the credit policy. Status: Charged Off, Late (31-120 days))
  - `0` = Good loan

---

## Workflow

```
Data Ingestion
    ↓
Remove Useless Features
    ↓
Define Target Variable  (bad loan = 1 / good loan = 0)
    ↓
Cleaning & Feature Engineering
    ↓
Handle Missing Values
    ↓
Encoding & Scaling
    ↓
Train / Test Split  (80 / 20, stratified)
    ↙               ↘
Baseline Models    SMOTE Models
(class_weight)   (oversampling)
    ↘               ↙
       Evaluation
           ↓
     Best Model Selected
```

---

## Models Used

Three models were selected to cover different angles of credit risk modeling:

| Model                   | Role                                                            |
| ----------------------- | --------------------------------------------------------------- |
| **Logistic Regression** | Interpretability (decisions that can be audited and explained)  |
| **Random Forest**       | Robustness (handles messy, real-world data without breaking)    |
| **Gradient Boosting**   | Performance (catches borderline cases that are easiest to miss) |

Each model is trained **twice**: once on the original data (with `class_weight='balanced'`) and once on SMOTE-balanced data, giving **6 models total** for a fair comparison.

---

## Key Preprocessing Decisions

- **Date features:** Converted to "months since reference date" (Dec 2014) to capture recency
- **High-correlation removal:** Features with pairwise correlation > 0.7 were dropped to reduce multicollinearity
- **Missing values:** Strategy varies by column (mean for continuous, 0 for count columns, -1 as sentinel for "never delinquent")
- **Data leakage prevention:** SMOTE applied only to the training set; scaler fitted on training data and applied to the test set

---

## Evaluation Metrics

Accuracy alone is misleading on imbalanced credit data. The following metrics are used:

| Metric       | Why it matters                                                                    |
| ------------ | --------------------------------------------------------------------------------- |
| ROC-AUC      | Overall ability to separate good from bad loans, regardless of threshold          |
| KS Statistic | Industry standard in credit scoring (measures score separation at each decile)    |
| Recall       | Fraction of actual bad loans correctly identified (most critical metric)          |
| FNR          | Bad loans missed by the model, directly linked to financial loss (lower = better) |
| Precision    | Of predicted bad loans, how many are actually bad                                 |
| F1 Score     | Balance between Precision and Recall                                              |

> Threshold optimization is applied per model to maximize F1-Score while monitoring FNR.

---

## Results

| Model                       | Method   | AUC        | KS         | Threshold | Precision  | Recall     | F1         | FNR        |
| --------------------------- | -------- | ---------- | ---------- | --------- | ---------- | ---------- | ---------- | ---------- |
| Gradient Boosting           | Original | **0.8796** | **0.6088** | 0.30      | 0.8857     | **0.5309** | **0.6639** | **0.4691** |
| Random Forest               | Original | 0.8739     | 0.5982     | 0.32      | 0.8990     | 0.5195     | 0.6585     | 0.4805     |
| Random Forest (SMOTE)       | SMOTE    | 0.8686     | 0.5882     | 0.54      | 0.9096     | 0.5091     | 0.6528     | 0.4909     |
| Gradient Boosting (SMOTE)   | SMOTE    | 0.8584     | 0.5710     | 0.61      | **0.9413** | 0.4965     | 0.6501     | 0.5035     |
| Logistic Regression         | Original | 0.8510     | 0.5594     | 0.73      | 0.8468     | 0.4879     | 0.6191     | 0.5121     |
| Logistic Regression (SMOTE) | SMOTE    | 0.8264     | 0.5114     | 0.43      | 0.8657     | 0.4679     | 0.6074     | 0.5321     |

**Gradient Boosting (original)** ranked first on AUC, KS, Recall, F1, and FNR. The five metrics most critical to credit risk. SMOTE did not improve performance here because Gradient Boosting already handles class imbalance through its iterative loss-reweighting mechanism.

---

## Tech Stack

| Library                  | Purpose                             |
| ------------------------ | ----------------------------------- |
| `pandas` / `numpy`       | Data manipulation                   |
| `scikit-learn`           | Preprocessing, modeling, evaluation |
| `matplotlib` / `seaborn` | Visualization                       |
| `imbalanced-learn`       | SMOTE                               |

---

## Repository Structure

```
├── Credit_Risk.ipynb
└── README.md
```

---

## How to Run

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
   ```
3. Download the dataset via the `gdown` command in the notebook (Google Drive link included)
4. Run all cells in `Credit_Risk.ipynb`

---

_This project is part of the **Final Project for Virtual Intern: ID/X Partners x Rakamin Academy**._
