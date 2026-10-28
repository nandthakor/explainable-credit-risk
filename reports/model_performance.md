# Model Performance Summary — Phase 2

> All numbers below come directly from `src/train_classification.py` and
> `src/train_regression.py` running on the real cleaned dataset. Nothing is
> estimated or assumed.

---

## 1. Classification — Predicting Credit Default

### Task
Binary classification: predict `target_default` (0 = no default, 1 = defaulted
within 2 years) on the 149,730-row cleaned dataset.

### Class Imbalance and Why It Matters for Metric Choice

The dataset has a **93.40% / 6.60%** class split (139,851 non-defaults vs 9,879
defaults) in the full cleaned set. After stratified 80/20 splitting, both
train and test folds preserve exactly **6.60%** positive rate.

A "dumb" classifier that predicts *no default* for every borrower would achieve
**93.4% accuracy** while catching exactly zero defaults — useless for a lender.
This is why:

- **ROC-AUC** is the primary comparison metric: it measures how well the model
  ranks defaulters above non-defaulters regardless of the decision threshold,
  and is insensitive to class imbalance.
- **F1 Score** (harmonic mean of precision and recall) is the secondary metric:
  it penalises both missed defaults (false negatives) and false alarms (false
  positives) equally, making it more informative than accuracy on imbalanced data.
- Both models were trained with `class_weight='balanced'` so the loss function
  up-weights the minority class during training.

### Test-Set Results (20% held-out, 29,946 rows)

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | 0.8064 | 0.9354 |
| Precision | 0.2144 | 0.5430 |
| Recall | 0.7262 | 0.1311 |
| F1 Score | 0.3311 | 0.2112 |
| **ROC-AUC** | **0.8496** | **0.8333** |

### Winner: Logistic Regression (ROC-AUC margin: 0.0163)

The **Logistic Regression** pipeline (StandardScaler → LR) beats Random Forest
on both ROC-AUC (0.8496 vs 0.8333) and F1 (0.3311 vs 0.2112).

The result is instructive: the Random Forest achieves 93.54% accuracy by
essentially predicting "no default" for most borrowers (it caught only 259 of
1,976 actual defaults — a recall of just 13.1%). Logistic Regression sacrifices
raw accuracy (80.6%) but catches 72.6% of actual defaulters, which is far more
useful for a credit-risk application. **In lending, missing a default is
typically far more costly than a false alarm.**

Saved to: `models/classification_model.joblib`

### Confusion Matrices

**Logistic Regression** (winner)

| | Predicted 0 | Predicted 1 |
|---|---|---|
| Actual 0 | 22,713 | 5,257 |
| Actual 1 | 541 | 1,435 |

**Random Forest**

| | Predicted 0 | Predicted 1 |
|---|---|---|
| Actual 0 | 27,752 | 218 |
| Actual 1 | 1,717 | 259 |

### Top Classification Features (Random Forest MDI importances — not SHAP yet)

| Rank | Feature | MDI Importance |
|---|---|---|
| 1 | revolving_util | 0.2546 |
| 2 | debt_ratio | 0.1388 |
| 3 | age | 0.1186 |
| 4 | monthly_income | 0.1127 |
| 5 | late_30_59_days | 0.0901 |
| 6 | late_90_days | 0.0860 |
| 7 | open_credit_lines | 0.0807 |

`revolving_util` (25.5% of MDI importance) is the single most influential
feature — this measures how much of a borrower's available revolving credit is
in use. High utilisation is a well-known predictor of financial distress.
`debt_ratio` (13.9%) and `age` (11.9%) follow, with the three late-payment
count columns collectively contributing about 22% of importance. Notably,
`monthly_income_missing` (the engineered flag) contributes only 0.45% —
the imputed income value itself carries most of the income signal.

> MDI importances can be biased toward continuous features and high-cardinality
> columns. Phase 3 will replace these with SHAP values, which are more reliable
> and provide per-prediction local explanations.

---

## 2. Regression — Estimating Monthly Income

### Task
Predict `monthly_income` (USD/month) using the 120,120 rows where income is
known as training data (~80% of the full dataset). The trained model will later
be applied to the 29,610 rows with missing income to estimate it — simulating
the real banking problem of unverified income estimation.

### Test-Set Results (20% held-out, 24,024 rows)

| Metric | Linear Regression | Random Forest Regressor |
|---|---|---|
| MAE (USD) | 3,120.44 | 2,566.80 |
| RMSE (USD) | 7,136.46 | 9,677.14 |
| **R²** | **0.0793** | **-0.6929** |

### Winner: Linear Regression (R² margin: 0.7722)

**Linear Regression** wins decisively by R² (0.0793 vs −0.6929). Both models
struggle — income is highly right-skewed and the available features have low
signal for income level. However, the results tell a clear story:

- The Random Forest achieved lower MAE (2,567 vs 3,120) but a *negative* R²
  (−0.6929), which means it explains *less* variance than simply predicting the
  mean income for everyone. This is a classic sign of **overfitting to extreme
  outlier values**: the RF memorised specific high-income outliers in training
  but generalised poorly to the test set, leading to enormous prediction errors
  (RMSE \$9,677 vs \$7,136 for Linear Regression).
- Both models' low R² values (max 0.08) reflect the fundamental limitation: the
  features available (payment history, credit lines, debt ratio) are proxies for
  creditworthiness, not direct income predictors. Phase 5 will revisit this with
  better feature engineering.

Saved to: `models/regression_model.joblib`

### Top Regression Features (Random Forest MDI importances — not SHAP yet)

| Rank | Feature | MDI Importance |
|---|---|---|
| 1 | debt_ratio | 0.5956 |
| 2 | open_credit_lines | 0.1354 |
| 3 | revolving_util | 0.1185 |
| 4 | age | 0.0660 |
| 5 | real_estate_loans | 0.0404 |

`debt_ratio` dominates (59.6%), which makes intuitive sense: a person with more
debt relative to income has a higher debt ratio, so the ratio partially reflects
income level. `open_credit_lines` (13.5%) suggests that higher-income borrowers
tend to have more credit accounts. These MDI values should be treated cautiously
given the negative test-set R² — Phase 3 SHAP analysis will provide a more
reliable picture.

---

## 3. Files Produced in Phase 2

| File | Description |
|---|---|
| `data/processed/X_train_clf.csv` | 119,784 rows × 11 features — classification train |
| `data/processed/X_test_clf.csv` | 29,946 rows × 11 features — classification test |
| `data/processed/y_train_clf.csv` / `y_test_clf.csv` | Classification targets |
| `data/processed/X_train_reg.csv` | 96,096 rows × 10 features — regression train |
| `data/processed/X_test_reg.csv` | 24,024 rows × 10 features — regression test |
| `data/processed/y_train_reg.csv` / `y_test_reg.csv` | Regression targets |
| `reports/feature_columns_clf.json` | Ordered list of 11 classification features |
| `reports/feature_columns_reg.json` | Ordered list of 10 regression features |
| `reports/prep_metadata.json` | Train-fold income median + prep constants |
| `reports/classification_performance.md` | Full classification metrics table |
| `reports/regression_performance.md` | Full regression metrics table |
| `models/classification_model.joblib` | Best classifier (Logistic Regression pipeline) |
| `models/regression_model.joblib` | Best regressor (Linear Regression) |
| `reports/figures/logistic_regression_confusion.png` | LR confusion matrix |
| `reports/figures/random_forest_confusion.png` | RF confusion matrix |
| `reports/figures/clf_feature_importance.png` | RF classification feature importances |
| `reports/figures/linear_regression_residuals.png` | Linear Reg residual plots |
| `reports/figures/random_forest_regressor_residuals.png` | RF Reg residual plots |
| `reports/figures/reg_feature_importance.png` | RF regression feature importances |
