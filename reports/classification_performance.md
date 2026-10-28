# Classification Model Performance — Phase 2

> All metrics were computed on the 20% held-out test set.
> No test-set statistics were used during training or hyperparameter selection.

## Models Trained

| Model | class_weight | max_iter / n_estimators | random_state |
|---|---|---|---|
| Logistic Regression (StandardScaler pipeline) | balanced | max_iter=2000 | 42 |
| Random Forest | balanced | n_estimators=200 | 42 |

## Test-Set Metrics

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy  | 0.8064 | 0.9354 |
| Precision | 0.2144 | 0.5430 |
| Recall    | 0.7262 | 0.1311 |
| F1 Score  | 0.3311 | 0.2112 |
| **ROC-AUC** | **0.8496** | **0.8333** |

## Confusion Matrices

**Logistic Regression**

| | Predicted 0 | Predicted 1 |
|---|---|---|
| Actual 0 | 22,713 | 5,257 |
| Actual 1 | 541 | 1,435 |

**Random Forest**

| | Predicted 0 | Predicted 1 |
|---|---|---|
| Actual 0 | 27,752 | 218 |
| Actual 1 | 1,717 | 259 |

## Winner

**Logistic Regression** wins by ROC-AUC: 0.8496 vs 0.8333 (margin: 0.0163).

Saved to: `models/classification_model.joblib`

## Top-5 Random Forest Feature Importances (MDI, not SHAP)

| Rank | Feature | MDI Importance |
|---|---|---|
| 1 | revolving_util | 0.2546 |
| 2 | debt_ratio | 0.1388 |
| 3 | age | 0.1186 |
| 4 | monthly_income | 0.1127 |
| 5 | late_30_59_days | 0.0901 |

> Note: MDI importances can be biased toward high-cardinality features.
> Phase 3 will use SHAP for more reliable feature attribution.