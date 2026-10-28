# Regression Model Performance — Phase 2

> All metrics were computed on the 20% held-out test set.
> Task: predict monthly income using rows where income is known (~80% of data).

## Models Trained

| Model | n_estimators | random_state |
|---|---|---|
| Linear Regression | N/A | N/A |
| Random Forest Regressor | 200 | 42 |

## Test-Set Metrics

| Metric | Linear Regression | Random Forest |
|---|---|---|
| MAE  (USD) | 3,120.44 | 2,566.80 |
| RMSE (USD) | 7,136.46 | 9,677.14 |
| **R²** | **0.0793** | **-0.6929** |

## Winner

**Linear Regression** wins by R²: 0.0793 vs -0.6929 (margin: 0.7722).

Saved to: `models/regression_model.joblib`

## Top-5 Random Forest Feature Importances (MDI, not SHAP)

| Rank | Feature | MDI Importance |
|---|---|---|
| 1 | debt_ratio | 0.5956 |
| 2 | open_credit_lines | 0.1354 |
| 3 | revolving_util | 0.1185 |
| 4 | age | 0.0660 |
| 5 | real_estate_loans | 0.0404 |

> Note: MDI importances can be biased toward high-cardinality features.
> Phase 3 will use SHAP for more reliable feature attribution.