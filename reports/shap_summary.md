# SHAP Explanation Summary — Phase 3

> All SHAP values were computed using the real trained models on the
> held-out test data. No values are estimated or fabricated.

## Global Feature Importance

### Classification (mean |SHAP value| over 2,000 test-set rows)

| Rank | Feature | Mean |SHAP Value| |
|---|---|---|
| 1 | revolving_util | 0.62386 |
| 2 | late_30_59_days | 0.25155 |
| 3 | age | 0.24449 |
| 4 | late_90_days | 0.17415 |
| 5 | open_credit_lines | 0.16212 |

### Regression — Income Estimation (mean |SHAP value| over 2,000 test-set rows)

| Rank | Feature | Mean |SHAP Value| |
|---|---|---|
| 1 | real_estate_loans | 971.60235 |
| 2 | num_dependents | 709.33434 |
| 3 | open_credit_lines | 451.53913 |
| 4 | age | 362.14440 |
| 5 | revolving_util | 197.89627 |

---

## Selected Instance Explanations

> For each instance the top-2 or top-3 features by |SHAP value|
> are described, along with the direction of their contribution.
> A positive SHAP value pushes the prediction higher (toward
> default / higher income); a negative SHAP value pushes it lower.

### True Positive (instance ID 34, classification)

- **Actual:** 1  **Predicted prob (default):** 0.804

**Top contributing features (by |SHAP value|):**

- `revolving_util` = 1 → SHAP +1.41355 (increases predicted default probability)
- `late_30_59_days` = 1 → SHAP +0.50774 (increases predicted default probability)
- `age` = 35 → SHAP +0.33829 (increases predicted default probability)

### True Negative (instance ID 3, classification)

- **Actual:** 0  **Predicted prob (default):** 0.119

**Top contributing features (by |SHAP value|):**

- `revolving_util` = 0.005186 → SHAP -0.65013 (decreases predicted default probability)
- `late_30_59_days` = 0 → SHAP -0.14665 (decreases predicted default probability)
- `real_estate_loans` = 0 → SHAP -0.13340 (decreases predicted default probability)

### False Positive (instance ID 2, classification)

- **Actual:** 0  **Predicted prob (default):** 0.783

**Top contributing features (by |SHAP value|):**

- `late_30_59_days` = 3 → SHAP +1.81651 (increases predicted default probability)
- `revolving_util` = 0.5853 → SHAP +0.55328 (increases predicted default probability)
- `real_estate_loans` = 0 → SHAP -0.13340 (decreases predicted default probability)

### False Negative (instance ID 73, classification)

- **Actual:** 1  **Predicted prob (default):** 0.222

**Top contributing features (by |SHAP value|):**

- `revolving_util` = 0.102 → SHAP -0.44922 (decreases predicted default probability)
- `open_credit_lines` = 13 → SHAP +0.18174 (increases predicted default probability)
- `late_30_59_days` = 0 → SHAP -0.14665 (decreases predicted default probability)

### Borderline (instance ID 1, classification)

- **Actual:** 0  **Predicted prob (default):** 0.522

**Top contributing features (by |SHAP value|):**

- `revolving_util` = 0.5683 → SHAP +0.51799 (increases predicted default probability)
- `late_30_59_days` = 1 → SHAP +0.50774 (increases predicted default probability)
- `monthly_income` = 1.2e+04 → SHAP -0.13084 (decreases predicted default probability)

### Low Income Low Error (instance ID 21769, regression)

- **Actual income:** $3,000  **Predicted:** $2,998  **Abs error:** $2

**Top contributing features (by |SHAP value|):**

- `real_estate_loans` = 0 → SHAP -1295.11376 (decreases predicted income estimate)
- `age` = 26 → SHAP -762.30382 (decreases predicted income estimate)
- `open_credit_lines` = 3 → SHAP -655.32701 (decreases predicted income estimate)

### High Income Low Error (instance ID 1721, regression)

- **Actual income:** $8,400  **Predicted:** $8,401  **Abs error:** $1

**Top contributing features (by |SHAP value|):**

- `real_estate_loans` = 2 → SHAP +1172.27822 (increases predicted income estimate)
- `num_dependents` = 2 → SHAP +858.60945 (increases predicted income estimate)
- `age` = 37 → SHAP -431.80145 (decreases predicted income estimate)

### High Error (instance ID 2015, regression)

- **Actual income:** $629,000  **Predicted:** $5,390  **Abs error:** $623,610

**Top contributing features (by |SHAP value|):**

- `real_estate_loans` = 0 → SHAP -1295.11376 (decreases predicted income estimate)
- `age` = 83 → SHAP +950.29937 (increases predicted income estimate)
- `open_credit_lines` = 3 → SHAP -655.32701 (decreases predicted income estimate)
