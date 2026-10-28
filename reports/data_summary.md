# Data Summary — Credit Risk XAI Project (Phase 1)

> **All statistics below were produced by running `src/data_prep.py` and
> `notebooks/01_eda.py` on the real dataset. No number is estimated or assumed.**

---

## 1. Dataset Overview

| Metric | Value |
|---|---|
| Source | Give Me Some Credit — Kaggle |
| Raw file | `data/raw/cs-training.csv` |
| Rows on load | 150,000 |
| Columns on load | 11 |
| Data-entry artifact rows flagged & removed | 270 |
| **Rows in cleaned file** (`credit_data_clean.csv`) | **149,730** |

---

## 2. Target Class Imbalance

The dataset is significantly imbalanced, as expected for real-world credit default data.

| Class | Label | Count | Percentage |
|---|---|---|---|
| 0 | No default within 2 years | 139,851 | **93.40%** |
| 1 | Defaulted within 2 years | 9,879 | **6.60%** |

This ~14:1 ratio between the majority and minority class is a key modelling
challenge that will need to be addressed in Phase 2 (e.g. SMOTE, class weights,
or threshold tuning).

---

## 3. Missingness (Before Any Imputation)

Only two columns have missing values:

| Column | Missing Count | Missing % | Treatment |
|---|---|---|---|
| `monthly_income` | 29,731 | **19.82%** | **Left as-is intentionally** — rows with known income form the training set for an income-estimation regression model in a later phase |
| `num_dependents` | 3,924 | **2.62%** | **Imputed with column median (0.0)** — a small fraction, and median is appropriate for a count variable |

---

## 4. Data-Entry Artifacts Flagged

The following known data-entry issues in this dataset were flagged and saved to
`data/processed/flagged_rows.csv`. They were **removed** from the main cleaned file.

| Artifact type | Rows affected |
|---|---|
| `age == 0` (physiologically impossible) | 1 |
| Late-payment columns containing sentinel codes 96 or 98 | 269 |
| **Total flagged (union)** | **270** |

The 270 flagged rows (0.18% of the dataset) are preserved in
`data/processed/flagged_rows.csv` for audit purposes.

---

## 5. EDA Observations

All observations below are based on what the plots in `reports/figures/` actually show.

### 5.1 The dataset is severely class-imbalanced
Only 6.60% of borrowers defaulted. Any model trained without handling this imbalance
will be biased toward predicting "no default" for almost everyone and will achieve
high accuracy while missing the minority class entirely.

### 5.2 `revolving_util` and `debt_ratio` are extremely right-skewed with wild outliers
- `revolving_util` has a median of **0.15** but a maximum of **50,708** and a
  skewness of **97.54** — the vast majority of borrowers have utilisation below 1.0
  (i.e. below 100%), but a handful of extreme values drag the mean to **6.06**.
- `debt_ratio` similarly has a median of **0.37** but a maximum of **329,664** and
  skewness of **95.08**. These extreme values are likely data errors or edge cases
  (e.g. retired borrowers with near-zero income).
- Both columns will need log-transformation or robust scaling before being used
  as model features.

### 5.3 `monthly_income` is right-skewed with a median of \$5,400
The income distribution is right-skewed (skewness = **113.99**), which is typical for
income data — most borrowers cluster in a middle range (median \$5,400/month) while
a long right tail of high earners extends the distribution. Missing income data
(19.82%) is concentrated in a meaningful fraction of the population and will be
estimated in a later phase.

### 5.4 `age` is approximately normally distributed, centred around 52
Age ranges from ~21 to ~87 (in the cleaned data) with a median of **52 years** and
very low skewness (0.19). The distribution is roughly bell-shaped, making it
the most well-behaved feature in the dataset with no preprocessing concerns.

### 5.5 The three late-payment columns are the strongest predictors of default
From the correlation heatmap:
- `late_90_days` has the highest correlation with `target_default` (r = **0.31**)
- `late_30_59_days` follows at r = **0.27**, and `late_60_89_days` at r = **0.26**
- The three late-payment columns are also moderately correlated with each other
  (e.g. `late_30_59_days` vs `late_60_89_days` r = **0.31**), which is expected —
  borrowers who miss one payment are likely to miss others.
- `open_credit_lines` and `real_estate_loans` are strongly correlated with each
  other (r = **0.43**), suggesting potential multicollinearity to monitor in Phase 2.
- `debt_ratio` shows near-zero correlation with almost everything, likely due to
  the extreme outliers suppressing meaningful signal in Pearson's r.

---

## 6. Output Files

| File | Description |
|---|---|
| `data/processed/credit_data_clean.csv` | 149,730 rows, cleaned, ready for Phase 2 |
| `data/processed/flagged_rows.csv` | 270 rows with data-entry artifacts, for audit |
| `reports/figures/01_class_balance.png` | Bar chart of target class distribution |
| `reports/figures/02_missingness.png` | Horizontal bar chart of % missing per column |
| `reports/figures/03_histograms.png` | Histograms: age, debt_ratio, revolving_util, monthly_income |
| `reports/figures/04_correlation_heatmap.png` | Pearson correlation heatmap of all numeric features |
