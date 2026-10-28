# 5. IMPLEMENTATION SCREENSHOTS WITH EXPLANATIONS

> All numerical values in this chapter are drawn directly from the
> project's output files (reports/data_summary.md,
> reports/model_performance.md, reports/shap_summary.md,
> reports/counterfactual_summary.md, reports/llm_generation_summary.md,
> reports/evaluation_summary.md). No value is fabricated.

---

## 5.1 Data Preparation

### Dataset and Initial Loading

The Give Me Some Credit dataset was obtained from Kaggle and placed at
`data/raw/cs-training.csv`. On load, the raw file contained 150,000 rows
and 11 columns. All column names were standardised to lowercase
snake_case identifiers for consistent handling throughout the pipeline.

[SCREENSHOT 5.1: Terminal output of data_prep.py showing column names
and dtypes confirmed on load.]

### Data-Entry Artifact Removal

The dataset is known to contain data-quality artefacts. Two categories
were identified and addressed:

- One row with `age = 0` was removed as physiologically impossible.
- 269 rows containing sentinel codes 96 or 98 in the three late-payment
  count columns were removed, as these represent data-entry placeholders
  rather than genuine payment history values.

Following removal of these 270 rows (0.18% of the dataset), the cleaned
file contained **149,730 rows**, written to
`data/processed/credit_data_clean.csv`.

[SCREENSHOT 5.2: data_prep.py output confirming 270 flagged rows removed
and 149,730 rows retained in the cleaned file.]

### Class Imbalance

The target variable `target_default` is severely imbalanced. Of the
149,730 cleaned rows, 139,851 (93.40%) belong to class 0 (no default
within two years) and 9,879 (6.60%) belong to class 1 (defaulted within
two years). This approximately 14:1 ratio requires class-weighted
training and precludes accuracy as a meaningful performance metric.

[FIGURE 5.1: Class balance bar chart — reports/figures/01_class_balance.png]

### Missingness

Two columns contained missing values:

- `monthly_income`: 29,731 missing (19.82%) — retained intentionally,
  as these rows define the held-out population for the income-estimation
  regression task. A binary flag `monthly_income_missing` was added.
- `num_dependents`: 3,924 missing (2.62%) — imputed with the column
  median (0.0), appropriate for a count variable.

[FIGURE 5.2: Missingness bar chart — reports/figures/02_missingness.png]

### EDA Observations

Exploratory analysis revealed that `revolving_util` and `debt_ratio`
contain extreme outliers: `revolving_util` has a median of 0.15 but a
maximum of 50,708 (skewness 97.54), and `debt_ratio` has a median of
0.37 but a maximum of 329,664 (skewness 95.08). These are addressed by
Winsorisation in Phase 2. The three late-payment columns show the
highest correlations with the default target: `late_90_days` r = 0.31,
`late_30_59_days` r = 0.27, and `late_60_89_days` r = 0.26.

[FIGURE 5.3: Histogram grid — reports/figures/03_histograms.png]
[FIGURE 5.4: Pearson correlation heatmap —
reports/figures/04_correlation_heatmap.png]

---

## 5.2 Predictive Models

### Outlier Handling

Before model training, five columns (`debt_ratio`, `revolving_util`,
`late_30_59_days`, `late_60_89_days`, `late_90_days`) were Winsorised
by capping values above the 99th percentile at that percentile's value.
This approach was chosen over row deletion to preserve sample size; the
extreme values in these columns are likely data errors, not genuine
signals, and capping limits their leverage on the model's parameters.

### Classification Model Selection

Two models were trained on the 119,784-row training split (stratified
80/20) and evaluated on the 29,946-row test split.

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | 0.8064 | 0.9354 |
| Precision | 0.2144 | 0.5430 |
| Recall | 0.7262 | 0.1311 |
| F1 Score | 0.3311 | 0.2112 |
| **ROC-AUC** | **0.8496** | **0.8333** |

The **Logistic Regression Pipeline** (StandardScaler followed by
LogisticRegression with `class_weight='balanced'`, `max_iter=2000`,
`random_state=42`) was selected as the winning model (ROC-AUC 0.8496).
Although Random Forest achieves 93.54% accuracy, this is achieved by
predicting "no default" for most borrowers: it catches only 259 of
1,976 actual defaults (recall 13.1%), which is inadequate for a
credit-risk application. Logistic Regression sacrifices raw accuracy
(80.6%) in exchange for catching 72.6% of actual defaulters — a far
more useful trade-off in lending, where a missed default is typically
more costly than a false alarm.

[FIGURE 5.5: Logistic Regression confusion matrix —
reports/figures/logistic_regression_confusion.png]
[FIGURE 5.6: Random Forest confusion matrix —
reports/figures/random_forest_confusion.png]

### Regression Model Selection

For income estimation, 96,096 rows with known income formed the training
set (stratified 80/20 split on income quartiles), evaluated on 24,024
held-out rows.

| Metric | Linear Regression | Random Forest Regressor |
|---|---|---|
| MAE (USD) | 3,120.44 | 2,566.80 |
| RMSE (USD) | 7,136.46 | 9,677.14 |
| **R²** | **0.0793** | **-0.6929** |

**Linear Regression** was selected. Both models achieve low R²,
reflecting the fundamental limitation that credit-history features are
proxies for creditworthiness rather than direct income predictors.
However, the Random Forest's negative R² (meaning it explains less
variance than the mean predictor) reveals classic overfitting: it
memorised extreme income outliers in training and generalised poorly,
producing an RMSE of USD 9,677 versus Linear Regression's USD 7,136.

---

## 5.3 SHAP Explanations

### Explainer Selection

Because the winning classification model is a Logistic Regression inside
a StandardScaler Pipeline, `shap.LinearExplainer` was selected. The
inner estimator was extracted from the Pipeline, the test data was
scaled with the saved StandardScaler, and a one-row mean background
dataset was passed to the explainer. This matches the SHAP-recommended
approach for linear models.

### Global Feature Importance (Classification)

Global SHAP values were computed over 2,000 randomly sampled test rows.
The top-five features by mean absolute SHAP value are:

| Rank | Feature | Mean |SHAP Value| |
|---|---|---|
| 1 | revolving_util | 0.62386 |
| 2 | late_30_59_days | 0.25155 |
| 3 | age | 0.24449 |
| 4 | late_90_days | 0.17415 |
| 5 | open_credit_lines | 0.16212 |

`revolving_util` is the dominant feature by a substantial margin —
nearly 2.5 times the impact of the second-ranked feature. This is
consistent with the well-established finding that credit utilisation is
one of the strongest predictors of financial distress.

[FIGURE 5.7: SHAP summary bar chart (classification) —
reports/figures/shap_summary_bar_classification.png]
[FIGURE 5.8: SHAP beeswarm plot (classification) —
reports/figures/shap_beeswarm_classification.png]

### Case-Study Instance Selection

Eight instances were fixed as the project's reference set — five for
classification (covering True Positive, True Negative, False Positive,
False Negative, and Borderline outcomes) and three for regression
(low-income low-error, high-income low-error, and high-error). These
instances were selected once in Phase 3 and never regenerated; all
downstream phases use the same fixed set.

### Local SHAP Explanations (Selected Cases)

Waterfall plots and structured SHAP JSON files were produced for all
eight instances. Two representative examples:

**True Positive (instance 34, P(default) = 0.804):**
The model's top contributing feature was `revolving_util = 1.0`
(SHAP +1.41, strongly increases predicted default probability),
followed by `late_30_59_days = 1` (SHAP +0.51) and `age = 35`
(SHAP +0.34).

**False Positive (instance 2, P(default) = 0.783, actual = no default):**
`late_30_59_days = 3` dominates with SHAP +1.82, explaining why the
model incorrectly flags this applicant as high-risk despite their not
having defaulted.

[FIGURE 5.9: SHAP waterfall — True Positive (instance 34) —
reports/figures/shap_local/34_true_positive_waterfall.png]
[FIGURE 5.10: SHAP waterfall — False Positive (instance 2) —
reports/figures/shap_local/2_false_positive_waterfall.png]

---

## 5.4 Counterfactual Explanations

### Generation and Feasibility Filtering

DiCE (`method='genetic'`) was run for all five classification instances,
requesting five counterfactuals per instance. Of 24 raw counterfactuals
generated (one instance produced four rather than five), all 24 (100%)
were model-valid — meaning they genuinely flipped the model's prediction.

After applying the directional feasibility filter, only **5 of 24
(20.8%)** were retained as feasible. This result is the project's most
directly actionable finding: without domain knowledge encoded as a
filter, four out of every five suggested changes would be unrealistic
or counterproductive.

| Instance | Feasible CFs | Notes |
|---|---|---|
| 34 — True Positive | 2 / 4 | Feasible CFs available |
| 3 — True Negative | 0 / 5 | All CFs require worsening the profile |
| 2 — False Positive | 2 / 5 | Feasible CFs available |
| 73 — False Negative | 0 / 5 | All CFs require worsening the profile |
| 1 — Borderline | 1 / 5 | One feasible CF available |

The True Negative and False Negative instances yielded zero feasible
counterfactuals. For the True Negative (correctly cleared applicant),
flipping to "high risk" requires worsening the financial profile —
which is not useful advice. For the False Negative (actual defaulter
predicted as low risk), the same issue applies: the model's confidence
is low but the counterfactuals needed to push it further toward "default"
require perverse changes.

**Best feasible CF for True Positive (instance 34):** Reduce
`revolving_util` from 1.0 to 0.6, clear the `late_30_59_days` account,
reduce `debt_ratio` from 0.36 to 0.20, and close several credit lines
(from 10 to 4). Proximity score: 0.2007.

[SCREENSHOT 5.3: DiCE output and feasibility filter results from
src/counterfactual_explain.py]

---

## 5.5 LLM-Generated Explanations (Three-Arm Comparison)

### Generation Details

All 15 classification explanations (5 instances × 3 arms) plus 3
regression explanations were generated via the Groq API using
`openai/gpt-oss-120b` at temperature 0.2. The low temperature was
chosen to produce consistent, factual output rather than creative text.

### Readability Results

| Arm | Avg FK Grade | Avg Reading Ease | N |
|---|---|---|---|
| shap_only | 10.18 | 56.14 | 5 |
| cf_only | 10.50 | 53.58 | 5 |
| combined | 9.68 | 56.15 | 5 |

The combined arm produced the lowest (easiest) Flesch-Kincaid grade
(9.53 in the evaluation-phase measurement, 9.68 in the generation-phase
measurement), likely because its longer, more structured format produces
more natural sentence patterns. The CF-only arm, despite being the
shortest on average, had the highest grade level, possibly because
action-oriented language tends to use more complex conditional
constructions.

### Illustrative Comparison (Instance 34 — True Positive)

The three arms for the same instance illustrate the design clearly:

- **SHAP-only** (121 words, FK 9.6): Explains the decision in terms of
  credit utilisation at maximum, a recent late payment, and age of 35,
  without suggesting any corrective action.
- **CF-only** (120 words, FK 10.3): Lists four practical steps — pay
  down balances, clear the late-payment account, reduce overall debt,
  simplify the credit profile — without explaining which factor drove
  the original decision.
- **Combined** (252 words, FK 8.9): Addresses both questions in a
  structured two-section format: "Why the decision came back high risk"
  followed by "What the applicant can do to improve the outlook."

[SCREENSHOT 5.4: Three-column LLM explanation display in the Streamlit
demo interface — app.py, Replay mode, True Positive case.]

---

## 5.6 Evaluation Framework

### Rule-Based Faithfulness

The automated faithfulness check measured what fraction of the top-three
SHAP features were mentioned in each generated explanation, using a
feature-alias dictionary to account for paraphrase (e.g. "credit
utilisation" matching `revolving_util`).

| Arm | Avg Top-Feature Recall | Contamination Rate |
|---|---|---|
| shap_only | 0.867 | 0.0 |
| cf_only | 0.767 | 0.0 |
| combined | **0.967** | 0.0 |

Zero contamination was observed across all 15 explanations: the
shap_only arm never accidentally suggested what the applicant should do,
and the cf_only arm never accidentally explained why the decision was
made. The prompt template design successfully enforced arm separation.

### LLM-as-Judge Scores

The judge model (`qwen/qwen3.8-27b`) scored each explanation on three
dimensions (1–10 scale):

| Arm | Faithfulness | Understandability | Actionability |
|---|---|---|---|
| shap_only | 7.8 | 9.0 | 2.0 |
| cf_only | **9.4** | **9.8** | 6.4 |
| combined | 8.8 | **9.8** | **9.0** |

The actionability gap is the most decisive: shap_only's score of 2.0
is structurally expected — its prompt explicitly prohibits advice. The
CF-only arm scores 6.4 rather than higher because two of the five
instances had zero feasible counterfactuals, leaving the LLM with
nothing actionable to say. The combined arm's 9.0 score reflects its
ability to fall back to SHAP-based reasoning when no counterfactuals
are available.

Notably, the cf_only arm scored highest on faithfulness (9.4 vs. 8.8
for combined). A focused single-source explanation has less surface area
for the judge to identify minor inaccuracies, whereas the combined arm
occasionally introduced features that appeared in the broader dataset
but not in the strict top-3 SHAP ranking.

[SCREENSHOT 5.5: Aggregate evaluation table — master_comparison_table.md]

### Human Survey

The blinded survey instrument (`reports/evaluation/survey.html`) was
built and randomised using `random_state=42`. Each of the five instances
received a randomly shuffled A/B/C label assignment that differs from
the arm labels, ensuring participants cannot guess which arm produced
which explanation. The `survey_key.json` file records the true mapping
and is held separately from the survey form.

Human survey responses have not yet been collected. The aggregate
evaluation script (`src/aggregate_evaluation.py`) prints a warning and
skips the human section if no response files are present, and will
incorporate responses automatically once they are collected into
`reports/evaluation/survey_responses/`.

---

## 5.7 Demo Interface

The Streamlit interface (`app.py`) provides two modes of interaction,
accessible at `http://localhost:8501` by running `streamlit run app.py`
from the project root.

**Replay mode** presents a dropdown of the five classification case
types. Selecting a case study loads and displays all pre-computed
results: the applicant's feature values as a table, the model's
predicted probability and actual outcome, the SHAP waterfall plot, the
feasible counterfactuals, and all three LLM explanations in side-by-side
columns with their Flesch-Kincaid grade as a caption. The Phase 6 master
comparison table is accessible via a collapsible expander at the bottom
of the page.

**Live mode** allows a user to either pull a random applicant from the
held-out test set (excluding the five saved case studies) or enter
custom feature values using number inputs pre-filled with training-data
medians and bounded by training-data min/max values. Clicking "Generate
Explanation" runs the full pipeline in real time, with Streamlit spinners
labelling each step. The LLM generation step is wrapped in error
handling; if the Groq API is unavailable, the interface displays the
SHAP and counterfactual results alone with a descriptive warning.

[SCREENSHOT 5.6: Streamlit Replay mode — sidebar and case study dropdown.]
[SCREENSHOT 5.7: Streamlit Replay mode — feature table, prediction
banner, and SHAP waterfall for True Positive instance.]
[SCREENSHOT 5.8: Streamlit Replay mode — three-column LLM explanation
display with FK grade captions.]
[SCREENSHOT 5.9: Streamlit Live mode — custom feature value inputs.]
[SCREENSHOT 5.10: Streamlit Live mode — results display after pipeline
execution.]
