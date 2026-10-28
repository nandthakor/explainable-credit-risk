# 4. SYSTEM DESIGN

---

## 4.1 System Architecture Overview

The system is structured as a sequential seven-phase pipeline in which
each phase's outputs become the inputs to the next. All intermediate
artefacts — cleaned data, trained models, SHAP values, counterfactual
JSON files, LLM-generated explanations, evaluation scores, and the demo
interface — are persisted to disk, making each phase independently
reproducible and auditable.

[FIGURE 4.1: System architecture diagram — a linear pipeline showing:
Raw CSV → Data Preparation → Classification + Regression Models →
SHAP Explainer → DiCE Counterfactuals → Three-Arm LLM Generation →
Evaluation Framework → Streamlit Demo.
Arrows between each stage indicate file-based hand-off.
Draw this in Word or PowerPoint and insert here.]

---

## 4.2 Pipeline Description

The pipeline proceeds in seven phases, corresponding to the seven
implementation chapters described in Chapter 5.

### Phase 1 — Data Preparation

The raw Give Me Some Credit CSV file (150,000 rows, 11 columns) is
loaded by `src/data_prep.py`. Column names are renamed to snake_case
identifiers. Rows with physiologically impossible values (age = 0) and
rows containing sentinel codes 96 or 98 in late-payment columns are
flagged and removed, reducing the dataset to 149,730 rows. Missing
values in `num_dependents` (2.62%) are imputed with the column median.
Missing values in `monthly_income` (19.82%) are retained intentionally,
as they define the population for the income-estimation regression task.
A binary flag column `monthly_income_missing` is added. The cleaned
dataset is written to `data/processed/credit_data_clean.csv`.

### Phase 2 — Predictive Model Training

`src/model_prep.py` applies Winsorisation (99th-percentile capping) to
five columns with extreme outliers before train/test splitting. Two
separate modelling tasks are executed:

- **Classification** (`src/train_classification.py`): Logistic Regression
  (inside a StandardScaler Pipeline) and Random Forest are trained on
  119,784 rows and evaluated on 29,946 held-out rows. The winning model
  is saved to `models/classification_model.joblib`.

- **Regression** (`src/train_regression.py`): Linear Regression and
  Random Forest Regressor are trained on the 96,096 rows with known
  income and evaluated on 24,024 held-out rows. The winning model is
  saved to `models/regression_model.joblib`.

### Phase 3 — SHAP Explanations

`src/shap_explain.py` loads both saved models and constructs
`shap.LinearExplainer` instances (appropriate for linear models). Global
SHAP summary plots are generated for both tasks over 2,000 sampled test
rows. Eight case-study instances (five classification, three regression)
are selected to cover the full range of prediction outcomes: True
Positive, True Negative, False Positive, False Negative, Borderline,
and three regression error profiles. For each instance, a waterfall plot
PNG and a structured JSON file of SHAP values are saved. The instance
metadata is persisted to `reports/selected_instances.json` as the fixed
reference set for all downstream phases.

### Phase 4 — Counterfactual Explanations

`src/counterfactual_explain.py` uses DiCE with the `genetic` method to
generate up to five counterfactuals per classification instance. A
directional feasibility filter — encoding domain knowledge about which
feature changes are realistic — retains only feasible suggestions. Raw
and filtered counterfactual JSON files are written to
`reports/counterfactuals/`.

### Phase 5 — Three-Arm LLM Explanation Generation

`src/generate_explanations.py` formats prompts from the Phase 3 and
Phase 4 outputs using three template files in `prompts/`. Three LLM
calls per classification instance — one per arm — are made via the Groq
API. Readability metrics are computed for each response. Results are
saved to `reports/llm_explanations/`.

### Phase 6 — Evaluation Framework

Four evaluation scripts score the 15 classification explanations:

- `src/faithfulness_rulebased.py` — rule-based feature recall and
  contamination check.
- `src/llm_judge.py` — LLM-as-judge scoring using a separate model.
- `src/build_survey.py` — builds the blinded HTML survey instrument.
- `src/aggregate_evaluation.py` — aggregates all signals into a master
  comparison table.

### Phase 7 — Streamlit Demo Interface

`app.py` exposes the entire pipeline — including cached model loading,
live SHAP computation, live DiCE search, and live LLM generation — as
a browser-based interactive interface. Replay mode loads all saved
Phase 3-6 artefacts for reliable demonstration. Live mode re-runs the
full pipeline on any new applicant.

---

## 4.3 Data Flow and File Conventions

All intermediate files follow a consistent naming convention:
`{instance_id}_{case_type}_{arm}.json` for LLM outputs and
`{instance_id}_{case_type}_waterfall.png` for SHAP figures. This
convention ensures traceability: given any output file, the originating
instance and explanation arm can be identified immediately.

The project root contains:

```
credit-xai-project/
├── data/
│   ├── raw/              cs-training.csv (input)
│   └── processed/        cleaned data, train/test splits
├── models/               saved joblib model files
├── prompts/              LLM prompt templates
├── reports/
│   ├── figures/          EDA and SHAP plots
│   ├── shap_local/       per-instance SHAP JSONs
│   ├── counterfactuals/  per-instance CF JSONs
│   ├── llm_explanations/ per-instance LLM output JSONs
│   ├── evaluation/       evaluation CSVs and survey files
│   └── final_report/     this report
├── src/                  all Python source modules
└── app.py                Streamlit entry point
```
