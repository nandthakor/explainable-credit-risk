# 2. TECHNOLOGY USED

---

## 2.1 Programming Language and Core Libraries

**Python 3** (version 3.14) was used as the sole implementation language.
Python was chosen for its extensive ecosystem of scientific computing,
machine learning, and natural language processing libraries, all of which
are employed in this project.

Key supporting libraries included:

- **pandas 3.0.1** — used for all data loading, cleaning, and
  transformation in `src/data_prep.py` and `src/model_prep.py`. The
  entire cleaned dataset of 149,730 rows is stored and manipulated as
  pandas DataFrames throughout the pipeline.
- **NumPy 2.4.3** — used for numerical operations, array handling in
  SHAP value computation, and for computing proximity scores in the
  counterfactual filter.
- **matplotlib / seaborn** — used in `notebooks/01_eda.py` to produce
  the class-balance bar chart, missingness plot, histogram grid, and
  Pearson correlation heatmap that form the EDA artefacts.
- **joblib 1.5.3** — used to persist and load the trained model
  pipelines (`models/classification_model.joblib`,
  `models/regression_model.joblib`).
- **textstat** — used in `src/generate_explanations.py` to compute
  Flesch-Kincaid grade level and Flesch Reading Ease scores for every
  generated explanation, providing objective readability metrics.
- **python-dotenv** — used to load the Groq API key from the `.env`
  file without hardcoding credentials in source files.

---

## 2.2 Machine Learning Modelling (scikit-learn)

**scikit-learn 1.8.0** provides the complete modelling infrastructure
for both tasks in Phase 2.

For the **classification task** (predicting loan default), two models
were trained and compared:

- A `Pipeline(StandardScaler → LogisticRegression(class_weight='balanced',
  max_iter=2000, random_state=42))` — selected as the winner with
  ROC-AUC 0.8496.
- A `RandomForestClassifier(class_weight='balanced', random_state=42)` —
  compared as a baseline, achieving ROC-AUC 0.8333.

The Pipeline wrapper ensures the scaler is always applied consistently,
including during inference in the demo interface. The `class_weight=
'balanced'` parameter causes the loss function to up-weight the 6.6%
minority class during training, which is essential given the severe class
imbalance in the dataset.

For the **regression task** (estimating monthly income), two models were
similarly compared:

- `LinearRegression` — selected as the winner with R² 0.0793.
- `RandomForestRegressor` — discarded due to overfitting (R² −0.6929
  on the test set).

`train_test_split` with `stratify=y` and `random_state=42` was used for
both tasks to ensure reproducible, stratified splits.

---

## 2.3 Explainability Techniques: SHAP

**SHAP (SHapley Additive exPlanations) version 0.52.0** was used in
`src/shap_explain.py` to compute both global and local feature-importance
explanations.

Because the winning classifier is a Logistic Regression inside a
scikit-learn Pipeline, `shap.LinearExplainer` was selected. The inner
`LogisticRegression` estimator was extracted from the Pipeline via
`model.named_steps`, the test data was scaled using the saved
`StandardScaler`, and a mean background dataset of one row was passed to
the explainer as the reference distribution. This is the
SHAP-recommended approach for linear models.

For **global explanations**, SHAP values were computed over a random
sample of 2,000 test-set rows (to manage runtime), producing summary bar
charts (mean |SHAP value| per feature) and beeswarm plots saved to
`reports/figures/`. For **local explanations**, waterfall plots were
generated for each of the eight selected case-study instances (five
classification, three regression) and saved as PNG files.

Structured JSON files containing feature name, feature value, and SHAP
value for each instance — sorted by absolute SHAP value descending —
were written to `reports/shap_local/` for use by downstream phases.

---

## 2.4 Explainability Techniques: DiCE Counterfactuals

**DiCE (Diverse Counterfactual Explanations, dice-ml)** was used in
`src/counterfactual_explain.py` to generate counterfactual instances for
the five classification case studies. DiCE's `method='genetic'` search
was used, which directly optimises for prediction-class flip (validity)
and diversity, and works with scikit-learn Pipeline models via the
`backend='sklearn'` setting.

A critical contribution of this phase is the **directional feasibility
filter** applied after DiCE generation. Raw DiCE counterfactuals are
mathematically valid (they flip the model's prediction) but not
necessarily actionable in the real world. For example, a counterfactual
that reduces a borrower's income, or increases their count of late
payments, would be absurd advice. The filter encodes domain knowledge:
late-payment counts and revolving utilisation may only decrease;
monthly income may only increase; age and number of dependents are
treated as immutable. Only counterfactuals in which every changed feature
moves in its allowed direction are retained. Counterfactual data in the
format `{feature_name, original_value, cf_value, direction}` is written
to `reports/counterfactuals/` for each instance.

---

## 2.5 LLM Integration: Groq API

**Groq** (accessed via the `groq` Python library) provides the
inference API used for both explanation generation and LLM-as-judge
evaluation.

The generation model used throughout this project was
`openai/gpt-oss-120b`, the largest model available on the free-tier
account used for this research. Calls are made via the `groq.Groq`
client, wrapped in a retry loop in `src/llm_client.py` that handles rate
limits with exponential backoff.

Three prompt templates were crafted in `prompts/` — one for each
explanation arm:

- `shap_only_template.txt` — instructs the model to explain the
  decision in terms of feature contributions only, and explicitly
  prohibits actionable advice.
- `cf_only_template.txt` — instructs the model to translate
  counterfactual changes into actionable advice only, without explaining
  why the original decision was made.
- `combined_template.txt` — instructs the model to address both the
  why (from SHAP data) and the what-to-do (from counterfactual data) in
  a single coherent explanation.

For **LLM-as-judge evaluation** in Phase 6, a deliberately different
model — `qwen/qwen3.8-27b` — was used to score the generated
explanations on faithfulness, understandability, and actionability
(1-10 scale), reducing self-preference bias that would arise if the same
model both generated and judged.

---

## 2.6 Development Environment: Google Antigravity

**Google Antigravity** served as the AI-assisted integrated development
environment for this project. It provided an agentic coding assistant
that could read, write, and execute files within the project workspace
under researcher direction. This enabled a structured, phase-by-phase
development workflow in which each phase's code was written, tested, and
its outputs verified before the next phase began. The environment
supports both file editing and terminal command execution, and its
conversation history provided a persistent record of all design decisions.

---

## 2.7 Deployment: Streamlit

**Streamlit 1.64.0** is used in `app.py` to provide an interactive web
demonstration interface accessible at `http://localhost:8501` by running
`streamlit run app.py` from the project root.

The interface offers two modes. In **Replay mode**, all five pre-computed
classification case studies are available via a dropdown. Selecting a
case study loads and displays the applicant's feature values, the model's
prediction and confidence, the saved SHAP waterfall plot, the feasible
counterfactuals, and all three LLM-generated explanations side by side
with their Flesch-Kincaid grade captions. In **Live mode**, the user may
either pull a random test-set applicant or enter custom feature values
via number inputs pre-filled with training-data medians. Clicking
"Generate Explanation" triggers the full pipeline in real time — model
prediction, SHAP, DiCE counterfactuals, and three LLM calls — with
Streamlit spinners indicating progress at each step. The LLM call is
wrapped in a `try/except` block so that failures (rate limits, network
errors) display a clear warning rather than crashing the page.

`@st.cache_resource` is used for all model-loading and explainer-setup
operations, ensuring these run only once at startup and are reused across
all user interactions.
