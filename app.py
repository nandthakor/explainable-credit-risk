# Credit Risk XAI — 
# ======================================================
#
# Run:  streamlit run app.py
#       (from the project root: c:\Users\HP\Desktop\XAI Project)
#
# Modes:
#   Replay — shows the 5 pre-computed case studies with saved results
#   Live   — runs the real pipeline on a new or custom applicant

import os
import sys
import json
import random
import warnings
warnings.filterwarnings("ignore")

# ---- Ensure src/ is importable ----------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Set NLTK proxy flag before any textstat import (needed for readability metrics)
os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend, required for Streamlit
import matplotlib.pyplot as plt
import joblib
import streamlit as st
from dotenv import load_dotenv

load_dotenv()   # picks up GROQ_API_KEY from .env

# ---- Streamlit page config --------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Explainability",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Paths ------------------------------------------------------------------
REPORTS      = "reports"
MODELS       = "models"
PROCESSED    = os.path.join("data", "processed")
SHAP_LOCAL   = os.path.join(REPORTS, "shap_local")
SHAP_FIGS    = os.path.join(REPORTS, "figures", "shap_local")
CF_DIR       = os.path.join(REPORTS, "counterfactuals")
LLM_DIR      = os.path.join(REPORTS, "llm_explanations")
EVAL_DIR     = os.path.join(REPORTS, "evaluation")
PROMPTS_DIR  = "prompts"

# ---- Human-readable case-type labels ----------------------------------------
CASE_LABELS = {
    "true_positive":  "True Positive (correctly flagged as high risk)",
    "true_negative":  "True Negative (correctly cleared as low risk)",
    "false_positive": "False Positive (incorrectly flagged as high risk)",
    "false_negative": "False Negative (missed — actually defaulted)",
    "borderline":     "Borderline (model was uncertain)",
}

# Reverse lookup: label -> case_type
LABEL_TO_CASE = {v: k for k, v in CASE_LABELS.items()}

# Features that are immutable (same as Phase 4)
IMMUTABLE_FEATURES = ["age", "num_dependents", "monthly_income_missing"]


# =============================================================================
# Cached resource loaders — run ONCE at startup, reused every interaction
# =============================================================================

@st.cache_resource(show_spinner="Loading classification model...")
def load_clf_model():
    """Load the trained Pipeline (StandardScaler + LogisticRegression)."""
    from sklearn.pipeline import Pipeline
    model = joblib.load(os.path.join(MODELS, "classification_model.joblib"))
    # Extract scaler and inner estimator
    steps = list(model.named_steps.values())
    scaler = steps[0]
    return model, scaler


@st.cache_resource(show_spinner="Building SHAP explainer...")
def load_shap_explainer(feature_cols):
    """
    Build the LinearExplainer from the test set background.
    Reuses the same approach as Phase 3 — load once, reuse for all calls.
    """
    import shap
    # Convert tuple back to list — passed as tuple for cache-key hashability,
    # but pandas requires a list for multi-column selection.
    feature_cols = list(feature_cols)

    model, scaler = load_clf_model()
    inner_clf = list(model.named_steps.values())[-1]

    X_test = pd.read_csv(os.path.join(PROCESSED, "X_test_clf.csv"))
    X_test = X_test[feature_cols]
    X_scaled = scaler.transform(X_test)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    background = X_scaled_df.mean().values.reshape(1, -1)

    explainer = shap.LinearExplainer(inner_clf, masker=background)
    return explainer


@st.cache_resource(show_spinner="Setting up DiCE counterfactual engine...")
def load_dice_setup(feature_cols):
    """
    Initialise DiCE objects using the training data.
    Returns (dice_exp, actionable_features, feat_ranges).
    """
    import dice_ml
    # Convert tuple back to list — passed as tuple for cache-key hashability.
    feature_cols = list(feature_cols)

    model, _ = load_clf_model()

    X_train = pd.read_csv(os.path.join(PROCESSED, "X_train_clf.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED, "y_train_clf.csv")).squeeze()
    X_train = X_train[feature_cols]

    # Feature ranges for proximity normalisation
    feat_ranges = {}
    for col in feature_cols:
        r = X_train[col].max() - X_train[col].min()
        feat_ranges[col] = r if r > 0 else 1.0

    bg = X_train.copy()
    bg["target_default"] = y_train.values

    actionable = [f for f in feature_cols if f not in IMMUTABLE_FEATURES]

    d = dice_ml.Data(
        dataframe=bg,
        continuous_features=feature_cols,
        outcome_name="target_default",
    )
    m = dice_ml.Model(model=model, backend="sklearn")
    exp = dice_ml.Dice(d, m, method="genetic")

    return exp, actionable, feat_ranges


@st.cache_data
def load_selected_instances():
    """Load the 5 fixed classification case-study instances."""
    with open(os.path.join(REPORTS, "selected_instances.json")) as f:
        all_insts = json.load(f)
    return [i for i in all_insts if i["task"] == "classification"]


@st.cache_data
def load_feature_cols():
    """Load the ordered list of classification feature names."""
    with open(os.path.join(REPORTS, "feature_columns_clf.json")) as f:
        return json.load(f)


@st.cache_data
def load_master_table_md():
    """Load the master comparison table markdown text."""
    path = os.path.join(EVAL_DIR, "master_comparison_table.md")
    with open(path, encoding="utf-8") as f:
        return f.read()


@st.cache_data
def load_evaluation_summary():
    """Load the evaluation summary markdown text."""
    path = os.path.join(REPORTS, "evaluation_summary.md")
    with open(path, encoding="utf-8") as f:
        return f.read()


@st.cache_data
def load_training_stats(feature_cols):
    """Load per-feature min/median/max from training data for Live mode inputs."""
    feature_cols = list(feature_cols)   # tuple -> list for pandas
    X_train = pd.read_csv(os.path.join(PROCESSED, "X_train_clf.csv"))
    X_train = X_train[feature_cols]
    return {
        "min":    X_train.min().to_dict(),
        "median": X_train.median().to_dict(),
        "max":    X_train.max().to_dict(),
    }


# =============================================================================
# Helper: display feature table
# =============================================================================
def show_feature_table(features: dict):
    """Render the applicant's feature values as a clean two-column table."""
    # Friendly display names
    friendly = {
        "revolving_util":       "Revolving Credit Utilization",
        "age":                  "Age (years)",
        "late_30_59_days":      "Times 30-59 Days Late",
        "debt_ratio":           "Debt Ratio",
        "monthly_income":       "Monthly Income ($)",
        "open_credit_lines":    "Open Credit Lines",
        "late_90_days":         "Times 90+ Days Late",
        "real_estate_loans":    "Real Estate Loans",
        "late_60_89_days":      "Times 60-89 Days Late",
        "num_dependents":       "Number of Dependents",
        "monthly_income_missing": "Income Data Missing (flag)",
    }
    rows = []
    for k, v in features.items():
        label = friendly.get(k, k.replace("_", " ").title())
        rows.append({"Feature": label, "Value": f"{v:.4g}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# =============================================================================
# Helper: display SHAP records as a table
# =============================================================================
def show_shap_table(shap_records: list, top_n: int = 5):
    """Show top-N SHAP features as a readable table."""
    rows = []
    for rec in shap_records[:top_n]:
        direction = "▲ increases risk" if rec["shap_value"] > 0 else "▼ decreases risk"
        rows.append({
            "Feature":   rec["feature_name"].replace("_", " ").title(),
            "Value":     f"{rec['feature_value']:.4g}",
            "SHAP Impact": f"{rec['shap_value']:+.4f}",
            "Direction": direction,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# =============================================================================
# Helper: display feasible counterfactuals
# =============================================================================
def show_counterfactuals(feasible_cfs: list):
    """Display feasible counterfactuals as a readable list."""
    if not feasible_cfs:
        st.info(
            "No feasible counterfactuals found for this applicant. "
            "This can happen when the model's prediction is already very "
            "confident, or when all possible changes would violate feasibility "
            "constraints (e.g., requiring more late payments)."
        )
        return

    best = min(feasible_cfs, key=lambda c: c["proximity_score"])
    st.write(
        f"**Best counterfactual** "
        f"({best['num_features_changed']} feature(s) changed, "
        f"proximity score: {best['proximity_score']:.4f})"
    )
    for ch in best["changed_features"]:
        feat = ch["feature_name"].replace("_", " ").title()
        arrow = "⬆" if ch["direction"] == "increase" else "⬇"
        st.write(
            f"- {arrow} **{feat}**: "
            f"{ch['original_value']:.4g} → {ch['cf_value']:.4g}"
        )

    if len(feasible_cfs) > 1:
        with st.expander(f"Show all {len(feasible_cfs)} feasible counterfactuals"):
            for i, cf in enumerate(feasible_cfs):
                st.write(f"**Option {i+1}** "
                         f"({cf['num_features_changed']} changes, "
                         f"proximity: {cf['proximity_score']:.4f})")
                for ch in cf["changed_features"]:
                    feat = ch["feature_name"].replace("_", " ").title()
                    arrow = "⬆" if ch["direction"] == "increase" else "⬇"
                    st.write(f"  - {arrow} {feat}: "
                             f"{ch['original_value']:.4g} → {ch['cf_value']:.4g}")


# =============================================================================
# Helper: display three LLM explanation columns
# =============================================================================
def show_three_explanations(shap_text, shap_fk, cf_text, cf_fk,
                             combined_text, combined_fk):
    """Render the three explanation arms in side-by-side columns."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("SHAP Only")
        st.caption(f"Explains WHY the decision was made  |  "
                   f"FK Grade: {shap_fk}")
        st.write(shap_text)

    with col2:
        st.subheader("Counterfactual Only")
        st.caption(f"Explains WHAT to change  |  FK Grade: {cf_fk}")
        st.write(cf_text)

    with col3:
        st.subheader("Combined")
        st.caption(f"Explains both WHY and WHAT  |  FK Grade: {combined_fk}")
        st.write(combined_text)


# =============================================================================
# REPLAY MODE: display a saved case study
# =============================================================================
def replay_mode(instances, feature_cols):
    """Show the pre-computed results for one of the 5 saved case studies."""
    st.header("Replay Saved Case Studies")
    st.write(
        "Select one of the 5 pre-computed case-study instances"
        "All results (SHAP, counterfactuals, LLM explanations) were generated "
        "by the real pipeline and are loaded from saved files."
    )

    # Dropdown of case-type labels
    case_label_options = [CASE_LABELS[inst["case_type"]] for inst in instances]
    selected_label = st.selectbox("Select case study:", case_label_options)
    selected_case = LABEL_TO_CASE[selected_label]
    inst = next(i for i in instances if i["case_type"] == selected_case)
    iid  = inst["instance_id"]

    st.divider()

    # ---- Applicant features -------------------------------------------------
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Applicant Profile")
        show_feature_table(inst["features"])

    with col_right:
        st.subheader("Model Prediction")
        pred_proba = inst["predicted_proba"]
        pred_class = 1 if pred_proba >= 0.5 else 0
        actual     = int(inst["actual"])

        if pred_class == 1:
            st.error(f"**HIGH RISK** — Predicted probability of default: "
                     f"**{pred_proba:.1%}**")
        else:
            st.success(f"**LOW RISK** — Predicted probability of default: "
                       f"**{pred_proba:.1%}**")

        verdict_icon = "✅" if pred_class == actual else "❌"
        st.write(f"Dataset outcome: {'Defaulted' if actual == 1 else 'Did not default'} "
                 f"{verdict_icon}")

    st.divider()

    # ---- SHAP ---------------------------------------------------------------
    st.subheader("SHAP Explanation (why this decision?)")

    shap_path = os.path.join(SHAP_LOCAL, f"{iid}_{selected_case}.json")
    waterfall_path = os.path.join(SHAP_FIGS, f"{iid}_{selected_case}_waterfall.png")

    with open(shap_path) as f:
        shap_records = json.load(f)

    col_s1, col_s2 = st.columns([1, 1])
    with col_s1:
        st.write("**Top SHAP Features**")
        show_shap_table(shap_records)

    with col_s2:
        if os.path.exists(waterfall_path):
            st.image(waterfall_path, caption="SHAP Waterfall Plot",
                     use_container_width=True)
        else:
            st.info("Waterfall plot image not found.")

    st.divider()

    # ---- Counterfactuals ----------------------------------------------------
    st.subheader("Counterfactual Explanation (what would change it?)")

    cf_path = os.path.join(CF_DIR, f"{iid}_{selected_case}_feasible.json")
    with open(cf_path) as f:
        feasible_cfs = json.load(f)
    show_counterfactuals(feasible_cfs)

    st.divider()

    # ---- Three LLM explanations ---------------------------------------------
    st.subheader("LLM Explanations — Three Arms")
    st.caption("All text was generated using the real Groq API. "
               "FK Grade = Flesch-Kincaid readability grade (lower = easier reading).")

    arms = ["shap_only", "cf_only", "combined"]
    llm_texts = {}
    llm_fk    = {}
    for arm in arms:
        path = os.path.join(LLM_DIR, f"{iid}_{selected_case}_{arm}.json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        llm_texts[arm] = data["raw_response"]
        llm_fk[arm]    = data["flesch_kincaid_grade"]

    show_three_explanations(
        shap_text=llm_texts["shap_only"], shap_fk=llm_fk["shap_only"],
        cf_text=llm_texts["cf_only"],    cf_fk=llm_fk["cf_only"],
        combined_text=llm_texts["combined"], combined_fk=llm_fk["combined"],
    )


# =============================================================================
# LIVE MODE: run the real pipeline on a new applicant
# =============================================================================
def live_mode(instances, feature_cols):
    """Run SHAP + CF + LLM on a user-specified or random new applicant."""
    import shap as shap_lib
    from shap_explain import explain_instance_live
    from counterfactual_explain import get_counterfactuals_live
    from generate_explanations import generate_all_three_arms

    st.header("Live: New Applicant")
    st.write(
        "Enter feature values for a new applicant, or pick a random test-set row. "
        "The full pipeline (model prediction → SHAP → counterfactuals → LLM) runs live."
    )

    # Load resources (cached after first call)
    clf_model, clf_scaler = load_clf_model()
    clf_explainer         = load_shap_explainer(tuple(feature_cols))
    dice_exp, actionable, feat_ranges = load_dice_setup(tuple(feature_cols))
    stats                 = load_training_stats(tuple(feature_cols))

    # Already-saved instance IDs — exclude from random picks
    saved_ids = {inst["instance_id"] for inst in instances}

    # ---- Input method toggle ------------------------------------------------
    input_mode = st.radio(
        "Input method:",
        ["Pick a random test-set applicant", "Enter custom values"],
        horizontal=True,
    )

    feature_dict = {}

    if input_mode == "Pick a random test-set applicant":
        if st.button("🎲 Pick random applicant"):
            X_test = pd.read_csv(os.path.join(PROCESSED, "X_test_clf.csv"))
            X_test = X_test[feature_cols]
            candidates = X_test[~X_test.index.isin(saved_ids)]
            if len(candidates) == 0:
                st.warning("No unsaved test rows available.")
                return
            row = candidates.sample(n=1, random_state=random.randint(0, 9999))
            st.session_state["live_features"] = row.iloc[0].to_dict()
            st.success(f"Loaded test-set applicant (row index {row.index[0]})")

        if "live_features" not in st.session_state:
            st.info("Click the button above to load a random applicant.")
            return

        feature_dict = st.session_state["live_features"]

        # Show as editable inputs (pre-filled with random row values)
        st.write("**Pre-filled feature values (you can adjust):**")
        updated = {}
        cols_per_row = 3
        feat_list = list(feature_cols)
        for i in range(0, len(feat_list), cols_per_row):
            chunk = feat_list[i:i + cols_per_row]
            cols = st.columns(cols_per_row)
            for j, feat in enumerate(chunk):
                current = feature_dict.get(feat, stats["median"][feat])
                updated[feat] = cols[j].number_input(
                    feat.replace("_", " ").title(),
                    min_value=float(stats["min"][feat]),
                    max_value=float(stats["max"][feat]),
                    value=float(current),
                    step=1.0 if feat in ("age", "num_dependents",
                                          "monthly_income_missing",
                                          "late_30_59_days", "late_60_89_days",
                                          "late_90_days", "open_credit_lines",
                                          "real_estate_loans") else 0.001,
                    key=f"live_{feat}",
                )
        feature_dict = updated

    else:  # custom values
        st.write("**Enter applicant values** (defaults are training-data medians):")
        updated = {}
        cols_per_row = 3
        feat_list = list(feature_cols)
        for i in range(0, len(feat_list), cols_per_row):
            chunk = feat_list[i:i + cols_per_row]
            cols = st.columns(cols_per_row)
            for j, feat in enumerate(chunk):
                updated[feat] = cols[j].number_input(
                    feat.replace("_", " ").title(),
                    min_value=float(stats["min"][feat]),
                    max_value=float(stats["max"][feat]),
                    value=float(stats["median"][feat]),
                    step=1.0 if feat in ("age", "num_dependents",
                                          "monthly_income_missing",
                                          "late_30_59_days", "late_60_89_days",
                                          "late_90_days", "open_credit_lines",
                                          "real_estate_loans") else 0.001,
                    key=f"custom_{feat}",
                )
        feature_dict = updated

    # ---- Generate button ----------------------------------------------------
    if not feature_dict:
        return

    if not st.button("🚀 Generate Explanation"):
        return

    st.divider()

    # ---- Step 1: Model prediction + SHAP ------------------------------------
    with st.spinner("Running SHAP explanation..."):
        shap_result = explain_instance_live(
            feature_dict=feature_dict,
            clf_model=clf_model,
            clf_explainer=clf_explainer,
            clf_scaler=clf_scaler,
            feature_cols=feature_cols,
        )

    pred_class = shap_result["predicted_class"]
    pred_proba = shap_result["predicted_proba"]
    shap_records = shap_result["shap_records"]

    # Display prediction
    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.subheader("Applicant Profile")
        show_feature_table(feature_dict)

    with col_r:
        st.subheader("Model Prediction")
        if pred_class == 1:
            st.error(f"**HIGH RISK** — P(default): **{pred_proba:.1%}**")
        else:
            st.success(f"**LOW RISK** — P(default): **{pred_proba:.1%}**")

    st.divider()

    # Display SHAP
    st.subheader("SHAP Explanation")
    col_s1, col_s2 = st.columns([1, 1])
    with col_s1:
        show_shap_table(shap_records)
    with col_s2:
        st.pyplot(shap_result["waterfall_fig"])
        plt.close(shap_result["waterfall_fig"])

    st.divider()

    # ---- Step 2: Counterfactuals --------------------------------------------
    with st.spinner("Generating counterfactuals (this may take 15-30 seconds)..."):
        feasible_cfs = get_counterfactuals_live(
            feature_dict=feature_dict,
            feature_cols=feature_cols,
            dice_exp=dice_exp,
            actionable=actionable,
            feat_ranges=feat_ranges,
            original_pred=pred_class,
        )

    st.subheader("Counterfactual Explanation")
    show_counterfactuals(feasible_cfs)

    st.divider()

    # ---- Step 3: LLM explanations -------------------------------------------
    predicted_label = (
        "HIGH RISK (likely to default)" if pred_class == 1
        else "LOW RISK (unlikely to default)"
    )
    confidence = f"{pred_proba:.1%}"

    st.subheader("LLM Explanations — Three Arms")

    try:
        with st.spinner("Calling LLM (this may take 10-20 seconds)..."):
            llm_results = generate_all_three_arms(
                feature_dict=feature_dict,
                predicted_label=predicted_label,
                confidence=confidence,
                shap_records=shap_records,
                feasible_cfs=feasible_cfs,
            )

        show_three_explanations(
            shap_text=llm_results["shap_only"]["text"],
            shap_fk=llm_results["shap_only"]["flesch_kincaid_grade"],
            cf_text=llm_results["cf_only"]["text"],
            cf_fk=llm_results["cf_only"]["flesch_kincaid_grade"],
            combined_text=llm_results["combined"]["text"],
            combined_fk=llm_results["combined"]["flesch_kincaid_grade"],
        )

    except Exception as e:
        st.warning(
            f"Live LLM generation failed — showing SHAP and counterfactual "
            f"results only.\n\n"
            f"Error: {e}\n\n"
            "Check that your GROQ_API_KEY is set in the .env file and that "
            "you have a working internet connection."
        )


# =============================================================================
# Research Finding Expander (shown in both modes)
# =============================================================================
def show_research_finding():
    """Collapsible section showing the master comparison table and conclusion."""
    with st.expander("Overall Research Finding ", expanded=False):
        st.markdown(load_master_table_md())
        st.divider()
        st.markdown(load_evaluation_summary())


# =============================================================================
# Main app layout
# =============================================================================
def main():
    # Sidebar
    st.sidebar.title("🏦 Credit Risk XAI")
    st.sidebar.write(
        "Explaining automated credit decisions using "
        "SHAP, counterfactuals, and LLM-generated natural language."
    )
    st.sidebar.divider()

    mode = st.sidebar.radio(
        "Mode",
        ["Replay saved case studies", "Live: new applicant"],
    )

    st.sidebar.divider()
    st.sidebar.caption(
        " "
       
    )

    # Load shared data
    feature_cols = load_feature_cols()
    instances    = load_selected_instances()

    # Page title
    st.title("Explainable Credit Decisions")
    st.caption(

        "Model: Logistic Regression | "
        "Dataset: Give Me Some Credit (Kaggle)"
    )

    st.divider()

    # Route to mode
    if mode == "Replay saved case studies":
        replay_mode(instances, feature_cols)
    else:
        live_mode(instances, feature_cols)

    st.divider()

    # Research finding — always visible at the bottom
    show_research_finding()


if __name__ == "__main__":
    main()
