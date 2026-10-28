# Credit Risk XAI — Phase 3: SHAP Explanations
# =============================================
# Generates global and local SHAP explanations for both the classification
# and regression models trained in Phase 2.
#
# Run: python src/shap_explain.py
# Prerequisites:
#   - python src/model_prep.py  (splits must exist)
#   - python src/train_classification.py
#   - python src/train_regression.py

import os
import json
import warnings
warnings.filterwarnings("ignore")   # suppress shap/matplotlib minor warnings

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")               # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import shap

from sklearn.pipeline          import Pipeline
from sklearn.linear_model      import LogisticRegression, LinearRegression
from sklearn.ensemble          import (RandomForestClassifier,
                                       RandomForestRegressor)

# ---- Paths ------------------------------------------------------------------
PROCESSED     = os.path.join("data", "processed")
REPORTS       = os.path.join("reports")
FIGURES       = os.path.join("reports", "figures")
LOCAL_FIGS    = os.path.join("reports", "figures", "shap_local")
LOCAL_JSON    = os.path.join("reports", "shap_local")
MODELS        = os.path.join("models")

for d in [LOCAL_FIGS, LOCAL_JSON]:
    os.makedirs(d, exist_ok=True)

RANDOM_STATE  = 42
GLOBAL_SAMPLE = 2000   # max rows to use for global SHAP plots


# =============================================================================
# Step 1 — Load models and detect inner estimator type
# =============================================================================
def load_models():
    """
    Load both saved joblib models. If the model is wrapped in a sklearn
    Pipeline (as the classifier is), extract the inner estimator so the
    correct SHAP explainer can be chosen.

    Returns a dict with the raw model, the inner estimator, and a flag
    indicating whether a Pipeline scaler needs to be applied before SHAP.
    """
    print("\n" + "=" * 60)
    print("STEP 1: Loading models")
    print("=" * 60)

    clf_model = joblib.load(os.path.join(MODELS, "classification_model.joblib"))
    reg_model = joblib.load(os.path.join(MODELS, "regression_model.joblib"))

    # Unwrap Pipeline if needed to get the actual estimator
    def unwrap(model):
        if isinstance(model, Pipeline):
            steps = list(model.named_steps.values())
            inner = steps[-1]            # last step is always the estimator
            scaler = steps[0]            # first step is StandardScaler
            return inner, scaler
        return model, None

    clf_inner, clf_scaler = unwrap(clf_model)
    reg_inner, reg_scaler = unwrap(reg_model)

    print(f"\n  Classification model : {type(clf_model).__name__}")
    print(f"    -> inner estimator : {type(clf_inner).__name__}")
    print(f"    -> scaler          : {type(clf_scaler).__name__ if clf_scaler else 'None'}")

    print(f"\n  Regression model     : {type(reg_model).__name__}")
    print(f"    -> inner estimator : {type(reg_inner).__name__}")

    return {
        "clf_model":  clf_model,
        "clf_inner":  clf_inner,
        "clf_scaler": clf_scaler,
        "reg_model":  reg_model,
        "reg_inner":  reg_inner,
        "reg_scaler": reg_scaler,
    }


# =============================================================================
# Step 2 — Load test sets and feature column lists
# =============================================================================
def load_test_data():
    """Load the held-out test splits and feature column names."""
    print("\n" + "=" * 60)
    print("STEP 2: Loading test data")
    print("=" * 60)

    X_test_clf = pd.read_csv(os.path.join(PROCESSED, "X_test_clf.csv"))
    X_test_reg = pd.read_csv(os.path.join(PROCESSED, "X_test_reg.csv"))
    y_test_clf = pd.read_csv(
        os.path.join(PROCESSED, "y_test_clf.csv")).squeeze()
    y_test_reg = pd.read_csv(
        os.path.join(PROCESSED, "y_test_reg.csv")).squeeze()

    with open(os.path.join(REPORTS, "feature_columns_clf.json")) as f:
        clf_feat_cols = json.load(f)
    with open(os.path.join(REPORTS, "feature_columns_reg.json")) as f:
        reg_feat_cols = json.load(f)

    print(f"  X_test_clf : {X_test_clf.shape}  (clf feature cols: {len(clf_feat_cols)})")
    print(f"  X_test_reg : {X_test_reg.shape}  (reg feature cols: {len(reg_feat_cols)})")

    # Ensure column order matches the saved feature list exactly
    X_test_clf = X_test_clf[clf_feat_cols]
    X_test_reg = X_test_reg[reg_feat_cols]

    return X_test_clf, y_test_clf, X_test_reg, y_test_reg, clf_feat_cols, reg_feat_cols


# =============================================================================
# Step 3 — Build SHAP explainers
# =============================================================================
def build_explainers(models: dict, X_test_clf: pd.DataFrame,
                     X_test_reg: pd.DataFrame):
    """
    Choose the correct SHAP explainer for each model type.

    Rule:
      - RandomForest*  -> shap.TreeExplainer
          Fast, exact Shapley values using the tree structure directly.
          No background dataset needed.
      - LogisticRegression / LinearRegression wrapped in Pipeline
          -> shap.LinearExplainer
          Analytically correct for linear models; requires a background
          dataset (we use the mean of the scaled test data as a 1-row
          background for speed, which is the SHAP recommended approach
          for LinearExplainer).

    For the Pipeline classifier, we MUST:
      1. Scale X_test_clf with the saved StandardScaler BEFORE computing
         SHAP values (the inner LR operates in scaled space).
      2. Pass the scaled data and the inner LR to LinearExplainer.
      The resulting SHAP values are in scaled-feature space but we map
      them back to original feature names for interpretability — the
      relative magnitudes are preserved.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Building SHAP explainers")
    print("=" * 60)

    clf_inner  = models["clf_inner"]
    clf_scaler = models["clf_scaler"]
    reg_inner  = models["reg_inner"]

    # --- Classification explainer --------------------------------------------
    if isinstance(clf_inner, (RandomForestClassifier,)):
        print("\n  [CLF] TreeExplainer selected — model is RandomForest")
        clf_explainer   = shap.TreeExplainer(clf_inner)
        X_clf_for_shap  = X_test_clf          # raw features, no scaling needed
    else:
        # LogisticRegression inside a StandardScaler Pipeline
        print("\n  [CLF] LinearExplainer selected — model is LogisticRegression")
        print("        Scaling X_test_clf with the saved StandardScaler first.")
        X_clf_scaled    = clf_scaler.transform(X_test_clf)
        X_clf_scaled_df = pd.DataFrame(X_clf_scaled,
                                        columns=X_test_clf.columns,
                                        index=X_test_clf.index)
        # Background: mean of scaled training data (single row, fast)
        background      = X_clf_scaled_df.mean().values.reshape(1, -1)
        clf_explainer   = shap.LinearExplainer(
            clf_inner,
            masker=background,
        )
        X_clf_for_shap  = X_clf_scaled_df

    # --- Regression explainer ------------------------------------------------
    if isinstance(reg_inner, (RandomForestRegressor,)):
        print("\n  [REG] TreeExplainer selected — model is RandomForestRegressor")
        reg_explainer   = shap.TreeExplainer(reg_inner)
        X_reg_for_shap  = X_test_reg
    else:
        # LinearRegression
        print("\n  [REG] LinearExplainer selected — model is LinearRegression")
        background_reg  = X_test_reg.mean().values.reshape(1, -1)
        reg_explainer   = shap.LinearExplainer(
            reg_inner,
            masker=background_reg,
        )
        X_reg_for_shap  = X_test_reg

    return clf_explainer, X_clf_for_shap, reg_explainer, X_reg_for_shap


# =============================================================================
# Step 4 — Sample for global plots if test set is large
# =============================================================================
def maybe_sample(X: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    If X has more than GLOBAL_SAMPLE rows, take a reproducible random sample
    of GLOBAL_SAMPLE rows for global SHAP plots only.
    Local explanations still use the full test set.
    """
    if len(X) > GLOBAL_SAMPLE:
        print(f"\n  [{label}] Test set has {len(X):,} rows — sampling "
              f"{GLOBAL_SAMPLE:,} for global plots (random_state={RANDOM_STATE}).")
        return X.sample(n=GLOBAL_SAMPLE, random_state=RANDOM_STATE)
    else:
        print(f"\n  [{label}] Test set has {len(X):,} rows — using all rows.")
        return X


# =============================================================================
# Step 5 — Compute SHAP values and generate global plots
# =============================================================================
def compute_and_plot_global(
    explainer, X_sample: pd.DataFrame,
    feature_cols: list, task: str
) -> shap.Explanation:
    """
    Compute SHAP values for X_sample, then save:
      - A summary bar chart (mean |SHAP|)
      - A beeswarm plot (shows direction and distribution)

    Returns the Explanation object so callers can extract mean |SHAP| rankings.
    """
    print(f"\n  Computing SHAP values for {task} ({len(X_sample)} rows) ...")
    shap_values = explainer(X_sample)

    # For binary classification, shap returns shape (n, features, 2) for
    # TreeExplainer or (n, features) for LinearExplainer.
    # We always want the positive-class (class 1) slice.
    if task == "classification" and len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    # ---- Summary bar chart --------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.bar(shap_values, show=False, ax=ax,
                   max_display=len(feature_cols))
    ax.set_title(f"Global Feature Importance ({task.title()}) — Mean |SHAP Value|")
    fname_bar = f"shap_summary_bar_{task}.png"
    fpath_bar = os.path.join(FIGURES, fname_bar)
    plt.savefig(fpath_bar, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  [saved] {fpath_bar}")

    # ---- Beeswarm plot ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.beeswarm(shap_values, show=False, max_display=len(feature_cols))
    plt.title(f"SHAP Beeswarm ({task.title()}) — Feature Value vs Impact")
    fname_bee = f"shap_beeswarm_{task}.png"
    fpath_bee = os.path.join(FIGURES, fname_bee)
    plt.savefig(fpath_bee, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  [saved] {fpath_bee}")

    return shap_values


# =============================================================================
# Step 6 — Select case-study instances
# =============================================================================
def select_clf_instances(clf_model, X_test_clf: pd.DataFrame,
                          y_test_clf: pd.Series) -> list:
    """
    Deliberately select 5 classification instances that cover the full range
    of prediction outcomes.  Uses the full test set (not the sampled subset).
    """
    print("\n" + "=" * 60)
    print("STEP 6a: Selecting classification case-study instances")
    print("=" * 60)

    proba = clf_model.predict_proba(X_test_clf)[:, 1]
    proba_s = pd.Series(proba, index=X_test_clf.index)
    y = y_test_clf

    def pick_first(mask, label):
        """Return the index of the first row matching mask, or None."""
        matches = X_test_clf.index[mask]
        if len(matches) == 0:
            print(f"  WARNING: No row found for '{label}' — will pick closest.")
            return None
        idx = matches[0]
        print(f"  [{label}] -> row index {idx}  proba={proba_s[idx]:.4f}  "
              f"actual={int(y[idx])}")
        return idx

    # --- 1. True Positive  (predicted high, actually defaulted) -------------
    tp_mask  = (proba_s > 0.70) & (y == 1)
    tp_idx   = pick_first(tp_mask, "true_positive")
    if tp_idx is None:           # fallback: highest-proba actual positive
        tp_idx = y[y == 1].index[proba_s[y == 1].argmax()]

    # --- 2. True Negative  (predicted low, actually did not default) ---------
    tn_mask  = (proba_s < 0.30) & (y == 0)
    tn_idx   = pick_first(tn_mask, "true_negative")
    if tn_idx is None:
        tn_idx = y[y == 0].index[proba_s[y == 0].argmin()]

    # --- 3. False Positive (predicted high, actually did not default) --------
    fp_mask  = (proba_s > 0.60) & (y == 0)
    fp_idx   = pick_first(fp_mask, "false_positive")
    if fp_idx is None:
        fp_idx = y[y == 0].index[proba_s[y == 0].argmax()]

    # --- 4. False Negative (predicted low, actually defaulted) ---------------
    fn_mask  = (proba_s < 0.40) & (y == 1)
    fn_idx   = pick_first(fn_mask, "false_negative")
    if fn_idx is None:
        fn_idx = y[y == 1].index[proba_s[y == 1].argmin()]

    # --- 5. Borderline  (probability near 0.50) ------------------------------
    bl_mask  = (proba_s >= 0.45) & (proba_s <= 0.55)
    bl_idx   = pick_first(bl_mask, "borderline")
    if bl_idx is None:
        bl_idx = (proba_s - 0.50).abs().idxmin()

    cases = [
        (tp_idx, "true_positive"),
        (tn_idx, "true_negative"),
        (fp_idx, "false_positive"),
        (fn_idx, "false_negative"),
        (bl_idx, "borderline"),
    ]
    return cases


def select_reg_instances(reg_model, X_test_reg: pd.DataFrame,
                          y_test_reg: pd.Series) -> list:
    """
    Deliberately select 3 regression instances:
      - Low-error on a low-income row (actual in bottom 25th percentile)
      - Low-error on a high-income row (actual in top 25th percentile)
      - High-error (largest absolute residual in the test set)
    """
    print("\n" + "=" * 60)
    print("STEP 6b: Selecting regression case-study instances")
    print("=" * 60)

    preds     = reg_model.predict(X_test_reg)
    residuals = np.abs(preds - y_test_reg.values)
    res_s     = pd.Series(residuals, index=X_test_reg.index)
    pred_s    = pd.Series(preds,     index=X_test_reg.index)

    p25 = y_test_reg.quantile(0.25)
    p75 = y_test_reg.quantile(0.75)

    # Low-error on low income
    low_inc_mask  = y_test_reg <= p25
    low_inc_resid = res_s[low_inc_mask]
    low_idx       = low_inc_resid.idxmin()
    print(f"  [low_income_low_error] -> row {low_idx}  "
          f"actual={y_test_reg[low_idx]:,.0f}  "
          f"pred={pred_s[low_idx]:,.0f}  "
          f"abs_err={res_s[low_idx]:,.0f}")

    # Low-error on high income
    high_inc_mask  = y_test_reg >= p75
    high_inc_resid = res_s[high_inc_mask]
    high_idx       = high_inc_resid.idxmin()
    print(f"  [high_income_low_error] -> row {high_idx}  "
          f"actual={y_test_reg[high_idx]:,.0f}  "
          f"pred={pred_s[high_idx]:,.0f}  "
          f"abs_err={res_s[high_idx]:,.0f}")

    # Highest-error row overall
    err_idx = res_s.idxmax()
    print(f"  [high_error]           -> row {err_idx}  "
          f"actual={y_test_reg[err_idx]:,.0f}  "
          f"pred={pred_s[err_idx]:,.0f}  "
          f"abs_err={res_s[err_idx]:,.0f}")

    cases = [
        (low_idx,  "low_income_low_error"),
        (high_idx, "high_income_low_error"),
        (err_idx,  "high_error"),
    ]
    return cases


# =============================================================================
# Step 7 — Compute local SHAP, save waterfall PNGs + structured JSON
# =============================================================================
def explain_instance(
    explainer,
    X_for_shap: pd.DataFrame,
    row_index: int,
    instance_id: int,
    case_type: str,
    task: str,
    y_actual,
    y_pred,
    y_proba=None,
    X_raw: pd.DataFrame = None,
):
    """
    For a single selected instance:
      1. Compute its SHAP values (local explanation).
      2. Save a waterfall plot as a PNG.
      3. Save a structured JSON with feature name, value, and shap_value
         sorted by abs(shap_value) descending.
      4. Return a dict for selected_instances.json.

    Parameters
    ----------
    X_for_shap : the data in the form the explainer expects (may be scaled)
    X_raw      : the original unscaled features (for human-readable JSON)
    row_index  : the original DataFrame index of this row
    """
    # Pull the single row from X_for_shap using its original index
    row_shap = X_for_shap.loc[[row_index]]   # keep as DataFrame (1 row)

    # Compute SHAP values for this one row
    local_shap = explainer(row_shap)

    # For binary classification TreeExplainer returns (1, features, 2)
    if task == "classification" and len(local_shap.shape) == 3:
        local_shap = local_shap[:, :, 1]

    # ---- Waterfall plot -----------------------------------------------------
    waterfall_path = os.path.join(
        LOCAL_FIGS, f"{instance_id}_{case_type}_waterfall.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.waterfall(local_shap[0], show=False)
    title_str = (f"{task.title()} | {case_type.replace('_',' ').title()}"
                 f" | ID={instance_id}\n"
                 f"Actual={y_actual}  Predicted={y_pred}")
    if y_proba is not None:
        title_str += f"  P(default)={y_proba:.3f}"
    plt.title(title_str, fontsize=10)
    plt.savefig(waterfall_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    [saved] {waterfall_path}")

    # ---- Structured JSON ----------------------------------------------------
    shap_vals   = local_shap.values[0]          # array of shape (n_features,)
    feat_names  = local_shap.feature_names
    raw_row     = (X_raw if X_raw is not None else X_for_shap).loc[row_index]

    shap_records = []
    for feat, sv in zip(feat_names, shap_vals):
        shap_records.append({
            "feature_name":  feat,
            "feature_value": float(raw_row[feat]),
            "shap_value":    float(sv),
        })
    # Sort by absolute SHAP value descending
    shap_records.sort(key=lambda r: abs(r["shap_value"]), reverse=True)

    json_path = os.path.join(
        LOCAL_JSON, f"{instance_id}_{case_type}.json")
    with open(json_path, "w") as f:
        json.dump(shap_records, f, indent=2)
    print(f"    [saved] {json_path}")

    # ---- Build selected_instances entry ------------------------------------
    entry = {
        "instance_id":    int(instance_id),
        "task":           task,
        "case_type":      case_type,
        "actual":         float(y_actual),
        "predicted":      float(y_pred),
        "features":       {feat: float(raw_row[feat])
                           for feat in feat_names},
    }
    if y_proba is not None:
        entry["predicted_proba"] = float(y_proba)

    return entry, shap_records


# =============================================================================
# Step 8 — Generate all local explanations
# =============================================================================
def explain_all_instances(
    models, explainers_data, test_data, clf_cases, reg_cases
):
    """
    Iterate over the 5 classification and 3 regression selected instances,
    generate waterfall plots and JSON files for each, then return the full
    list of instance dicts for selected_instances.json.
    """
    print("\n" + "=" * 60)
    print("STEP 7: Generating local waterfall plots and SHAP JSONs")
    print("=" * 60)

    clf_model, reg_model = models["clf_model"], models["reg_model"]
    clf_explainer, X_clf_for_shap = explainers_data["clf_explainer"], explainers_data["X_clf_for_shap"]
    reg_explainer, X_reg_for_shap = explainers_data["reg_explainer"], explainers_data["X_reg_for_shap"]
    X_test_clf, y_test_clf = test_data["X_test_clf"], test_data["y_test_clf"]
    X_test_reg, y_test_reg = test_data["X_test_reg"], test_data["y_test_reg"]

    proba_all = clf_model.predict_proba(X_test_clf)[:, 1]
    pred_all_clf = (proba_all >= 0.5).astype(int)
    pred_all_reg = reg_model.predict(X_test_reg)

    all_instances = []
    all_shap_by_instance = {}   # for shap_summary.md

    # --- Classification cases ------------------------------------------------
    print("\n  -- Classification instances --")
    for (row_idx, case_type) in clf_cases:
        print(f"\n  [{case_type}] row_index={row_idx}")
        y_actual = int(y_test_clf.loc[row_idx])
        y_proba  = float(proba_all[X_test_clf.index.get_loc(row_idx)])
        y_pred   = int(y_proba >= 0.5)

        entry, shap_records = explain_instance(
            explainer   = clf_explainer,
            X_for_shap  = X_clf_for_shap,
            row_index   = row_idx,
            instance_id = row_idx,
            case_type   = case_type,
            task        = "classification",
            y_actual    = y_actual,
            y_pred      = y_pred,
            y_proba     = y_proba,
            X_raw       = X_test_clf,
        )
        all_instances.append(entry)
        all_shap_by_instance[f"{row_idx}_{case_type}"] = shap_records

    # --- Regression cases ----------------------------------------------------
    print("\n  -- Regression instances --")
    for (row_idx, case_type) in reg_cases:
        print(f"\n  [{case_type}] row_index={row_idx}")
        y_actual  = float(y_test_reg.loc[row_idx])
        pred_loc  = list(X_test_reg.index).index(row_idx)
        y_pred    = float(pred_all_reg[pred_loc])

        entry, shap_records = explain_instance(
            explainer   = reg_explainer,
            X_for_shap  = X_reg_for_shap,
            row_index   = row_idx,
            instance_id = row_idx,
            case_type   = case_type,
            task        = "regression",
            y_actual    = y_actual,
            y_pred      = y_pred,
            X_raw       = X_test_reg,
        )
        all_instances.append(entry)
        all_shap_by_instance[f"{row_idx}_{case_type}"] = shap_records

    return all_instances, all_shap_by_instance


# =============================================================================
# Step 9 — Save selected_instances.json
# =============================================================================
def save_selected_instances(all_instances: list) -> None:
    """Save the fixed reference set to reports/selected_instances.json."""
    path = os.path.join(REPORTS, "selected_instances.json")
    with open(path, "w") as f:
        json.dump(all_instances, f, indent=2)
    print(f"\n  [saved] {path}  ({len(all_instances)} instances)")


# =============================================================================
# Step 10 — Write shap_summary.md
# =============================================================================
def write_shap_summary(
    clf_shap_vals,
    reg_shap_vals,
    clf_feat_cols: list,
    reg_feat_cols: list,
    clf_sample_n:  int,
    reg_sample_n:  int,
    all_instances: list,
    all_shap_by_instance: dict,
    clf_cases: list,
    reg_cases: list,
) -> None:
    """
    Write reports/shap_summary.md with:
      - Top-5 global features (classification and regression)
      - Per-instance narrative describing the 2-3 largest SHAP contributors
    All numbers come directly from the computed SHAP values — nothing is invented.
    """
    print("\n" + "=" * 60)
    print("STEP 8: Writing shap_summary.md")
    print("=" * 60)

    # ---- Global feature rankings (mean |SHAP|) ------------------------------
    clf_mean_abs = np.abs(clf_shap_vals.values).mean(axis=0)
    reg_mean_abs = np.abs(reg_shap_vals.values).mean(axis=0)

    clf_ranking = sorted(zip(clf_mean_abs, clf_feat_cols), reverse=True)
    reg_ranking = sorted(zip(reg_mean_abs, reg_feat_cols), reverse=True)

    lines = [
        "# SHAP Explanation Summary — Phase 3",
        "",
        "> All SHAP values were computed using the real trained models on the",
        "> held-out test data. No values are estimated or fabricated.",
        "",
        "## Global Feature Importance",
        "",
        f"### Classification (mean |SHAP value| over {clf_sample_n:,} test-set rows)",
        "",
        "| Rank | Feature | Mean |SHAP Value| |",
        "|---|---|---|",
    ]
    for rank, (mv, feat) in enumerate(clf_ranking[:5], 1):
        lines.append(f"| {rank} | {feat} | {mv:.5f} |")

    lines += [
        "",
        f"### Regression — Income Estimation "
        f"(mean |SHAP value| over {reg_sample_n:,} test-set rows)",
        "",
        "| Rank | Feature | Mean |SHAP Value| |",
        "|---|---|---|",
    ]
    for rank, (mv, feat) in enumerate(reg_ranking[:5], 1):
        lines.append(f"| {rank} | {feat} | {mv:.5f} |")

    # ---- Per-instance narratives --------------------------------------------
    lines += [
        "",
        "---",
        "",
        "## Selected Instance Explanations",
        "",
        "> For each instance the top-2 or top-3 features by |SHAP value|",
        "> are described, along with the direction of their contribution.",
        "> A positive SHAP value pushes the prediction higher (toward",
        "> default / higher income); a negative SHAP value pushes it lower.",
        "",
    ]

    # Helper: describe sign direction
    def direction(sv: float) -> str:
        return "increases" if sv > 0 else "decreases"

    case_labels_clf = {
        "true_positive":  "True Positive",
        "true_negative":  "True Negative",
        "false_positive": "False Positive",
        "false_negative": "False Negative",
        "borderline":     "Borderline",
    }

    for inst in all_instances:
        iid   = inst["instance_id"]
        ctype = inst["case_type"]
        task  = inst["task"]
        key   = f"{iid}_{ctype}"
        recs  = all_shap_by_instance[key]   # sorted by |SHAP| desc

        label = case_labels_clf.get(ctype, ctype.replace("_", " ").title())
        lines.append(f"### {label} (instance ID {iid}, {task})")
        lines.append("")

        if task == "classification":
            lines.append(
                f"- **Actual:** {int(inst['actual'])}  "
                f"**Predicted prob (default):** {inst['predicted_proba']:.3f}"
            )
        else:
            lines.append(
                f"- **Actual income:** ${inst['actual']:,.0f}  "
                f"**Predicted:** ${inst['predicted']:,.0f}  "
                f"**Abs error:** ${abs(inst['predicted'] - inst['actual']):,.0f}"
            )

        lines.append("")
        lines.append("**Top contributing features (by |SHAP value|):**")
        lines.append("")

        for r in recs[:3]:
            lines.append(
                f"- `{r['feature_name']}` = {r['feature_value']:.4g} "
                f"→ SHAP {r['shap_value']:+.5f} "
                f"({direction(r['shap_value'])} predicted "
                + ("default probability" if task == "classification"
                   else "income estimate") + ")"
            )
        lines.append("")

    report_path = os.path.join(REPORTS, "shap_summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [saved] {report_path}")


# =============================================================================
# Main pipeline
# =============================================================================
def run_shap_pipeline() -> None:
    """Run the full Phase 3 SHAP explanation pipeline end-to-end."""

    # 1 & 2: Load models and test data
    models = load_models()
    (X_test_clf, y_test_clf,
     X_test_reg, y_test_reg,
     clf_feat_cols, reg_feat_cols) = load_test_data()

    # 3: Build explainers (handles Pipeline unwrapping)
    clf_explainer, X_clf_for_shap, reg_explainer, X_reg_for_shap = \
        build_explainers(models, X_test_clf, X_test_reg)

    # 4: Sample for global plots if needed
    print("\n" + "=" * 60)
    print("STEP 4: Sampling for global plots")
    print("=" * 60)
    X_clf_sample = maybe_sample(X_clf_for_shap, "CLF")
    X_reg_sample = maybe_sample(X_reg_for_shap, "REG")

    # 5: Global SHAP plots
    print("\n" + "=" * 60)
    print("STEP 5: Computing global SHAP values and saving plots")
    print("=" * 60)
    clf_shap_vals = compute_and_plot_global(
        clf_explainer, X_clf_sample, clf_feat_cols, "classification")
    reg_shap_vals = compute_and_plot_global(
        reg_explainer, X_reg_sample, reg_feat_cols, "regression")

    # 6: Select instances (use FULL test set, not sample)
    clf_cases = select_clf_instances(models["clf_model"], X_test_clf, y_test_clf)
    reg_cases = select_reg_instances(models["reg_model"], X_test_reg, y_test_reg)

    # 7: Local explanations — waterfall PNGs + structured JSON
    explainers_data = {
        "clf_explainer":   clf_explainer,
        "X_clf_for_shap":  X_clf_for_shap,
        "reg_explainer":   reg_explainer,
        "X_reg_for_shap":  X_reg_for_shap,
    }
    test_data = {
        "X_test_clf": X_test_clf,
        "y_test_clf": y_test_clf,
        "X_test_reg": X_test_reg,
        "y_test_reg": y_test_reg,
    }
    all_instances, all_shap_by_instance = explain_all_instances(
        models, explainers_data, test_data, clf_cases, reg_cases)

    # 8: Save selected_instances.json
    save_selected_instances(all_instances)

    # 9: Write shap_summary.md
    write_shap_summary(
        clf_shap_vals     = clf_shap_vals,
        reg_shap_vals     = reg_shap_vals,
        clf_feat_cols     = clf_feat_cols,
        reg_feat_cols     = reg_feat_cols,
        clf_sample_n      = len(X_clf_sample),
        reg_sample_n      = len(X_reg_sample),
        all_instances     = all_instances,
        all_shap_by_instance = all_shap_by_instance,
        clf_cases         = clf_cases,
        reg_cases         = reg_cases,
    )

    print("\n" + "=" * 60)
    print("PHASE 3 COMPLETE")
    print("=" * 60)
    print("  Outputs:")
    print(f"    reports/figures/shap_summary_bar_classification.png")
    print(f"    reports/figures/shap_beeswarm_classification.png")
    print(f"    reports/figures/shap_summary_bar_regression.png")
    print(f"    reports/figures/shap_beeswarm_regression.png")
    print(f"    reports/figures/shap_local/<id>_<case>_waterfall.png  (x8)")
    print(f"    reports/shap_local/<id>_<case>.json                   (x8)")
    print(f"    reports/selected_instances.json")
    print(f"    reports/shap_summary.md")


# =============================================================================
# Phase 7 — Live inference helper (used by app.py, not Phase 3 pipeline)
# =============================================================================
def explain_instance_live(
    feature_dict: dict,
    clf_model,
    clf_explainer,
    clf_scaler,
    feature_cols: list,
) -> dict:
    """
    Compute SHAP values for a single new applicant (not from the saved test set).
    Called by the Streamlit app in Live mode.

    Parameters
    ----------
    feature_dict : dict mapping feature name -> value for ONE applicant
    clf_model    : the full Pipeline (StandardScaler + LogisticRegression)
    clf_explainer: shap.LinearExplainer already built on the scaled background
    clf_scaler   : the StandardScaler extracted from the pipeline
    feature_cols : ordered list of feature names

    Returns
    -------
    dict with keys:
      "shap_records" - list of {feature_name, feature_value, shap_value}
                       sorted by abs(shap_value) descending
      "waterfall_fig"- matplotlib Figure object (caller must close/display it)
      "predicted_class"  - 0 or 1
      "predicted_proba"  - float probability of default
    """
    # Build a single-row DataFrame in the correct column order
    row_df = pd.DataFrame([feature_dict], columns=feature_cols)

    # Model prediction (uses the full Pipeline including scaler)
    pred_class = int(clf_model.predict(row_df)[0])
    pred_proba = float(clf_model.predict_proba(row_df)[0, 1])

    # Scale the row the same way the explainer expects
    row_scaled = clf_scaler.transform(row_df)
    row_scaled_df = pd.DataFrame(row_scaled, columns=feature_cols)

    # Compute SHAP values for this single row
    local_shap = clf_explainer(row_scaled_df)
    if len(local_shap.shape) == 3:      # binary TreeExplainer returns (1, f, 2)
        local_shap = local_shap[:, :, 1]

    # Build structured records sorted by |SHAP| descending
    shap_vals  = local_shap.values[0]
    feat_names = local_shap.feature_names
    shap_records = sorted(
        [
            {
                "feature_name":  feat,
                "feature_value": float(feature_dict[feat]),
                "shap_value":    float(sv),
            }
            for feat, sv in zip(feat_names, shap_vals)
        ],
        key=lambda r: abs(r["shap_value"]),
        reverse=True,
    )

    # Waterfall figure — caller is responsible for closing it
    fig, _ = plt.subplots(figsize=(9, 5))
    shap.plots.waterfall(local_shap[0], show=False)
    title = (
        f"SHAP Waterfall — Live Applicant\n"
        f"Predicted: {'HIGH RISK' if pred_class == 1 else 'LOW RISK'}  "
        f"P(default)={pred_proba:.3f}"
    )
    plt.title(title, fontsize=10)
    plt.tight_layout()

    return {
        "shap_records":    shap_records,
        "waterfall_fig":   fig,
        "predicted_class": pred_class,
        "predicted_proba": pred_proba,
    }


if __name__ == "__main__":
    run_shap_pipeline()
