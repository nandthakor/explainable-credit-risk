# =============================================================================
# Structured Evidence Contract & Case Export
# =============================================================================
# Defines a clean JSON schema for an instance's complete explanation
# evidence: prediction details, SHAP attributions, and verified
# counterfactual recourse. Then selects a stratified sample of 50 test
# cases and generates the full evidence record for each.
#
# This is the "Structured Evidence Contract" — the fixed data interface
# between the XAI computation layer and the LLM explanation generator.
# By defining this contract explicitly, the LLM prompt templates can
# reference a guaranteed schema rather than ad-hoc data formats.
#
# Usage:
#   python src/evaluation/evidence_contract.py
#
# No external API calls. No model retraining.

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import shap

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from xai.taxonomy import ALL_FEATURE_COLS, get_features_to_vary, get_permitted_ranges
from xai.recourse_engine import (
    setup_constrained_dice,
    generate_constrained_cfs,
    run_funnel,
    NUM_CFS,
)

# ---- Paths ------------------------------------------------------------------
PROCESSED    = os.path.join(PROJECT_ROOT, "data", "processed")
REPORTS      = os.path.join(PROJECT_ROOT, "reports")
EVAL_DIR     = os.path.join(REPORTS, "evaluation")
EVIDENCE_DIR = os.path.join(EVAL_DIR, "evidence_records")
MODELS       = os.path.join(PROJECT_ROOT, "models")

os.makedirs(EVIDENCE_DIR, exist_ok=True)

# ---- Constants --------------------------------------------------------------
RANDOM_STATE   = 42
SAMPLE_SIZE    = 50   # total cases to export
SHAP_TOP_N     = 5    # top-N SHAP features to include
DECISION_THRESHOLD = 0.5  # standard threshold for classification


# =============================================================================
# Evidence Schema Definition
# =============================================================================
EVIDENCE_SCHEMA = {
    "case_id": "int — unique test-set row index",
    "prediction": {
        "predicted_class": "int — 0 or 1",
        "predicted_proba": "float — P(default)",
        "raw_log_odds": "float — decision_function output",
        "threshold": "float — decision threshold used",
        "actual_label": "int — ground truth",
        "case_category": "str — TP/TN/FP/FN/borderline",
    },
    "shap_attributions": {
        "base_value": "float — expected value (log-odds)",
        "top_features": [
            {
                "rank": "int",
                "feature_name": "str",
                "feature_value": "float",
                "shap_value": "float",
                "direction": "str — increases_risk / decreases_risk",
                "abs_shap": "float — |shap_value|",
            },
        ],
    },
    "counterfactual_recourse": {
        "num_requested": "int",
        "num_generated": "int",
        "num_passed_funnel": "int",
        "candidates": [
            {
                "cf_index": "int",
                "changed_features": [
                    {
                        "feature_name": "str",
                        "original_value": "float",
                        "cf_value": "float",
                        "direction": "str",
                        "delta": "float",
                    },
                ],
                "cf_predicted_class": "int",
                "cf_predicted_proba": "float",
                "verification": {
                    "model_valid": "bool",
                    "constraints_satisfied": "bool",
                    "sparsity_l0": "int",
                    "gower_distance": "float",
                    "violations": "list of str",
                    "final_status": "str — PASSED / REJECTED",
                },
            },
        ],
    },
}


# =============================================================================
# Step 1 — Stratified test-case sampling
# =============================================================================
def sample_stratified_cases(
    X_test, y_test, clf_model, n_total=SAMPLE_SIZE, seed=RANDOM_STATE
):
    """
    Select a stratified sample of test cases covering:
      - True Positives  (predicted=1, actual=1)
      - False Positives (predicted=1, actual=0)
      - True Negatives  (predicted=0, actual=0)
      - False Negatives (predicted=0, actual=1)
      - Borderline      (predicted_proba near threshold)

    Returns a list of dicts with case_id, features, actual_label, category.
    """
    rng = np.random.RandomState(seed)

    # Get predictions
    y_pred = clf_model.predict(X_test)
    y_proba = clf_model.predict_proba(X_test)[:, 1]

    # Categorise
    is_tp = (y_pred == 1) & (y_test.values == 1)
    is_fp = (y_pred == 1) & (y_test.values == 0)
    is_tn = (y_pred == 0) & (y_test.values == 0)
    is_fn = (y_pred == 0) & (y_test.values == 1)
    is_borderline = (y_proba >= 0.40) & (y_proba <= 0.60)

    # Get indices for each category
    tp_idx = np.where(is_tp)[0]
    fp_idx = np.where(is_fp)[0]
    tn_idx = np.where(is_tn)[0]
    fn_idx = np.where(is_fn)[0]
    bl_idx = np.where(is_borderline)[0]

    # Target: ~10 per category, with borderline as a bonus stratum
    per_cat = max(1, n_total // 5)

    def safe_sample(idx_arr, n, label):
        n = min(n, len(idx_arr))
        if n == 0:
            return []
        chosen = rng.choice(idx_arr, size=n, replace=False)
        return [(int(c), label) for c in chosen]

    sampled = []
    sampled.extend(safe_sample(tp_idx, per_cat, "true_positive"))
    sampled.extend(safe_sample(fp_idx, per_cat, "false_positive"))
    sampled.extend(safe_sample(tn_idx, per_cat, "true_negative"))
    sampled.extend(safe_sample(fn_idx, per_cat, "false_negative"))
    sampled.extend(safe_sample(bl_idx, per_cat, "borderline"))

    # Deduplicate (borderline can overlap with other categories)
    seen = set()
    unique = []
    for idx, cat in sampled:
        if idx not in seen:
            seen.add(idx)
            unique.append((idx, cat))

    # Trim or pad to exactly n_total
    if len(unique) > n_total:
        unique = unique[:n_total]

    # Build output
    cases = []
    for idx, category in unique:
        row = X_test.iloc[idx]
        cases.append({
            "case_id": int(X_test.index[idx]) if hasattr(X_test, 'index') else idx,
            "test_row_index": idx,
            "features": {feat: float(row[feat]) for feat in X_test.columns},
            "actual_label": int(y_test.values[idx]),
            "case_category": category,
        })

    return cases


# =============================================================================
# Step 2 — Build SHAP explainer (reuses existing pattern)
# =============================================================================
def build_shap_explainer(clf_model, X_train, feature_cols):
    """
    Build a SHAP LinearExplainer using the same approach as the original
    Phase 3 code: unwrap Pipeline, scale data, create explainer.
    """
    from sklearn.pipeline import Pipeline as SkPipeline

    if isinstance(clf_model, SkPipeline):
        scaler = clf_model.named_steps.get("scaler")
        inner_model = clf_model.named_steps.get("clf")
    else:
        scaler = None
        inner_model = clf_model

    # Scale training data for background
    X_train_feat = X_train[feature_cols]
    if scaler is not None:
        X_scaled = pd.DataFrame(
            scaler.transform(X_train_feat),
            columns=feature_cols,
        )
    else:
        X_scaled = X_train_feat

    # Background: mean of scaled training data (1 row)
    background = X_scaled.mean(axis=0).values.reshape(1, -1)

    explainer = shap.LinearExplainer(inner_model, background)

    return explainer, scaler


def compute_shap_for_instance(explainer, scaler, instance_features, feature_cols, top_n=SHAP_TOP_N):
    """
    Compute SHAP values for a single instance and return the top-N
    features with their signed values and directions.
    """
    row = pd.DataFrame([instance_features], columns=feature_cols)

    if scaler is not None:
        row_scaled = pd.DataFrame(
            scaler.transform(row), columns=feature_cols
        )
    else:
        row_scaled = row

    shap_values = explainer.shap_values(row_scaled.values)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # class 1 (default)
    shap_values = shap_values.flatten()
    base_value = float(explainer.expected_value)
    if isinstance(base_value, np.ndarray):
        base_value = float(base_value[0])

    # Build ranked list
    abs_shap = np.abs(shap_values)
    ranked_idx = np.argsort(abs_shap)[::-1][:top_n]

    top_features = []
    for rank, idx in enumerate(ranked_idx):
        feat_name = feature_cols[idx]
        sv = float(shap_values[idx])
        top_features.append({
            "rank": rank + 1,
            "feature_name": feat_name,
            "feature_value": float(instance_features[feat_name]),
            "shap_value": round(sv, 5),
            "direction": "increases_risk" if sv > 0 else "decreases_risk",
            "abs_shap": round(float(abs_shap[idx]), 5),
        })

    return {
        "base_value": round(base_value, 5),
        "top_features": top_features,
        "all_shap_values": {
            feature_cols[i]: round(float(shap_values[i]), 5)
            for i in range(len(feature_cols))
        },
    }


# =============================================================================
# Step 3 — Build complete evidence record for one instance
# =============================================================================
def build_evidence_record(
    case, clf_model, shap_explainer, shap_scaler, feature_cols,
    dice_exp, features_to_vary, permitted_ranges, X_train
):
    """
    Build the complete structured evidence record for a single case.
    """
    features = case["features"]
    case_id = case["case_id"]

    # ---- Prediction block ----------------------------------------------------
    row_df = pd.DataFrame([features], columns=feature_cols)
    pred_class = int(clf_model.predict(row_df)[0])
    pred_proba = float(clf_model.predict_proba(row_df)[0, 1])

    # Raw log-odds from decision_function (Pipeline handles scaling)
    try:
        raw_score = float(clf_model.decision_function(row_df)[0])
    except AttributeError:
        raw_score = None

    prediction = {
        "predicted_class": pred_class,
        "predicted_proba": round(pred_proba, 5),
        "raw_log_odds": round(raw_score, 5) if raw_score is not None else None,
        "threshold": DECISION_THRESHOLD,
        "actual_label": case["actual_label"],
        "case_category": case["case_category"],
    }

    # ---- SHAP block ----------------------------------------------------------
    shap_block = compute_shap_for_instance(
        shap_explainer, shap_scaler, features, feature_cols
    )

    # ---- Counterfactual block ------------------------------------------------
    cf_records, cf_meta = generate_constrained_cfs(
        dice_exp, clf_model, features, feature_cols,
        features_to_vary, permitted_ranges,
    )

    if cf_records:
        verified, funnel_stats = run_funnel(
            cf_records, clf_model, feature_cols, X_train
        )
        # Merge verification info into CF records
        candidates = []
        for cf_rec, ver_rec in zip(cf_records, verified):
            candidates.append({
                "cf_index": cf_rec["cf_index"],
                "changed_features": cf_rec["changed_features"],
                "cf_predicted_class": cf_rec["cf_predicted_class"],
                "cf_predicted_proba": round(cf_rec["cf_predicted_proba"], 5),
                "verification": {
                    "model_valid": ver_rec["stage1_model_valid"],
                    "constraints_satisfied": ver_rec["stage2_constraints_satisfied"],
                    "sparsity_l0": ver_rec["stage3_sparsity_l0"],
                    "gower_distance": ver_rec["stage3_gower_distance"],
                    "violations": ver_rec["stage2_violations"],
                    "final_status": ver_rec["final_status"],
                },
            })

        cf_block = {
            "num_requested": cf_meta["num_requested"],
            "num_generated": cf_meta["num_generated"],
            "num_passed_funnel": funnel_stats["passed_all"],
            "candidates": candidates,
        }
    else:
        reason = cf_meta.get("reason_skipped", "unknown")
        cf_block = {
            "num_requested": NUM_CFS,
            "num_generated": 0,
            "num_passed_funnel": 0,
            "candidates": [],
            "reason_skipped": reason,
        }

    # ---- Assemble final record -----------------------------------------------
    evidence = {
        "case_id": case_id,
        "prediction": prediction,
        "shap_attributions": shap_block,
        "counterfactual_recourse": cf_block,
    }

    return evidence


# =============================================================================
# Main pipeline
# =============================================================================
def run_evidence_pipeline():
    """
    Select 50 stratified test cases, generate full evidence records,
    and export to reports/evaluation/evidence_records/.
    """
    print("\n" + "=" * 60)
    print("STRUCTURED EVIDENCE CONTRACT - CASE EXPORT")
    print("=" * 60)

    # ---- Load model and data -------------------------------------------------
    clf_model = joblib.load(os.path.join(MODELS, "classification_model.joblib"))
    with open(os.path.join(REPORTS, "feature_columns_clf.json")) as f:
        feature_cols = json.load(f)

    X_train = pd.read_csv(os.path.join(PROCESSED, "X_train_clf.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED, "y_train_clf.csv")).squeeze()
    X_test = pd.read_csv(os.path.join(PROCESSED, "X_test_clf.csv"))
    y_test = pd.read_csv(os.path.join(PROCESSED, "y_test_clf.csv")).squeeze()

    print(f"  Model: {type(clf_model).__name__}")
    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")

    # ---- Stratified sampling -------------------------------------------------
    print(f"\n  Sampling {SAMPLE_SIZE} stratified test cases...")
    cases = sample_stratified_cases(X_test, y_test, clf_model, n_total=SAMPLE_SIZE)

    # Print category distribution
    from collections import Counter
    cat_counts = Counter(c["case_category"] for c in cases)
    for cat, count in sorted(cat_counts.items()):
        print(f"    {cat}: {count}")
    print(f"    Total: {len(cases)}")

    # ---- Set up SHAP ---------------------------------------------------------
    print("\n  Setting up SHAP explainer...")
    shap_explainer, shap_scaler = build_shap_explainer(clf_model, X_train, feature_cols)
    print("  [OK] SHAP LinearExplainer ready")

    # ---- Set up constrained DiCE ---------------------------------------------
    print("  Setting up constrained DiCE...")
    dice_exp, features_to_vary, permitted_ranges = setup_constrained_dice(
        clf_model, X_train, y_train, feature_cols
    )
    print(f"  [OK] DiCE ready (features_to_vary: {features_to_vary})")

    # ---- Generate evidence for each case -------------------------------------
    print(f"\n  Generating evidence for {len(cases)} cases...")
    all_evidence = []
    funnel_totals = {
        "total_adverse": 0,
        "total_requested": 0,
        "total_generated": 0,
        "total_model_valid": 0,
        "total_constraint_ok": 0,
        "total_passed": 0,
    }

    for i, case in enumerate(cases):
        cid = case["case_id"]
        cat = case["case_category"]
        print(f"\n  [{i+1}/{len(cases)}] case_id={cid} ({cat})", end="")

        evidence = build_evidence_record(
            case, clf_model, shap_explainer, shap_scaler, feature_cols,
            dice_exp, features_to_vary, permitted_ranges, X_train,
        )

        # Track funnel stats
        cf_block = evidence["counterfactual_recourse"]
        if evidence["prediction"]["predicted_class"] == 1:
            funnel_totals["total_adverse"] += 1
            funnel_totals["total_requested"] += cf_block["num_requested"]
            funnel_totals["total_generated"] += cf_block["num_generated"]
            for cand in cf_block.get("candidates", []):
                ver = cand["verification"]
                if ver["model_valid"]:
                    funnel_totals["total_model_valid"] += 1
                if ver["constraints_satisfied"]:
                    funnel_totals["total_constraint_ok"] += 1
                if ver["final_status"] == "PASSED":
                    funnel_totals["total_passed"] += 1

        # Save individual evidence file
        out_path = os.path.join(EVIDENCE_DIR, f"case_{cid}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(evidence, f, indent=2, ensure_ascii=False)

        status = "ADVERSE" if evidence["prediction"]["predicted_class"] == 1 else "non-adverse"
        n_passed = cf_block.get("num_passed_funnel", 0)
        print(f"  [{status}] CFs passed: {n_passed}")

        all_evidence.append(evidence)

    # ---- Export summary -------------------------------------------------------
    print("\n" + "=" * 60)
    print("AGGREGATE FUNNEL STATISTICS")
    print("=" * 60)
    for k, v in funnel_totals.items():
        print(f"  {k}: {v}")

    # Conversion rates
    if funnel_totals["total_generated"] > 0:
        mv_rate = funnel_totals["total_model_valid"] / funnel_totals["total_generated"]
        cs_rate = funnel_totals["total_constraint_ok"] / funnel_totals["total_generated"]
        pass_rate = funnel_totals["total_passed"] / funnel_totals["total_generated"]
        print(f"\n  Model-validity rate:  {mv_rate:.1%}")
        print(f"  Constraint-pass rate: {cs_rate:.1%}")
        print(f"  Overall pass rate:    {pass_rate:.1%}")

    # Save aggregate summary
    summary_path = os.path.join(EVAL_DIR, "evidence_summary.json")
    summary = {
        "total_cases_exported": len(all_evidence),
        "category_distribution": dict(cat_counts),
        "funnel_totals": funnel_totals,
        "schema_version": "1.0",
        "evidence_directory": EVIDENCE_DIR,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  [saved] {summary_path}")
    print(f"  [saved] {len(all_evidence)} evidence records in {EVIDENCE_DIR}/")

    return all_evidence


if __name__ == "__main__":
    run_evidence_pipeline()
