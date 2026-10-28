# =============================================================================
# Domain-Constrained Counterfactual Recourse Engine
# =============================================================================
# Generates counterfactual explanations using DiCE with strict domain
# constraints from the Feature Actionability Taxonomy, then passes every
# candidate through an independent multi-stage verification funnel.
#
# Key differences from the original Phase 4 counterfactual_explain.py:
#   1. Historical features (late-payment counts) are now FROZEN, not just
#      directionally constrained. A borrower cannot rewrite their credit
#      bureau history.
#   2. Permitted ranges are derived from training-data percentiles,
#      intersected with domain knowledge (e.g., utilisation in [0, 1]).
#   3. An independent verification funnel checks EVERY candidate CF after
#      generation — it does not trust DiCE's internal assertions.
#   4. The funnel tracks conversion at each stage for Paper Table 1.
#
# Usage:
#   from src.xai.recourse_engine import generate_constrained_cfs, run_funnel
#
# Run standalone:
#   python src/xai/recourse_engine.py
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
import dice_ml

# Add project root to path so we can import taxonomy
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from xai.taxonomy import (
    get_features_to_vary,
    get_permitted_ranges,
    get_direction_constraints,
    get_all_frozen_features,
    ACTIONABLE_FEATURES,
    ALL_FEATURE_COLS,
    IMMUTABLE_FEATURES,
    HISTORICAL_FEATURES,
)

# ---- Paths ------------------------------------------------------------------
PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
REPORTS   = os.path.join(PROJECT_ROOT, "reports")
EVAL_DIR  = os.path.join(REPORTS, "evaluation")
MODELS    = os.path.join(PROJECT_ROOT, "models")

os.makedirs(EVAL_DIR, exist_ok=True)

# ---- Constants --------------------------------------------------------------
NUM_CFS       = 3       # counterfactuals per instance (paper uses 3)
RANDOM_STATE  = 42
DICE_METHOD   = "genetic"


# =============================================================================
# Step 1 — Load model and training data
# =============================================================================
def load_model_and_data():
    """Load the saved classification pipeline and training data."""
    clf_model = joblib.load(os.path.join(MODELS, "classification_model.joblib"))

    with open(os.path.join(REPORTS, "feature_columns_clf.json")) as f:
        feature_cols = json.load(f)

    X_train = pd.read_csv(os.path.join(PROCESSED, "X_train_clf.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED, "y_train_clf.csv")).squeeze()

    return clf_model, X_train, y_train, feature_cols


# =============================================================================
# Step 2 — Set up constrained DiCE
# =============================================================================
def setup_constrained_dice(clf_model, X_train, y_train, feature_cols):
    """
    Create DiCE objects with taxonomy-derived constraints:
      - features_to_vary: only actionable features (5 of 11)
      - permitted_range: data-driven bounds intersected with domain rules
    """
    # Build background dataframe
    bg = X_train[feature_cols].copy()
    bg["target_default"] = y_train.values

    # DiCE Data object
    d = dice_ml.Data(
        dataframe=bg,
        continuous_features=feature_cols,
        outcome_name="target_default",
    )

    # DiCE Model object
    m = dice_ml.Model(model=clf_model, backend="sklearn")

    # DiCE explainer
    exp = dice_ml.Dice(d, m, method=DICE_METHOD)

    # Compute permitted ranges from training data
    permitted_ranges = get_permitted_ranges(X_train)

    # Features DiCE is allowed to touch
    features_to_vary = get_features_to_vary()

    return exp, features_to_vary, permitted_ranges


# =============================================================================
# Step 3 — Generate constrained counterfactuals for one instance
# =============================================================================
def generate_constrained_cfs(
    exp, clf_model, instance_features, feature_cols,
    features_to_vary, permitted_ranges, num_cfs=NUM_CFS
):
    """
    Generate counterfactuals for a single instance using DiCE with
    domain constraints applied at the generator level.

    Parameters
    ----------
    exp : dice_ml.Dice
        The DiCE explainer object.
    clf_model : sklearn model
        The classification pipeline.
    instance_features : dict
        Feature name -> value for the query instance.
    feature_cols : list
        Ordered list of feature column names.
    features_to_vary : list
        Only actionable features.
    permitted_ranges : dict
        {feature: [lo, hi]} bounds.
    num_cfs : int
        Number of counterfactuals to request.

    Returns
    -------
    list of dict
        Each dict has: cf_features, original_features, changed_features,
        cf_predicted_class, cf_predicted_proba.
    """
    # Build the query row
    query_row = pd.DataFrame([instance_features], columns=feature_cols)

    original_pred = clf_model.predict(query_row)[0]
    original_proba = clf_model.predict_proba(query_row)[0, 1]

    # Only generate recourse for adverse predictions (predicted default)
    if int(original_pred) == 0:
        return [], {
            "original_pred": int(original_pred),
            "original_proba": float(original_proba),
            "reason_skipped": "not_adverse",
        }

    # Generate counterfactuals
    try:
        cf_result = exp.generate_counterfactuals(
            query_row,
            total_CFs=num_cfs,
            desired_class="opposite",
            features_to_vary=features_to_vary,
            permitted_range=permitted_ranges,
        )
    except Exception as e:
        return [], {
            "original_pred": int(original_pred),
            "original_proba": float(original_proba),
            "reason_skipped": f"dice_error: {str(e)}",
        }

    # Extract CF dataframe
    cf_df = cf_result.cf_examples_list[0].final_cfs_df
    if cf_df is None or len(cf_df) == 0:
        return [], {
            "original_pred": int(original_pred),
            "original_proba": float(original_proba),
            "reason_skipped": "no_cfs_generated",
        }

    # Drop target column if present
    if "target_default" in cf_df.columns:
        cf_df = cf_df.drop(columns=["target_default"])
    cf_df = cf_df[feature_cols]

    # Build structured CF records
    cf_records = []
    for idx in range(len(cf_df)):
        cf_row = cf_df.iloc[idx]
        cf_row_df = cf_df.iloc[[idx]]

        # Model predictions on the CF
        cf_pred = int(clf_model.predict(cf_row_df)[0])
        cf_proba = float(clf_model.predict_proba(cf_row_df)[0, 1])

        # Identify changed features
        changed = []
        for feat in feature_cols:
            orig_val = instance_features[feat]
            cf_val = float(cf_row[feat])
            if not np.isclose(orig_val, cf_val, atol=1e-6):
                direction = "increase" if cf_val > orig_val else "decrease"
                changed.append({
                    "feature_name": feat,
                    "original_value": float(orig_val),
                    "cf_value": cf_val,
                    "direction": direction,
                    "delta": cf_val - float(orig_val),
                })

        cf_records.append({
            "cf_index": idx,
            "cf_features": {feat: float(cf_row[feat]) for feat in feature_cols},
            "original_features": {feat: float(instance_features[feat]) for feat in feature_cols},
            "changed_features": changed,
            "cf_predicted_class": cf_pred,
            "cf_predicted_proba": cf_proba,
            "num_features_changed": len(changed),
        })

    meta = {
        "original_pred": int(original_pred),
        "original_proba": float(original_proba),
        "num_requested": num_cfs,
        "num_generated": len(cf_records),
    }

    return cf_records, meta


# =============================================================================
# Step 4 — Independent Multi-Stage Verification Funnel
# =============================================================================
def verify_single_cf(cf_record, clf_model, feature_cols, X_train):
    """
    Run a single counterfactual through the independent verification funnel.

    Returns a dict with pass/fail at each stage plus computed metrics.
    """
    original = cf_record["original_features"]
    cf_feats = cf_record["cf_features"]
    frozen_features = get_all_frozen_features()
    direction_rules = get_direction_constraints()

    result = {
        "cf_index": cf_record["cf_index"],
        "stage1_model_valid": False,
        "stage2_constraints_satisfied": False,
        "stage2_violations": [],
        "stage3_sparsity_l0": 0,
        "stage3_gower_distance": 0.0,
        "final_status": "REJECTED",
    }

    # ---- Stage 1: Model-Validity Check --------------------------------------
    # Does the CF actually flip the model prediction?
    cf_row_df = pd.DataFrame([cf_feats], columns=feature_cols)
    cf_pred = int(clf_model.predict(cf_row_df)[0])
    original_row_df = pd.DataFrame([original], columns=feature_cols)
    original_pred = int(clf_model.predict(original_row_df)[0])

    if cf_pred != original_pred:
        result["stage1_model_valid"] = True
    else:
        result["stage2_violations"].append(
            f"Model did not flip: original_pred={original_pred}, cf_pred={cf_pred}"
        )
        return result  # Fail fast

    # ---- Stage 2: Constraint-Satisfaction Check ------------------------------
    violations = []

    # 2a: Frozen features (immutable + historical) must be identical
    for feat in frozen_features:
        orig_val = original[feat]
        cf_val = cf_feats[feat]
        if not np.isclose(orig_val, cf_val, atol=1e-6):
            violations.append(
                f"FROZEN feature '{feat}' changed: {orig_val} -> {cf_val}"
            )

    # 2b: Directional constraints on actionable features
    for feat, allowed_dir in direction_rules.items():
        orig_val = original[feat]
        cf_val = cf_feats[feat]
        if np.isclose(orig_val, cf_val, atol=1e-6):
            continue  # not changed, no violation

        if allowed_dir == "decrease" and cf_val > orig_val + 1e-6:
            violations.append(
                f"Direction violation: '{feat}' must decrease but "
                f"went {orig_val:.4f} -> {cf_val:.4f}"
            )
        elif allowed_dir == "increase" and cf_val < orig_val - 1e-6:
            violations.append(
                f"Direction violation: '{feat}' must increase but "
                f"went {orig_val:.4f} -> {cf_val:.4f}"
            )

    # 2c: Domain-specific cross-feature constraint
    # real_estate_loans must not exceed open_credit_lines
    if cf_feats["real_estate_loans"] > cf_feats["open_credit_lines"]:
        violations.append(
            f"Cross-feature violation: real_estate_loans ({cf_feats['real_estate_loans']}) "
            f"> open_credit_lines ({cf_feats['open_credit_lines']})"
        )

    # 2d: Integer constraints
    for feat, rules in ACTIONABLE_FEATURES.items():
        if rules["is_integer"]:
            cf_val = cf_feats[feat]
            if not np.isclose(cf_val, round(cf_val), atol=0.01):
                violations.append(
                    f"Integer violation: '{feat}' = {cf_val} (not integer)"
                )

    # 2e: Range bounds
    for feat, rules in ACTIONABLE_FEATURES.items():
        cf_val = cf_feats[feat]
        if rules["domain_min"] is not None and cf_val < rules["domain_min"] - 1e-6:
            violations.append(
                f"Range violation: '{feat}' = {cf_val} < domain_min {rules['domain_min']}"
            )
        if rules["domain_max"] is not None and cf_val > rules["domain_max"] + 1e-6:
            violations.append(
                f"Range violation: '{feat}' = {cf_val} > domain_max {rules['domain_max']}"
            )

    result["stage2_violations"] = violations
    result["stage2_constraints_satisfied"] = (len(violations) == 0)

    # ---- Stage 3: Plausibility & Sparsity Metrics ----------------------------
    # Compute even if stage 2 failed (for analysis)

    # L0 sparsity: number of features changed
    l0 = sum(
        1 for feat in feature_cols
        if not np.isclose(original[feat], cf_feats[feat], atol=1e-6)
    )
    result["stage3_sparsity_l0"] = l0

    # Gower-like normalised L1 distance
    # Normalise each feature's absolute change by its training-data range
    gower_sum = 0.0
    n_features = len(feature_cols)
    for feat in feature_cols:
        feat_range = X_train[feat].max() - X_train[feat].min()
        if feat_range < 1e-10:
            feat_range = 1.0  # constant feature
        gower_sum += abs(cf_feats[feat] - original[feat]) / feat_range
    result["stage3_gower_distance"] = round(gower_sum / n_features, 6)

    # Final status
    if result["stage1_model_valid"] and result["stage2_constraints_satisfied"]:
        result["final_status"] = "PASSED"

    return result


def run_funnel(cf_records, clf_model, feature_cols, X_train):
    """
    Run the full verification funnel on a list of CF records.

    Returns:
        verified_results: list of verification dicts
        funnel_stats: dict with counts at each stage
    """
    verified = []
    for cf_rec in cf_records:
        v = verify_single_cf(cf_rec, clf_model, feature_cols, X_train)
        verified.append(v)

    # Aggregate funnel statistics
    n_generated = len(verified)
    n_model_valid = sum(1 for v in verified if v["stage1_model_valid"])
    n_constraint_ok = sum(1 for v in verified if v["stage2_constraints_satisfied"])
    n_passed = sum(1 for v in verified if v["final_status"] == "PASSED")

    # Average metrics for passed CFs
    passed = [v for v in verified if v["final_status"] == "PASSED"]
    avg_sparsity = np.mean([v["stage3_sparsity_l0"] for v in passed]) if passed else 0
    avg_gower = np.mean([v["stage3_gower_distance"] for v in passed]) if passed else 0

    funnel_stats = {
        "requested": NUM_CFS,
        "generated": n_generated,
        "model_valid": n_model_valid,
        "constraint_satisfied": n_constraint_ok,
        "passed_all": n_passed,
        "avg_sparsity_l0": round(float(avg_sparsity), 2),
        "avg_gower_distance": round(float(avg_gower), 6),
    }

    return verified, funnel_stats


# =============================================================================
# Step 5 — Run on a batch of test instances and export funnel summary
# =============================================================================
def run_batch(instances, clf_model, X_train, feature_cols,
              exp, features_to_vary, permitted_ranges):
    """
    Generate constrained CFs for a batch of instances and run the
    verification funnel. Returns per-instance results and aggregate stats.
    """
    all_results = []
    aggregate_funnel = {
        "total_requested": 0,
        "total_generated": 0,
        "total_model_valid": 0,
        "total_constraint_satisfied": 0,
        "total_passed": 0,
    }

    for i, inst in enumerate(instances):
        case_id = inst.get("case_id", i)
        features = inst["features"]

        print(f"\n  [{i+1}/{len(instances)}] case_id={case_id}")

        # Generate constrained CFs
        cf_records, meta = generate_constrained_cfs(
            exp, clf_model, features, feature_cols,
            features_to_vary, permitted_ranges,
        )

        if not cf_records:
            reason = meta.get("reason_skipped", "unknown")
            print(f"    Skipped: {reason}")
            all_results.append({
                "case_id": case_id,
                "original_pred": meta.get("original_pred"),
                "original_proba": meta.get("original_proba"),
                "num_cfs_generated": 0,
                "funnel_stats": None,
                "verified_cfs": [],
                "reason_skipped": reason,
            })
            continue

        # Run verification funnel
        verified, funnel_stats = run_funnel(
            cf_records, clf_model, feature_cols, X_train
        )

        print(f"    Generated: {funnel_stats['generated']}  "
              f"Model-valid: {funnel_stats['model_valid']}  "
              f"Constraint-OK: {funnel_stats['constraint_satisfied']}  "
              f"PASSED: {funnel_stats['passed_all']}")

        # Update aggregates
        aggregate_funnel["total_requested"] += funnel_stats["requested"]
        aggregate_funnel["total_generated"] += funnel_stats["generated"]
        aggregate_funnel["total_model_valid"] += funnel_stats["model_valid"]
        aggregate_funnel["total_constraint_satisfied"] += funnel_stats["constraint_satisfied"]
        aggregate_funnel["total_passed"] += funnel_stats["passed_all"]

        all_results.append({
            "case_id": case_id,
            "original_pred": meta["original_pred"],
            "original_proba": meta["original_proba"],
            "num_cfs_generated": len(cf_records),
            "funnel_stats": funnel_stats,
            "verified_cfs": verified,
            "cf_records": cf_records,
        })

    return all_results, aggregate_funnel


def export_funnel_summary(all_results, aggregate_funnel, output_path):
    """
    Export the funnel summary CSV for Paper Table 1.
    One row per instance plus an aggregate row.
    """
    rows = []
    for res in all_results:
        fs = res.get("funnel_stats")
        if fs:
            rows.append({
                "case_id": res["case_id"],
                "original_pred": res["original_pred"],
                "original_proba": res["original_proba"],
                "requested": fs["requested"],
                "generated": fs["generated"],
                "model_valid": fs["model_valid"],
                "constraint_satisfied": fs["constraint_satisfied"],
                "passed_all": fs["passed_all"],
                "avg_sparsity_l0": fs["avg_sparsity_l0"],
                "avg_gower_distance": fs["avg_gower_distance"],
            })
        else:
            rows.append({
                "case_id": res["case_id"],
                "original_pred": res["original_pred"],
                "original_proba": res["original_proba"],
                "requested": NUM_CFS,
                "generated": 0,
                "model_valid": 0,
                "constraint_satisfied": 0,
                "passed_all": 0,
                "avg_sparsity_l0": "",
                "avg_gower_distance": "",
            })

    # Aggregate row
    n_with_data = sum(1 for r in all_results if r.get("funnel_stats"))
    rows.append({
        "case_id": "AGGREGATE",
        "original_pred": "",
        "original_proba": "",
        "requested": aggregate_funnel["total_requested"],
        "generated": aggregate_funnel["total_generated"],
        "model_valid": aggregate_funnel["total_model_valid"],
        "constraint_satisfied": aggregate_funnel["total_constraint_satisfied"],
        "passed_all": aggregate_funnel["total_passed"],
        "avg_sparsity_l0": "",
        "avg_gower_distance": "",
    })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\n  [saved] {output_path}")

    return df


# =============================================================================
# Standalone runner
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DOMAIN-CONSTRAINED COUNTERFACTUAL RECOURSE ENGINE")
    print("=" * 60)

    # Load model and data
    clf_model, X_train, y_train, feature_cols = load_model_and_data()
    print(f"  Model: {type(clf_model).__name__}")
    print(f"  X_train: {X_train.shape}")
    print(f"  Features: {feature_cols}")

    # Set up constrained DiCE
    exp, features_to_vary, permitted_ranges = setup_constrained_dice(
        clf_model, X_train, y_train, feature_cols
    )
    print(f"\n  Features to vary ({len(features_to_vary)}): {features_to_vary}")
    print(f"  Permitted ranges:")
    for feat, bounds in permitted_ranges.items():
        print(f"    {feat}: {bounds}")

    # Load original 5 case studies for a quick test
    with open(os.path.join(REPORTS, "selected_instances.json")) as f:
        all_instances = json.load(f)
    clf_instances = [
        {"case_id": inst["instance_id"], "features": inst["features"]}
        for inst in all_instances if inst["task"] == "classification"
    ]

    print(f"\n  Running on {len(clf_instances)} case studies...")

    results, aggregate = run_batch(
        clf_instances, clf_model, X_train, feature_cols,
        exp, features_to_vary, permitted_ranges,
    )

    # Export funnel summary
    output_csv = os.path.join(EVAL_DIR, "counterfactual_funnel_summary.csv")
    export_funnel_summary(results, aggregate, output_csv)

    # Print aggregate
    print("\n" + "=" * 60)
    print("AGGREGATE FUNNEL RESULTS")
    print("=" * 60)
    for k, v in aggregate.items():
        print(f"  {k}: {v}")
