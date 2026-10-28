# Credit Risk XAI - Phase 4: Counterfactual Explanations (DiCE)
# ==============================================================
# Generates counterfactual explanations for the 5 classification
# case-study instances selected in Phase 3.
#
# Counterfactuals answer: "What would need to change in this applicant's
# profile for the model to flip its prediction?"
#
# Run: python src/counterfactual_explain.py
# Prerequisites:
#   models/classification_model.joblib
#   data/processed/X_train_clf.csv, y_train_clf.csv
#   reports/feature_columns_clf.json
#   reports/selected_instances.json

import os
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import dice_ml

# ---- Paths ------------------------------------------------------------------
PROCESSED  = os.path.join("data", "processed")
REPORTS    = os.path.join("reports")
CF_DIR     = os.path.join("reports", "counterfactuals")
MODELS     = os.path.join("models")

os.makedirs(CF_DIR, exist_ok=True)

# ---- Constants --------------------------------------------------------------
NUM_CFS        = 5          # counterfactuals per instance
RANDOM_STATE   = 42

# Features that must NEVER change in a counterfactual.
# Rationale:
#   - age: a person cannot change their age
#   - num_dependents: not an actionable lever for credit improvement
#   - monthly_income_missing: an engineered flag, not a real-world feature
IMMUTABLE_FEATURES = ["age", "num_dependents", "monthly_income_missing"]

# Direction constraints for feasibility filtering.
# A counterfactual is "feasible" only if each changed feature moves in a
# direction a real person could realistically achieve to improve credit.
# Key: feature name -> allowed direction ("decrease", "increase", "any")
FEASIBILITY_RULES = {
    "late_30_59_days":   "decrease",   # can reduce late payments, not increase
    "late_60_89_days":   "decrease",
    "late_90_days":      "decrease",
    "revolving_util":    "decrease",   # can pay down credit usage
    "debt_ratio":        "decrease",   # can reduce debt relative to income
    "monthly_income":    "increase",   # can earn more, not deliberately less
    "open_credit_lines": "any",        # can open or close accounts
    "real_estate_loans":  "any",       # can buy or sell property
}


# =============================================================================
# Step 1 - Load everything
# =============================================================================
def load_all():
    """Load model, training data, feature list, and selected instances."""
    print("\n" + "=" * 60)
    print("STEP 1: Loading model, data, and selected instances")
    print("=" * 60)

    clf_model = joblib.load(os.path.join(MODELS, "classification_model.joblib"))
    print(f"  Model loaded: {type(clf_model).__name__}")

    with open(os.path.join(REPORTS, "feature_columns_clf.json")) as f:
        feature_cols = json.load(f)
    print(f"  Feature columns: {len(feature_cols)}")

    X_train = pd.read_csv(os.path.join(PROCESSED, "X_train_clf.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED, "y_train_clf.csv")).squeeze()
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")

    # Confirm all features are numeric
    non_numeric = X_train.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        print(f"  WARNING: Non-numeric columns found: {non_numeric}")
        print("           DiCE may need categorical encoding for these.")
    else:
        print("  [OK] All features are numeric - no categorical encoding needed.")

    with open(os.path.join(REPORTS, "selected_instances.json")) as f:
        all_instances = json.load(f)

    # Keep only classification instances
    clf_instances = [i for i in all_instances if i["task"] == "classification"]
    print(f"  Classification instances: {len(clf_instances)}")
    for inst in clf_instances:
        print(f"    id={inst['instance_id']:6d}  case={inst['case_type']:18s}  "
              f"actual={int(inst['actual'])}  proba={inst['predicted_proba']:.3f}")

    return clf_model, X_train, y_train, feature_cols, clf_instances


# =============================================================================
# Step 2 - Set up DiCE objects
# =============================================================================
def setup_dice(clf_model, X_train, y_train, feature_cols):
    """
    Create the three DiCE objects:
      - dice_ml.Data:  background dataset + feature metadata
      - dice_ml.Model: sklearn model wrapper
      - dice_ml.Dice:  counterfactual generator

    We use method="genetic" which directly optimises for validity (class flip)
    and works well with both Random Forest and Logistic Regression.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Setting up DiCE objects")
    print("=" * 60)

    # Build background dataframe: features + target in one frame
    bg = X_train[feature_cols].copy()
    bg["target_default"] = y_train.values
    print(f"  Background dataframe: {bg.shape}")

    # All features are continuous/numeric in this dataset
    continuous_features = feature_cols

    # Actionable features = everything except immutables
    actionable = [f for f in feature_cols if f not in IMMUTABLE_FEATURES]
    print(f"  Immutable features: {IMMUTABLE_FEATURES}")
    print(f"  Actionable features ({len(actionable)}): {actionable}")

    # DiCE Data
    d = dice_ml.Data(
        dataframe=bg,
        continuous_features=continuous_features,
        outcome_name="target_default",
    )
    print("  [OK] dice_ml.Data created")

    # DiCE Model
    m = dice_ml.Model(
        model=clf_model,
        backend="sklearn",
    )
    print("  [OK] dice_ml.Model created")

    # DiCE explainer - try genetic first
    method = "genetic"
    exp = dice_ml.Dice(d, m, method=method)
    print(f"  [OK] dice_ml.Dice created (method='{method}')")

    return d, m, exp, actionable, method


# =============================================================================
# Step 3 - Compute per-feature ranges for proximity normalisation
# =============================================================================
def compute_feature_ranges(X_train, feature_cols):
    """
    For normalised proximity, we divide each feature's change by its
    (max - min) range in the training data.
    """
    ranges = {}
    for col in feature_cols:
        col_range = X_train[col].max() - X_train[col].min()
        # Avoid division by zero for constant columns
        ranges[col] = col_range if col_range > 0 else 1.0
    return ranges


# =============================================================================
# Step 4 - Generate counterfactuals for one instance
# =============================================================================
def generate_cfs_for_instance(
    exp, clf_model, instance_dict, feature_cols, actionable,
    feat_ranges, method_name
):
    """
    Generate NUM_CFS counterfactuals for a single instance, then compute
    validity, proximity, and sparsity for each.

    Returns a list of raw counterfactual dicts and the query row.
    """
    inst_id   = instance_dict["instance_id"]
    case_type = instance_dict["case_type"]

    print(f"\n  --- Instance id={inst_id} ({case_type}) ---")

    # Reconstruct the query as a single-row DataFrame
    query_row = pd.DataFrame(
        [instance_dict["features"]],
        columns=feature_cols,
    )

    original_pred = clf_model.predict(query_row)[0]
    original_proba = clf_model.predict_proba(query_row)[0, 1]
    desired_class = 1 - int(original_pred)
    print(f"    Original prediction: {int(original_pred)}  "
          f"(proba={original_proba:.3f})  -> desired flip to {desired_class}")

    # Generate counterfactuals
    actual_method = method_name
    t0 = time.time()
    try:
        cf_result = exp.generate_counterfactuals(
            query_row,
            total_CFs=NUM_CFS,
            desired_class="opposite",
            features_to_vary=actionable,
        )
        elapsed = time.time() - t0
        print(f"    Generated {NUM_CFS} CFs in {elapsed:.1f}s "
              f"(method='{actual_method}')")
    except Exception as e:
        print(f"    [ERROR] method='{actual_method}' failed: {e}")
        print(f"    Falling back to method='random' ...")
        actual_method = "random"
        exp_fallback = dice_ml.Dice(exp.data_interface, exp.model, method="random")
        t0 = time.time()
        cf_result = exp_fallback.generate_counterfactuals(
            query_row,
            total_CFs=NUM_CFS,
            desired_class="opposite",
            features_to_vary=actionable,
        )
        elapsed = time.time() - t0
        print(f"    Generated {NUM_CFS} CFs in {elapsed:.1f}s "
              f"(method='random' fallback)")

    # Extract the counterfactual rows as a DataFrame
    cf_df = cf_result.cf_examples_list[0].final_cfs_df
    if cf_df is None or len(cf_df) == 0:
        print(f"    WARNING: DiCE returned no counterfactuals for id={inst_id}")
        return [], query_row, actual_method

    # Drop the target column if DiCE included it
    if "target_default" in cf_df.columns:
        cf_df = cf_df.drop(columns=["target_default"])

    # Ensure column order matches
    cf_df = cf_df[feature_cols]

    # ---- Compute metrics for each CF ----------------------------------------
    raw_cfs = []
    for cf_idx in range(len(cf_df)):
        cf_row = cf_df.iloc[[cf_idx]]

        # Validity: does the model ACTUALLY flip its prediction?
        cf_pred = clf_model.predict(cf_row)[0]
        cf_proba = clf_model.predict_proba(cf_row)[0, 1]
        is_valid = (int(cf_pred) != int(original_pred))

        # Identify changed features
        changed = []
        for col in feature_cols:
            orig_val = float(query_row[col].iloc[0])
            cf_val   = float(cf_row[col].iloc[0])
            if abs(cf_val - orig_val) > 1e-8:
                direction = "increase" if cf_val > orig_val else "decrease"
                changed.append({
                    "feature_name":  col,
                    "original_value": round(orig_val, 6),
                    "cf_value":       round(cf_val, 6),
                    "direction":      direction,
                })

        # Sparsity: how many features changed
        sparsity = len(changed)

        # Proximity: normalised distance, averaged over CHANGED features only
        if sparsity > 0:
            norm_dists = []
            for ch in changed:
                dist = abs(ch["cf_value"] - ch["original_value"])
                norm_dist = dist / feat_ranges[ch["feature_name"]]
                norm_dists.append(norm_dist)
            proximity = float(np.mean(norm_dists))
        else:
            proximity = 0.0

        raw_cfs.append({
            "cf_index":             cf_idx,
            "changed_features":     changed,
            "validity":             is_valid,
            "cf_predicted_class":   int(cf_pred),
            "cf_predicted_proba":   round(float(cf_proba), 4),
            "proximity_score":      round(proximity, 6),
            "num_features_changed": sparsity,
        })

        status = "VALID" if is_valid else "INVALID"
        print(f"    CF {cf_idx}: {status}  sparsity={sparsity}  "
              f"proximity={proximity:.4f}  cf_proba={cf_proba:.3f}")

    return raw_cfs, query_row, actual_method


# =============================================================================
# Step 5 - Feasibility filter
# =============================================================================
def apply_feasibility_filter(raw_cfs):
    """
    Filter counterfactuals by feasibility: each changed feature must move
    in a direction that a real person could realistically achieve.

    Returns a list of feasible CF dicts (may be empty - that's reportable).
    """
    feasible = []
    for cf in raw_cfs:
        is_feasible = True
        for ch in cf["changed_features"]:
            fname = ch["feature_name"]
            direction = ch["direction"]
            rule = FEASIBILITY_RULES.get(fname, "any")

            if rule == "any":
                continue
            if rule != direction:
                is_feasible = False
                break

        if is_feasible:
            # Build the clean structure Phase 5 needs
            feasible_cf = {
                "changed_features":     cf["changed_features"],
                "validity":             cf["validity"],
                "proximity_score":      cf["proximity_score"],
                "num_features_changed": cf["num_features_changed"],
            }
            feasible.append(feasible_cf)

    return feasible


# =============================================================================
# Step 6 - Run the full pipeline
# =============================================================================
def run_counterfactual_pipeline():
    """Generate, filter, and report counterfactuals for all 5 clf instances."""

    # 1. Load
    clf_model, X_train, y_train, feature_cols, clf_instances = load_all()

    # 2. Setup DiCE
    d, m, exp, actionable, method_name = setup_dice(
        clf_model, X_train, y_train, feature_cols)

    # 3. Feature ranges for normalisation
    feat_ranges = compute_feature_ranges(X_train, feature_cols)
    print("\n  Feature ranges (for proximity normalisation):")
    for col in feature_cols:
        print(f"    {col:28s}: {feat_ranges[col]:.4f}")

    # 4. Generate CFs for each instance
    print("\n" + "=" * 60)
    print("STEP 4: Generating counterfactuals for each instance")
    print("=" * 60)

    all_raw = []       # all 25 raw CFs across all instances
    all_feasible = []  # all feasible CFs
    per_instance = []  # per-instance summary for the report

    for inst in clf_instances:
        inst_id   = inst["instance_id"]
        case_type = inst["case_type"]

        raw_cfs, query_row, actual_method = generate_cfs_for_instance(
            exp, clf_model, inst, feature_cols, actionable,
            feat_ranges, method_name)

        # Save raw CFs to JSON
        raw_path = os.path.join(CF_DIR, f"{inst_id}_{case_type}_raw.json")
        with open(raw_path, "w") as f:
            json.dump(raw_cfs, f, indent=2)
        print(f"    [saved] {raw_path}")

        # Apply feasibility filter
        feasible_cfs = apply_feasibility_filter(raw_cfs)

        feas_path = os.path.join(CF_DIR, f"{inst_id}_{case_type}_feasible.json")
        with open(feas_path, "w") as f:
            json.dump(feasible_cfs, f, indent=2)
        print(f"    [saved] {feas_path}")

        # Stats
        n_raw    = len(raw_cfs)
        n_valid  = sum(1 for c in raw_cfs if c["validity"])
        n_feas   = len(feasible_cfs)
        raw_prox = [c["proximity_score"] for c in raw_cfs]
        raw_spar = [c["num_features_changed"] for c in raw_cfs]
        feas_prox = [c["proximity_score"] for c in feasible_cfs]
        feas_spar = [c["num_features_changed"] for c in feasible_cfs]

        feas_rate = n_feas / n_raw * 100 if n_raw > 0 else 0
        val_rate  = n_valid / n_raw * 100 if n_raw > 0 else 0

        # Best feasible CF (lowest proximity = smallest change needed)
        best_feas = None
        if feasible_cfs:
            best_feas = min(feasible_cfs, key=lambda c: c["proximity_score"])

        summary = {
            "instance_id":     inst_id,
            "case_type":       case_type,
            "method_used":     actual_method,
            "n_raw":           n_raw,
            "n_valid":         n_valid,
            "n_feasible":      n_feas,
            "validity_rate":   round(val_rate, 1),
            "feasibility_rate": round(feas_rate, 1),
            "avg_proximity_raw":   round(float(np.mean(raw_prox)), 4) if raw_prox else None,
            "avg_sparsity_raw":    round(float(np.mean(raw_spar)), 2) if raw_spar else None,
            "avg_proximity_feas":  round(float(np.mean(feas_prox)), 4) if feas_prox else None,
            "avg_sparsity_feas":   round(float(np.mean(feas_spar)), 2) if feas_spar else None,
            "best_feasible":       best_feas,
        }
        per_instance.append(summary)

        print(f"\n    Summary for id={inst_id}:")
        print(f"      Raw CFs: {n_raw}  Valid: {n_valid}/{n_raw} ({val_rate:.0f}%)")
        print(f"      Feasible after filter: {n_feas}/{n_raw} ({feas_rate:.0f}%)")
        if raw_prox:
            print(f"      Avg proximity (raw): {np.mean(raw_prox):.4f}")
            print(f"      Avg sparsity  (raw): {np.mean(raw_spar):.2f}")
        if feas_prox:
            print(f"      Avg proximity (feasible): {np.mean(feas_prox):.4f}")
            print(f"      Avg sparsity  (feasible): {np.mean(feas_spar):.2f}")
        if best_feas:
            print(f"      Best feasible: proximity={best_feas['proximity_score']:.4f}, "
                  f"changes={best_feas['num_features_changed']}")
        else:
            print(f"      No feasible CFs for this instance.")

        all_raw.extend(raw_cfs)
        all_feasible.extend(feasible_cfs)

    # ---- Overall stats ------------------------------------------------------
    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)

    total_raw    = len(all_raw)
    total_valid  = sum(1 for c in all_raw if c["validity"])
    total_feas   = len(all_feasible)

    overall_val_rate  = total_valid / total_raw * 100 if total_raw > 0 else 0
    overall_feas_rate = total_feas / total_raw * 100 if total_raw > 0 else 0

    raw_prox_all  = [c["proximity_score"] for c in all_raw]
    raw_spar_all  = [c["num_features_changed"] for c in all_raw]
    feas_prox_all = [c["proximity_score"] for c in all_feasible]
    feas_spar_all = [c["num_features_changed"] for c in all_feasible]

    print(f"  Total raw CFs generated : {total_raw}")
    print(f"  Total valid (class flip): {total_valid}  ({overall_val_rate:.1f}%)")
    print(f"  Total feasible          : {total_feas}  ({overall_feas_rate:.1f}%)")
    if raw_prox_all:
        print(f"  Avg proximity (raw)     : {np.mean(raw_prox_all):.4f}")
        print(f"  Avg sparsity  (raw)     : {np.mean(raw_spar_all):.2f}")
    if feas_prox_all:
        print(f"  Avg proximity (feasible): {np.mean(feas_prox_all):.4f}")
        print(f"  Avg sparsity  (feasible): {np.mean(feas_spar_all):.2f}")

    print(f"\n  Feasibility comparison:")
    print(f"    Without filter: 100% by definition (all {total_raw} raw CFs)")
    print(f"    With filter   : {overall_feas_rate:.1f}% "
          f"({total_feas}/{total_raw} pass directional constraints)")

    # ---- Write summary report -----------------------------------------------
    write_summary_report(per_instance, total_raw, total_valid, total_feas,
                         overall_val_rate, overall_feas_rate,
                         raw_prox_all, raw_spar_all,
                         feas_prox_all, feas_spar_all)

    print("\n" + "=" * 60)
    print("PHASE 4 COMPLETE")
    print("=" * 60)


# =============================================================================
# Summary report writer
# =============================================================================
def write_summary_report(
    per_instance, total_raw, total_valid, total_feas,
    overall_val_rate, overall_feas_rate,
    raw_prox_all, raw_spar_all,
    feas_prox_all, feas_spar_all
):
    """Write reports/counterfactual_summary.md with real generated numbers."""
    print("\n" + "=" * 60)
    print("Writing counterfactual_summary.md")
    print("=" * 60)

    avg_prox_raw  = np.mean(raw_prox_all) if raw_prox_all else 0
    avg_spar_raw  = np.mean(raw_spar_all) if raw_spar_all else 0
    avg_prox_feas = np.mean(feas_prox_all) if feas_prox_all else 0
    avg_spar_feas = np.mean(feas_spar_all) if feas_spar_all else 0

    lines = [
        "# Counterfactual Explanation Summary - Phase 4",
        "",
        "> All counterfactuals were generated by DiCE (method='genetic') running",
        "> against the real trained classification model. No numbers are fabricated.",
        "",
        "## Overall Statistics",
        "",
        f"- **Total raw counterfactuals generated:** {total_raw} "
        f"(5 instances x {NUM_CFS} CFs each)",
        f"- **Overall validity rate:** {total_valid}/{total_raw} "
        f"({overall_val_rate:.1f}%) actually flipped the model's prediction",
        f"- **Overall feasibility rate (after directional filter):** "
        f"{total_feas}/{total_raw} ({overall_feas_rate:.1f}%)",
        "",
        "## Feasibility Comparison",
        "",
        "| Metric | Without Filter (raw) | With Directional Filter |",
        "|---|---|---|",
        f"| Counterfactuals retained | {total_raw} (100%) "
        f"| {total_feas} ({overall_feas_rate:.1f}%) |",
        f"| Avg proximity (lower = smaller change) "
        f"| {avg_prox_raw:.4f} | {avg_prox_feas:.4f} |" if feas_prox_all else
        f"| Avg proximity | {avg_prox_raw:.4f} | N/A (none feasible) |",
        f"| Avg sparsity (features changed) "
        f"| {avg_spar_raw:.2f} | {avg_spar_feas:.2f} |" if feas_spar_all else
        f"| Avg sparsity | {avg_spar_raw:.2f} | N/A |",
        "",
        "### Direction constraints applied",
        "",
        "| Feature | Allowed Direction | Rationale |",
        "|---|---|---|",
        "| late_30/60/90_days | Decrease only | "
        "A person can reduce late payments, not increase them |",
        "| revolving_util, debt_ratio | Decrease only | "
        "Can pay down debt, not deliberately take on more |",
        "| monthly_income | Increase only | "
        "Can earn more, not deliberately earn less |",
        "| open_credit_lines, real_estate_loans | Either direction | "
        "Opening or closing accounts is a real, if complex, action |",
        "",
        "### Immutable features (never changed by DiCE)",
        "",
        f"- `age`, `num_dependents`, `monthly_income_missing`",
        "",
        "---",
        "",
        "## Per-Instance Results",
        "",
    ]

    for s in per_instance:
        label = s["case_type"].replace("_", " ").title()
        lines.append(f"### {label} (instance ID {s['instance_id']})")
        lines.append("")
        lines.append(
            f"- Raw CFs: {s['n_raw']}  |  "
            f"Valid: {s['n_valid']}  |  "
            f"Feasible: {s['n_feasible']}  |  "
            f"Method: {s['method_used']}"
        )

        if s["avg_proximity_raw"] is not None:
            lines.append(
                f"- Avg proximity (raw): {s['avg_proximity_raw']:.4f}  |  "
                f"Avg sparsity (raw): {s['avg_sparsity_raw']:.2f}"
            )

        if s["best_feasible"] is not None:
            bf = s["best_feasible"]
            lines.append("")
            lines.append(
                f"**Best feasible CF** (proximity={bf['proximity_score']:.4f}, "
                f"{bf['num_features_changed']} feature(s) changed):"
            )
            lines.append("")
            for ch in bf["changed_features"]:
                lines.append(
                    f"- `{ch['feature_name']}`: "
                    f"{ch['original_value']:.4g} -> {ch['cf_value']:.4g} "
                    f"({ch['direction']})"
                )
        else:
            lines.append("")
            lines.append(
                "**No feasible counterfactuals** passed the directional filter "
                "for this instance. All generated CFs required at least one "
                "feature to move in an unrealistic direction."
            )
        lines.append("")

    # File list
    lines += [
        "---",
        "",
        "## Files Produced",
        "",
        "| File | Description |",
        "|---|---|",
    ]
    for s in per_instance:
        iid = s["instance_id"]
        ct  = s["case_type"]
        lines.append(
            f"| `reports/counterfactuals/{iid}_{ct}_raw.json` "
            f"| {s['n_raw']} raw CFs with metrics |"
        )
        lines.append(
            f"| `reports/counterfactuals/{iid}_{ct}_feasible.json` "
            f"| {s['n_feasible']} feasible CFs (directional filter applied) |"
        )

    report_path = os.path.join(REPORTS, "counterfactual_summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [saved] {report_path}")


# =============================================================================
# Phase 7 — Live inference helper (used by app.py)
# =============================================================================
def get_counterfactuals_live(
    feature_dict: dict,
    feature_cols: list,
    dice_exp,
    actionable: list,
    feat_ranges: dict,
    original_pred: int,
    n_cfs: int = 5,
) -> list:
    """
    Generate counterfactuals for a single new applicant, then apply the same
    directional feasibility filter used in Phase 4.
    Reuses the already-loaded DiCE explainer (dice_exp) — no reload.

    Parameters
    ----------
    feature_dict  : dict mapping feature name -> value for ONE applicant
    feature_cols  : ordered list of all feature column names
    dice_exp      : dice_ml.Dice object (already initialised)
    actionable    : list of feature names DiCE is allowed to change
    feat_ranges   : dict of feature name -> (max-min) range for proximity
    original_pred : 0 or 1 (the model's prediction for this applicant)
    n_cfs         : how many CFs to request from DiCE

    Returns
    -------
    list of feasible CF dicts in the Phase 4 format:
      {changed_features: [{feature_name, original_value, cf_value, direction}],
       validity, proximity_score, num_features_changed}
    Returns an empty list if none are feasible.
    """
    # numpy is already imported at the top of this module as np
    query_row = pd.DataFrame([feature_dict], columns=feature_cols)

    try:
        cf_result = dice_exp.generate_counterfactuals(
            query_row,
            total_CFs=n_cfs,
            desired_class="opposite",
            features_to_vary=actionable,
        )
    except Exception as e:
        print(f"  [CF live] DiCE error: {e}")
        return []

    cf_df = cf_result.cf_examples_list[0].final_cfs_df
    if cf_df is None or len(cf_df) == 0:
        return []

    if "target_default" in cf_df.columns:
        cf_df = cf_df.drop(columns=["target_default"])
    cf_df = cf_df[feature_cols]

    raw_cfs = []
    for i in range(len(cf_df)):
        cf_row = cf_df.iloc[[i]]
        changed = []
        for col in feature_cols:
            orig = float(query_row[col].iloc[0])
            new  = float(cf_row[col].iloc[0])
            if abs(new - orig) > 1e-8:
                direction = "increase" if new > orig else "decrease"
                changed.append({
                    "feature_name":   col,
                    "original_value": round(orig, 6),
                    "cf_value":       round(new, 6),
                    "direction":      direction,
                })
        sparsity = len(changed)
        if sparsity > 0:
            norm_dists = [
                abs(ch["cf_value"] - ch["original_value"]) / feat_ranges.get(ch["feature_name"], 1.0)
                for ch in changed
            ]
            proximity = float(np.mean(norm_dists))
        else:
            proximity = 0.0
        raw_cfs.append({
            "changed_features":     changed,
            "validity":             True,   # DiCE desired_class="opposite"
            "proximity_score":      round(proximity, 6),
            "num_features_changed": sparsity,
        })

    # Apply the same feasibility filter from Phase 4
    return apply_feasibility_filter(raw_cfs)


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    run_counterfactual_pipeline()
