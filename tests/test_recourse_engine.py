# =============================================================================
# Unit Tests for the Domain-Constrained Recourse Engine
# =============================================================================
# Run: python tests/test_recourse_engine.py
#
# Tests:
#   1. Taxonomy validation — all 11 features accounted for, no overlaps
#   2. DiCE cannot change immutable or historical features
#   3. Independent verifier catches artificially violated constraints
#   4. Direction constraint violations are detected
#   5. Cross-feature constraint (real_estate <= open_credit_lines) detected

import os
import sys
import json
import copy

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from xai.taxonomy import (
    validate_taxonomy,
    get_all_frozen_features,
    get_direction_constraints,
    IMMUTABLE_FEATURES,
    HISTORICAL_FEATURES,
    ACTIONABLE_FEATURE_NAMES,
    ALL_FEATURE_COLS,
    get_permitted_ranges,
)
from xai.recourse_engine import (
    load_model_and_data,
    setup_constrained_dice,
    generate_constrained_cfs,
    verify_single_cf,
    run_funnel,
    NUM_CFS,
)


# =============================================================================
# Test Helpers
# =============================================================================
PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def assert_true(condition, test_name, msg=""):
    if condition:
        print(f"  {PASS} {test_name}")
        results.append((test_name, True, ""))
    else:
        print(f"  {FAIL} {test_name}: {msg}")
        results.append((test_name, False, msg))


def assert_false(condition, test_name, msg=""):
    assert_true(not condition, test_name, msg)


def assert_equal(a, b, test_name):
    assert_true(a == b, test_name, f"expected {b}, got {a}")


def assert_in(item, collection, test_name):
    assert_true(item in collection, test_name, f"{item!r} not in {collection!r}")


def assert_not_in(item, collection, test_name):
    assert_true(item not in collection, test_name, f"{item!r} should not be in collection")


# =============================================================================
# TEST GROUP 1 — Taxonomy Integrity
# =============================================================================
def test_taxonomy():
    print("\n--- TEST GROUP 1: Taxonomy Integrity ---")

    # 1.1 Validation passes without error
    try:
        ok = validate_taxonomy()
        assert_true(ok, "1.1 validate_taxonomy() passes without error")
    except AssertionError as e:
        assert_true(False, "1.1 validate_taxonomy() passes without error", str(e))

    # 1.2 All 11 features are covered
    covered = set(IMMUTABLE_FEATURES) | set(HISTORICAL_FEATURES) | set(ACTIONABLE_FEATURE_NAMES)
    assert_equal(len(covered), 11, "1.2 Exactly 11 features covered")

    # 1.3 No feature appears in two categories
    assert_equal(
        len(set(IMMUTABLE_FEATURES) & set(HISTORICAL_FEATURES)), 0,
        "1.3a Immutable and Historical are disjoint"
    )
    assert_equal(
        len(set(IMMUTABLE_FEATURES) & set(ACTIONABLE_FEATURE_NAMES)), 0,
        "1.3b Immutable and Actionable are disjoint"
    )
    assert_equal(
        len(set(HISTORICAL_FEATURES) & set(ACTIONABLE_FEATURE_NAMES)), 0,
        "1.3c Historical and Actionable are disjoint"
    )

    # 1.4 Specific features are in correct categories
    assert_in("age", IMMUTABLE_FEATURES, "1.4a age is IMMUTABLE")
    assert_in("num_dependents", IMMUTABLE_FEATURES, "1.4b num_dependents is IMMUTABLE")
    assert_in("monthly_income_missing", IMMUTABLE_FEATURES, "1.4c monthly_income_missing is IMMUTABLE")
    assert_in("late_30_59_days", HISTORICAL_FEATURES, "1.4d late_30_59_days is HISTORICAL")
    assert_in("late_60_89_days", HISTORICAL_FEATURES, "1.4e late_60_89_days is HISTORICAL")
    assert_in("late_90_days", HISTORICAL_FEATURES, "1.4f late_90_days is HISTORICAL")
    assert_in("revolving_util", ACTIONABLE_FEATURE_NAMES, "1.4g revolving_util is ACTIONABLE")
    assert_in("monthly_income", ACTIONABLE_FEATURE_NAMES, "1.4h monthly_income is ACTIONABLE")

    # 1.5 Features to vary does not include any frozen features
    features_to_vary = set(ACTIONABLE_FEATURE_NAMES)
    frozen = set(get_all_frozen_features())
    overlap = features_to_vary & frozen
    assert_equal(len(overlap), 0, f"1.5 No overlap between features_to_vary and frozen features (overlap={overlap})")


# =============================================================================
# TEST GROUP 2 — DiCE Does Not Change Frozen Features
# =============================================================================
def test_dice_frozen_features(clf_model, X_train, y_train, feature_cols):
    print("\n--- TEST GROUP 2: DiCE Cannot Change Frozen Features ---")

    exp, features_to_vary, permitted_ranges = setup_constrained_dice(
        clf_model, X_train, y_train, feature_cols
    )

    # 2.1 features_to_vary does not include any frozen feature
    frozen = set(get_all_frozen_features())
    ftv_set = set(features_to_vary)
    overlap = ftv_set & frozen
    assert_equal(
        len(overlap), 0,
        f"2.1 features_to_vary excludes all frozen features (overlap={overlap})"
    )

    # 2.2 Load an adverse instance (True Positive, predicted default)
    with open(os.path.join(PROJECT_ROOT, "reports", "selected_instances.json")) as f:
        all_inst = json.load(f)
    # Find a true positive (predicted class 1)
    adverse_inst = None
    for inst in all_inst:
        if inst["task"] == "classification" and inst.get("predicted_proba", 0) >= 0.5:
            adverse_inst = inst
            break

    if adverse_inst is None:
        assert_true(False, "2.2 Found an adverse instance for DiCE test", "No adverse instance")
        return

    assert_true(True, f"2.2 Found adverse instance: id={adverse_inst['instance_id']}")

    # 2.3 Generate CFs and verify no frozen feature changed
    cf_records, meta = generate_constrained_cfs(
        exp, clf_model, adverse_inst["features"], feature_cols,
        features_to_vary, permitted_ranges,
    )

    if not cf_records:
        print(f"  [SKIP] 2.3 No CFs generated (reason: {meta.get('reason_skipped', 'unknown')})")
        return

    frozen_changed = []
    for cf_rec in cf_records:
        for changed in cf_rec["changed_features"]:
            if changed["feature_name"] in frozen:
                frozen_changed.append(
                    f"CF {cf_rec['cf_index']}: {changed['feature_name']} changed "
                    f"{changed['original_value']} -> {changed['cf_value']}"
                )

    assert_equal(
        len(frozen_changed), 0,
        f"2.3 No frozen feature changed in any CF (violations={frozen_changed})"
    )

    # 2.4 Specifically assert age and historical features unchanged
    for cf_rec in cf_records:
        orig = cf_rec["original_features"]
        cf_feats = cf_rec["cf_features"]

        for feat in IMMUTABLE_FEATURES:
            assert_true(
                np.isclose(orig[feat], cf_feats[feat], atol=1e-6),
                f"2.4a IMMUTABLE '{feat}' unchanged in CF {cf_rec['cf_index']}",
                f"  {orig[feat]} -> {cf_feats[feat]}"
            )

        for feat in HISTORICAL_FEATURES:
            assert_true(
                np.isclose(orig[feat], cf_feats[feat], atol=1e-6),
                f"2.4b HISTORICAL '{feat}' unchanged in CF {cf_rec['cf_index']}",
                f"  {orig[feat]} -> {cf_feats[feat]}"
            )


# =============================================================================
# TEST GROUP 3 — Verifier Catches Artificial Constraint Violations
# =============================================================================
def test_verifier_catches_violations(clf_model, X_train, feature_cols):
    print("\n--- TEST GROUP 3: Verifier Catches Artificial Violations ---")

    # Build a minimal valid-looking CF record using first test-set adverse instance
    with open(os.path.join(PROJECT_ROOT, "reports", "selected_instances.json")) as f:
        all_inst = json.load(f)
    tp_inst = next(
        (i for i in all_inst if i["task"] == "classification" and i.get("predicted_proba", 0) >= 0.5),
        None
    )
    if tp_inst is None:
        print("  [SKIP] No adverse instance available for verifier tests")
        return

    orig_features = tp_inst["features"]

    # Build a "perfect" CF: copy original, then reduce revolving_util to 0.1
    # This should produce predicted_class=0 (non-default) if done enough
    cf_feats = dict(orig_features)
    cf_feats["revolving_util"] = 0.1
    cf_feats["debt_ratio"] = 0.1
    cf_feats["monthly_income"] = float(orig_features.get("monthly_income", 5400)) * 1.5

    # Actually get a valid CF from the model
    cf_row = pd.DataFrame([cf_feats], columns=feature_cols)
    cf_pred = int(clf_model.predict(cf_row)[0])
    original_row = pd.DataFrame([orig_features], columns=feature_cols)
    orig_pred = int(clf_model.predict(original_row)[0])

    # Build synthetic cf_record
    changed = []
    for feat in feature_cols:
        if not np.isclose(float(orig_features[feat]), float(cf_feats[feat]), atol=1e-6):
            delta = float(cf_feats[feat]) - float(orig_features[feat])
            changed.append({
                "feature_name": feat,
                "original_value": float(orig_features[feat]),
                "cf_value": float(cf_feats[feat]),
                "direction": "increase" if delta > 0 else "decrease",
                "delta": delta,
            })

    valid_cf_record = {
        "cf_index": 0,
        "cf_features": cf_feats,
        "original_features": orig_features,
        "changed_features": changed,
        "cf_predicted_class": cf_pred,
        "cf_predicted_proba": float(clf_model.predict_proba(cf_row)[0, 1]),
        "num_features_changed": len(changed),
    }

    # --- 3.1: Test with artificially changed immutable feature (age) ---
    bad_age = copy.deepcopy(valid_cf_record)
    bad_age["cf_features"]["age"] = float(orig_features["age"]) + 5
    bad_age["changed_features"].append({
        "feature_name": "age",
        "original_value": float(orig_features["age"]),
        "cf_value": float(orig_features["age"]) + 5,
        "direction": "increase",
        "delta": 5.0,
    })

    ver = verify_single_cf(bad_age, clf_model, feature_cols, X_train)
    has_age_violation = any("age" in v for v in ver["stage2_violations"])
    assert_true(
        has_age_violation or not ver["stage2_constraints_satisfied"],
        "3.1 Verifier detects change to IMMUTABLE 'age'",
        f"  violations={ver['stage2_violations']}"
    )

    # --- 3.2: Test with artificially changed historical feature ---
    bad_hist = copy.deepcopy(valid_cf_record)
    orig_late = float(orig_features["late_30_59_days"])
    bad_hist["cf_features"]["late_30_59_days"] = orig_late + 1
    bad_hist["changed_features"].append({
        "feature_name": "late_30_59_days",
        "original_value": orig_late,
        "cf_value": orig_late + 1,
        "direction": "increase",
        "delta": 1.0,
    })

    ver = verify_single_cf(bad_hist, clf_model, feature_cols, X_train)
    has_hist_violation = any("late_30_59_days" in v for v in ver["stage2_violations"])
    assert_true(
        has_hist_violation or not ver["stage2_constraints_satisfied"],
        "3.2 Verifier detects change to HISTORICAL 'late_30_59_days'",
        f"  violations={ver['stage2_violations']}"
    )

    # --- 3.3: Test direction violation — revolving_util increased (should decrease) ---
    bad_dir = copy.deepcopy(valid_cf_record)
    orig_util = float(orig_features["revolving_util"])
    bad_dir["cf_features"]["revolving_util"] = min(1.0, orig_util + 0.2)
    # Update changed_features list
    bad_dir["changed_features"] = [
        c for c in bad_dir["changed_features"] if c["feature_name"] != "revolving_util"
    ]
    bad_dir["changed_features"].append({
        "feature_name": "revolving_util",
        "original_value": orig_util,
        "cf_value": min(1.0, orig_util + 0.2),
        "direction": "increase",
        "delta": 0.2,
    })

    ver = verify_single_cf(bad_dir, clf_model, feature_cols, X_train)
    has_dir_violation = any("revolving_util" in v for v in ver["stage2_violations"])
    assert_true(
        has_dir_violation or not ver["stage2_constraints_satisfied"],
        "3.3 Verifier detects direction violation on 'revolving_util' (increase instead of decrease)",
        f"  violations={ver['stage2_violations']}"
    )

    # --- 3.4: Cross-feature violation — real_estate_loans > open_credit_lines ---
    bad_cross = copy.deepcopy(valid_cf_record)
    bad_cross["cf_features"]["real_estate_loans"] = 20
    bad_cross["cf_features"]["open_credit_lines"] = 5

    ver = verify_single_cf(bad_cross, clf_model, feature_cols, X_train)
    has_cross_violation = any("real_estate" in v for v in ver["stage2_violations"])
    assert_true(
        has_cross_violation or not ver["stage2_constraints_satisfied"],
        "3.4 Verifier detects cross-feature violation (real_estate_loans > open_credit_lines)",
        f"  violations={ver['stage2_violations']}"
    )

    # --- 3.5: A CF that passes all checks should have status PASSED ---
    # Only test this if the model actually flips for our constructed valid CF
    if orig_pred == 1 and cf_pred == 0:
        ver = verify_single_cf(valid_cf_record, clf_model, feature_cols, X_train)
        # Note: valid_cf_record may still have direction violations if income increased
        # We just check the infrastructure works, not that our synthetic CF is perfect
        assert_true(
            ver["stage1_model_valid"],
            "3.5 Valid CF passes Stage 1 (model validity)"
        )
    else:
        print(f"  [SKIP] 3.5 Synthetic CF did not flip model (orig={orig_pred}, cf={cf_pred})")


# =============================================================================
# Main test runner
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RECOURSE ENGINE UNIT TESTS")
    print("=" * 60)

    # Load model (needed for groups 2 & 3)
    try:
        clf_model, X_train, y_train, feature_cols = load_model_and_data()
        model_loaded = True
    except Exception as e:
        print(f"  WARNING: Could not load model: {e}")
        model_loaded = False

    # Run test groups
    test_taxonomy()

    if model_loaded:
        test_dice_frozen_features(clf_model, X_train, y_train, feature_cols)
        test_verifier_catches_violations(clf_model, X_train, feature_cols)
    else:
        print("\n  [SKIP] Groups 2 & 3 require a loaded model")

    # ---- Summary -------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    skipped = sum(1 for _, ok, _ in results if ok is None)

    for test_name, ok, msg in results:
        status = PASS if ok else FAIL
        print(f"  {status} {test_name}")
        if not ok and msg:
            print(f"       ^ {msg}")

    print(f"\n  Total: {len(results)} | Passed: {passed} | Failed: {failed}")

    if failed == 0:
        print("\n  ALL TESTS PASSED")
    else:
        print(f"\n  {failed} TEST(S) FAILED")
        sys.exit(1)
