# =============================================================================
# Feature Actionability Taxonomy
# =============================================================================
# Defines explicit domain-knowledge rules for which features can appear in
# counterfactual recourse recommendations and in which direction.
#
# This taxonomy is the core of the "domain-constrained recourse" contribution:
# rather than letting the DiCE optimizer change any feature in any direction,
# we encode real-world financial knowledge about what a borrower CAN and
# CANNOT change.
#
# Three categories:
#   1. IMMUTABLE   - cannot be changed by the applicant (age, dependents, flags)
#   2. HISTORICAL  - record of past events; cannot be retroactively altered
#   3. ACTIONABLE  - the applicant CAN realistically modify these, subject
#                    to directional and range constraints
#
# Usage:
#   from src.xai.taxonomy import (
#       IMMUTABLE_FEATURES, HISTORICAL_FEATURES, ACTIONABLE_FEATURES,
#       get_features_to_vary, get_permitted_ranges, get_direction_constraints,
#       ALL_FEATURE_COLS,
#   )

import os
import json
import pandas as pd
import numpy as np

# ---- The canonical 11 classification features --------------------------------
# (matches reports/feature_columns_clf.json exactly)
ALL_FEATURE_COLS = [
    "revolving_util",
    "age",
    "late_30_59_days",
    "debt_ratio",
    "monthly_income",
    "open_credit_lines",
    "late_90_days",
    "real_estate_loans",
    "late_60_89_days",
    "num_dependents",
    "monthly_income_missing",
]


# =============================================================================
# Category 1: IMMUTABLE features — recourse MUST NOT modify these
# =============================================================================
# age:                    A person cannot change their age.
# num_dependents:         Not an actionable financial lever; also involves
#                         other people's lives and cannot be "optimised."
# monthly_income_missing: An engineered binary flag, not a real-world feature.
#                         Changing it would be meaningless.
IMMUTABLE_FEATURES = [
    "age",
    "num_dependents",
    "monthly_income_missing",
]


# =============================================================================
# Category 2: HISTORICAL features — past records, not retroactively changeable
# =============================================================================
# These features record how many times a borrower was late in specific
# delinquency windows. A credit bureau report reflects historical fact:
# a borrower cannot go back in time and un-miss a payment.
#
# NOTE: In the original Phase 4 design, late-payment features were allowed
# to DECREASE (as if the borrower could reduce them). This is debatable:
# technically a bureau record can improve over time as old items age off,
# but the record at the time of application is fixed. For the research
# paper's stricter taxonomy, we classify them as historical/immutable
# for recourse purposes, which is the more defensible position.
HISTORICAL_FEATURES = [
    "late_30_59_days",
    "late_60_89_days",
    "late_90_days",
]


# =============================================================================
# Category 3: ACTIONABLE features — permitted for recourse, with constraints
# =============================================================================
# Each entry: (feature_name, allowed_direction, min_bound, max_bound, is_integer)
#
# allowed_direction:
#   "decrease" = CF value must be <= original (e.g., pay down utilisation)
#   "increase" = CF value must be >= original (e.g., earn more income)
#   "any"      = CF value can move in either direction
#
# Bounds are set dynamically from training data (1st-99th percentile) in
# get_permitted_ranges(). The static constraints below are domain floors/ceilings.
ACTIONABLE_FEATURES = {
    "revolving_util": {
        "direction": "decrease",     # can pay down credit card balances
        "domain_min": 0.0,           # utilisation cannot be negative
        "domain_max": 1.0,           # capped at 100% for recourse purposes
        "is_integer": False,
        "rationale": "Borrower can pay down revolving credit balances.",
    },
    "debt_ratio": {
        "direction": "decrease",     # can reduce debt relative to income
        "domain_min": 0.0,           # ratio cannot be negative
        "domain_max": None,          # derived from training data
        "is_integer": False,
        "rationale": "Borrower can pay off debts or increase income to lower ratio.",
    },
    "monthly_income": {
        "direction": "increase",     # can earn more, not deliberately less
        "domain_min": None,          # derived from training data
        "domain_max": None,          # derived from training data
        "is_integer": False,
        "rationale": "Borrower can seek higher-paying employment or additional income.",
    },
    "open_credit_lines": {
        "direction": "any",          # can open or close credit accounts
        "domain_min": 0,             # cannot have negative accounts
        "domain_max": None,          # derived from training data
        "is_integer": True,
        "rationale": "Borrower can open or close credit lines/loans.",
    },
    "real_estate_loans": {
        "direction": "any",          # can buy or sell property
        "domain_min": 0,             # cannot have negative mortgages
        "domain_max": None,          # derived from training data; also
                                     # must not exceed open_credit_lines
        "is_integer": True,
        "rationale": "Borrower can buy or sell real estate.",
    },
}

# Convenience: just the names of actionable features
ACTIONABLE_FEATURE_NAMES = list(ACTIONABLE_FEATURES.keys())


# =============================================================================
# Derived helpers
# =============================================================================

def get_features_to_vary():
    """
    Return the list of feature names DiCE is permitted to modify.
    This is the ACTIONABLE set only — immutable and historical are excluded.
    """
    return list(ACTIONABLE_FEATURES.keys())


def get_permitted_ranges(X_train, percentile_lo=1, percentile_hi=99):
    """
    Compute permitted value ranges for each actionable feature based on
    the training data distribution (1st-99th percentile), intersected
    with domain constraints.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data (features only).
    percentile_lo : int
        Lower percentile for range floor (default 1st).
    percentile_hi : int
        Upper percentile for range ceiling (default 99th).

    Returns
    -------
    dict
        {feature_name: [lower_bound, upper_bound]} suitable for DiCE's
        permitted_range parameter.
    """
    ranges = {}
    for feat, rules in ACTIONABLE_FEATURES.items():
        if feat not in X_train.columns:
            continue

        # Data-derived bounds
        data_lo = float(np.percentile(X_train[feat].dropna(), percentile_lo))
        data_hi = float(np.percentile(X_train[feat].dropna(), percentile_hi))

        # Intersect with domain constraints
        lo = rules["domain_min"] if rules["domain_min"] is not None else data_lo
        hi = rules["domain_max"] if rules["domain_max"] is not None else data_hi
        lo = max(lo, data_lo) if rules["domain_min"] is not None else data_lo
        hi = min(hi, data_hi) if rules["domain_max"] is not None else data_hi

        # Ensure lo <= hi
        if lo > hi:
            lo, hi = hi, lo

        # For integer features, round bounds
        if rules["is_integer"]:
            lo = int(np.floor(lo))
            hi = int(np.ceil(hi))

        ranges[feat] = [lo, hi]

    return ranges


def get_direction_constraints():
    """
    Return a dict mapping each actionable feature to its allowed direction.
    Used by the post-generation verification funnel.
    """
    return {
        feat: rules["direction"]
        for feat, rules in ACTIONABLE_FEATURES.items()
    }


def get_all_frozen_features():
    """
    Return the combined list of features that must NOT change in a
    counterfactual: immutable + historical.
    """
    return IMMUTABLE_FEATURES + HISTORICAL_FEATURES


def validate_taxonomy():
    """
    Sanity check: ensure all 11 features are accounted for in exactly
    one category (immutable, historical, or actionable).
    """
    immutable_set = set(IMMUTABLE_FEATURES)
    historical_set = set(HISTORICAL_FEATURES)
    actionable_set = set(ACTIONABLE_FEATURE_NAMES)
    all_set = set(ALL_FEATURE_COLS)

    # No overlaps
    assert immutable_set.isdisjoint(historical_set), \
        f"Overlap: immutable & historical: {immutable_set & historical_set}"
    assert immutable_set.isdisjoint(actionable_set), \
        f"Overlap: immutable & actionable: {immutable_set & actionable_set}"
    assert historical_set.isdisjoint(actionable_set), \
        f"Overlap: historical & actionable: {historical_set & actionable_set}"

    # Full coverage
    covered = immutable_set | historical_set | actionable_set
    missing = all_set - covered
    extra = covered - all_set
    assert not missing, f"Features not assigned to any category: {missing}"
    assert not extra, f"Features in taxonomy but not in feature list: {extra}"

    return True


# Run validation on import to catch errors early
validate_taxonomy()


if __name__ == "__main__":
    print("Feature Actionability Taxonomy")
    print("=" * 50)
    print(f"\nIMMUTABLE ({len(IMMUTABLE_FEATURES)}):")
    for f in IMMUTABLE_FEATURES:
        print(f"  - {f}")
    print(f"\nHISTORICAL ({len(HISTORICAL_FEATURES)}):")
    for f in HISTORICAL_FEATURES:
        print(f"  - {f}")
    print(f"\nACTIONABLE ({len(ACTIONABLE_FEATURE_NAMES)}):")
    for name, rules in ACTIONABLE_FEATURES.items():
        print(f"  - {name}: direction={rules['direction']}, "
              f"integer={rules['is_integer']}")
        print(f"    Rationale: {rules['rationale']}")
    print(f"\nTotal features accounted for: "
          f"{len(IMMUTABLE_FEATURES) + len(HISTORICAL_FEATURES) + len(ACTIONABLE_FEATURE_NAMES)}"
          f" / {len(ALL_FEATURE_COLS)}")
    print("\nTaxonomy validation: PASSED")
