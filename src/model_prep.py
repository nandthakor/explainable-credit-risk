# Credit Risk XAI — Phase 2: Model Preparation
# ==============================================
# This script prepares two separate feature matrices:
#   1. Classification task  — predict target_default (0/1)
#   2. Regression task      — estimate monthly_income (continuous)
#
# It handles:
#   - Outlier capping (winsorisation at the 99th percentile)
#   - Feature engineering (monthly_income_missing flag)
#   - Median imputation for monthly_income with NO data leakage
#     (imputer is fit only on the training fold, then applied to test)
#   - Stratified 80/20 train/test split
#   - Saving splits to data/processed/ and feature lists to reports/
#
# Run: python src/model_prep.py

import os
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# ---- Paths ------------------------------------------------------------------
CLEAN_CSV    = os.path.join("data", "processed", "credit_data_clean.csv")
PROCESSED    = os.path.join("data", "processed")
REPORTS      = os.path.join("reports")

os.makedirs(PROCESSED, exist_ok=True)
os.makedirs(REPORTS,   exist_ok=True)
os.makedirs("models",  exist_ok=True)

# ---- Constants --------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE    = 0.20

# Columns whose extreme outliers we cap at the 99th percentile.
# Rationale: debt_ratio reaches 329,664 and revolving_util reaches 50,708 in
# this dataset — these are almost certainly data errors or edge cases (e.g.
# retirees with ~$0 income inflating a ratio). Dropping these rows would
# lose information about the DEFAULT LABEL still present in those rows; capping
# (winsorisation) is the standard compromise. We cap BEFORE splitting so the
# cap threshold is computed on the full cleaned dataset — this is intentional
# because we are using a fixed historical dataset (not a live streaming scenario).
WINSOR_COLS = [
    "debt_ratio",
    "revolving_util",
    "late_30_59_days",
    "late_90_days",
    "late_60_89_days",
]


# =============================================================================
# Step 1 — Load
# =============================================================================
def load_clean_data(path: str = CLEAN_CSV) -> pd.DataFrame:
    """Load the Phase 1 cleaned dataset and print a quick sanity check."""
    print("\n" + "=" * 60)
    print("STEP 1: Loading cleaned dataset")
    print("=" * 60)

    df = pd.read_csv(path)

    print(f"\n  Shape : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Null counts (only monthly_income should be non-zero):")
    print(df.isnull().sum().to_string())

    return df


# =============================================================================
# Step 2 — Winsorise (cap) extreme outliers
# =============================================================================
def winsorise(df: pd.DataFrame, cols: list = WINSOR_COLS) -> pd.DataFrame:
    """
    Cap values above the 99th percentile for specified columns.

    WHY WE WINSORISE RATHER THAN DROP:
      - Dropping rows with extreme values would silently remove both the
        feature value AND the default label for those borrowers.
      - Winsorising replaces only the extreme feature values; all rows and
        labels are retained.
      - The 99th percentile is a standard, conservative cut-off that removes
        the most egregious outliers (e.g. debt_ratio = 329,664) while keeping
        the 99% of the distribution intact.

    NOTE: Caps are computed on the full cleaned dataset BEFORE splitting.
    This is acceptable here because we are working with a fixed historical
    benchmark dataset, not a production pipeline where the test set must
    remain completely unseen.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Winsorising extreme outliers (capping at 99th percentile)")
    print("=" * 60)

    cap_values = {}

    for col in cols:
        cap = df[col].quantile(0.99)
        n_above = (df[col] > cap).sum()
        pct_above = n_above / len(df) * 100

        print(f"\n  {col}:")
        print(f"    99th pct cap : {cap:.4f}")
        print(f"    Max before   : {df[col].max():.1f}")
        print(f"    Rows above   : {n_above:,}  ({pct_above:.3f}% of dataset)")

        df[col] = df[col].clip(upper=cap)
        cap_values[col] = cap

        print(f"    Max after    : {df[col].max():.4f}  [capped]")

    return df, cap_values


# =============================================================================
# Step 3 — Build feature matrices for both tasks
# =============================================================================
def build_feature_sets(df: pd.DataFrame) -> dict:
    """
    Construct two separate feature/target pairs:

    CLASSIFICATION:
      Features: all columns except target_default and monthly_income,
                PLUS monthly_income_missing (engineered binary flag),
                PLUS monthly_income (nulls filled with train-fold median later).
      Target:   target_default (0/1)

    REGRESSION:
      Features: all columns except target_default and monthly_income,
                used only on rows where monthly_income is NOT null.
      Target:   monthly_income (continuous)

    Note: monthly_income imputation for classification is done AFTER splitting
    to avoid any leakage from test fold statistics into training.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Building feature sets")
    print("=" * 60)

    # ---- 3a: Classification feature set ----
    # Engineer the binary missing-income flag BEFORE imputing income
    df["monthly_income_missing"] = df["monthly_income"].isnull().astype(int)
    print(f"\n  Engineered 'monthly_income_missing':")
    print(f"    Rows with flag=1 : {df['monthly_income_missing'].sum():,}")
    print(f"    Rows with flag=0 : {(df['monthly_income_missing'] == 0).sum():,}")

    # Classification: all features except the target
    clf_feature_cols = [
        c for c in df.columns
        if c != "target_default"
    ]
    # monthly_income is included but still has nulls at this stage — imputed later

    X_clf = df[clf_feature_cols].copy()
    y_clf = df["target_default"].copy()

    print(f"\n  Classification:")
    print(f"    Feature columns ({len(clf_feature_cols)}): {clf_feature_cols}")
    print(f"    X shape: {X_clf.shape}   y shape: {y_clf.shape}")
    print(f"    Class balance — 0: {(y_clf==0).sum():,}  1: {(y_clf==1).sum():,}")

    # ---- 3b: Regression feature set ----
    # Keep only rows where monthly_income is known (the ~80% with verified income)
    df_reg = df[df["monthly_income"].notna()].copy()

    reg_feature_cols = [
        c for c in df_reg.columns
        if c not in ("target_default", "monthly_income")
    ]
    # NOTE: monthly_income_missing will always be 0 in this subset,
    # so it is included but carries no information for regression —
    # it is kept to maintain a consistent feature list for later phases.

    X_reg = df_reg[reg_feature_cols].copy()
    y_reg = df_reg["monthly_income"].copy()

    print(f"\n  Regression (income-known subset only):")
    print(f"    Rows in subset   : {len(df_reg):,}")
    print(f"    Feature columns ({len(reg_feature_cols)}): {reg_feature_cols}")
    print(f"    X shape: {X_reg.shape}   y shape: {y_reg.shape}")

    return {
        "X_clf": X_clf,
        "y_clf": y_clf,
        "X_reg": X_reg,
        "y_reg": y_reg,
        "clf_feature_cols": clf_feature_cols,
        "reg_feature_cols": reg_feature_cols,
    }


# =============================================================================
# Step 4 — Train / test split
# =============================================================================
def split_data(feature_sets: dict) -> dict:
    """
    Perform an 80/20 stratified split for classification and a plain 80/20
    split for regression. random_state=42 for reproducibility.

    Classification uses stratify=y to preserve the ~6.6% minority class
    proportion in both folds — without stratification the test set could
    end up with a very different imbalance ratio by chance.
    """
    print("\n" + "=" * 60)
    print("STEP 4: Train / test split  (80% train, 20% test, random_state=42)")
    print("=" * 60)

    X_clf = feature_sets["X_clf"]
    y_clf = feature_sets["y_clf"]
    X_reg = feature_sets["X_reg"]
    y_reg = feature_sets["y_reg"]

    # ---- Classification split (stratified) ----------------------------------
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_clf,          # preserve class imbalance ratio
    )

    print(f"\n  Classification split:")
    print(f"    Train rows : {len(X_train_clf):,}")
    print(f"    Test rows  : {len(X_test_clf):,}")
    train_pos_rate = y_train_clf.mean() * 100
    test_pos_rate  = y_test_clf.mean()  * 100
    print(f"    Train class 1 rate : {train_pos_rate:.2f}%  (target_default=1)")
    print(f"    Test  class 1 rate : {test_pos_rate:.2f}%  (target_default=1)")
    print(f"    -> Stratification preserved the ~6.6% minority class in both folds.")

    # ---- Median imputation for monthly_income — NO LEAKAGE ------------------
    # We fit the median imputer ONLY on training rows, then transform both
    # train and test.  Using the full-dataset median would leak test statistics.
    print(f"\n  Imputing monthly_income nulls with TRAIN-FOLD median (no leakage):")
    income_imputer = SimpleImputer(strategy="median")
    income_col_idx = list(X_train_clf.columns).index("monthly_income")

    # Fit on train, transform both
    X_train_clf = X_train_clf.copy()
    X_test_clf  = X_test_clf.copy()
    train_median = X_train_clf["monthly_income"].median()
    X_train_clf["monthly_income"] = X_train_clf["monthly_income"].fillna(train_median)
    X_test_clf["monthly_income"]  = X_test_clf["monthly_income"].fillna(train_median)
    print(f"    Train-fold median used : {train_median:,.1f}")
    print(f"    Nulls remaining in X_train_clf: {X_train_clf['monthly_income'].isnull().sum()}")
    print(f"    Nulls remaining in X_test_clf : {X_test_clf['monthly_income'].isnull().sum()}")

    # ---- Regression split (plain, not stratified) ---------------------------
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"\n  Regression split:")
    print(f"    Train rows : {len(X_train_reg):,}")
    print(f"    Test rows  : {len(X_test_reg):,}")
    print(f"    Train income median : {y_train_reg.median():,.1f}")
    print(f"    Test  income median : {y_test_reg.median():,.1f}")

    return {
        "X_train_clf": X_train_clf,
        "X_test_clf":  X_test_clf,
        "y_train_clf": y_train_clf,
        "y_test_clf":  y_test_clf,
        "X_train_reg": X_train_reg,
        "X_test_reg":  X_test_reg,
        "y_train_reg": y_train_reg,
        "y_test_reg":  y_test_reg,
        "train_median_income": train_median,
    }


# =============================================================================
# Step 5 — Save splits and feature lists
# =============================================================================
def save_splits(splits: dict, feature_sets: dict) -> None:
    """
    Persist all eight train/test CSVs and both feature-column JSON files.
    The JSON files are critical for later phases (SHAP, DiCE) to align
    feature names and ordering correctly without ambiguity.
    """
    print("\n" + "=" * 60)
    print("STEP 5: Saving splits to data/processed/ and feature lists to reports/")
    print("=" * 60)

    # ---- CSV splits ---------------------------------------------------------
    csv_files = {
        "X_train_clf": splits["X_train_clf"],
        "X_test_clf":  splits["X_test_clf"],
        "y_train_clf": splits["y_train_clf"].to_frame(),
        "y_test_clf":  splits["y_test_clf"].to_frame(),
        "X_train_reg": splits["X_train_reg"],
        "X_test_reg":  splits["X_test_reg"],
        "y_train_reg": splits["y_train_reg"].to_frame(),
        "y_test_reg":  splits["y_test_reg"].to_frame(),
    }

    for name, data in csv_files.items():
        path = os.path.join(PROCESSED, f"{name}.csv")
        data.to_csv(path, index=False)
        print(f"  [OK] {path}  ({len(data):,} rows x {data.shape[1]} cols)")

    # ---- Feature column JSON files ------------------------------------------
    clf_json_path = os.path.join(REPORTS, "feature_columns_clf.json")
    reg_json_path = os.path.join(REPORTS, "feature_columns_reg.json")

    with open(clf_json_path, "w") as f:
        json.dump(feature_sets["clf_feature_cols"], f, indent=2)
    print(f"\n  [OK] {clf_json_path}")

    with open(reg_json_path, "w") as f:
        json.dump(feature_sets["reg_feature_cols"], f, indent=2)
    print(f"  [OK] {reg_json_path}")

    # Also save the train-fold income median so later phases can replicate
    meta = {
        "train_median_income_used_for_imputation": splits["train_median_income"],
        "winsor_percentile": 99,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }
    meta_path = os.path.join(REPORTS, "prep_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  [OK] {meta_path}  (imputation median + prep constants)")


# =============================================================================
# Pipeline entry point
# =============================================================================
def run_prep_pipeline() -> dict:
    """Run the full model-prep pipeline and return the splits dict."""
    df = load_clean_data()
    df, cap_values = winsorise(df)
    feature_sets   = build_feature_sets(df)
    splits         = split_data(feature_sets)
    save_splits(splits, feature_sets)

    print("\n" + "=" * 60)
    print("MODEL PREP COMPLETE")
    print("=" * 60)
    print("  Next steps:")
    print("    python src/train_classification.py")
    print("    python src/train_regression.py")

    return splits, feature_sets


if __name__ == "__main__":
    run_prep_pipeline()
