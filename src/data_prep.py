# Credit Risk XAI — Phase 1: Data Preparation
# =============================================
# This script contains clean, well-commented helper functions for loading,
# validating, and cleaning the "Give Me Some Credit" Kaggle dataset.
#
# Run order:
#   load_raw_data() -> rename_columns() -> report_missingness()
#   -> impute_dependents() -> flag_artifacts() -> save_cleaned_data()
#
# Or call the high-level pipeline: run_pipeline()

import os
import pandas as pd
import numpy as np

# ---- Paths ------------------------------------------------------------------
RAW_PATH       = os.path.join("data", "raw", "cs-training.csv")
CLEAN_PATH     = os.path.join("data", "processed", "credit_data_clean.csv")
FLAGGED_PATH   = os.path.join("data", "processed", "flagged_rows.csv")

# ---- Column rename mapping --------------------------------------------------
# Original Kaggle names -> readable snake_case names
COLUMN_RENAME = {
    "SeriousDlqin2yrs":                       "target_default",
    "RevolvingUtilizationOfUnsecuredLines":    "revolving_util",
    "age":                                     "age",
    "NumberOfTime30-59DaysPastDueNotWorse":    "late_30_59_days",
    "DebtRatio":                               "debt_ratio",
    "MonthlyIncome":                           "monthly_income",
    "NumberOfOpenCreditLinesAndLoans":         "open_credit_lines",
    "NumberOfTimes90DaysLate":                 "late_90_days",
    "NumberRealEstateLoansOrLines":            "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse":    "late_60_89_days",
    "NumberOfDependents":                      "num_dependents",
}

# Artifact sentinel codes known to be data-entry errors in this dataset
ARTIFACT_CODES = {96, 98}


# ---- Step 1: Load -----------------------------------------------------------
def load_raw_data(path: str = RAW_PATH) -> pd.DataFrame:
    """
    Load the raw CSV, drop the unnamed row-index column, and print
    the actual column names + dtypes so we can confirm they match spec.

    The copy of this dataset has two quirks handled automatically:
      1. 'Unnamed: 0' row-index column — dropped if present.
      2. SeriousDlqin2yrs values stored as Python bytes literals, e.g. b'1'
         — the b'...' wrapper is stripped and the column is cast to int.
    """
    print("\n" + "=" * 60)
    print("STEP 1: Loading raw data")
    print("=" * 60)

    df = pd.read_csv(path)

    print(f"\nShape on load : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("\nActual columns and dtypes (raw, before any fixes):")
    print(df.dtypes.to_string())

    # --- Fix 1: Drop the Kaggle-generated row index column if present --------
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        print("\n[OK] Dropped 'Unnamed: 0' index column.")
    else:
        print("\n[INFO] 'Unnamed: 0' not present - skipping drop.")

    # --- Fix 2: Handle byte-string artifact in the target column -------------
    # The CSV was saved with Python bytes repr: b'1', b'0' instead of 1, 0.
    # Newer pandas (2.x) loads this as StringDtype (dtype.kind == 'O' or 'U')
    # rather than numpy object, so we check dtype.kind instead of == object.
    target_col = "SeriousDlqin2yrs"
    if target_col in df.columns:
        col_dtype = df[target_col].dtype
        # kind 'O' = numpy object, 'U' = unicode string (StringDtype in newer pandas)
        is_string_col = (col_dtype == object) or (str(col_dtype) in ("str", "string"))
        if is_string_col:
            sample_val = str(df[target_col].iloc[0])
            if sample_val.startswith("b'"):
                print("\n[INFO] Target column has byte-string values (e.g. b'1').")
                print("       Stripping b'...' notation and casting to int.")
                # Strip leading b' and trailing '
                df[target_col] = (
                    df[target_col]
                    .astype(str)
                    .str.replace(r"^b'(.*)'$", r"\1", regex=True)
                )
            else:
                print("\n[INFO] Target column loaded as string (non-byte cause).")

            # Cast to numeric (int) regardless of which string format it was
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
            print(f"       Target dtype after fix: {df[target_col].dtype}")

    print(f"\nShape after column fixes: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


# ---- Step 2: Rename columns -------------------------------------------------
def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename Kaggle column names to readable snake_case equivalents.
    Prints a before/after mapping so the transformation is transparent.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Renaming columns")
    print("=" * 60)

    # Only rename columns that actually exist (defensive check)
    rename_map = {k: v for k, v in COLUMN_RENAME.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    print("\nColumn mapping applied:")
    for old, new in rename_map.items():
        print(f"  {old:50s} -> {new}")

    return df


# ---- Step 3: Report missingness ---------------------------------------------
def report_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Print a table showing null count and percentage for every column.
    No data is changed here — this is purely diagnostic.
    Returns a DataFrame with the missingness stats for easy reuse.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Missingness report (BEFORE any imputation)")
    print("=" * 60)

    total = len(df)
    null_counts = df.isnull().sum()
    null_pct    = (null_counts / total * 100).round(2)

    miss_df = pd.DataFrame({
        "null_count": null_counts,
        "null_pct":   null_pct,
    })
    # Only show columns that actually have missing values
    miss_df_filtered = miss_df[miss_df["null_count"] > 0]

    if miss_df_filtered.empty:
        print("\n[OK] No missing values found.")
    else:
        print(f"\nTotal rows: {total:,}")
        print("\nColumns with missing values:")
        print(miss_df_filtered.to_string())

    # Specifically call out monthly_income per project spec
    if "monthly_income" in df.columns:
        mi_pct = df["monthly_income"].isnull().mean() * 100
        print(f"\n  -> monthly_income missing: {mi_pct:.2f}%")
        print("     (NOT imputing — rows with known income = training set for")
        print("      income-estimation regression in a later phase.)")

    if "num_dependents" in df.columns:
        nd_pct = df["num_dependents"].isnull().mean() * 100
        print(f"  -> num_dependents missing: {nd_pct:.2f}% (will impute with median)")

    return miss_df


# ---- Step 4: Impute num_dependents (median) ---------------------------------
def impute_dependents(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute the small fraction of missing num_dependents values with
    the column median.  Monthly income is intentionally left untouched.
    """
    print("\n" + "=" * 60)
    print("STEP 4: Imputing num_dependents (median)")
    print("=" * 60)

    col = "num_dependents"
    missing_before = df[col].isnull().sum()

    if missing_before == 0:
        print(f"\n[OK] No nulls in '{col}' — nothing to impute.")
        return df

    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

    missing_after = df[col].isnull().sum()
    print(f"\n  Nulls before imputation : {missing_before:,}")
    print(f"  Median used for fill    : {median_val}")
    print(f"  Nulls after imputation  : {missing_after:,}")

    return df


# ---- Step 5: Flag data-entry artifacts --------------------------------------
def flag_artifacts(df: pd.DataFrame) -> tuple:
    """
    Identify rows with known data-entry artifacts:
      - age == 0              (physiologically impossible)
      - late_30_59_days, late_90_days, or late_60_89_days in {96, 98}
        (these are sentinel codes, not real delinquency counts)

    Returns:
      cleaned_df  — main DataFrame with flagged rows REMOVED
      flagged_df  — the flagged rows saved separately for audit
    """
    print("\n" + "=" * 60)
    print("STEP 5: Flagging data-entry artifacts")
    print("=" * 60)

    late_cols = ["late_30_59_days", "late_90_days", "late_60_89_days"]

    # Build boolean masks
    age_zero_mask    = df["age"] == 0
    late_artifact_mask = df[late_cols].isin(ARTIFACT_CODES).any(axis=1)
    combined_mask    = age_zero_mask | late_artifact_mask

    # Report counts
    print(f"\n  Rows flagged for age == 0              : {age_zero_mask.sum():,}")
    print(f"  Rows flagged for late-payment codes    : {late_artifact_mask.sum():,}")
    print(f"  Rows flagged in total (union)          : {combined_mask.sum():,}")

    flagged_df = df[combined_mask].copy()
    cleaned_df = df[~combined_mask].copy()

    print(f"\n  Rows in main dataset after removal     : {len(cleaned_df):,}")

    return cleaned_df, flagged_df


# ---- Step 6: Save outputs ---------------------------------------------------
def save_cleaned_data(
    cleaned_df:   pd.DataFrame,
    flagged_df:   pd.DataFrame,
    clean_path:   str = CLEAN_PATH,
    flagged_path: str = FLAGGED_PATH,
) -> None:
    """
    Persist the cleaned DataFrame and the flagged rows to disk.
    Creates the output directory automatically if it doesn't exist.
    """
    print("\n" + "=" * 60)
    print("STEP 6: Saving outputs")
    print("=" * 60)

    os.makedirs(os.path.dirname(clean_path),   exist_ok=True)
    os.makedirs(os.path.dirname(flagged_path), exist_ok=True)

    cleaned_df.to_csv(clean_path,   index=False)
    flagged_df.to_csv(flagged_path, index=False)

    print(f"\n  [OK] Cleaned data  -> {clean_path}  ({len(cleaned_df):,} rows)")
    print(f"  [OK] Flagged rows  -> {flagged_path} ({len(flagged_df):,} rows)")


# ---- High-level pipeline ----------------------------------------------------
def run_pipeline() -> dict:
    """
    Run the full Phase 1 data-prep pipeline end-to-end.
    Returns a dictionary of key statistics for use in the EDA notebook
    and the data_summary.md report.
    """
    # 1. Load
    df_raw   = load_raw_data()
    n_rows_raw = len(df_raw)

    # 2. Rename
    df = rename_columns(df_raw)

    # 3. Missingness report (before any changes)
    miss_report = report_missingness(df)

    # Capture missingness percentages before any imputation
    monthly_income_pct = df["monthly_income"].isnull().mean() * 100
    num_dependents_pct = df["num_dependents"].isnull().mean() * 100

    # 4. Impute num_dependents
    df = impute_dependents(df)

    # 5. Flag artifacts
    df_clean, df_flagged = flag_artifacts(df)
    n_rows_clean   = len(df_clean)
    n_rows_flagged = len(df_flagged)

    # 6. Save
    save_cleaned_data(df_clean, df_flagged)

    # Compute class balance on the cleaned dataset
    class_counts    = df_clean["target_default"].value_counts()
    default_pct     = class_counts.get(1, 0) / len(df_clean) * 100
    non_default_pct = class_counts.get(0, 0) / len(df_clean) * 100

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE — Summary Statistics")
    print("=" * 60)
    print(f"  Raw rows               : {n_rows_raw:,}")
    print(f"  Cleaned rows           : {n_rows_clean:,}")
    print(f"  Flagged/removed rows   : {n_rows_flagged:,}")
    print(f"  monthly_income missing : {monthly_income_pct:.2f}%")
    print(f"  num_dependents missing : {num_dependents_pct:.2f}% (before imputation)")
    print(f"  Class 1 (default)      : {default_pct:.2f}%")
    print(f"  Class 0 (no default)   : {non_default_pct:.2f}%")

    return {
        "n_rows_raw":          n_rows_raw,
        "n_rows_clean":        n_rows_clean,
        "n_rows_flagged":      n_rows_flagged,
        "monthly_income_pct":  monthly_income_pct,
        "num_dependents_pct":  num_dependents_pct,
        "default_pct":         default_pct,
        "non_default_pct":     non_default_pct,
        "class_counts":        class_counts,
        "miss_report":         miss_report,
    }


# ---- Entry point ------------------------------------------------------------
if __name__ == "__main__":
    stats = run_pipeline()
