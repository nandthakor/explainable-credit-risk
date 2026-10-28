# Phase 1 EDA — Credit Risk XAI Project
# =========================================
# Generates and saves EDA figures to reports/figures/.
# Reads the cleaned dataset produced by src/data_prep.py.
# Run from the project root:  python notebooks/01_eda.py

import os
import sys

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# ---- Paths ------------------------------------------------------------------
CLEAN_CSV   = os.path.join("data", "processed", "credit_data_clean.csv")
FIGURES_DIR = os.path.join("reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Use the raw CSV for the missingness chart so we see pre-imputation state
RAW_CSV = os.path.join("data", "raw", "cs-training.csv")

# ---- Shared style -----------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi":     150,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


# ---- Helper: save and close -------------------------------------------------
def savefig(filename: str) -> None:
    """Save the current matplotlib figure to FIGURES_DIR and close it."""
    fpath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(fpath, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {fpath}")


# =============================================================================
# Load cleaned data
# =============================================================================
print("\nLoading cleaned dataset ...")
df = pd.read_csv(CLEAN_CSV)
print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")


# =============================================================================
# Figure 1 — Target class balance bar chart
# =============================================================================
print("\n[Figure 1] Class balance ...")

class_counts = df["target_default"].value_counts().sort_index()
class_labels  = {0: "No Default (0)", 1: "Default (1)"}
pct_0 = class_counts[0] / len(df) * 100
pct_1 = class_counts[1] / len(df) * 100

print(f"  Class 0 (no default): {class_counts[0]:,} rows  ({pct_0:.2f}%)")
print(f"  Class 1 (default)   : {class_counts[1]:,} rows  ({pct_1:.2f}%)")

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(
    [class_labels[k] for k in class_counts.index],
    class_counts.values,
    color=["#4C72B0", "#DD8452"],
    edgecolor="white",
    width=0.5,
)

# Annotate bars with counts and percentages
for bar, (label, count, pct) in zip(bars, [
    (class_labels[0], class_counts[0], pct_0),
    (class_labels[1], class_counts[1], pct_1),
]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 800,
        f"{count:,}\n({pct:.1f}%)",
        ha="center", va="bottom", fontsize=10,
    )

ax.set_title("Target Class Balance\n(credit_data_clean.csv — after artifact removal)")
ax.set_ylabel("Number of Rows")
ax.set_xlabel("Class")
ax.set_ylim(0, max(class_counts.values) * 1.18)
savefig("01_class_balance.png")


# =============================================================================
# Figure 2 — Missingness bar chart (raw data, before imputation)
# =============================================================================
print("\n[Figure 2] Missingness chart (raw data) ...")

# Reload raw data with the same fix used in data_prep.py
df_raw = pd.read_csv(RAW_CSV)
target_raw = "SeriousDlqin2yrs"
if str(df_raw[target_raw].dtype) in ("str", "string"):
    df_raw[target_raw] = (
        df_raw[target_raw]
        .astype(str)
        .str.replace(r"^b'(.*)'$", r"\1", regex=True)
    )
    df_raw[target_raw] = pd.to_numeric(df_raw[target_raw], errors="coerce")

# Rename for readability on the chart
RENAME = {
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
df_raw = df_raw.rename(columns=RENAME)

null_pct = (df_raw.isnull().mean() * 100).sort_values(ascending=False)
null_pct = null_pct[null_pct > 0]  # show only columns with any missing

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh(
    null_pct.index,
    null_pct.values,
    color="#4C72B0",
    edgecolor="white",
)
for bar, val in zip(bars, null_pct.values):
    ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}%", va="center", fontsize=10)

ax.set_xlabel("% Missing")
ax.set_title("Missingness by Column\n(raw data, before imputation)")
ax.set_xlim(0, null_pct.max() * 1.2)
savefig("02_missingness.png")


# =============================================================================
# Figure 3 — Histograms: age, debt_ratio, revolving_util, monthly_income
# =============================================================================
print("\n[Figure 3] Histograms ...")

hist_cols = {
    "age":            "Age (years)",
    "debt_ratio":     "Debt Ratio",
    "revolving_util": "Revolving Utilisation of Unsecured Lines",
    "monthly_income": "Monthly Income (USD)",
}

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for ax, (col, label) in zip(axes, hist_cols.items()):
    data = df[col].dropna()

    # Cap extreme outliers at 99th percentile for readability
    p99 = data.quantile(0.99)
    data_capped = data[data <= p99]

    ax.hist(data_capped, bins=60, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.set_title(f"{label}\n(capped at 99th pct = {p99:,.1f})")
    ax.set_xlabel(label)
    ax.set_ylabel("Count")

    # Annotate with median
    median = data.median()
    ax.axvline(median, color="#DD8452", linestyle="--", linewidth=1.5,
               label=f"Median = {median:,.1f}")
    ax.legend(fontsize=9)

fig.suptitle("Feature Distributions (cleaned data)", fontsize=14, y=1.01)
fig.tight_layout()
savefig("03_histograms.png")


# =============================================================================
# Figure 4 — Correlation heatmap
# =============================================================================
print("\n[Figure 4] Correlation heatmap ...")

# Drop rows where monthly_income is NaN for the correlation (avoids NaN spread)
df_corr = df.dropna(subset=["monthly_income"])

corr = df_corr.select_dtypes(include=[np.number]).corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))   # hide upper triangle (redundant)

sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    ax=ax,
    annot_kws={"size": 8},
)
ax.set_title("Pearson Correlation Heatmap — Numeric Features\n(rows with missing monthly_income excluded)")
fig.tight_layout()
savefig("04_correlation_heatmap.png")

print("\n[DONE] All figures saved to", FIGURES_DIR)
