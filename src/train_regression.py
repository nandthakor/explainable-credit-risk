# Credit Risk XAI — Phase 2: Income Regression Model Training
# =============================================================
# Trains two regression models on the prepared splits from model_prep.py:
#   1. Linear Regression    (interpretable baseline)
#   2. Random Forest Regressor  (main model, 200 trees)
#
# Task: Estimate monthly_income for the ~20% of applicants with no verified
# income, using the ~80% with known income as training data.
#
# Run: python src/train_regression.py
# Prerequisites: python src/model_prep.py must have been run first.

import os
import time

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model  import LinearRegression
from sklearn.ensemble      import RandomForestRegressor
from sklearn.metrics       import mean_absolute_error, mean_squared_error, r2_score

# ---- Paths ------------------------------------------------------------------
PROCESSED   = os.path.join("data", "processed")
REPORTS     = os.path.join("reports")
FIGURES     = os.path.join("reports", "figures")
MODELS      = os.path.join("models")

os.makedirs(MODELS,  exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

RANDOM_STATE = 42


# =============================================================================
# Load pre-split data
# =============================================================================
def load_splits() -> tuple:
    """Load the regression train/test splits saved by model_prep.py."""
    print("\n" + "=" * 60)
    print("Loading regression splits")
    print("=" * 60)

    X_train = pd.read_csv(os.path.join(PROCESSED, "X_train_reg.csv"))
    X_test  = pd.read_csv(os.path.join(PROCESSED, "X_test_reg.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED, "y_train_reg.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(PROCESSED, "y_test_reg.csv")).squeeze()

    print(f"  X_train: {X_train.shape}   y_train: {y_train.shape}")
    print(f"  X_test : {X_test.shape}    y_test : {y_test.shape}")
    print(f"  Train income — median: {y_train.median():,.0f}  mean: {y_train.mean():,.0f}")
    print(f"  Test  income — median: {y_test.median():,.0f}  mean: {y_test.mean():,.0f}")

    return X_train, X_test, y_train, y_test


# =============================================================================
# Evaluate a fitted regressor
# =============================================================================
def evaluate_regressor(model, X_test, y_test, model_name: str) -> dict:
    """
    Compute MAE, RMSE, and R² on the test set.

    - MAE (Mean Absolute Error): average absolute dollar error in income prediction.
      Easy to interpret: "on average, the model is off by $X/month".
    - RMSE (Root Mean Squared Error): penalises large errors more than MAE.
      Useful to know if there are systematic large mispredictions.
    - R² (Coefficient of Determination): proportion of variance in income
      explained by the features.  R²=1.0 is perfect; R²=0.0 is equivalent
      to always predicting the mean income; negative R² means worse than mean.
    """
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"\n  --- {model_name} ---")
    print(f"    MAE  : {mae:,.2f}   (avg absolute dollar error per prediction)")
    print(f"    RMSE : {rmse:,.2f}  (penalises large errors more)")
    print(f"    R2   : {r2:.4f}    (proportion of income variance explained)")

    return {
        "model_name": model_name,
        "mae":        mae,
        "rmse":       rmse,
        "r2":         r2,
        "y_pred":     y_pred,
    }


# =============================================================================
# Residual plot
# =============================================================================
def save_residual_plot(metrics: dict, y_test, model_name: str) -> None:
    """Scatter plot of actual vs predicted income for the test set."""
    y_pred = metrics["y_pred"]

    # Cap display at 99th percentile for readability (income is right-skewed)
    cap = np.percentile(y_test, 99)
    mask = y_test <= cap

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Actual vs Predicted
    ax = axes[0]
    ax.scatter(y_test[mask], y_pred[mask], alpha=0.2, s=4, color="#4C72B0")
    max_val = max(y_test[mask].max(), y_pred[mask].max())
    ax.plot([0, max_val], [0, max_val], "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Monthly Income (USD)")
    ax.set_ylabel("Predicted Monthly Income (USD)")
    ax.set_title(f"{model_name} — Actual vs Predicted\n"
                 f"(capped at 99th pct, R²={metrics['r2']:.4f})")
    ax.legend(fontsize=9)

    # Residuals
    ax = axes[1]
    residuals = y_pred - y_test.values
    ax.scatter(y_pred[mask], residuals[mask], alpha=0.2, s=4, color="#DD8452")
    ax.axhline(0, color="red", linestyle="--", lw=1.5)
    ax.set_xlabel("Predicted Monthly Income (USD)")
    ax.set_ylabel("Residual (Predicted - Actual)")
    ax.set_title(f"{model_name} — Residual Plot\n"
                 f"(MAE={metrics['mae']:,.0f}, RMSE={metrics['rmse']:,.0f})")

    fig.tight_layout()
    fname = model_name.replace(" ", "_").lower() + "_residuals.png"
    fpath = os.path.join(FIGURES, fname)
    plt.savefig(fpath, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  [saved] {fpath}")


# =============================================================================
# Feature importances (Random Forest)
# =============================================================================
def save_reg_feature_importance_plot(model, feature_cols: list) -> None:
    """Bar chart of MDI feature importances for the regression model."""
    importances = model.feature_importances_
    pairs = sorted(zip(importances, feature_cols), reverse=True)
    imps, names = zip(*pairs)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names[::-1], imps[::-1], color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Mean Decrease in Impurity (MDI importance)")
    ax.set_title("Feature Importances — Random Forest Regressor\n(built-in, not SHAP yet)")
    ax.grid(axis="x", alpha=0.4)

    fpath = os.path.join(FIGURES, "reg_feature_importance.png")
    plt.savefig(fpath, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  [saved] {fpath}")

    print("\n  Feature importances (MDI, descending):")
    for name, imp in zip(names, imps):
        bar = "#" * int(imp * 300)
        print(f"    {name:28s}: {imp:.4f}  {bar}")


# =============================================================================
# Save markdown performance report
# =============================================================================
def save_performance_report(lr_metrics: dict, rf_metrics: dict,
                             feature_cols: list, rf_model) -> None:
    """Write the regression performance markdown report with real numbers only."""

    winner = rf_metrics if rf_metrics["r2"] >= lr_metrics["r2"] else lr_metrics
    runner = lr_metrics if winner is rf_metrics else rf_metrics

    imps = sorted(
        zip(rf_model.feature_importances_, feature_cols),
        reverse=True,
    )

    lines = [
        "# Regression Model Performance — Phase 2",
        "",
        "> All metrics were computed on the 20% held-out test set.",
        "> Task: predict monthly income using rows where income is known (~80% of data).",
        "",
        "## Models Trained",
        "",
        "| Model | n_estimators | random_state |",
        "|---|---|---|",
        "| Linear Regression | N/A | N/A |",
        "| Random Forest Regressor | 200 | 42 |",
        "",
        "## Test-Set Metrics",
        "",
        "| Metric | Linear Regression | Random Forest |",
        "|---|---|---|",
        f"| MAE  (USD) | {lr_metrics['mae']:,.2f} | {rf_metrics['mae']:,.2f} |",
        f"| RMSE (USD) | {lr_metrics['rmse']:,.2f} | {rf_metrics['rmse']:,.2f} |",
        f"| **R²** | **{lr_metrics['r2']:.4f}** | **{rf_metrics['r2']:.4f}** |",
        "",
        "## Winner",
        "",
        f"**{winner['model_name']}** wins by R²: "
        f"{winner['r2']:.4f} vs {runner['r2']:.4f} "
        f"(margin: {abs(winner['r2'] - runner['r2']):.4f}).",
        "",
        f"Saved to: `models/regression_model.joblib`",
        "",
        "## Top-5 Random Forest Feature Importances (MDI, not SHAP)",
        "",
        "| Rank | Feature | MDI Importance |",
        "|---|---|---|",
    ]

    for rank, (imp, feat) in enumerate(imps[:5], 1):
        lines.append(f"| {rank} | {feat} | {imp:.4f} |")

    lines += [
        "",
        "> Note: MDI importances can be biased toward high-cardinality features.",
        "> Phase 3 will use SHAP for more reliable feature attribution.",
    ]

    report_path = os.path.join(REPORTS, "regression_performance.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  [saved] {report_path}")


# =============================================================================
# Main training pipeline
# =============================================================================
def run_regression_pipeline() -> None:
    """Train, evaluate, and save both regressors."""

    # 1. Load data
    X_train, X_test, y_train, y_test = load_splits()
    feature_cols = list(X_train.columns)

    # 2. Linear Regression (baseline)
    print("\n" + "=" * 60)
    print("Training: Linear Regression (baseline)")
    print("=" * 60)
    t0 = time.time()
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print(f"  Fit time: {time.time()-t0:.2f}s")

    # 3. Random Forest Regressor (main model)
    print("\n" + "=" * 60)
    print("Training: Random Forest Regressor (main model, 200 trees)")
    print("=" * 60)
    t0 = time.time()
    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    print(f"  Fit time: {time.time()-t0:.1f}s")

    # 4. Evaluate both
    print("\n" + "=" * 60)
    print("Evaluation on held-out test set")
    print("=" * 60)
    lr_metrics = evaluate_regressor(lr, X_test, y_test, "Linear Regression")
    rf_metrics = evaluate_regressor(rf, X_test, y_test, "Random Forest Regressor")

    # 5. Save residual plots
    print("\nSaving figures ...")
    save_residual_plot(lr_metrics, y_test, "Linear Regression")
    save_residual_plot(rf_metrics, y_test, "Random Forest Regressor")

    # 6. Save RF feature importances
    save_reg_feature_importance_plot(rf, feature_cols)

    # 7. Pick winner by R²
    print("\n" + "=" * 60)
    print("Selecting best model by R2")
    print("=" * 60)
    if rf_metrics["r2"] >= lr_metrics["r2"]:
        best_model = rf
        print(f"  Winner: Random Forest Regressor  (R2={rf_metrics['r2']:.4f})")
    else:
        best_model = lr
        print(f"  Winner: Linear Regression  (R2={lr_metrics['r2']:.4f})")

    model_path = os.path.join(MODELS, "regression_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"  [saved] {model_path}")

    # 8. Save markdown report
    save_performance_report(lr_metrics, rf_metrics, feature_cols, rf)

    print("\n" + "=" * 60)
    print("REGRESSION TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_regression_pipeline()
