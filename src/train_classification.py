# Credit Risk XAI — Phase 2: Classification Model Training
# ==========================================================
# Trains two classifiers on the prepared splits from model_prep.py:
#   1. Logistic Regression  (interpretable baseline)
#   2. Random Forest        (main model, 200 trees)
#
# Both use class_weight='balanced' because the dataset is heavily imbalanced
# (~6.6% positive class).  Without this, models tend to predict "no default"
# for everyone and still achieve 93% accuracy — which is useless for a lender.
#
# Run: python src/train_classification.py
# Prerequisites: python src/model_prep.py must have been run first.

import os
import json
import time

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay,
)

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
    """Load the classification train/test splits saved by model_prep.py."""
    print("\n" + "=" * 60)
    print("Loading classification splits")
    print("=" * 60)

    X_train = pd.read_csv(os.path.join(PROCESSED, "X_train_clf.csv"))
    X_test  = pd.read_csv(os.path.join(PROCESSED, "X_test_clf.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED, "y_train_clf.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(PROCESSED, "y_test_clf.csv")).squeeze()

    print(f"  X_train: {X_train.shape}   y_train: {y_train.shape}")
    print(f"  X_test : {X_test.shape}    y_test : {y_test.shape}")
    print(f"  Train class-1 rate: {y_train.mean()*100:.2f}%")
    print(f"  Test  class-1 rate: {y_test.mean()*100:.2f}%")

    return X_train, X_test, y_train, y_test


# =============================================================================
# Evaluate a fitted classifier
# =============================================================================
def evaluate_classifier(model, X_test, y_test, model_name: str) -> dict:
    """
    Compute accuracy, precision, recall, F1, ROC-AUC, and confusion matrix.
    Returns a dict of all metrics plus the label-predicted arrays.

    NOTE on metric choice:
      - Accuracy is reported for completeness but is misleading on an
        imbalanced dataset (a naive "always predict 0" model gets 93.4%).
      - ROC-AUC and F1 are the primary metrics we compare models on because
        they are robust to class imbalance and measure different aspects of
        discrimination quality.
    """
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]   # probability of default

    acc      = accuracy_score(y_test, y_pred)
    prec     = precision_score(y_test, y_pred, zero_division=0)
    rec      = recall_score(y_test, y_pred, zero_division=0)
    f1       = f1_score(y_test, y_pred, zero_division=0)
    roc_auc  = roc_auc_score(y_test, y_prob)
    cm       = confusion_matrix(y_test, y_pred)

    print(f"\n  --- {model_name} ---")
    print(f"    Accuracy  : {acc:.4f}  (misleading here — see ROC-AUC/F1)")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1 Score  : {f1:.4f}")
    print(f"    ROC-AUC   : {roc_auc:.4f}")
    print(f"    Confusion matrix (rows=actual, cols=predicted):")
    print(f"      TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"      FN={cm[1,0]:,}  TP={cm[1,1]:,}")

    return {
        "model_name": model_name,
        "accuracy":   acc,
        "precision":  prec,
        "recall":     rec,
        "f1":         f1,
        "roc_auc":    roc_auc,
        "confusion_matrix": cm.tolist(),
        "y_pred":     y_pred,
        "y_prob":     y_prob,
    }


# =============================================================================
# Save confusion matrix as a figure
# =============================================================================
def save_confusion_matrix_plot(metrics: dict) -> None:
    """Save a labelled confusion matrix heatmap to reports/figures/."""
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Default (0)", "Default (1)"],
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {metrics['model_name']}\n"
                 f"(ROC-AUC={metrics['roc_auc']:.4f}, F1={metrics['f1']:.4f})")
    fname = metrics["model_name"].replace(" ", "_").lower() + "_confusion.png"
    fpath = os.path.join(FIGURES, fname)
    plt.savefig(fpath, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  [saved] {fpath}")


# =============================================================================
# Save feature importances (Random Forest only)
# =============================================================================
def save_feature_importance_plot(model, feature_cols: list, model_name: str) -> None:
    """Bar chart of mean decrease in impurity (MDI) importances."""
    importances = model.feature_importances_
    pairs = sorted(zip(importances, feature_cols), reverse=True)
    imps, names = zip(*pairs)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4C72B0"] * len(names)
    ax.barh(names[::-1], imps[::-1], color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Mean Decrease in Impurity (MDI importance)")
    ax.set_title(f"Feature Importances — {model_name}\n(built-in, not SHAP yet)")
    ax.grid(axis="x", alpha=0.4)

    fpath = os.path.join(FIGURES, "clf_feature_importance.png")
    plt.savefig(fpath, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  [saved] {fpath}")

    print("\n  Feature importances (MDI, descending):")
    for name, imp in zip(names, imps):
        bar = "#" * int(imp * 300)
        print(f"    {name:28s}: {imp:.4f}  {bar}")


# =============================================================================
# Save performance report
# =============================================================================
def save_performance_report(lr_metrics: dict, rf_metrics: dict,
                             feature_cols: list, rf_model) -> None:
    """Write a markdown report with all real metrics from the evaluation run."""

    winner = rf_metrics if rf_metrics["roc_auc"] >= lr_metrics["roc_auc"] else lr_metrics
    runner = lr_metrics if winner is rf_metrics else rf_metrics

    # Feature importances (top 5 for narrative)
    imps = sorted(
        zip(rf_model.feature_importances_, feature_cols),
        reverse=True,
    )

    lines = [
        "# Classification Model Performance — Phase 2",
        "",
        "> All metrics were computed on the 20% held-out test set.",
        "> No test-set statistics were used during training or hyperparameter selection.",
        "",
        "## Models Trained",
        "",
        "| Model | class_weight | max_iter / n_estimators | random_state |",
        "|---|---|---|---|",
        "| Logistic Regression (StandardScaler pipeline) | balanced | max_iter=2000 | 42 |",
        "| Random Forest | balanced | n_estimators=200 | 42 |",
        "",
        "## Test-Set Metrics",
        "",
        "| Metric | Logistic Regression | Random Forest |",
        "|---|---|---|",
        f"| Accuracy  | {lr_metrics['accuracy']:.4f} | {rf_metrics['accuracy']:.4f} |",
        f"| Precision | {lr_metrics['precision']:.4f} | {rf_metrics['precision']:.4f} |",
        f"| Recall    | {lr_metrics['recall']:.4f} | {rf_metrics['recall']:.4f} |",
        f"| F1 Score  | {lr_metrics['f1']:.4f} | {rf_metrics['f1']:.4f} |",
        f"| **ROC-AUC** | **{lr_metrics['roc_auc']:.4f}** | **{rf_metrics['roc_auc']:.4f}** |",
        "",
        "## Confusion Matrices",
        "",
        "**Logistic Regression**",
        "",
        "| | Predicted 0 | Predicted 1 |",
        "|---|---|---|",
        f"| Actual 0 | {lr_metrics['confusion_matrix'][0][0]:,} | {lr_metrics['confusion_matrix'][0][1]:,} |",
        f"| Actual 1 | {lr_metrics['confusion_matrix'][1][0]:,} | {lr_metrics['confusion_matrix'][1][1]:,} |",
        "",
        "**Random Forest**",
        "",
        "| | Predicted 0 | Predicted 1 |",
        "|---|---|---|",
        f"| Actual 0 | {rf_metrics['confusion_matrix'][0][0]:,} | {rf_metrics['confusion_matrix'][0][1]:,} |",
        f"| Actual 1 | {rf_metrics['confusion_matrix'][1][0]:,} | {rf_metrics['confusion_matrix'][1][1]:,} |",
        "",
        "## Winner",
        "",
        f"**{winner['model_name']}** wins by ROC-AUC: "
        f"{winner['roc_auc']:.4f} vs {runner['roc_auc']:.4f} "
        f"(margin: {abs(winner['roc_auc'] - runner['roc_auc']):.4f}).",
        "",
        f"Saved to: `models/classification_model.joblib`",
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

    report_path = os.path.join(REPORTS, "classification_performance.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  [saved] {report_path}")


# =============================================================================
# Main training pipeline
# =============================================================================
def run_classification_pipeline() -> None:
    """Train, evaluate, and save both classifiers."""

    # 1. Load data
    X_train, X_test, y_train, y_test = load_splits()
    feature_cols = list(X_train.columns)

    # 2. Logistic Regression (baseline)
    # Wrapped in a Pipeline with StandardScaler: lbfgs converges much faster
    # on scaled features and avoids ConvergenceWarning. The Pipeline is a
    # drop-in: .predict(), .predict_proba(), and joblib.dump() all work as-is.
    print("\n" + "=" * 60)
    print("Training: Logistic Regression (baseline, with StandardScaler pipeline)")
    print("=" * 60)
    t0 = time.time()
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        )),
    ])
    lr.fit(X_train, y_train)
    print(f"  Fit time: {time.time()-t0:.1f}s")

    # 3. Random Forest (main model)
    print("\n" + "=" * 60)
    print("Training: Random Forest (main model, 200 trees)")
    print("=" * 60)
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,          # use all available CPU cores
    )
    rf.fit(X_train, y_train)
    print(f"  Fit time: {time.time()-t0:.1f}s")

    # 4. Evaluate both
    print("\n" + "=" * 60)
    print("Evaluation on held-out test set")
    print("=" * 60)
    lr_metrics = evaluate_classifier(lr, X_test, y_test, "Logistic Regression")
    rf_metrics = evaluate_classifier(rf, X_test, y_test, "Random Forest")

    # 5. Save confusion matrix figures
    print("\nSaving figures ...")
    save_confusion_matrix_plot(lr_metrics)
    save_confusion_matrix_plot(rf_metrics)

    # 6. Save RF feature importances
    save_feature_importance_plot(rf, feature_cols, "Random Forest")

    # 7. Pick winner by ROC-AUC
    print("\n" + "=" * 60)
    print("Selecting best model by ROC-AUC")
    print("=" * 60)
    if rf_metrics["roc_auc"] >= lr_metrics["roc_auc"]:
        best_model = rf
        print(f"  Winner: Random Forest  (ROC-AUC {rf_metrics['roc_auc']:.4f})")
    else:
        best_model = lr
        print(f"  Winner: Logistic Regression  (ROC-AUC {lr_metrics['roc_auc']:.4f})")

    model_path = os.path.join(MODELS, "classification_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"  [saved] {model_path}")

    # 8. Save markdown performance report
    save_performance_report(lr_metrics, rf_metrics, feature_cols, rf)

    print("\n" + "=" * 60)
    print("CLASSIFICATION TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_classification_pipeline()
