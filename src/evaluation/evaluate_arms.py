# =============================================================================
# Evaluation Metrics — Arms 1-4
# =============================================================================
# Reads all cached generation JSONs from raw_generations/ and computes
# per-arm aggregate metrics for three paper tables:
#
#   Table 2 — Faithfulness metrics (directional recall, hallucination rate,
#              numerical accuracy, recourse fidelity, post-verifier violation)
#   Table 3 — Readability metrics  (Flesch-Kincaid Grade, Flesch Reading Ease,
#              mean word count) via textstat
#   Verifier Ablation — pre/post verifier violation rates, retry & fallback counts
#
# No LLM calls; pure metric computation from cached JSON artefacts.
#
# Usage:
#   python src/evaluation/evaluate_arms.py

import os
import sys
import json
import csv
import math

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# textstat may need proxy env var on some systems
os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")
try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False
    print("[WARN] textstat not installed — readability metrics will be 0. "
          "Run: pip install textstat")

GEN_DIR  = os.path.join(PROJECT_ROOT, "reports", "evaluation", "raw_generations")
EVAL_DIR = os.path.join(PROJECT_ROOT, "reports", "evaluation")

ARMS = ["shap_only", "cf_only", "combined", "deterministic_template"]

ARM_DISPLAY = {
    "shap_only":              "SHAP-Only (Arm 1)",
    "cf_only":                "CF-Only (Arm 2)",
    "combined":               "Combined (Arm 3)",
    "deterministic_template": "Deterministic Template (Arm 4)",
}

# ---- Helpers ----------------------------------------------------------------

def safe_div(numerator, denominator, default=0.0):
    if denominator == 0:
        return default
    return numerator / denominator


def load_arm_records(arm: str) -> list:
    arm_dir = os.path.join(GEN_DIR, arm)
    if not os.path.isdir(arm_dir):
        print(f"[WARN] No cache directory found for arm '{arm}': {arm_dir}")
        return []
    records = []
    for fname in sorted(os.listdir(arm_dir)):
        if fname.startswith("case_") and fname.endswith(".json"):
            fpath = os.path.join(arm_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                records.append(json.load(f))
    return records


# ---- Table 2: Faithfulness --------------------------------------------------

def compute_faithfulness_row(arm: str, records: list) -> dict:
    """
    Compute per-arm faithfulness metrics from verification_result fields.

    Metrics
    -------
    directional_recall      : fraction of directional checks that are correct
                              (avg over cases that have ≥1 directional check)
    hallucinated_feature_rate: fraction of cases with ≥1 hallucinated feature
    numerical_accuracy       : fraction of extracted numbers that match evidence
                               (avg over cases that had ≥1 number extracted)
    recourse_fidelity        : fraction of recourse actions verified
                               (avg over cases where recourse was applicable)
    post_verifier_violation_rate: fraction of final stored records that FAILED
                                  the verifier (after retry + fallback)
    n_cases                  : number of cases evaluated
    """
    dir_recalls         = []
    halluc_case_flags   = []
    num_accuracies      = []
    recourse_fidelities = []
    violations          = 0
    n                   = len(records)

    for rec in records:
        vr = rec.get("verification_result", {})

        # Directional recall
        total_dir = vr.get("directional_checks", 0)
        if total_dir > 0:
            dir_recalls.append(safe_div(vr.get("directional_correct", 0), total_dir))

        # Hallucination: at least one hallucinated feature?
        halluc_case_flags.append(1 if vr.get("hallucinated_features") else 0)

        # Numerical accuracy
        nums_extracted = vr.get("numbers_extracted", 0)
        if nums_extracted > 0:
            num_accuracies.append(safe_div(vr.get("numbers_matched", 0), nums_extracted))

        # Recourse fidelity (only for cases where recourse was applicable)
        if vr.get("recourse_checks_applicable"):
            fid = vr.get("recourse_fidelity")
            if fid is not None:
                recourse_fidelities.append(float(fid))

        # Post-verifier violation (final stored text)
        if not vr.get("passed", False):
            violations += 1

    return {
        "Arm":                         ARM_DISPLAY[arm],
        "N_Cases":                     n,
        "Directional_Recall":          round(sum(dir_recalls) / len(dir_recalls), 4) if dir_recalls else "N/A",
        "Hallucinated_Feature_Rate":   round(sum(halluc_case_flags) / n, 4) if n else "N/A",
        "Numerical_Accuracy":          round(sum(num_accuracies) / len(num_accuracies), 4) if num_accuracies else "N/A",
        "Recourse_Fidelity":           round(sum(recourse_fidelities) / len(recourse_fidelities), 4) if recourse_fidelities else "N/A",
        "Post_Verifier_Violation_Rate":round(violations / n, 4) if n else "N/A",
    }


# ---- Table 3: Readability ---------------------------------------------------

def compute_readability_row(arm: str, records: list) -> dict:
    fk_grades     = []
    fk_ease       = []
    word_counts   = []

    for rec in records:
        text = rec.get("text", "")
        if not text:
            continue
        word_counts.append(rec.get("word_count", len(text.split())))
        if TEXTSTAT_AVAILABLE:
            try:
                fk_grades.append(textstat.flesch_kincaid_grade(text))
                fk_ease.append(textstat.flesch_reading_ease(text))
            except Exception:
                pass

    n = len(records)
    return {
        "Arm":                   ARM_DISPLAY[arm],
        "N_Cases":               n,
        "Flesch_Kincaid_Grade":  round(sum(fk_grades) / len(fk_grades), 2) if fk_grades else "N/A",
        "Flesch_Reading_Ease":   round(sum(fk_ease) / len(fk_ease), 2) if fk_ease else "N/A",
        "Mean_Word_Count":       round(sum(word_counts) / len(word_counts), 1) if word_counts else "N/A",
    }


# ---- Verifier Ablation ------------------------------------------------------

def compute_ablation_row(arm: str, records: list) -> dict:
    """
    Pre-verifier violation rate: fraction of cases where the first LLM
    response (before retry/fallback) failed.  For template arm this equals
    the post-verifier rate (no retry).

    We recover the pre-verifier state from retry_triggered and fallback_triggered:
      - If retry_triggered=True → the first LLM call failed (pre=FAIL)
      - If retry_triggered=False and fallback_triggered=False → first call passed (pre=PASS)
      - If fallback_triggered=True → retry also failed (post=FAIL from template)
    """
    pre_violations  = 0
    post_violations = 0
    retry_count     = 0
    fallback_count  = 0
    n               = len(records)

    for rec in records:
        vr             = rec.get("verification_result", {})
        retry_trig     = rec.get("retry_triggered", False)
        fallback_trig  = rec.get("fallback_triggered", False)

        # Count pre-verifier violations
        if arm == "deterministic_template":
            if not vr.get("passed", False):
                pre_violations += 1
        else:
            if retry_trig:
                pre_violations += 1

        # Count post-verifier violations
        if not vr.get("passed", False):
            post_violations += 1

        if retry_trig:
            retry_count += 1
        if fallback_trig:
            fallback_count += 1

    return {
        "Arm":                        ARM_DISPLAY[arm],
        "N_Cases":                    n,
        "Pre_Verifier_Violation_Rate":round(pre_violations / n, 4) if n else "N/A",
        "Post_Verifier_Violation_Rate":round(post_violations / n, 4) if n else "N/A",
        "Retry_Count":                retry_count,
        "Fallback_Count":             fallback_count,
    }


# ---- CSV writer -------------------------------------------------------------

def write_csv(path: str, rows: list):
    if not rows:
        print(f"[WARN] No rows to write for {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}  ({len(rows)} rows)")


# ---- Main -------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Evaluation Metrics — 4-Arm Pipeline")
    print("=" * 70)

    table2_rows    = []
    table3_rows    = []
    ablation_rows  = []

    total_cases_evaluated = 0

    for arm in ARMS:
        print(f"\n[ARM] {ARM_DISPLAY[arm]}")
        records = load_arm_records(arm)
        if not records:
            print(f"  [SKIP] No cached records found — run generator_pipeline.py first.")
            continue
        print(f"  Records loaded: {len(records)}")
        total_cases_evaluated += len(records)

        faith_row    = compute_faithfulness_row(arm, records)
        readab_row   = compute_readability_row(arm, records)
        ablat_row    = compute_ablation_row(arm, records)

        table2_rows.append(faith_row)
        table3_rows.append(readab_row)
        ablation_rows.append(ablat_row)

        # Print summary
        print(f"  Directional Recall      : {faith_row['Directional_Recall']}")
        print(f"  Hallucinated Feat. Rate : {faith_row['Hallucinated_Feature_Rate']}")
        print(f"  Numerical Accuracy      : {faith_row['Numerical_Accuracy']}")
        print(f"  Recourse Fidelity       : {faith_row['Recourse_Fidelity']}")
        print(f"  Post-Verifier Violation : {faith_row['Post_Verifier_Violation_Rate']}")
        print(f"  FK Grade                : {readab_row['Flesch_Kincaid_Grade']}")
        print(f"  Reading Ease            : {readab_row['Flesch_Reading_Ease']}")
        print(f"  Mean Word Count         : {readab_row['Mean_Word_Count']}")

    # Save tables
    print(f"\n{'='*70}")
    print("Saving paper tables ...")

    table2_path   = os.path.join(EVAL_DIR, "paper_table2_faithfulness.csv")
    table3_path   = os.path.join(EVAL_DIR, "paper_table3_readability.csv")
    ablation_path = os.path.join(EVAL_DIR, "verifier_ablation_summary.csv")

    write_csv(table2_path,   table2_rows)
    write_csv(table3_path,   table3_rows)
    write_csv(ablation_path, ablation_rows)

    print(f"\nDone. Total case-arm records evaluated: {total_cases_evaluated}")
    print("Paper Tables 2, 3, and Verifier Ablation Summary are ready.")


if __name__ == "__main__":
    main()
