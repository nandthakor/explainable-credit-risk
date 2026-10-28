# Credit Risk XAI - Phase 6: Rule-Based Faithfulness Evaluation
# ===============================================================
# For each of the 15 classification explanations (5 instances x 3 arms),
# checks whether the LLM's explanation faithfully references the
# ground-truth features from SHAP and/or counterfactual data.
#
# Two checks:
#   1. Top-feature recall: does the explanation mention the features
#      that the XAI method identified as important?
#   2. Arm contamination: does a SHAP-only explanation accidentally
#      include counterfactual-style advice, or vice versa?
#
# Run: python src/faithfulness_rulebased.py

import os
import sys
import json
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_aliases import FEATURE_ALIASES

# ---- Paths ------------------------------------------------------------------
REPORTS     = "reports"
SHAP_LOCAL  = os.path.join(REPORTS, "shap_local")
CF_DIR      = os.path.join(REPORTS, "counterfactuals")
LLM_DIR     = os.path.join(REPORTS, "llm_explanations")
EVAL_DIR    = os.path.join(REPORTS, "evaluation")

os.makedirs(EVAL_DIR, exist_ok=True)

# ---- Contamination phrase lists ---------------------------------------------
# Phrases that indicate counterfactual-style advice (should NOT appear in
# shap_only explanations):
CF_CONTAMINATION_PHRASES = [
    "if you", "if the applicant", "would change", "in order to improve",
    "to qualify", "to improve", "could do", "steps to", "to change the",
    "what the applicant", "to flip", "to reverse", "to get approved",
    "action steps", "what you can do",
]

# Phrases that indicate SHAP-style reasoning (should NOT appear in
# cf_only explanations):
SHAP_CONTAMINATION_PHRASES = [
    "contributed", "was a factor", "influenced the decision",
    "top reason", "main reason", "key factor", "contributing factor",
    "most important factor", "biggest factor", "primary driver",
    "drove the decision", "pushed the risk",
]


def feature_mentioned(feature_name, text_lower):
    """
    Check if any alias of the given feature appears in the text.
    Returns True if at least one alias is found.
    """
    aliases = FEATURE_ALIASES.get(feature_name, [feature_name])
    for alias in aliases:
        if alias.lower() in text_lower:
            return True
    return False


def get_ground_truth_features(instance_id, case_type, arm):
    """
    Load the ground-truth features that the explanation SHOULD mention.

    - shap_only: top-3 SHAP features by |shap_value|
    - cf_only: changed features from the best feasible counterfactual
    - combined: both SHAP top-3 AND CF changed features (deduplicated)
    """
    features = set()

    # SHAP features (for shap_only and combined)
    if arm in ("shap_only", "combined"):
        shap_path = os.path.join(SHAP_LOCAL,
                                 f"{instance_id}_{case_type}.json")
        with open(shap_path) as f:
            shap_records = json.load(f)
        # Top 3 by |shap_value| (already sorted descending in the file)
        for rec in shap_records[:3]:
            features.add(rec["feature_name"])

    # CF features (for cf_only and combined)
    if arm in ("cf_only", "combined"):
        cf_path = os.path.join(CF_DIR,
                               f"{instance_id}_{case_type}_feasible.json")
        with open(cf_path) as f:
            feasible_cfs = json.load(f)
        if feasible_cfs:
            # Pick the best (lowest proximity) feasible CF
            best = min(feasible_cfs, key=lambda c: c["proximity_score"])
            for ch in best["changed_features"]:
                features.add(ch["feature_name"])

    return features


def check_contamination(arm, text_lower):
    """
    Check if the explanation text contains phrases from the WRONG arm.
    Returns (is_contaminated: bool, phrases_found: list).
    """
    phrases_found = []

    if arm == "shap_only":
        # Should NOT contain CF-style advice phrases
        for phrase in CF_CONTAMINATION_PHRASES:
            if phrase.lower() in text_lower:
                phrases_found.append(phrase)

    elif arm == "cf_only":
        # Should NOT contain SHAP-style reasoning phrases
        for phrase in SHAP_CONTAMINATION_PHRASES:
            if phrase.lower() in text_lower:
                phrases_found.append(phrase)

    # combined arm: no contamination check (it's allowed to have both)

    return len(phrases_found) > 0, phrases_found


def run_faithfulness_check():
    """Evaluate all 15 classification explanations."""
    print("\n" + "=" * 60)
    print("RULE-BASED FAITHFULNESS EVALUATION")
    print("=" * 60)

    # Load selected instances to get the 5 classification IDs
    with open(os.path.join(REPORTS, "selected_instances.json")) as f:
        all_insts = json.load(f)
    clf_insts = [i for i in all_insts if i["task"] == "classification"]

    arms = ["shap_only", "cf_only", "combined"]
    results = []

    for inst in clf_insts:
        iid = inst["instance_id"]
        case = inst["case_type"]

        for arm in arms:
            # Load the LLM explanation
            llm_path = os.path.join(LLM_DIR, f"{iid}_{case}_{arm}.json")
            with open(llm_path, encoding="utf-8") as f:
                llm_data = json.load(f)
            text_lower = llm_data["raw_response"].lower()

            # Get ground-truth features
            gt_features = get_ground_truth_features(iid, case, arm)

            # Compute top-feature recall
            if gt_features:
                mentioned = sum(1 for feat in gt_features
                                if feature_mentioned(feat, text_lower))
                recall = mentioned / len(gt_features)
            else:
                # No ground truth (e.g., no feasible CFs for cf_only)
                mentioned = 0
                recall = None  # undefined

            # Check contamination
            is_contaminated, bad_phrases = check_contamination(arm, text_lower)

            results.append({
                "instance_id":      iid,
                "case_type":        case,
                "arm":              arm,
                "ground_truth_features": sorted(gt_features) if gt_features else [],
                "features_mentioned":    mentioned,
                "features_expected":     len(gt_features),
                "top_feature_recall":    round(recall, 4) if recall is not None else "",
                "contamination_flag":    is_contaminated,
                "contaminating_phrases_found": bad_phrases,
            })

            recall_str = f"{recall:.2%}" if recall is not None else "N/A (no GT)"
            contam_str = "CONTAMINATED" if is_contaminated else "clean"
            print(f"  id={iid:3d}  {case:18s}  {arm:10s}  "
                  f"recall={recall_str:>8s}  ({mentioned}/{len(gt_features)})  "
                  f"{contam_str}")
            if bad_phrases:
                print(f"    contaminating phrases: {bad_phrases}")

    # Save to CSV
    csv_path = os.path.join(EVAL_DIR, "rulebased_faithfulness.csv")
    fieldnames = [
        "instance_id", "case_type", "arm", "top_feature_recall",
        "contamination_flag", "contaminating_phrases_found",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "instance_id":      r["instance_id"],
                "case_type":        r["case_type"],
                "arm":              r["arm"],
                "top_feature_recall": r["top_feature_recall"],
                "contamination_flag": r["contamination_flag"],
                "contaminating_phrases_found": json.dumps(r["contaminating_phrases_found"]),
            })
    print(f"\n  [saved] {csv_path}")

    # Print summary
    print("\n  --- Summary by arm ---")
    for arm in arms:
        arm_results = [r for r in results if r["arm"] == arm]
        recalls = [r["top_feature_recall"] for r in arm_results
                   if r["top_feature_recall"] != ""]
        contam_count = sum(1 for r in arm_results if r["contamination_flag"])
        avg_recall = sum(recalls) / len(recalls) if recalls else 0
        print(f"  {arm:10s}:  avg recall={avg_recall:.2%}  "
              f"contaminated={contam_count}/{len(arm_results)}")


if __name__ == "__main__":
    run_faithfulness_check()
