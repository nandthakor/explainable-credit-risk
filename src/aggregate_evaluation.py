# Credit Risk XAI - Phase 6: Aggregate Evaluation
# ==================================================
# Pulls together all evaluation signals into one master comparison table:
#   - Rule-based faithfulness (top-feature recall, contamination)
#   - LLM judge scores (faithfulness, understandability, actionability)
#   - Flesch-Kincaid readability (from Phase 5)
#   - Counterfactual quality (validity, feasibility from Phase 4)
#   - Human survey responses (if collected)
#
# Run: python src/aggregate_evaluation.py

import os
import json
import csv
import numpy as np

# ---- Paths ------------------------------------------------------------------
REPORTS   = "reports"
EVAL_DIR  = os.path.join(REPORTS, "evaluation")
LLM_DIR   = os.path.join(REPORTS, "llm_explanations")
CF_DIR    = os.path.join(REPORTS, "counterfactuals")
RESP_DIR  = os.path.join(EVAL_DIR, "survey_responses")

ARMS = ["shap_only", "cf_only", "combined"]


def load_rulebased():
    """Load rule-based faithfulness CSV."""
    path = os.path.join(EVAL_DIR, "rulebased_faithfulness.csv")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["top_feature_recall"] = (
                float(row["top_feature_recall"])
                if row["top_feature_recall"] != ""
                else None
            )
            row["contamination_flag"] = row["contamination_flag"] == "True"
            rows.append(row)
    return rows


def load_judge_scores():
    """Load LLM judge scores CSV."""
    path = os.path.join(EVAL_DIR, "llm_judge_scores.csv")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ["faithfulness_score", "understandability_score",
                        "actionability_score"]:
                row[key] = float(row[key])
            rows.append(row)
    return rows


def load_flesch_kincaid():
    """Load FK grades from Phase 5 explanation JSONs."""
    fk_data = []
    for fname in os.listdir(LLM_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(LLM_DIR, fname)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data["arm"] in ARMS:
            fk_data.append({
                "instance_id": data["instance_id"],
                "case_type":   data["case_type"],
                "arm":         data["arm"],
                "flesch_kincaid_grade": data["flesch_kincaid_grade"],
            })
    return fk_data


def load_cf_quality():
    """
    Load counterfactual validity and feasibility rates from Phase 4 JSONs.
    Returns per-instance rates for cf_only and combined arms.
    """
    with open(os.path.join(REPORTS, "selected_instances.json")) as f:
        all_insts = json.load(f)
    clf_insts = [i for i in all_insts if i["task"] == "classification"]

    cf_quality = []
    for inst in clf_insts:
        iid = inst["instance_id"]
        case = inst["case_type"]

        # Load raw CFs for validity
        raw_path = os.path.join(CF_DIR, f"{iid}_{case}_raw.json")
        with open(raw_path) as f:
            raw_cfs = json.load(f)

        # Load feasible CFs
        feas_path = os.path.join(CF_DIR, f"{iid}_{case}_feasible.json")
        with open(feas_path) as f:
            feas_cfs = json.load(f)

        n_raw = len(raw_cfs)
        n_valid = sum(1 for c in raw_cfs if c["validity"])
        n_feasible = len(feas_cfs)

        validity_rate = n_valid / n_raw if n_raw > 0 else 0
        feasibility_rate = n_feasible / n_raw if n_raw > 0 else 0

        cf_quality.append({
            "instance_id":      iid,
            "case_type":        case,
            "validity_rate":    validity_rate,
            "feasibility_rate": feasibility_rate,
        })

    return cf_quality


def load_survey_responses():
    """
    Load and un-blind survey responses, if any exist.
    Returns (responses_list, arm_scores, preference_votes) or
    (None, None, None) if no responses found.
    """
    if not os.path.exists(RESP_DIR):
        return None, None, None

    response_files = [f for f in os.listdir(RESP_DIR) if f.endswith(".json")]
    if not response_files:
        return None, None, None

    # Load survey key
    with open(os.path.join(EVAL_DIR, "survey_key.json")) as f:
        survey_key = json.load(f)

    # Aggregate scores per arm
    arm_scores = {arm: {"understanding": [], "trust": [], "knows_next_step": []}
                  for arm in ARMS}
    preference_votes = {arm: 0 for arm in ARMS}

    n_responses = 0
    for fname in response_files:
        path = os.path.join(RESP_DIR, fname)
        with open(path, encoding="utf-8") as f:
            resp = json.load(f)
        n_responses += 1

        for iid_str, mapping in survey_key.items():
            if iid_str not in resp:
                continue
            inst_resp = resp[iid_str]

            # Un-blind the labels -> arms
            for label in ["A", "B", "C"]:
                arm = mapping[label]
                if label in inst_resp and isinstance(inst_resp[label], dict):
                    for dim in ["understanding", "trust", "knows_next_step"]:
                        val = inst_resp[label].get(dim)
                        if val is not None:
                            arm_scores[arm][dim].append(val)

            # Preference vote
            pref_label = inst_resp.get("most_helpful")
            if pref_label and pref_label in mapping:
                pref_arm = mapping[pref_label]
                preference_votes[pref_arm] += 1

    print(f"  Loaded {n_responses} survey response(s)")
    return n_responses, arm_scores, preference_votes


def build_master_table():
    """Build the ONE master comparison table, aggregated per arm."""
    print("\n" + "=" * 60)
    print("AGGREGATING EVALUATION RESULTS")
    print("=" * 60)

    # Load all data sources
    rulebased = load_rulebased()
    judge     = load_judge_scores()
    fk_data   = load_flesch_kincaid()
    cf_qual   = load_cf_quality()

    print(f"  Rule-based rows: {len(rulebased)}")
    print(f"  Judge rows:      {len(judge)}")
    print(f"  FK data rows:    {len(fk_data)}")
    print(f"  CF quality rows: {len(cf_qual)}")

    # Check for survey responses
    n_resp, arm_scores, pref_votes = load_survey_responses()
    has_survey = n_resp is not None and n_resp > 0
    if has_survey:
        print(f"  Survey responses: {n_resp}")
    else:
        print("  No survey responses found yet -- human evaluation "
              "section will be skipped.")
        print("  Run this script again after collecting responses.")

    # Build per-arm aggregates
    master = []
    for arm in ARMS:
        row = {"arm": arm}

        # Rule-based: avg top-feature recall
        arm_rb = [r for r in rulebased if r["arm"] == arm]
        recalls = [r["top_feature_recall"] for r in arm_rb
                   if r["top_feature_recall"] is not None]
        row["avg_top_feature_recall"] = (
            round(np.mean(recalls), 4) if recalls else ""
        )

        # Rule-based: contamination rate
        contam_count = sum(1 for r in arm_rb if r["contamination_flag"])
        row["contamination_rate"] = (
            round(contam_count / len(arm_rb), 4) if arm_rb else ""
        )

        # LLM judge: averages
        arm_j = [r for r in judge if r["arm"] == arm]
        for dim in ["faithfulness", "understandability", "actionability"]:
            key = f"{dim}_score"
            vals = [r[key] for r in arm_j]
            row[f"avg_llm_{dim}"] = (
                round(np.mean(vals), 2) if vals else ""
            )

        # Flesch-Kincaid grade
        arm_fk = [r for r in fk_data if r["arm"] == arm]
        fk_vals = [r["flesch_kincaid_grade"] for r in arm_fk]
        row["avg_flesch_kincaid_grade"] = (
            round(np.mean(fk_vals), 2) if fk_vals else ""
        )

        # CF quality (only for cf_only and combined — they USE counterfactuals)
        if arm in ("cf_only", "combined"):
            val_rates = [c["validity_rate"] for c in cf_qual]
            feas_rates = [c["feasibility_rate"] for c in cf_qual]
            row["avg_cf_validity_rate"] = round(np.mean(val_rates), 4)
            row["avg_cf_feasibility_rate"] = round(np.mean(feas_rates), 4)
        else:
            row["avg_cf_validity_rate"] = "N/A"
            row["avg_cf_feasibility_rate"] = "N/A"

        # Human survey (if available)
        if has_survey:
            for dim in ["understanding", "trust", "knows_next_step"]:
                vals = arm_scores[arm][dim]
                row[f"avg_human_{dim}"] = (
                    round(np.mean(vals), 2) if vals else ""
                )
            total_votes = sum(pref_votes.values())
            row["pct_most_helpful_votes"] = (
                round(pref_votes[arm] / total_votes * 100, 1)
                if total_votes > 0 else ""
            )
        else:
            row["avg_human_understanding"] = "pending"
            row["avg_human_trust"] = "pending"
            row["avg_human_knows_next_step"] = "pending"
            row["pct_most_helpful_votes"] = "pending"

        master.append(row)

    # Save as CSV
    csv_path = os.path.join(EVAL_DIR, "master_comparison_table.csv")
    fieldnames = [
        "arm", "avg_top_feature_recall", "contamination_rate",
        "avg_llm_faithfulness", "avg_llm_understandability",
        "avg_llm_actionability", "avg_flesch_kincaid_grade",
        "avg_cf_validity_rate", "avg_cf_feasibility_rate",
        "avg_human_understanding", "avg_human_trust",
        "avg_human_knows_next_step", "pct_most_helpful_votes",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(master)
    print(f"\n  [saved] {csv_path}")

    # Save as Markdown
    md_path = os.path.join(EVAL_DIR, "master_comparison_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Master Comparison Table\n\n")
        f.write("> All numbers are real, computed from actual model outputs.\n\n")

        # Header
        f.write("| Metric | shap_only | cf_only | combined |\n")
        f.write("|---|---|---|---|\n")

        # Rows — transpose for readability
        metric_labels = {
            "avg_top_feature_recall": "Avg Top-Feature Recall",
            "contamination_rate": "Contamination Rate",
            "avg_llm_faithfulness": "Avg LLM Faithfulness (1-10)",
            "avg_llm_understandability": "Avg LLM Understandability (1-10)",
            "avg_llm_actionability": "Avg LLM Actionability (1-10)",
            "avg_flesch_kincaid_grade": "Avg Flesch-Kincaid Grade",
            "avg_cf_validity_rate": "Avg CF Validity Rate",
            "avg_cf_feasibility_rate": "Avg CF Feasibility Rate",
            "avg_human_understanding": "Avg Human Understanding (1-5)",
            "avg_human_trust": "Avg Human Trust (1-5)",
            "avg_human_knows_next_step": "Avg Human Knows Next Step (1-5)",
            "pct_most_helpful_votes": "% Most Helpful Votes",
        }

        for key, label in metric_labels.items():
            vals = []
            for arm in ARMS:
                arm_row = [r for r in master if r["arm"] == arm][0]
                v = arm_row.get(key, "")
                vals.append(str(v))
            f.write(f"| {label} | {' | '.join(vals)} |\n")

    print(f"  [saved] {md_path}")

    # Print the table
    print("\n  --- Master Comparison Table ---")
    for key, label in metric_labels.items():
        line = f"  {label:40s}"
        for arm in ARMS:
            arm_row = [r for r in master if r["arm"] == arm][0]
            v = arm_row.get(key, "")
            line += f"  {str(v):>10s}"
        print(line)

    return master


if __name__ == "__main__":
    master = build_master_table()
