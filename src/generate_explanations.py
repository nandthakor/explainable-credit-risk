# Credit Risk XAI - Phase 5: Three-Arm LLM Explanation Generator
# ===============================================================
# Generates natural-language explanations for all 8 case-study instances
# using three explanation "arms" for classification and one for regression:
#
#   Classification (5 instances x 3 arms = 15 calls):
#     1. shap_only   - explains WHY the decision was made (SHAP features only)
#     2. cf_only     - explains WHAT would change it (counterfactuals only)
#     3. combined    - explains both WHY and WHAT (SHAP + counterfactuals)
#
#   Regression (3 instances x 1 arm = 3 calls):
#     4. shap_only_regression - explains the income estimate (SHAP features only)
#
#   Total: 18 LLM API calls
#
# Run: python src/generate_explanations.py
# Prerequisites:
#   - .env file with GROQ_API_KEY
#   - All Phase 3 and Phase 4 output files

import os
import sys
import json
import time
from datetime import datetime, timezone

# textstat uses NLTK's cmudict for syllable counting.
# If running behind a proxy, NLTK blocks downloads by default (CWE-918).
# This env var tells NLTK to trust the proxy for corpus downloads.
os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")

import textstat

# Add project root to path so we can import llm_client
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_client import generate_explanation, MODEL_NAME

# ---- Paths ------------------------------------------------------------------
REPORTS       = os.path.join("reports")
SHAP_LOCAL    = os.path.join("reports", "shap_local")
CF_DIR        = os.path.join("reports", "counterfactuals")
LLM_DIR       = os.path.join("reports", "llm_explanations")
PROMPTS_DIR   = os.path.join("prompts")

os.makedirs(LLM_DIR, exist_ok=True)

# ---- Constants --------------------------------------------------------------
TEMPERATURE = 0.2
# Delay between API calls to stay under Groq free-tier rate limits
API_DELAY   = 3   # seconds


# =============================================================================
# Load all input data
# =============================================================================
def load_inputs():
    """Load selected instances, SHAP local JSONs, and CF feasible JSONs."""
    print("\n" + "=" * 60)
    print("Loading inputs")
    print("=" * 60)

    with open(os.path.join(REPORTS, "selected_instances.json")) as f:
        all_instances = json.load(f)

    clf_instances = [i for i in all_instances if i["task"] == "classification"]
    reg_instances = [i for i in all_instances if i["task"] == "regression"]

    print(f"  Classification instances: {len(clf_instances)}")
    print(f"  Regression instances:     {len(reg_instances)}")

    # Load SHAP local JSONs
    shap_data = {}
    for inst in all_instances:
        key = f"{inst['instance_id']}_{inst['case_type']}"
        path = os.path.join(SHAP_LOCAL, f"{key}.json")
        with open(path) as f:
            shap_data[key] = json.load(f)
    print(f"  SHAP local JSONs loaded: {len(shap_data)}")

    # Load CF feasible JSONs (classification only)
    cf_data = {}
    for inst in clf_instances:
        key = f"{inst['instance_id']}_{inst['case_type']}"
        path = os.path.join(CF_DIR, f"{key}_feasible.json")
        with open(path) as f:
            cf_data[key] = json.load(f)
    print(f"  CF feasible JSONs loaded: {len(cf_data)}")

    return clf_instances, reg_instances, shap_data, cf_data


# =============================================================================
# Load prompt templates
# =============================================================================
def load_templates():
    """Load the four plain-text prompt templates."""
    templates = {}
    template_files = {
        "shap_only":            "shap_only_template.txt",
        "cf_only":              "cf_only_template.txt",
        "combined":             "combined_template.txt",
        "shap_only_regression": "shap_only_regression_template.txt",
    }
    for name, fname in template_files.items():
        path = os.path.join(PROMPTS_DIR, fname)
        with open(path) as f:
            templates[name] = f.read()
        print(f"  Loaded template: {name} ({len(templates[name])} chars)")
    return templates


# =============================================================================
# Format SHAP features into human-readable text
# =============================================================================
def format_shap_features(shap_records, top_n=5):
    """
    Convert SHAP JSON records into a numbered, readable list for the prompt.
    Example output:
      1. Revolving credit utilization (value: 1.00) - strongly increases
         the likelihood of default (SHAP impact: +1.41)
    """
    lines = []
    for i, rec in enumerate(shap_records[:top_n], 1):
        feat = rec["feature_name"].replace("_", " ")
        val  = rec["feature_value"]
        shap = rec["shap_value"]

        # Human-readable direction
        if shap > 0:
            direction = "increases"
        else:
            direction = "decreases"

        # Strength descriptor
        abs_shap = abs(shap)
        if abs_shap > 0.5:
            strength = "strongly"
        elif abs_shap > 0.1:
            strength = "moderately"
        else:
            strength = "slightly"

        lines.append(
            f"  {i}. {feat.title()} (value: {val:.4g}) - "
            f"{strength} {direction} the prediction "
            f"(impact: {shap:+.4f})"
        )

    return "\n".join(lines)


# =============================================================================
# Format CF changed features into human-readable text
# =============================================================================
def format_cf_features(feasible_cfs):
    """
    Pick the best (lowest-proximity) feasible CF and format its changes
    into a readable list for the prompt.

    If no feasible CFs exist, return a note saying so.
    """
    if not feasible_cfs:
        return "(No feasible changes were found that would flip the decision.)"

    # Pick the best CF (lowest proximity = smallest total change)
    best = min(feasible_cfs, key=lambda c: c["proximity_score"])

    lines = []
    for ch in best["changed_features"]:
        feat = ch["feature_name"].replace("_", " ")
        orig = ch["original_value"]
        new  = ch["cf_value"]
        direction = ch["direction"]

        lines.append(
            f"  - {feat.title()}: {orig:.4g} -> {new:.4g} ({direction})"
        )

    return "\n".join(lines)


# =============================================================================
# Compute readability metrics
# =============================================================================
def compute_readability(text):
    """
    Compute Flesch-Kincaid Grade Level and Flesch Reading Ease.

    - FK Grade Level: the US school grade needed to understand the text.
      Lower = easier. A score of 8 means an 8th-grader could understand it.
    - Flesch Reading Ease: 0-100 scale. Higher = easier reading.
      60-70 = standard, 70-80 = fairly easy, 80-90 = easy.
    """
    fk_grade = textstat.flesch_kincaid_grade(text)
    fk_ease  = textstat.flesch_reading_ease(text)
    word_count = textstat.lexicon_count(text, removepunct=True)
    return fk_grade, fk_ease, word_count


# =============================================================================
# Generate and save one explanation
# =============================================================================
def generate_and_save(
    instance_id, case_type, arm, prompt, call_num, total_calls
):
    """
    Call the LLM, compute readability, and save the result JSON.
    Returns the result dict.
    """
    print(f"\n  [{call_num}/{total_calls}] id={instance_id} "
          f"case={case_type} arm={arm}")

    # Call the LLM
    t0 = time.time()
    raw_response = generate_explanation(prompt, temperature=TEMPERATURE)
    elapsed = time.time() - t0
    print(f"    LLM response received ({elapsed:.1f}s)")

    # Compute readability
    fk_grade, fk_ease, word_count = compute_readability(raw_response)
    print(f"    Words: {word_count}  FK Grade: {fk_grade:.1f}  "
          f"Reading Ease: {fk_ease:.1f}")

    # Build result
    result = {
        "instance_id":         instance_id,
        "case_type":           case_type,
        "arm":                 arm,
        "model_name":          MODEL_NAME,
        "temperature":         TEMPERATURE,
        "prompt_used":         prompt,
        "raw_response":        raw_response,
        "flesch_kincaid_grade": round(fk_grade, 2),
        "flesch_reading_ease":  round(fk_ease, 2),
        "word_count":          word_count,
        "timestamp":           datetime.now(timezone.utc).isoformat(),
    }

    # Save
    fname = f"{instance_id}_{case_type}_{arm}.json"
    path  = os.path.join(LLM_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"    [saved] {path}")

    return result


# =============================================================================
# Main pipeline
# =============================================================================
def run_explanation_pipeline():
    """Generate all 18 LLM explanations across the three arms."""

    # Load inputs
    clf_instances, reg_instances, shap_data, cf_data = load_inputs()
    templates = load_templates()

    total_calls = len(clf_instances) * 3 + len(reg_instances) * 1
    print(f"\n  Total API calls planned: {total_calls}")

    all_results = []
    call_num = 0

    # ---- Classification instances: 3 arms each -----------------------------
    print("\n" + "=" * 60)
    print("CLASSIFICATION EXPLANATIONS (3 arms per instance)")
    print("=" * 60)

    for inst in clf_instances:
        iid       = inst["instance_id"]
        case_type = inst["case_type"]
        key       = f"{iid}_{case_type}"

        # Predicted label as human-readable text
        pred_class = int(inst["predicted"] if "predicted" in inst
                         else (1 if inst["predicted_proba"] >= 0.5 else 0))
        predicted_label = ("HIGH RISK (likely to default)"
                           if pred_class == 1
                           else "LOW RISK (unlikely to default)")
        confidence = f"{inst['predicted_proba']:.1%}"

        # Format SHAP features
        shap_formatted = format_shap_features(shap_data[key])

        # Format CF features
        cf_formatted = format_cf_features(cf_data[key])

        # --- Arm 1: SHAP only -----------------------------------------------
        call_num += 1
        prompt = templates["shap_only"].format(
            predicted_label=predicted_label,
            confidence=confidence,
            shap_features_formatted=shap_formatted,
        )
        result = generate_and_save(iid, case_type, "shap_only",
                                    prompt, call_num, total_calls)
        all_results.append(result)
        time.sleep(API_DELAY)

        # --- Arm 2: CF only -------------------------------------------------
        call_num += 1
        prompt = templates["cf_only"].format(
            predicted_label=predicted_label,
            cf_features_formatted=cf_formatted,
        )
        result = generate_and_save(iid, case_type, "cf_only",
                                    prompt, call_num, total_calls)
        all_results.append(result)
        time.sleep(API_DELAY)

        # --- Arm 3: Combined ------------------------------------------------
        call_num += 1
        prompt = templates["combined"].format(
            predicted_label=predicted_label,
            confidence=confidence,
            shap_features_formatted=shap_formatted,
            cf_features_formatted=cf_formatted,
        )
        result = generate_and_save(iid, case_type, "combined",
                                    prompt, call_num, total_calls)
        all_results.append(result)
        time.sleep(API_DELAY)

    # ---- Regression instances: 1 arm each -----------------------------------
    print("\n" + "=" * 60)
    print("REGRESSION EXPLANATIONS (shap_only_regression)")
    print("=" * 60)

    for inst in reg_instances:
        iid       = inst["instance_id"]
        case_type = inst["case_type"]
        key       = f"{iid}_{case_type}"

        predicted_income = f"${inst['predicted']:,.0f}/month"
        actual_income    = f"${inst['actual']:,.0f}/month"

        shap_formatted = format_shap_features(shap_data[key])

        call_num += 1
        prompt = templates["shap_only_regression"].format(
            predicted_income=predicted_income,
            actual_income=actual_income,
            shap_features_formatted=shap_formatted,
        )
        result = generate_and_save(iid, case_type, "shap_only_regression",
                                    prompt, call_num, total_calls)
        all_results.append(result)
        time.sleep(API_DELAY)

    # ---- Write summary report -----------------------------------------------
    write_summary_report(all_results, clf_instances)

    print("\n" + "=" * 60)
    print("PHASE 5 COMPLETE")
    print("=" * 60)
    print(f"  {len(all_results)} explanations generated and saved")
    print(f"  Output: reports/llm_explanations/*.json")
    print(f"  Report: reports/llm_generation_summary.md")


# =============================================================================
# Summary report
# =============================================================================
def write_summary_report(all_results, clf_instances):
    """Write reports/llm_generation_summary.md with real metrics."""
    print("\n" + "=" * 60)
    print("Writing llm_generation_summary.md")
    print("=" * 60)

    lines = [
        "# LLM Explanation Generation Summary - Phase 5",
        "",
        "> All explanations were generated by actual Groq API calls to",
        f"> {MODEL_NAME} at temperature={TEMPERATURE}.",
        "> No outputs are fabricated.",
        "",
        "## All Generated Explanations",
        "",
        "| Instance ID | Case Type | Arm | Word Count | FK Grade | Reading Ease |",
        "|---|---|---|---|---|---|",
    ]

    for r in all_results:
        lines.append(
            f"| {r['instance_id']} | {r['case_type']} | {r['arm']} "
            f"| {r['word_count']} | {r['flesch_kincaid_grade']:.1f} "
            f"| {r['flesch_reading_ease']:.1f} |"
        )

    # ---- Average FK grade per arm (classification only) ---------------------
    lines += [
        "",
        "## Average Flesch-Kincaid Grade by Arm (Classification Only)",
        "",
        "> Lower grade level generally indicates easier reading.",
        "> These are measured values; interpretation is deferred to Phase 6.",
        "",
    ]

    clf_arms = ["shap_only", "cf_only", "combined"]
    arm_grades = {arm: [] for arm in clf_arms}
    arm_ease   = {arm: [] for arm in clf_arms}

    for r in all_results:
        if r["arm"] in clf_arms:
            arm_grades[r["arm"]].append(r["flesch_kincaid_grade"])
            arm_ease[r["arm"]].append(r["flesch_reading_ease"])

    lines.append("| Arm | Avg FK Grade | Avg Reading Ease | N |")
    lines.append("|---|---|---|---|")
    for arm in clf_arms:
        if arm_grades[arm]:
            avg_g = sum(arm_grades[arm]) / len(arm_grades[arm])
            avg_e = sum(arm_ease[arm]) / len(arm_ease[arm])
            lines.append(f"| {arm} | {avg_g:.2f} | {avg_e:.2f} | {len(arm_grades[arm])} |")

    # ---- Side-by-side comparison for one instance ---------------------------
    # Pick the first classification instance for the comparison
    example_id   = clf_instances[0]["instance_id"]
    example_case = clf_instances[0]["case_type"]

    lines += [
        "",
        "---",
        "",
        f"## Side-by-Side Comparison: Instance {example_id} "
        f"({example_case.replace('_', ' ').title()})",
        "",
        "> The same instance explained by all three arms, "
        "to illustrate the differences.",
        "",
    ]

    for arm in clf_arms:
        matching = [r for r in all_results
                    if r["instance_id"] == example_id and r["arm"] == arm]
        if matching:
            r = matching[0]
            arm_label = arm.replace("_", " ").title()
            lines += [
                f"### {arm_label}",
                "",
                f"> FK Grade: {r['flesch_kincaid_grade']:.1f} | "
                f"Reading Ease: {r['flesch_reading_ease']:.1f} | "
                f"Words: {r['word_count']}",
                "",
                r["raw_response"],
                "",
            ]

    report_path = os.path.join(REPORTS, "llm_generation_summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [saved] {report_path}")


# =============================================================================
# Entry point
# =============================================================================
# Phase 7 — Live generation helper (used by app.py)
# =============================================================================
def generate_all_three_arms(
    feature_dict: dict,
    predicted_label: str,
    confidence: str,
    shap_records: list,
    feasible_cfs: list,
    temperature: float = 0.2,
) -> dict:
    """
    Generate all three explanation arms for a live applicant.
    Reuses the existing prompt templates and llm_client.generate_explanation().
    Called by the Streamlit app after SHAP and CF results are available.

    Parameters
    ----------
    feature_dict    : dict of feature name -> value (for context only)
    predicted_label : human-readable predicted outcome string
    confidence      : e.g. "78.4%"
    shap_records    : list of {feature_name, feature_value, shap_value}
    feasible_cfs    : list of feasible CF dicts (may be empty)
    temperature     : LLM temperature (default 0.2)

    Returns
    -------
    dict with keys "shap_only", "cf_only", "combined",
    each a dict: {text, flesch_kincaid_grade, flesch_reading_ease, word_count}
    """
    # Load templates (same files used in Phase 5)
    templates = load_templates()

    shap_formatted = format_shap_features(shap_records, top_n=5)
    cf_formatted   = format_cf_features(feasible_cfs)

    results = {}

    # --- Arm 1: shap_only ---
    prompt_shap = templates["shap_only"].format(
        predicted_label=predicted_label,
        confidence=confidence,
        shap_features_formatted=shap_formatted,
    )
    text_shap = generate_explanation(prompt_shap, temperature=temperature)
    fk_g, fk_e, wc = compute_readability(text_shap)
    results["shap_only"] = {
        "text": text_shap,
        "flesch_kincaid_grade": round(fk_g, 2),
        "flesch_reading_ease":  round(fk_e, 2),
        "word_count":           wc,
    }

    # --- Arm 2: cf_only ---
    prompt_cf = templates["cf_only"].format(
        predicted_label=predicted_label,
        cf_features_formatted=cf_formatted,
    )
    text_cf = generate_explanation(prompt_cf, temperature=temperature)
    fk_g, fk_e, wc = compute_readability(text_cf)
    results["cf_only"] = {
        "text": text_cf,
        "flesch_kincaid_grade": round(fk_g, 2),
        "flesch_reading_ease":  round(fk_e, 2),
        "word_count":           wc,
    }

    # --- Arm 3: combined ---
    prompt_combined = templates["combined"].format(
        predicted_label=predicted_label,
        confidence=confidence,
        shap_features_formatted=shap_formatted,
        cf_features_formatted=cf_formatted,
    )
    text_combined = generate_explanation(prompt_combined, temperature=temperature)
    fk_g, fk_e, wc = compute_readability(text_combined)
    results["combined"] = {
        "text": text_combined,
        "flesch_kincaid_grade": round(fk_g, 2),
        "flesch_reading_ease":  round(fk_e, 2),
        "word_count":           wc,
    }

    return results


# =============================================================================
if __name__ == "__main__":
    run_explanation_pipeline()
