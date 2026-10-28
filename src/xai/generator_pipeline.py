# =============================================================================
# 4-Arm Explanation Generator Pipeline
# =============================================================================
# Generates natural language credit-risk explanations over 50 precomputed
# evidence-contract cases across four experimental arms:
#
#   Arm 1: shap_only           — SHAP attribution only, no recourse advice
#   Arm 2: cf_only             — Counterfactual recourse only, no attribution
#   Arm 3: combined            — SHAP attribution + counterfactual recourse
#   Arm 4: deterministic_template — Zero LLM; rule-based template (baseline)
#
# Key design properties:
#   - Disk-cached: re-runs skip LLM if cache file already exists
#   - Retry logic: one structured retry if verifier rejects output (arms 1-3)
#   - Fallback: if retry also fails, replace with template output (logged)
#   - Rate-limited: 2s sleep between Groq API calls
#   - Deterministic template arm: zero LLM calls, fully reproducible
#
# Usage:
#   python src/xai/generator_pipeline.py
#
# Output:
#   reports/evaluation/raw_generations/{arm}/case_{case_id}.json

import os
import sys
import json
import time
import re
import math

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from xai.verifier import verify_explanation
from xai.template_baseline import generate_template_explanation
from llm_client import generate_explanation

# ---- Paths ------------------------------------------------------------------
EVIDENCE_DIR  = os.path.join(PROJECT_ROOT, "reports", "evaluation", "evidence_records")
GEN_DIR       = os.path.join(PROJECT_ROOT, "reports", "evaluation", "raw_generations")

ARMS = ["shap_only", "cf_only", "combined", "deterministic_template"]

API_SLEEP_SECONDS = 2.5   # ~24 req/min comfortably within Groq free tier

# ---- Feature human labels (for prompt construction) -------------------------
FEATURE_LABELS = {
    "revolving_util":         "Revolving Credit Utilization",
    "age":                    "Age",
    "late_30_59_days":        "Times 30-59 Days Past Due",
    "late_60_89_days":        "Times 60-89 Days Past Due",
    "late_90_days":           "Times 90+ Days Past Due (Serious Delinquency)",
    "debt_ratio":             "Debt-to-Income Ratio",
    "monthly_income":         "Monthly Income",
    "open_credit_lines":      "Number of Open Credit Lines & Loans",
    "real_estate_loans":      "Number of Real Estate Loans / Mortgages",
    "num_dependents":         "Number of Financial Dependents",
    "monthly_income_missing": "Income Data Flag (missing indicator)",
}

# ---- Prompt Builders --------------------------------------------------------

def _format_prob(p: float) -> str:
    return f"{p * 100:.1f}%"


def _shap_lines(evidence: dict, top_n: int = 3) -> str:
    """Return a numbered list of top-N SHAP features."""
    lines = []
    features = evidence["shap_attributions"]["top_features"][:top_n]
    for i, feat in enumerate(features, 1):
        label   = FEATURE_LABELS.get(feat["feature_name"], feat["feature_name"])
        val     = feat["feature_value"]
        shap    = feat["shap_value"]
        dirn    = "INCREASES" if feat["direction"] == "increases_risk" else "DECREASES"
        lines.append(
            f"  {i}. {label}: observed value = {val}, "
            f"SHAP = {shap:+.3f} ({dirn} default risk)"
        )
    return "\n".join(lines)


def _cf_lines(evidence: dict) -> str:
    """Return a textual summary of the best verified counterfactual (if any)."""
    cf_data    = evidence.get("counterfactual_recourse", {})
    candidates = cf_data.get("candidates", [])
    passed_cfs = [c for c in candidates if c.get("verification", {}).get("final_status") == "PASSED"]

    if not passed_cfs:
        return "  No feasible and constraint-verified recourse pathway was found for this applicant."

    # Pick the CF with lowest Gower distance
    best = min(passed_cfs, key=lambda c: c["verification"].get("gower_distance", 999))
    changes = best.get("changed_features", [])
    target_prob = best.get("cf_predicted_proba", None)

    lines = [f"  Verified recourse (target default probability: {_format_prob(target_prob) if target_prob is not None else 'below 50%'}):"]
    for ch in changes:
        feat_key = ch.get("feature_name", ch.get("feature", "unknown_feature"))
        label    = FEATURE_LABELS.get(feat_key, feat_key)
        current  = ch.get("original_value", ch.get("current_value", "?"))
        suggest  = ch.get("cf_value", ch.get("suggested_value", "?"))
        lines.append(f"    - Change {label}: {current} \u2192 {suggest}")
    return "\n".join(lines)


def build_shap_only_prompt(evidence: dict) -> str:
    pred     = evidence["prediction"]
    prob     = _format_prob(pred["predicted_proba"])
    decision = "HIGH RISK (likely to default)" if pred["predicted_class"] == 1 else "LOW RISK (unlikely to default)"
    shap_blk = _shap_lines(evidence)

    return (
        f"You are explaining an automated credit-risk model decision to a loan officer.\n\n"
        f"APPLICANT INFORMATION\n"
        f"  Model decision  : {decision}\n"
        f"  Default prob.   : {prob}\n\n"
        f"TOP FACTORS (SHAP attribution — signed contribution to default risk)\n"
        f"{shap_blk}\n\n"
        f"TASK\n"
        f"Write a clear, plain-English explanation of WHY the model produced this decision.\n"
        f"Focus ONLY on the factors listed above and their impact on the risk score.\n"
        f"DO NOT suggest any actions the applicant can take to improve their score.\n"
        f"DO NOT mention any features not listed above.\n"
        f"Length: 80-150 words."
    )


def build_cf_only_prompt(evidence: dict) -> str:
    pred     = evidence["prediction"]
    prob     = _format_prob(pred["predicted_proba"])
    decision = "HIGH RISK (likely to default)" if pred["predicted_class"] == 1 else "LOW RISK (unlikely to default)"
    cf_blk   = _cf_lines(evidence)

    return (
        f"You are advising a loan applicant on how to improve their credit-risk profile.\n\n"
        f"APPLICANT INFORMATION\n"
        f"  Current model decision : {decision}\n"
        f"  Current default prob.  : {prob}\n\n"
        f"VERIFIED RECOURSE (changes that flip the model decision, independently verified)\n"
        f"{cf_blk}\n\n"
        f"TASK\n"
        f"Write a practical, plain-English explanation of the ACTIONS the applicant can take\n"
        f"to improve their credit score and likely be approved.\n"
        f"Base your advice STRICTLY on the verified recourse above.\n"
        f"DO NOT explain WHY the model made its decision or mention SHAP values.\n"
        f"DO NOT recommend changes to the applicant's age, number of dependents, or past late payment history.\n"
        f"Length: 80-150 words."
    )


def build_combined_prompt(evidence: dict) -> str:
    pred     = evidence["prediction"]
    prob     = _format_prob(pred["predicted_proba"])
    decision = "HIGH RISK (likely to default)" if pred["predicted_class"] == 1 else "LOW RISK (unlikely to default)"
    shap_blk = _shap_lines(evidence)
    cf_blk   = _cf_lines(evidence)

    return (
        f"You are a fair and empathetic financial advisor explaining a credit decision.\n\n"
        f"APPLICANT INFORMATION\n"
        f"  Model decision  : {decision}\n"
        f"  Default prob.   : {prob}\n\n"
        f"TOP RISK FACTORS (SHAP attribution)\n"
        f"{shap_blk}\n\n"
        f"VERIFIED RECOURSE (independently checked actions to flip the decision)\n"
        f"{cf_blk}\n\n"
        f"TASK\n"
        f"Write a plain-English credit explanation that:\n"
        f"  Part 1 — Explains WHY the model reached this decision (use the SHAP factors).\n"
        f"  Part 2 — Explains WHAT the applicant can do to improve (use the recourse, if available).\n"
        f"Use only the information provided above. DO NOT recommend changes to age, number of dependents,\n"
        f"or past late payment history. DO NOT mention features not listed.\n"
        f"Length: 120-200 words."
    )


def build_retry_prompt(original_prompt: str, flags: list) -> str:
    """Append structured failure flags to the original prompt for a retry."""
    flag_text = "\n".join(f"  - {f}" for f in flags)
    return (
        f"{original_prompt}\n\n"
        f"CORRECTION REQUIRED\n"
        f"Your previous response was rejected by a deterministic verifier.\n"
        f"Specific violations found:\n"
        f"{flag_text}\n\n"
        f"Please produce a revised explanation that avoids all of the above issues.\n"
        f"Strictly adhere to the instructions above."
    )


# ---- Cache helpers ----------------------------------------------------------

def _cache_path(arm: str, case_id: int) -> str:
    return os.path.join(GEN_DIR, arm, f"case_{case_id}.json")


def _load_cache(arm: str, case_id: int):
    path = _cache_path(arm, case_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_cache(arm: str, case_id: int, record: dict):
    path = _cache_path(arm, case_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


# ---- Load evidence ----------------------------------------------------------

def load_all_evidence() -> list:
    records = []
    for fname in sorted(os.listdir(EVIDENCE_DIR)):
        if fname.startswith("case_") and fname.endswith(".json"):
            fpath = os.path.join(EVIDENCE_DIR, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                records.append(json.load(f))
    return records


# ---- Per-arm generation -----------------------------------------------------

def run_deterministic_arm(evidence: dict, case_id: int) -> dict:
    """Generate template explanation and verify it (no LLM)."""
    cached = _load_cache("deterministic_template", case_id)
    if cached:
        print(f"    [CACHE HIT] deterministic_template / case {case_id}")
        return cached

    text   = generate_template_explanation(evidence)
    result = verify_explanation(text, evidence, "deterministic_template")

    record = {
        "case_id":              case_id,
        "arm":                  "deterministic_template",
        "text":                 text,
        "word_count":           len(text.split()),
        "verification_result":  result.to_dict(),
        "fallback_triggered":   False,
        "retry_triggered":      False,
    }
    _save_cache("deterministic_template", case_id, record)
    return record


def run_llm_arm(arm: str, evidence: dict, case_id: int,
                api_call_counter: list) -> dict:
    """
    Generate an LLM explanation with optional retry and fallback.
    api_call_counter is a mutable list [n] so callers can track total calls.
    """
    cached = _load_cache(arm, case_id)
    if cached:
        print(f"    [CACHE HIT] {arm} / case {case_id}")
        return cached

    # Build initial prompt
    if arm == "shap_only":
        prompt = build_shap_only_prompt(evidence)
    elif arm == "cf_only":
        prompt = build_cf_only_prompt(evidence)
    else:  # combined
        prompt = build_combined_prompt(evidence)

    retry_triggered  = False
    fallback_triggered = False

    # --- First LLM call ---
    print(f"    [LLM CALL #{api_call_counter[0]+1}] {arm} / case {case_id}")
    text = generate_explanation(prompt, temperature=0.2)
    api_call_counter[0] += 1
    time.sleep(API_SLEEP_SECONDS)

    result = verify_explanation(text, evidence, arm)

    # --- Retry if verifier rejects ---
    if not result.passed and result.flags:
        retry_triggered = True
        retry_prompt    = build_retry_prompt(prompt, result.flags)
        print(f"    [RETRY   #{api_call_counter[0]+1}] {arm} / case {case_id} — flags: {result.flags}")
        text_retry = generate_explanation(retry_prompt, temperature=0.1)
        api_call_counter[0] += 1
        time.sleep(API_SLEEP_SECONDS)

        result_retry = verify_explanation(text_retry, evidence, arm)

        if result_retry.passed:
            text   = text_retry
            result = result_retry
        else:
            # Both failed — fallback to template
            print(f"    [FALLBACK] {arm} / case {case_id} — using template output")
            fallback_triggered = True
            text               = generate_template_explanation(evidence, arm=arm)
            result             = verify_explanation(text, evidence, arm)

    record = {
        "case_id":              case_id,
        "arm":                  arm,
        "text":                 text,
        "word_count":           len(text.split()),
        "verification_result":  result.to_dict(),
        "fallback_triggered":   fallback_triggered,
        "retry_triggered":      retry_triggered,
    }
    _save_cache(arm, case_id, record)
    return record


# ---- Main pipeline ----------------------------------------------------------

def run_pipeline(dry_run: bool = False):
    """
    Run all 4 arms over all 50 evidence cases.
    dry_run=True skips actual LLM calls (for quick testing of plumbing).
    """
    print("=" * 70)
    print("4-Arm Explanation Generator Pipeline")
    print("=" * 70)

    evidence_records = load_all_evidence()
    print(f"Loaded {len(evidence_records)} evidence records.\n")

    api_call_counter = [0]   # mutable counter shared across calls
    results_summary  = []

    for idx, evidence in enumerate(evidence_records):
        case_id = evidence["case_id"]
        pred_class = evidence["prediction"]["predicted_class"]
        category   = evidence["prediction"].get("case_category", "unknown")
        print(f"\n[{idx+1:02d}/50] Case {case_id}  "
              f"(pred={pred_class}, category={category})")

        for arm in ARMS:
            if arm == "deterministic_template":
                rec = run_deterministic_arm(evidence, case_id)
            else:
                if dry_run:
                    # In dry-run mode, check if already cached; skip otherwise
                    cached = _load_cache(arm, case_id)
                    if cached:
                        rec = cached
                        print(f"    [DRY-RUN CACHE HIT] {arm} / case {case_id}")
                    else:
                        print(f"    [DRY-RUN SKIP] {arm} / case {case_id}")
                        continue
                else:
                    rec = run_llm_arm(arm, evidence, case_id, api_call_counter)

            vr = rec["verification_result"]
            status = "PASS" if vr["passed"] else "FAIL"
            print(f"    {arm:<30} words={rec['word_count']:>4}  "
                  f"verifier={status}  "
                  f"retry={rec['retry_triggered']}  "
                  f"fallback={rec['fallback_triggered']}")

            results_summary.append({
                "case_id":   case_id,
                "arm":       arm,
                "word_count":rec["word_count"],
                "passed":    vr["passed"],
                "flags":     vr["flags"],
                "retry":     rec["retry_triggered"],
                "fallback":  rec["fallback_triggered"],
            })

    print(f"\n{'='*70}")
    print(f"Pipeline complete. Total API calls made this run: {api_call_counter[0]}")
    print(f"Results saved to: {GEN_DIR}")

    return results_summary


# ---- Entry point ------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="4-Arm Explanation Generator Pipeline")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip LLM calls; only process already-cached results + template arm.")
    args = parser.parse_args()

    run_pipeline(dry_run=args.dry_run)
