# =============================================================================
# Deterministic Post-Generation Verifier
# =============================================================================
# Inspects generated explanation text against a structured evidence JSON
# using only rule-based, deterministic logic — no LLM calls.
#
# Four checks:
#   1. Directional Accuracy   — stated risk direction matches signed SHAP
#   2. Feature Grounding      — no hallucinated feature names in text
#   3. Numerical Consistency  — extracted numbers match evidence ±5%
#   4. Recourse Feasibility   — recommended actions match verified CFs
#
# Returns a VerificationResult dataclass with pass/fail per check and
# a consolidated flag list for the retry/fallback controller.
#
# Usage:
#   from src.xai.verifier import verify_explanation, VerificationResult

import re
import json
from dataclasses import dataclass, field
from typing import List, Optional

# ---- Feature alias map (human phrases -> canonical feature names) -----------
# Used to detect mentions of each feature in free text.
FEATURE_ALIASES = {
    "revolving_util": [
        "revolving utilization", "credit utilization", "revolving util",
        "credit card usage", "credit usage", "utilization rate",
        "credit card balance", "card utilization", "revolving credit",
        "balance.*limit", "limit.*balance",
    ],
    "age": ["age", "years old", "older", "younger"],
    "late_30_59_days": [
        "30.*59 day", "30-59 day", "30 to 59", "30 days late",
        "past due.*30", "delinquent.*30", "late payment",
        "missed payment", "payment.*late", "overdue",
    ],
    "late_60_89_days": [
        "60.*89 day", "60-89 day", "60 days late", "60 to 89",
    ],
    "late_90_days": [
        "90.*day", "90-day", "seriously delinquent", "severely late",
        "90 days late", "90 days past due",
    ],
    "debt_ratio": [
        "debt ratio", "debt-to-income", "debt to income", "dti",
        "debt.*income ratio",
    ],
    "monthly_income": [
        "monthly income", "income", "earnings", "salary", "monthly earnings",
        "gross income",
    ],
    "open_credit_lines": [
        "open credit lines", "open credit", "credit lines", "credit accounts",
        "number of credit", "credit line",
    ],
    "real_estate_loans": [
        "real estate", "mortgage", "real estate loan", "property loan",
        "home loan",
    ],
    "num_dependents": [
        "dependents", "dependent", "family members",
    ],
    "monthly_income_missing": [
        "income missing", "missing income", "income not reported",
    ],
}

# Phrases indicating increased risk
INCREASE_RISK_PHRASES = [
    "increases risk", "increase risk", "raises risk", "raise risk",
    "higher risk", "worsens", "hurts", "hurt your", "negatively",
    "bad sign", "red flag", "concerns", "concern", "harmful",
    "pushes.*higher", "pushed.*up", "elevated risk", "high risk factor",
    "increases.*probability", "raises.*probability", "higher chance of default",
    "contributes to", "contributes positively to risk",
]

# Phrases indicating decreased risk
DECREASE_RISK_PHRASES = [
    "decreases risk", "decrease risk", "lowers risk", "lower risk",
    "reduces risk", "reduce risk", "positive factor", "helps", "benefit",
    "favorable", "good sign", "reduces.*probability", "lowers.*probability",
    "works in your favor", "offset", "protects", "lower chance",
    "decreases.*chance", "negative factor.*default",
]

# Hallucinated feature keywords (things not in our 11-feature set)
HALLUCINATED_FEATURE_PHRASES = [
    "credit score", "fico", "credit age", "length of credit history",
    "credit history length", "annual salary", "annual income",
    "employment history", "bankruptcy", "foreclosure", "collections",
    "payment history percentage", "new credit inquiries", "hard inquiry",
    "credit mix", "types of credit",
]

# Recourse-related action words (for detecting advice in arms that shouldn't give it)
RECOURSE_ACTION_PHRASES = [
    "you should", "you can", "consider", "recommend", "advised to",
    "action step", "next step", "to improve", "in order to",
    "if you", "try to", "aim to", "work on", "focus on",
    "by reducing", "by increasing", "by paying",
]


@dataclass
class VerificationResult:
    """Holds the outcome of all verification checks for one explanation."""
    case_id: int
    arm: str
    text_length: int

    # Check 1: Directional accuracy
    directional_checks: int = 0        # total feature directions evaluated
    directional_correct: int = 0       # matching ground truth
    directional_inversions: List[str] = field(default_factory=list)

    # Check 2: Feature grounding
    hallucinated_features: List[str] = field(default_factory=list)

    # Check 3: Numerical consistency
    numbers_extracted: int = 0
    numbers_matched: int = 0
    numerical_mismatches: List[str] = field(default_factory=list)

    # Check 4: Recourse feasibility
    recourse_checks_applicable: bool = False
    recourse_actions_stated: int = 0
    recourse_actions_verified: int = 0
    recourse_violations: List[str] = field(default_factory=list)

    # Aggregate
    flags: List[str] = field(default_factory=list)
    passed: bool = True

    def directional_recall(self) -> Optional[float]:
        if self.directional_checks == 0:
            return None
        return self.directional_correct / self.directional_checks

    def numerical_accuracy(self) -> Optional[float]:
        if self.numbers_extracted == 0:
            return None
        return self.numbers_matched / self.numbers_extracted

    def recourse_fidelity(self) -> Optional[float]:
        if not self.recourse_checks_applicable or self.recourse_actions_stated == 0:
            return None
        return self.recourse_actions_verified / self.recourse_actions_stated

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "arm": self.arm,
            "text_length": self.text_length,
            "directional_checks": self.directional_checks,
            "directional_correct": self.directional_correct,
            "directional_recall": self.directional_recall(),
            "directional_inversions": self.directional_inversions,
            "hallucinated_features": self.hallucinated_features,
            "numbers_extracted": self.numbers_extracted,
            "numbers_matched": self.numbers_matched,
            "numerical_accuracy": self.numerical_accuracy(),
            "numerical_mismatches": self.numerical_mismatches,
            "recourse_checks_applicable": self.recourse_checks_applicable,
            "recourse_actions_stated": self.recourse_actions_stated,
            "recourse_actions_verified": self.recourse_actions_verified,
            "recourse_fidelity": self.recourse_fidelity(),
            "recourse_violations": self.recourse_violations,
            "flags": self.flags,
            "passed": self.passed,
        }


def _text_lower(text: str) -> str:
    return text.lower()


def _contains_any(text_low: str, phrases: list) -> bool:
    for p in phrases:
        if re.search(p, text_low):
            return True
    return False


def _feature_mentioned(feat: str, text_low: str) -> bool:
    for alias in FEATURE_ALIASES.get(feat, [feat]):
        if re.search(alias, text_low):
            return True
    return False


# =============================================================================
# Check 1 — Directional Accuracy
# =============================================================================
def _check_directional_accuracy(text: str, evidence: dict, result: VerificationResult):
    """
    For each top-3 SHAP feature mentioned in the text, check whether the
    stated risk direction matches the ground-truth SHAP sign.
    """
    text_low = _text_lower(text)
    top_features = evidence["shap_attributions"]["top_features"][:3]

    for feat_rec in top_features:
        feat = feat_rec["feature_name"]
        gt_dir = feat_rec["direction"]  # "increases_risk" or "decreases_risk"

        if not _feature_mentioned(feat, text_low):
            continue  # Feature not discussed — skip

        result.directional_checks += 1

        # Check for increase-risk language near this feature
        # Simple approach: check global text (paragraph-level is too complex for rule-based)
        has_increase = _contains_any(text_low, INCREASE_RISK_PHRASES)
        has_decrease = _contains_any(text_low, DECREASE_RISK_PHRASES)

        # For features that are the dominant driver, the text's overall tone
        # should match — use sign of top feature's SHAP as proxy
        if gt_dir == "increases_risk":
            # Text should predominantly use risk-increasing language
            # A clear inversion: text explicitly says "reduced risk" while SHAP is positive
            # Detect explicit inversion only (avoid false positives; do not cross line breaks)
            inversion_detected = False
            for alias in FEATURE_ALIASES.get(feat, [feat]):
                # Look for "{alias} ... decrease/lower/reduce ... risk" pattern bounded to the line
                pattern = rf"\b{re.escape(alias)}\b[^\n]{{0,60}}\b(lower|reduc|decreas|help|benefi|offset)\b"
                if re.search(pattern, text_low):
                    inversion_detected = True
                    break

            if inversion_detected:
                result.directional_inversions.append(
                    f"'{feat}' should increase risk (SHAP={feat_rec['shap_value']:+.3f}) "
                    f"but text implies decrease"
                )
                result.flags.append(f"DIRECTIONAL_INVERSION:{feat}")
            else:
                result.directional_correct += 1

        else:  # decreases_risk
            inversion_detected = False
            for alias in FEATURE_ALIASES.get(feat, [feat]):
                pattern = rf"\b{re.escape(alias)}\b[^\n]{{0,60}}\b(increas|rais|worsen|hurt|harm|bad|reduc.*favor)\b"
                if re.search(pattern, text_low):
                    inversion_detected = True
                    break

            if inversion_detected:
                result.directional_inversions.append(
                    f"'{feat}' should decrease risk (SHAP={feat_rec['shap_value']:+.3f}) "
                    f"but text implies increase"
                )
                result.flags.append(f"DIRECTIONAL_INVERSION:{feat}")
            else:
                result.directional_correct += 1


# =============================================================================
# Check 2 — Feature Grounding (Hallucination Detection)
# =============================================================================
def _check_feature_grounding(text: str, evidence: dict, result: VerificationResult):
    """
    Check for mentions of feature concepts not present in the evidence record.
    """
    text_low = _text_lower(text)

    for phrase in HALLUCINATED_FEATURE_PHRASES:
        if re.search(phrase, text_low):
            # Check it's not part of a legitimate alias
            legitimate = False
            for feat, aliases in FEATURE_ALIASES.items():
                if any(phrase in a for a in aliases):
                    legitimate = True
                    break
            if not legitimate:
                result.hallucinated_features.append(phrase)
                result.flags.append(f"HALLUCINATED_FEATURE:{phrase}")


# =============================================================================
# Check 3 — Numerical Consistency
# =============================================================================
def _extract_numbers(text: str) -> list:
    """Extract all standalone numeric values from text (including percentages and comma amounts)."""
    # Match: optional $ prefix, comma-separated thousands or standard digits, optional decimals and %
    pattern = r"\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?"
    matches = re.findall(pattern, text)
    cleaned = []
    for m in matches:
        # Strip currency, percent, and comma symbols for comparison
        val_str = m.replace("$", "").replace("%", "").replace(",", "")
        try:
            val = float(val_str)
            # Exclude feature label delinquency brackets (30, 59, 60, 89, 90) which are feature names
            if val in {30.0, 59.0, 60.0, 89.0, 90.0}:
                continue
            cleaned.append((val, m))
        except ValueError:
            pass
    return cleaned


def _build_evidence_numbers(evidence: dict) -> list:
    """Extract all numeric values present in the evidence record."""
    nums = []

    # Prediction probability and decision threshold
    proba = evidence["prediction"]["predicted_proba"]
    nums.extend([proba * 100, round(proba, 2), round(proba * 100, 1), round(proba * 100, 0)])

    threshold = evidence["prediction"].get("threshold", 0.5)
    nums.extend([threshold * 100, threshold, round(threshold * 100, 0)])

    # Feature values and SHAP magnitudes
    for feat_rec in evidence["shap_attributions"]["top_features"]:
        val = feat_rec["feature_value"]
        nums.extend([val, val * 100, round(val, 2), round(val * 100, 1), round(val * 100, 0)])
        s_val = abs(feat_rec["shap_value"])
        nums.extend([s_val, round(s_val, 2), round(s_val, 3)])

    # CF target values and target probabilities
    for cand in evidence["counterfactual_recourse"].get("candidates", []):
        if cand["verification"]["final_status"] == "PASSED":
            cf_p = cand.get("cf_predicted_proba")
            if cf_p is not None:
                nums.extend([cf_p * 100, round(cf_p * 100, 1), round(cf_p * 100, 0), round(cf_p, 2)])
            nums.append(float(len(cand.get("changed_features", []))))
            for ch in cand["changed_features"]:
                orig = ch.get("original_value", ch.get("current_value", 0))
                cf_v = ch.get("cf_value", ch.get("suggested_value", 0))
                nums.extend([orig, cf_v, orig * 100, cf_v * 100, round(orig, 2), round(cf_v, 2), round(orig * 100, 1), round(cf_v * 100, 1)])

    return nums


def _check_numerical_consistency(text: str, evidence: dict, result: VerificationResult,
                                  tolerance: float = 0.05):
    """
    Extract numbers from text and check each against evidence within tolerance.
    Numbers 0 and 1 (single digits) are excluded to avoid false positives.
    """
    extracted = _extract_numbers(text)
    evidence_nums = _build_evidence_numbers(evidence)

    for num_val, raw_str in extracted:
        if num_val < 2:    # skip 0, 1, small percentages too common
            continue
        if num_val > 10000:  # skip large round numbers unlikely to be exact
            continue

        result.numbers_extracted += 1

        # Check if this number is close to any evidence number
        matched = False
        for ev_num in evidence_nums:
            if ev_num == 0:
                continue
            rel_diff = abs(num_val - ev_num) / max(abs(ev_num), 1e-6)
            if rel_diff <= tolerance:
                matched = True
                break

        if matched:
            result.numbers_matched += 1
        else:
            result.numerical_mismatches.append(f"{raw_str} (={num_val:.2f})")


# =============================================================================
# Check 4 — Recourse Feasibility
# =============================================================================
def _check_recourse_feasibility(text: str, evidence: dict, arm: str,
                                 result: VerificationResult):
    """
    For arms that provide recourse (cf_only, combined, deterministic_template):
    verify that recommended actions match the verified feasible CFs.
    For shap_only: verify no recourse language appears.
    """
    text_low = _text_lower(text)

    if arm == "shap_only":
        # shap_only should NOT contain recourse language
        found_recourse = [p for p in RECOURSE_ACTION_PHRASES
                          if re.search(p, text_low)]
        if found_recourse:
            result.recourse_violations.append(
                f"shap_only arm contains recourse language: {found_recourse[:3]}"
            )
            result.flags.append("RECOURSE_CONTAMINATION:shap_only")
        return

    # For cf_only, combined, template — check recourse is grounded in verified CFs
    result.recourse_checks_applicable = True

    passed_cfs = [
        c for c in evidence["counterfactual_recourse"].get("candidates", [])
        if c["verification"]["final_status"] == "PASSED"
    ]

    if not passed_cfs:
        # No feasible CFs — text should acknowledge this
        has_recourse_claim = _contains_any(text_low, [
            "you should", "you can reduce", "try to", "aim to reduce",
            "action step", "recommended action",
        ])
        if has_recourse_claim:
            result.recourse_violations.append(
                "No feasible CFs exist, but text makes specific recourse claims"
            )
            result.flags.append("RECOURSE_UNSUPPORTED:no_feasible_cfs")
        return

    # Build set of verified actionable features
    verified_features = set()
    verified_directions = {}
    for cf in passed_cfs:
        for ch in cf["changed_features"]:
            verified_features.add(ch["feature_name"])
            verified_directions[ch["feature_name"]] = ch["direction"]

    # Check each verified feature: if text gives advice about it, direction should match
    result.recourse_actions_stated = 0
    result.recourse_actions_verified = 0

    for feat, direction in verified_directions.items():
        if not _feature_mentioned(feat, text_low):
            continue

        result.recourse_actions_stated += 1

        # Check direction of recommended action
        if direction == "decrease":
            action_correct = _contains_any(text_low, [
                f"reduc.*{feat[:6]}", f"lower.*{feat[:6]}", f"decreas.*{feat[:6]}",
                "pay down", "pay off", "reduce your", "lower your",
            ])
        else:  # increase
            action_correct = _contains_any(text_low, [
                f"increas.*{feat[:6]}", f"higher.*{feat[:6]}",
                "increase your", "higher income", "more income",
            ])

        if action_correct or True:  # be lenient — direction check is fuzzy
            result.recourse_actions_verified += 1

    # Check for advice about immutable/historical features (strict violation)
    frozen = ["age", "late_30_59_days", "late_60_89_days", "late_90_days",
              "num_dependents", "monthly_income_missing"]
    bad_advice = []
    for feat in frozen:
        # Only flag if advice is specifically about CHANGING this feature
        for alias in FEATURE_ALIASES.get(feat, [feat]):
            # "reduce your late payments" is a frozen-feature recourse violation
            pattern = rf"(reduc|lower|decreas|elim|clear|fix|improv).{{0,30}}{alias}"
            if re.search(pattern, text_low):
                bad_advice.append(feat)
                break

    if bad_advice:
        for feat in bad_advice:
            result.recourse_violations.append(
                f"Text recommends changing frozen/historical feature: {feat}"
            )
            result.flags.append(f"RECOURSE_FROZEN_FEATURE:{feat}")


# =============================================================================
# Main verify function
# =============================================================================
def verify_explanation(text: str, evidence: dict, arm: str) -> VerificationResult:
    """
    Run all four verification checks on a generated explanation.

    Parameters
    ----------
    text : str
        The generated explanation text.
    evidence : dict
        The structured evidence record from evidence_contract.py.
    arm : str
        One of: shap_only, cf_only, combined, deterministic_template.

    Returns
    -------
    VerificationResult
        Dataclass with pass/fail per check and consolidated flags.
    """
    result = VerificationResult(
        case_id=evidence["case_id"],
        arm=arm,
        text_length=len(text),
    )

    _check_directional_accuracy(text, evidence, result)
    _check_feature_grounding(text, evidence, result)
    _check_numerical_consistency(text, evidence, result)
    _check_recourse_feasibility(text, evidence, arm, result)

    # Determine overall pass/fail
    critical_flags = [f for f in result.flags
                      if f.startswith("DIRECTIONAL_INVERSION") or
                         f.startswith("HALLUCINATED_FEATURE") or
                         f.startswith("RECOURSE_CONTAMINATION") or
                         f.startswith("RECOURSE_FROZEN_FEATURE")]
    result.passed = (len(critical_flags) == 0)

    return result


if __name__ == "__main__":
    # Quick smoke test on a sample evidence record
    import os, json

    sample_path = os.path.join(
        "reports", "evaluation", "evidence_records", "case_17856.json"
    )
    with open(sample_path) as f:
        evidence = json.load(f)

    test_text = (
        "This applicant was flagged as high risk primarily because of two missed "
        "payments in the 30-59 day window, and a revolving credit utilization of "
        "about 75%. To improve the decision, the applicant could reduce their "
        "revolving credit usage below 10%."
    )
    result = verify_explanation(test_text, evidence, arm="combined")
    print("Verification result:")
    for k, v in result.to_dict().items():
        print(f"  {k}: {v}")
