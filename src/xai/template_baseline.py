# =============================================================================
# Deterministic Template Baseline — 4th Arm (Zero LLM calls)
# =============================================================================
# Generates structured, rule-based natural language explanations using
# only the evidence contract JSON. No LLM, no randomness, fully reproducible.
#
# Three sections:
#   1. Prediction  — predicted class, probability, reliability caveat
#   2. Diagnosis   — top-3 SHAP features with values and risk directions
#   3. Recourse    — verified CF actions, or explicit "no feasible recourse" statement
#
# This arm serves as a non-LLM baseline to distinguish the LLM's added value
# from simple template filling, and as the fallback when LLM verification fails.
#
# Usage:
#   from src.xai.template_baseline import generate_template_explanation

from typing import Optional

# Human-readable feature descriptions (for clean prose output)
FEATURE_DESCRIPTIONS = {
    "revolving_util": "revolving credit utilization (the share of available credit in use)",
    "age": "applicant age",
    "late_30_59_days": "number of times 30-59 days past due (recent mild delinquencies)",
    "late_60_89_days": "number of times 60-89 days past due (moderate delinquencies)",
    "late_90_days": "number of times 90+ days past due (serious delinquencies)",
    "debt_ratio": "debt-to-income ratio (total monthly debt obligations relative to income)",
    "monthly_income": "monthly income",
    "open_credit_lines": "number of open credit lines and loans",
    "real_estate_loans": "number of real estate loans or mortgages",
    "num_dependents": "number of financial dependents",
    "monthly_income_missing": "income data availability flag",
}

DIRECTION_PROSE = {
    "increases_risk": "increases the estimated default risk",
    "decreases_risk": "decreases the estimated default risk",
}


def _format_value(feat_name: str, value: float) -> str:
    """Format a feature value for human-readable display."""
    if feat_name == "revolving_util":
        return f"{value * 100:.1f}%"
    elif feat_name == "monthly_income":
        return f"${value:,.0f}/month"
    elif feat_name == "debt_ratio" and value > 100:
        return f"{value:,.0f} (extremely high ratio)"
    elif feat_name in ("late_30_59_days", "late_60_89_days", "late_90_days",
                       "open_credit_lines", "real_estate_loans",
                       "num_dependents", "age"):
        return f"{int(round(value))}"
    else:
        return f"{value:.4g}"


def _format_cf_value(feat_name: str, value: float) -> str:
    """Format a counterfactual target value."""
    return _format_value(feat_name, value)


def generate_template_explanation(evidence: dict, arm: str = "combined") -> str:
    """
    Generate a structured explanation from the evidence record.

    Parameters
    ----------
    evidence : dict
        Structured evidence record from evidence_contract.py.
    arm : str
        One of 'shap_only', 'cf_only', 'combined', 'deterministic_template'.
        Controls which sections are included to prevent cross-arm contamination.

    Returns
    -------
    str
        Multi-paragraph plain-English explanation.
    """
    pred = evidence["prediction"]
    shap = evidence["shap_attributions"]
    cf   = evidence["counterfactual_recourse"]

    paragraphs = []

    # =========================================================================
    # Section 1: Prediction Statement (Always included)
    # =========================================================================
    pred_class = pred["predicted_class"]
    proba = pred["predicted_proba"]
    proba_pct = proba * 100

    if pred_class == 1:
        decision_text = "HIGH RISK — likely to default"
        risk_word = "elevated"
    else:
        decision_text = "LOW RISK — unlikely to default"
        risk_word = "low"

    section1 = (
        f"CREDIT DECISION SUMMARY\n"
        f"{'=' * 40}\n"
        f"Model Assessment: {decision_text}\n"
        f"Estimated Default Probability: {proba_pct:.1f}%\n"
        f"Decision Threshold: {pred['threshold'] * 100:.0f}%\n\n"
        f"IMPORTANT: This is a statistical model score, not a guarantee. "
        f"The model estimates a {proba_pct:.1f}% probability of serious delinquency "
        f"within two years based on the provided credit data. "
        f"Actual outcomes depend on many factors not captured by this model."
    )
    paragraphs.append(section1)

    # =========================================================================
    # Section 2: Diagnosis — Top-3 Contributing Factors (SHAP)
    # (Included in shap_only, combined, and deterministic_template; OMITTED in cf_only)
    # =========================================================================
    if arm != "cf_only":
        top3 = shap["top_features"][:3]

        section2_lines = ["KEY FACTORS DRIVING THIS ASSESSMENT", "=" * 40]
        for i, feat_rec in enumerate(top3, 1):
            feat = feat_rec["feature_name"]
            val  = feat_rec["feature_value"]
            sv   = feat_rec["shap_value"]
            direction = feat_rec["direction"]

            desc  = FEATURE_DESCRIPTIONS.get(feat, feat)
            val_str = _format_value(feat, val)
            dir_str = DIRECTION_PROSE[direction]
            sign_str = f"({'+'}{sv:.3f})" if sv > 0 else f"({sv:.3f})"

            section2_lines.append(
                f"\n{i}. {desc.upper()}\n"
                f"   Observed value : {val_str}\n"
                f"   Risk impact    : This factor {dir_str} {sign_str}.\n"
                f"   Interpretation : "
                + _factor_interpretation(feat, val, direction)
            )

        paragraphs.append("\n".join(section2_lines))

    # =========================================================================
    # Section 3: Recourse Recommendations (Counterfactuals)
    # (Included in cf_only, combined, and deterministic_template; OMITTED in shap_only)
    # =========================================================================
    if arm != "shap_only":
        passed_cfs = [
            c for c in cf.get("candidates", [])
            if c["verification"]["final_status"] == "PASSED"
        ]

        section3_lines = ["RECOURSE RECOMMENDATIONS", "=" * 40]

        if pred_class == 0:
            section3_lines.append(
                "The model currently assesses this applicant as low risk. "
                "No recourse action is needed at this time. "
                "Maintaining current financial habits is recommended."
            )
        elif not passed_cfs:
            section3_lines.append(
                "After exhaustive search, no feasible recourse pathway was identified "
                "under the current domain constraints. This means that within the set "
                "of changes a borrower can realistically make (reducing debt, improving "
                "credit utilization, increasing income), no single combination was found "
                "that would flip the model's assessment to low risk while respecting all "
                "domain rules. This may reflect that the adverse factors are primarily "
                "driven by features that cannot be immediately improved."
            )
        else:
            # Use the best (lowest gower distance) passing CF
            best_cf = min(
                passed_cfs,
                key=lambda c: c["verification"]["gower_distance"]
            )
            changed = best_cf["changed_features"]
            cf_proba = best_cf["cf_predicted_proba"]
            sparsity = best_cf["verification"]["sparsity_l0"]

            section3_lines.append(
                f"A feasible recourse pathway exists. The following {sparsity} change(s) "
                f"would reduce the estimated default probability from {proba_pct:.1f}% "
                f"to {cf_proba * 100:.1f}% (below the {pred['threshold']*100:.0f}% threshold):\n"
            )

            for j, ch in enumerate(changed, 1):
                feat = ch["feature_name"]
                orig = ch["original_value"]
                target = ch["cf_value"]
                direction = ch["direction"]

                desc = FEATURE_DESCRIPTIONS.get(feat, feat)
                orig_str   = _format_cf_value(feat, orig)
                target_str = _format_cf_value(feat, target)
                action_verb = "Reduce" if direction == "decrease" else "Increase"

                section3_lines.append(
                    f"  Step {j}: {action_verb} {desc}\n"
                    f"    Current value : {orig_str}\n"
                    f"    Target value  : {target_str}\n"
                    f"    Action        : {_recourse_action(feat, orig, target, direction)}"
                )

            section3_lines.append(
                f"\nNote: These recommendations are derived from a domain-constrained "
                f"counterfactual search. All suggested changes are directionally feasible "
                f"(e.g., paying down debt, increasing income). The model predicts that "
                f"implementing all steps simultaneously would change the assessment to low risk."
            )

        paragraphs.append("\n".join(section3_lines))

    return "\n\n".join(paragraphs)


def _factor_interpretation(feat: str, val: float, direction: str) -> str:
    """Return a one-sentence interpretation of why this factor matters."""
    if feat == "revolving_util":
        pct = val * 100
        if direction == "increases_risk":
            return (f"A utilization of {pct:.1f}% means the applicant is using a "
                    f"significant portion of available revolving credit. "
                    f"High utilization is one of the strongest predictors of default risk.")
        else:
            return (f"A utilization of {pct:.1f}% indicates responsible credit usage, "
                    f"which reduces predicted default risk.")

    elif feat in ("late_30_59_days", "late_60_89_days", "late_90_days"):
        count = int(round(val))
        bucket = feat.replace("late_", "").replace("_days", "").replace("_", "-")
        if count == 0:
            return f"No recorded {bucket}-day late payments, which is a positive signal."
        else:
            return (f"{count} recorded {bucket}-day late payment(s). "
                    f"Payment delinquencies are strong indicators of future default.")

    elif feat == "debt_ratio":
        if val > 100:
            return (f"A debt ratio of {val:,.0f} is extremely high, often seen in "
                    f"edge cases such as near-zero income. This distorts the ratio "
                    f"but the model interprets the raw value.")
        elif direction == "increases_risk":
            return f"A debt ratio of {val:.2f} indicates high debt relative to income."
        else:
            return f"A debt ratio of {val:.2f} is within a manageable range."

    elif feat == "monthly_income":
        return (f"A monthly income of ${val:,.0f} {'supports' if direction == 'decreases_risk' else 'may limit'} "
                f"the applicant's capacity to service debt obligations.")

    elif feat == "age":
        return (f"At {int(round(val))} years old, the applicant's age "
                f"{'reflects an established credit history' if val > 45 else 'is associated with shorter credit history'}.")

    elif feat == "open_credit_lines":
        return (f"{int(round(val))} open credit lines. "
                f"{'A higher number' if direction == 'increases_risk' else 'This number'} "
                f"{'may indicate over-leveraging' if direction == 'increases_risk' else 'is within a normal range'}.")

    elif feat == "real_estate_loans":
        count = int(round(val))
        return (f"{count} real estate loan(s). "
                f"{'Real estate loans are generally lower-risk obligations.' if direction == 'decreases_risk' else 'These loans contribute to the overall debt profile.'}")

    elif feat == "num_dependents":
        return (f"{int(round(val))} financial dependent(s). "
                f"More dependents can reduce disposable income available for debt service.")

    else:
        return f"Value: {val}. This feature {DIRECTION_PROSE[direction]}."


def _recourse_action(feat: str, orig: float, target: float, direction: str) -> str:
    """Generate a specific plain-language recourse action sentence."""
    if feat == "revolving_util":
        return (f"Pay down revolving credit balances so total utilization drops "
                f"from {orig*100:.1f}% to below {target*100:.1f}%.")
    elif feat == "debt_ratio":
        if orig > 100:
            return "Improve income documentation or reduce outstanding debt balances."
        return (f"Reduce total monthly debt payments or increase income to bring "
                f"the debt-to-income ratio from {orig:.2f} to approximately {target:.2f}.")
    elif feat == "monthly_income":
        return (f"Increase documented monthly income from ${orig:,.0f} "
                f"to at least ${target:,.0f}.")
    elif feat == "open_credit_lines":
        if direction == "decrease":
            return (f"Close {int(orig - target)} unused credit account(s) to reduce "
                    f"total open lines from {int(orig)} to {int(target)}.")
        else:
            return f"Consider opening additional credit accounts to reach {int(target)} total lines."
    elif feat == "real_estate_loans":
        if direction == "decrease":
            return f"Reduce real estate loans from {int(orig)} to {int(target)}."
        else:
            return f"Adding a real estate loan (e.g., a mortgage) may improve the credit profile."
    else:
        if direction == "decrease":
            return f"Reduce {feat} from {orig:.4g} to {target:.4g}."
        else:
            return f"Increase {feat} from {orig:.4g} to {target:.4g}."


if __name__ == "__main__":
    import json, os
    sample = os.path.join("reports", "evaluation", "evidence_records", "case_17856.json")
    with open(sample) as f:
        evidence = json.load(f)

    text = generate_template_explanation(evidence)
    print(text)
    print(f"\n--- Word count: {len(text.split())} ---")
