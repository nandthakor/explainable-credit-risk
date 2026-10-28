# 6. FUTURE WORK

---

The project in its current form demonstrates a complete proof-of-concept
pipeline and produces the first controlled three-arm comparison of
LLM-generated credit-risk explanations. However, several limitations
inherent to the current scope suggest concrete directions for extension.

---

## 6.1 Expanding the Human Evaluation Sample

The blinded survey instrument is built and ready, but no human
responses have been collected as of this report. The automated
evaluation (rule-based faithfulness and LLM judge) covers only
observable proxies for what matters most: whether a real borrower
finds the explanation understandable, trustworthy, and actionable.
Collecting responses from at least eight to ten participants — ideally
including individuals without a technical background — would provide
an independent signal that may confirm or contradict the automated
metrics.

The current evaluation relies on five classification instances per arm.
This is a small number; per-arm averages are sensitive to individual
outliers. A meaningful user study would require a substantially larger
instance set (at least 20-30) and a larger respondent pool to achieve
statistical power.

## 6.2 Multi-Domain and Multi-Dataset Validation

The findings are specific to the Give Me Some Credit dataset and the
Logistic Regression model trained on it. Whether the superiority of the
combined arm on actionability — and the inferiority on faithfulness —
would replicate on other credit datasets (different national markets,
different feature sets) or on other model types (gradient boosting,
neural networks) is unknown. Replicating the experiment on at least one
additional dataset would strengthen the generalisability of the
conclusions.

Extension to other high-stakes decision domains — such as medical
triage or insurance underwriting — would test whether the directional
feasibility filtering concept, which was designed for credit features,
can be adapted to domain-specific constraint knowledge elsewhere.

## 6.3 Additional Explanation Strategies

The three-arm design compared SHAP, DiCE, and their combination. Several
additional strategies could be introduced as additional arms without
changing the evaluation infrastructure:

- **LIME-based explanations** — LIME produces local linear approximations
  that are computationally less principled than SHAP but more widely
  used; comparing LIME-prompted versus SHAP-prompted LLM explanations
  would test whether the explanation method or the LLM prompt design is
  the dominant factor.
- **Rule-based surrogate explanations** — decision-tree surrogates of
  the classifier could be used to generate IF-THEN explanations as an
  alternative to SHAP.
- **Anchors** — the Anchor explanation method produces sufficient
  conditions for a prediction, which might translate well to actionable
  language.

## 6.4 Model Complexity and Faithfulness

The current classifier is a Logistic Regression, which is itself
relatively interpretable. SHAP values for a linear model are exact and
additive. A higher-performing but less interpretable model — such as
XGBoost or a neural network — would make the SHAP values approximate
rather than exact and might change the faithfulness scores substantially.
Testing whether the combined arm's faithfulness disadvantage (8.8 vs.
9.4 for CF-only) widens or narrows when applied to a tree-based model
with TreeExplainer is a natural follow-up.

## 6.5 LLM Generation Diversity and Stability

All 15 explanations were generated at temperature 0.2 in a single pass.
Temperature variation and multi-sample generation with voting or
averaging could test whether the results are stable across different
generation runs. The use of a single proprietary LLM (accessible via
the Groq API) also limits reproducibility; future work should test
open-weight models that can be run locally without an API dependency.

## 6.6 Deployment and Regulatory Compliance

The current prototype is a research demonstration, not a
production-grade system. Deploying such a system in a real lending
context would require:

- Formal bias auditing across demographic subgroups.
- Alignment with the relevant regulatory framework (the EU AI Act, RBI
  guidelines, or equivalent).
- Rate-limit and cost management for LLM inference at scale.
- A feedback loop that incorporates borrower responses to explanations
  into model monitoring.

These considerations are outside the current scope but are essential
prerequisites for any real-world deployment.
