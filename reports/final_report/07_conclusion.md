# 7. CONCLUSION

---

This project set out to answer the following research question:

> **Does combining SHAP and counterfactual explanations produce more
> faithful, understandable, and actionable LLM-generated credit-risk
> explanations than using either explanation type alone?**

The answer, based on automated evaluation over five classification
instances, is: **yes — but not uniformly across every dimension.**

---

## 7.1 Summary of Work

A complete seven-phase research prototype was designed, implemented, and
evaluated. Beginning with the Give Me Some Credit dataset (149,730
cleaned borrower records), a Logistic Regression classifier was trained
and selected as the predictive model (ROC-AUC 0.850 on the held-out test
set). A Linear Regression income-estimation model was built as a
companion regression task.

SHAP local and global explanations were generated for eight selected
case-study instances, confirming that revolving credit utilisation is
the dominant predictor of default risk in this dataset. DiCE
counterfactuals were generated for the five classification instances and
passed through a directional feasibility filter that retained only 5 of
24 raw suggestions (20.8%), demonstrating that domain-knowledge
filtering is essential before counterfactual advice is communicated to
borrowers.

Three LLM-generated explanations were produced for each classification
instance using the Groq API — one per arm (SHAP-only, CF-only,
combined). The explanations were evaluated on four dimensions using
rule-based faithfulness scoring, LLM-as-judge assessment by an
independent model, and Flesch-Kincaid readability measurement. A blinded
human survey was built and is awaiting responses.

A Streamlit demo interface was developed exposing the full pipeline in
both a pre-computed Replay mode and a live inference mode that re-runs
the complete pipeline on new applicants in real time.

---

## 7.2 Answer to the Research Question

The automated evaluation results are as follows:

| Metric | shap_only | cf_only | combined |
|---|---|---|---|
| Avg Top-Feature Recall | 0.867 | 0.767 | **0.967** |
| Contamination Rate | 0.0 | 0.0 | 0.0 |
| Avg LLM Faithfulness (1-10) | 7.8 | **9.4** | 8.8 |
| Avg LLM Understandability (1-10) | 9.0 | **9.8** | **9.8** |
| Avg LLM Actionability (1-10) | 2.0 | 6.4 | **9.0** |
| Avg Flesch-Kincaid Grade (lower=easier) | 10.18 | 10.50 | **9.53** |

The combined arm achieved the highest scores on three of the six
measured dimensions: feature recall (0.967), actionability (9.0/10),
and readability (FK grade 9.53). It tied with CF-only on
understandability (both 9.8/10). However, the CF-only arm was judged
more faithful (9.4 vs 8.8), reflecting that a focused single-source
explanation has less opportunity to introduce minor inaccuracies than an
explanation drawing on two data sources simultaneously.

Zero contamination was observed across all 15 explanations, confirming
that the prompt templates successfully enforced arm separation: SHAP-only
explanations consistently addressed the "why" without giving advice,
and CF-only explanations consistently provided actionable guidance
without explaining the underlying decision mechanism.

The most practically significant finding is the actionability gap:
the SHAP-only arm scores 2.0/10 — structurally, because its prompt
prohibits advice — while the combined arm reaches 9.0/10. For a
borrower who needs to understand both why they were declined and what
they can do about it, the combined arm provides substantially greater
utility.

The conclusion is therefore nuanced: the combined arm is the strongest
overall choice for end-user communication, but practitioners who
prioritise faithfulness to a single data source (for example, in a
regulatory audit context) may prefer the CF-only arm's more focused
approach. The SHAP-only arm effectively serves its specific purpose —
explaining a decision — but is fundamentally incapable of providing
recourse guidance.

---

## 7.3 Pending Evaluation

Human evaluation results are not yet available. The blinded survey
instrument has been distributed, but no participant responses have been
collected and processed as of the submission of this report. The
aggregate evaluation script will incorporate survey responses
automatically once they are received, and the human metrics will provide
an independent validation of the automated findings. The conclusions
above are therefore based exclusively on automated metrics.
