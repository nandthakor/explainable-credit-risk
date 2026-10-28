# 1. INTRODUCTION OF PROJECT

---

## 1.1 Background and Motivation

The application of machine learning to credit-risk assessment has grown
substantially over the past decade. Banks, non-banking financial
companies, and online lending platforms increasingly deploy automated
scoring models to determine loan eligibility, set interest rates, and
flag potentially defaulting borrowers. These systems offer speed and
consistency at scale, but they introduce a fundamental tension: the
models that perform best — complex ensembles and neural networks — are
the least interpretable to the applicants they affect and to the
regulators that oversee them.

This project began as a more straightforward applied machine learning
exercise on the Give Me Some Credit dataset. After an initial review
conducted by the internal guide, it was identified that the preliminary
direction did not engage with any meaningful research gap and lacked the
originality expected of a Mini Project. In response to that feedback,
the project was substantially redesigned into its current form: a
systematic research prototype that evaluates different strategies for
generating natural-language explanations of credit-risk decisions using
explainable AI (XAI) techniques and large language models (LLMs). This
revision transformed a routine modelling exercise into a structured
empirical investigation with a clearly defined research question,
evaluation methodology, and reproducible results.

The resulting prototype covers the full pipeline from raw data through
model training, SHAP and counterfactual explanation generation,
LLM-mediated natural-language explanation, and a multi-dimensional
evaluation framework, concluded by an interactive Streamlit demonstration
interface.

---

## 1.2 Research Gap and Research Question

### Research Gap

Limited research systematically compares SHAP-only, counterfactual-only,
and combined SHAP+counterfactual LLM explanations for credit risk, in
terms of faithfulness, understandability, and actionable/feasible
recommendations.

Existing work frequently evaluates either SHAP-style feature attribution
or counterfactual recourse in isolation. Studies that use LLMs to
convert structured XAI outputs into natural language generally do not
hold the explanation type as a controlled variable and do not apply a
domain-constrained feasibility filter to counterfactual suggestions. The
absence of this systematic three-arm comparison leaves practitioners
without clear guidance on which explanation strategy best serves the
end-user in a credit-risk context.

### Research Question

Does combining SHAP and counterfactual explanations produce more
faithful, understandable, and actionable LLM-generated credit-risk
explanations than using either explanation type alone?

---

## 1.3 Significance of the Work

The significance of this project spans three dimensions.

**Regulatory and ethical relevance.** Regulatory frameworks in several
jurisdictions — including the European Union's AI Act and the Reserve
Bank of India's responsible-AI guidelines for the financial sector —
require that automated decisions affecting consumers be explainable in
terms the consumer can understand and act upon. This project directly
addresses the actionability dimension, which has received less attention
than faithfulness and understandability in the literature.

**Methodological contribution.** The project introduces a directional
feasibility filter applied to DiCE-generated counterfactuals before LLM
prompting. The empirical result — that only 20.8% of raw counterfactuals
survive this filter — demonstrates that feasibility screening is not
optional for responsible deployment. This finding is independent of
which LLM is used and should transfer to other domains where domain
knowledge can be encoded as directional constraints.

**Practical contribution.** The Streamlit demonstration interface allows
a practitioner or regulator to explore all three explanation arms for
both saved case studies and live new applicants, providing a concrete
artefact for further user research.

---

## 1.4 Organization of the Report

The remainder of this report is organized as follows.

**Chapter 2 — Technology Used** describes the tools, libraries, and
platforms employed, with specific reference to how each was used in this
project rather than as generic descriptions.

**Chapter 3 — Proposed Work** presents the problem statement, a review
of related literature, and the proposed three-arm experimental design
together with the evaluation methodology.

**Chapter 4 — System Design** describes the end-to-end system
architecture as a sequential pipeline and provides a placeholder for
the architecture diagram.

**Chapter 5 — Implementation** walks through each phase of the
implementation in prose, presenting the real numerical results produced
at each stage.

**Chapter 6 — Future Work** identifies grounded limitations and
directions for extending this research.

**Chapter 7 — Conclusion** synthesises the findings and provides a
direct answer to the research question based on the automated evaluation
results produced.

**Chapter 8 — References** lists the twelve sources cited in this report
in IEEE numbered format.
