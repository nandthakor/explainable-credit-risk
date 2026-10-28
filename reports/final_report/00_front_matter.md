# FRONT MATTER

---

## TITLE PAGE

**A Mini Project Report**

on

**Evaluating SHAP and Counterfactual Explanations for Faithful, Understandable,
and Actionable LLM-Generated Credit-Risk Explanations**

Submitted in partial fulfilment of the requirements for the degree of

**Bachelor of Technology (Computer Engineering)**

by

**Nand Thakor**
Roll No. 12302040501032
B.Tech (CP), Semester VII

Under the guidance of

**Dr. Namrata Dave**
Internal Guide

**G H Patel College of Engineering & Technology (GCET)**
The Charutar Vidya Mandal (CVM) University, Vallabh Vidyanagar

**September 2026**

---

## [PLACEHOLDER: COLLEGE CERTIFICATE]

*This page requires the official certificate bearing the signature of the
internal guide, the Head of Department, and the College Principal. This
section cannot be generated digitally and must be printed and signed
separately before binding.*

---

## ACKNOWLEDGEMENT

The author wishes to express sincere gratitude to **Dr. Namrata Dave**,
Internal Guide, for her invaluable guidance, constructive feedback, and
encouragement throughout the development of this project. Her critical
review of the initial project direction — and the subsequent revision
toward a more rigorous research contribution — directly shaped the final
scope and quality of the work presented here.

Gratitude is also extended to the faculty members of the Department of
Computer Engineering at G H Patel College of Engineering & Technology
for their continuous academic support and for providing an environment
conducive to research.

Finally, the author acknowledges the broader open-source community whose
tools and datasets — including the scikit-learn, SHAP, DiCE, and
Streamlit projects, as well as the Kaggle dataset contributors — made
this work possible.

---

## ABSTRACT

Credit-risk models increasingly rely on opaque machine learning
classifiers, raising regulatory and ethical demands for explainability.
This project investigates whether combining SHAP feature-attribution
explanations with counterfactual recourse recommendations produces more
faithful, understandable, and actionable natural-language explanations
than either technique alone, when an LLM is used as the generation
medium. Using the Give Me Some Credit dataset (149,730 borrowers), a
Logistic Regression classifier (ROC-AUC 0.850) and income-estimation
regression model were trained. SHAP, DiCE counterfactuals, and three
explanation arms were generated via the Groq API and evaluated using
rule-based faithfulness checks, an independent LLM judge, and
Flesch-Kincaid readability metrics. The combined arm achieved the
highest feature recall (0.967), actionability score (9.0/10), and
easiest readability (FK grade 9.53). However, the counterfactual-only
arm was rated more faithful (9.4 vs 8.8). The results show that
combining both explanation types offers the strongest overall profile
for end-user communication, with no single arm dominating uniformly.

**Word count: 149**

---

## [PLACEHOLDER: LIST OF FIGURES]

*To be completed once all figures from reports/figures/ are inserted into
the Word document. Number each figure sequentially (Figure 1.1, Figure 4.1,
Figure 5.1, etc.) with matching captions.*

---

## [PLACEHOLDER: LIST OF TABLES]

*To be completed once all tables are finalised and numbered in the Word
document. Number each table sequentially (Table 5.1, Table 5.2, etc.)
with matching captions.*

---

## TABLE OF CONTENTS

1. Introduction of Project
   - 1.1 Background and Motivation
   - 1.2 Research Gap and Research Question
   - 1.3 Significance of the Work
   - 1.4 Organization of the Report

2. Technology Used
   - 2.1 Programming Language and Core Libraries
   - 2.2 Machine Learning Modelling (scikit-learn)
   - 2.3 Explainability Techniques: SHAP
   - 2.4 Explainability Techniques: DiCE Counterfactuals
   - 2.5 LLM Integration: Groq API
   - 2.6 Development Environment: Google Antigravity
   - 2.7 Deployment: Streamlit

3. Proposed Work
   - 3.1 Problem Statement
   - 3.2 Literature Review
   - 3.3 Proposed Approach

4. System Design
   - 4.1 System Architecture Overview
   - 4.2 Pipeline Description

5. Implementation Screenshots with Explanations
   - 5.1 Data Preparation
   - 5.2 Predictive Models
   - 5.3 SHAP Explanations
   - 5.4 Counterfactual Explanations
   - 5.5 LLM-Generated Explanations (Three-Arm Comparison)
   - 5.6 Evaluation Framework
   - 5.7 Demo Interface

6. Future Work

7. Conclusion

8. References
