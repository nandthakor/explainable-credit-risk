# Notes for Student — Before Final Submission

---

## 1. Abstract Word Count

The abstract in `00_front_matter.md` contains **149 words** (verified
by manual count). The requested maximum was 150 words. No trimming is
needed, but do recount after any edits.

---

## 2. Complete List of [PLACEHOLDER] and [TODO] Items

The following items require action before the report is complete. They
are listed in document order.

### 00_front_matter.md

- **[PLACEHOLDER: COLLEGE CERTIFICATE]** — Print the title page and
  have the official certificate signed by Dr. Namrata Dave, the Head of
  Department, and the College Principal. Insert as Page 2 of the bound
  report.

- **[PLACEHOLDER: LIST OF FIGURES]** — After all figures are inserted
  into the Word document, create this page listing every figure by
  number and caption. Example format:
  ```
  Figure 5.1   Class balance bar chart .............. 15
  Figure 5.2   Missingness bar chart ................. 16
  ```

- **[PLACEHOLDER: LIST OF TABLES]** — Same process for tables.
  Example:
  ```
  Table 5.1    Classification model comparison ....... 18
  Table 5.2    Regression model comparison ........... 19
  ```

### 04_system_design.md

- **[FIGURE 4.1: System architecture diagram]** — Draw a flowchart
  showing the seven-phase pipeline. Recommended tool: PowerPoint or
  draw.io. The diagram should show:
  Raw CSV → Data Preparation → Classification + Regression Models →
  SHAP → DiCE → Three-Arm LLM Generation → Evaluation → Streamlit Demo.
  Include file-name hand-offs between phases as annotations.

### 05_implementation.md

The following screenshot and figure placeholders must be filled by
taking actual screenshots from the running system and saving the
referenced output images:

- **[SCREENSHOT 5.1]** — Terminal output of `python src/data_prep.py`
  showing column names and dtypes.
- **[SCREENSHOT 5.2]** — Terminal output confirming 270 rows removed
  and 149,730 rows retained.
- **[FIGURE 5.1]** — `reports/figures/01_class_balance.png`
- **[FIGURE 5.2]** — `reports/figures/02_missingness.png`
- **[FIGURE 5.3]** — `reports/figures/03_histograms.png`
- **[FIGURE 5.4]** — `reports/figures/04_correlation_heatmap.png`
- **[FIGURE 5.5]** — `reports/figures/logistic_regression_confusion.png`
- **[FIGURE 5.6]** — `reports/figures/random_forest_confusion.png`
- **[FIGURE 5.7]** — `reports/figures/shap_summary_bar_classification.png`
- **[FIGURE 5.8]** — `reports/figures/shap_beeswarm_classification.png`
- **[FIGURE 5.9]** — `reports/figures/shap_local/34_true_positive_waterfall.png`
- **[FIGURE 5.10]** — `reports/figures/shap_local/2_false_positive_waterfall.png`
- **[SCREENSHOT 5.3]** — DiCE terminal output (run
  `python src/counterfactual_explain.py` and capture output).
- **[SCREENSHOT 5.4]** — Streamlit Replay mode, three-column LLM
  display. Open `http://localhost:8501`, select True Positive, scroll
  to the explanation columns, take a screenshot.
- **[SCREENSHOT 5.5]** — Open
  `reports/evaluation/master_comparison_table.md` and screenshot or
  paste the table.
- **[SCREENSHOT 5.6 through 5.10]** — Various Streamlit screenshots
  (Replay dropdown, feature table, waterfall image, Live mode inputs,
  Live mode results). All require running `streamlit run app.py` and
  navigating the interface.

---

## 3. Reference Verification Checklist

Every reference in `08_references.md` is marked "Author(s) TBC" or
"DOI: TBC". Before submission:

1. Visit each arXiv link (e.g., https://arxiv.org/abs/2601.02224) to
   confirm the paper exists and to extract the exact author names, title,
   and submission date.

2. For non-arXiv sources ([5] MDPI, [6] Frontiers, [7] Springer,
   [8]-[12] various), search Google Scholar or the publisher's website
   using the description provided. Confirm title, authors, journal/
   conference, volume, issue, page numbers, and DOI.

3. Confirm with Dr. Namrata Dave whether IEEE numbered style is required
   or whether APA (Author, Year) is preferred by the department.

4. Update `08_references.md` with the verified details.

**Note:** Submitting references with "Author(s) TBC" or "DOI: TBC"
would be treated as an incomplete report. Do not skip this step.

---

## 4. Human Survey

The human evaluation survey at `reports/evaluation/survey.html` has
been built but no responses have been collected. Chapter 7 honestly
states this. If responses are collected before submission:

1. Save each participant's downloaded JSON file into
   `reports/evaluation/survey_responses/`.
2. Run `python src/aggregate_evaluation.py` to incorporate them.
3. Update the relevant rows in `05_implementation.md` Section 5.6
   and the `07_conclusion.md` with the human results.
4. Do not fabricate any responses or numbers.

---

## 5. Word Processor Paste Instructions

When pasting each `.md` file into Word:
- Use a Markdown-to-Word converter (e.g., Pandoc:
  `pandoc 01_introduction.md -o introduction.docx`) for clean
  formatting, or paste as plain text and apply Word heading styles
  manually.
- Chapter headings → Heading 1 style.
- Section headings (1.1, 1.2, ...) → Heading 2 style.
- Sub-section headings → Heading 3 style.
- Table content → use Word's Table style.
- Update the Table of Contents field in Word after inserting all
  chapters.
