# Credit Risk XAI - Phase 6: Blinded Survey Builder
# ====================================================
# Builds a blinded human evaluation survey for the 5 classification
# instances. For each instance, the three arm explanations (shap_only,
# cf_only, combined) are randomly assigned labels A/B/C so survey
# participants cannot guess which method produced which explanation.
#
# Outputs:
#   reports/evaluation/survey_key.json       - hidden true mapping
#   reports/evaluation/survey_questions.json  - participant-facing data
#   reports/evaluation/survey.html           - fillable HTML form
#
# Run: python src/build_survey.py

import os
import json
import random

# ---- Paths ------------------------------------------------------------------
REPORTS   = "reports"
LLM_DIR   = os.path.join(REPORTS, "llm_explanations")
EVAL_DIR  = os.path.join(REPORTS, "evaluation")
RESP_DIR  = os.path.join(EVAL_DIR, "survey_responses")

os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(RESP_DIR, exist_ok=True)

# ---- Constants --------------------------------------------------------------
ARMS = ["shap_only", "cf_only", "combined"]
LABELS = ["A", "B", "C"]
RANDOM_STATE = 42


def build_survey():
    """Build the blinded survey specification and HTML form."""
    print("\n" + "=" * 60)
    print("BUILDING BLINDED SURVEY")
    print("=" * 60)

    # Load instances
    with open(os.path.join(REPORTS, "selected_instances.json")) as f:
        all_insts = json.load(f)
    clf_insts = [i for i in all_insts if i["task"] == "classification"]

    rng = random.Random(RANDOM_STATE)

    survey_key = {}        # hidden: instance_id -> {A: arm, B: arm, C: arm}
    survey_questions = []  # participant-facing

    for inst in clf_insts:
        iid = inst["instance_id"]
        case = inst["case_type"]

        # Randomly shuffle the arm-to-label mapping for this instance
        shuffled_arms = ARMS.copy()
        rng.shuffle(shuffled_arms)
        label_map = dict(zip(LABELS, shuffled_arms))

        survey_key[str(iid)] = label_map
        print(f"  Instance {iid} ({case}): "
              f"A={label_map['A']}, B={label_map['B']}, C={label_map['C']}")

        # Predicted outcome in plain terms
        pred_class = 1 if inst["predicted_proba"] >= 0.5 else 0
        if pred_class == 1:
            outcome_text = (
                f"The model flagged this applicant as HIGH RISK "
                f"(probability of default: {inst['predicted_proba']:.0%})"
            )
        else:
            outcome_text = (
                f"The model flagged this applicant as LOW RISK "
                f"(probability of default: {inst['predicted_proba']:.0%})"
            )

        # Load the three explanations
        explanations = {}
        for label in LABELS:
            arm = label_map[label]
            path = os.path.join(LLM_DIR, f"{iid}_{case}_{arm}.json")
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            explanations[label] = data["raw_response"]

        question_block = {
            "instance_id":      iid,
            "case_type":        case,
            "predicted_outcome": outcome_text,
            "explanations": {
                "A": explanations["A"],
                "B": explanations["B"],
                "C": explanations["C"],
            },
            "likert_questions": [
                "I understand why this decision was made.",
                "I trust this explanation.",
                "I know what I could do differently, if anything.",
            ],
            "preference_question":
                "Which explanation (A, B, or C) did you find most helpful overall?",
        }
        survey_questions.append(question_block)

    # Save survey key (hidden)
    key_path = os.path.join(EVAL_DIR, "survey_key.json")
    with open(key_path, "w") as f:
        json.dump(survey_key, f, indent=2)
    print(f"\n  [saved] {key_path} (DO NOT share with participants)")

    # Save survey questions (participant-facing)
    q_path = os.path.join(EVAL_DIR, "survey_questions.json")
    with open(q_path, "w", encoding="utf-8") as f:
        json.dump(survey_questions, f, indent=2, ensure_ascii=False)
    print(f"  [saved] {q_path}")

    # Generate HTML
    html = generate_html(survey_questions)
    html_path = os.path.join(EVAL_DIR, "survey.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  [saved] {html_path}")

    print("\n" + "=" * 60)
    print("INSTRUCTIONS FOR HUMAN EVALUATION")
    print("=" * 60)
    print("""
  1. Send reports/evaluation/survey.html to at least 8-10 people.
  2. Each person opens it in any browser, fills it out, and clicks
     "Download my responses as JSON".
  3. Collect their downloaded response JSON files into:
     reports/evaluation/survey_responses/
  4. Then run: python src/aggregate_evaluation.py
     to un-blind and aggregate results.

  DO NOT share survey_key.json with participants — it reveals
  which explanation is which method.
""")


def generate_html(survey_questions):
    """Generate a simple, functional HTML survey form."""

    # Build the question blocks
    question_blocks = []
    for idx, q in enumerate(survey_questions):
        iid = q["instance_id"]
        case_label = q["case_type"].replace("_", " ").title()

        block = f"""
    <div class="instance-block">
      <h2>Scenario {idx+1}: {case_label} (ID {iid})</h2>
      <p class="outcome"><strong>Model decision:</strong> {q['predicted_outcome']}</p>
"""
        # Three explanations
        for label in ["A", "B", "C"]:
            text = q["explanations"][label].replace("\n", "<br>")
            block += f"""
      <div class="explanation">
        <h3>Explanation {label}</h3>
        <div class="explanation-text">{text}</div>
"""
            # Likert questions for this explanation
            for li, lq in enumerate(q["likert_questions"]):
                name = f"q{iid}_{label}_likert{li}"
                block += f"""
        <p class="likert-q">{lq}</p>
        <div class="likert-scale">
"""
                for val in range(1, 6):
                    labels_txt = {1: "Strongly Disagree", 2: "Disagree",
                                  3: "Neutral", 4: "Agree", 5: "Strongly Agree"}
                    block += f"""          <label>
            <input type="radio" name="{name}" value="{val}" required>
            {val} - {labels_txt[val]}
          </label>
"""
                block += "        </div>\n"

            block += "      </div>\n"

        # Preference question
        pref_name = f"q{iid}_preference"
        block += f"""
      <div class="preference">
        <p><strong>{q['preference_question']}</strong></p>
        <select name="{pref_name}" required>
          <option value="">-- Select --</option>
          <option value="A">Explanation A</option>
          <option value="B">Explanation B</option>
          <option value="C">Explanation C</option>
        </select>
      </div>
    </div>
"""
        question_blocks.append(block)

    all_blocks = "\n".join(question_blocks)

    # Collect all instance IDs for the JS
    instance_ids = [q["instance_id"] for q in survey_questions]
    js_ids = json.dumps(instance_ids)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Credit Decision Explanation Survey</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto;
           padding: 20px; background: #f9f9f9; }}
    h1 {{ color: #333; border-bottom: 2px solid #2196F3; padding-bottom: 10px; }}
    .instance-block {{ background: white; padding: 20px; margin: 20px 0;
                       border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    .explanation {{ background: #f5f5f5; padding: 15px; margin: 10px 0;
                    border-left: 4px solid #2196F3; border-radius: 4px; }}
    .explanation h3 {{ margin-top: 0; color: #2196F3; }}
    .explanation-text {{ line-height: 1.6; }}
    .likert-q {{ font-weight: bold; margin-bottom: 5px; }}
    .likert-scale {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px; }}
    .likert-scale label {{ background: #e3f2fd; padding: 5px 10px; border-radius: 4px;
                           cursor: pointer; font-size: 0.9em; }}
    .likert-scale input:checked + span,
    .likert-scale label:has(input:checked) {{ background: #2196F3; color: white; }}
    .preference {{ margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd; }}
    select {{ padding: 8px; font-size: 1em; }}
    .outcome {{ background: #fff3e0; padding: 10px; border-radius: 4px;
                border-left: 4px solid #ff9800; }}
    #submit-area {{ text-align: center; margin: 30px 0; }}
    button {{ background: #2196F3; color: white; border: none; padding: 15px 40px;
             font-size: 1.1em; border-radius: 8px; cursor: pointer; }}
    button:hover {{ background: #1976D2; }}
    .instructions {{ background: #e8f5e9; padding: 15px; border-radius: 8px;
                     border-left: 4px solid #4CAF50; margin-bottom: 20px; }}
  </style>
</head>
<body>
  <h1>Credit Decision Explanation Survey</h1>

  <div class="instructions">
    <p><strong>Instructions:</strong> You will see 5 scenarios. For each scenario,
    you will read 3 different explanations (A, B, C) of the same credit decision.
    Please rate each explanation on the 3 statements using the 1-5 scale, then
    choose which explanation you found most helpful overall.</p>
    <p>There are no right or wrong answers. We are interested in your honest opinion.</p>
    <p>When done, click "Download my responses as JSON" at the bottom and send
    the downloaded file back to the researcher.</p>
  </div>

  <form id="surveyForm">
{all_blocks}

    <div id="submit-area">
      <button type="button" onclick="downloadResponses()">
        Download my responses as JSON
      </button>
    </div>
  </form>

  <script>
    function downloadResponses() {{
      var form = document.getElementById('surveyForm');
      var data = {{}};
      var instanceIds = {js_ids};

      // Check all required fields
      var radios = form.querySelectorAll('input[type=radio]');
      var radioGroups = {{}};
      radios.forEach(function(r) {{
        radioGroups[r.name] = radioGroups[r.name] || false;
        if (r.checked) radioGroups[r.name] = true;
      }});
      var selects = form.querySelectorAll('select');
      var allFilled = true;
      for (var g in radioGroups) {{
        if (!radioGroups[g]) {{ allFilled = false; break; }}
      }}
      selects.forEach(function(s) {{
        if (!s.value) allFilled = false;
      }});

      if (!allFilled) {{
        alert('Please answer all questions before downloading.');
        return;
      }}

      instanceIds.forEach(function(iid) {{
        data[iid] = {{}};
        ['A', 'B', 'C'].forEach(function(label) {{
          data[iid][label] = {{}};
          for (var li = 0; li < 3; li++) {{
            var name = 'q' + iid + '_' + label + '_likert' + li;
            var checked = form.querySelector('input[name="' + name + '"]:checked');
            var keys = ['understanding', 'trust', 'knows_next_step'];
            data[iid][label][keys[li]] = checked ? parseInt(checked.value) : null;
          }}
        }});
        var prefSelect = form.querySelector('select[name="q' + iid + '_preference"]');
        data[iid]['most_helpful'] = prefSelect ? prefSelect.value : null;
      }});

      data['timestamp'] = new Date().toISOString();

      var blob = new Blob([JSON.stringify(data, null, 2)],
                          {{type: 'application/json'}});
      var url = URL.createObjectURL(blob);
      var a = document.createElement('a');
      a.href = url;
      a.download = 'survey_response_' + Date.now() + '.json';
      a.click();
      URL.revokeObjectURL(url);
      alert('Response downloaded! Please send the file to the researcher.');
    }}
  </script>
</body>
</html>"""
    return html


if __name__ == "__main__":
    build_survey()
