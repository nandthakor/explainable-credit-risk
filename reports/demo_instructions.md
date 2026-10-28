# Demo Instructions — Phase 7

## How to Run the App

Open a terminal in the project root and run:

```
streamlit run app.py
```

Streamlit will open a browser tab automatically (usually at http://localhost:8501).
If it doesn't, open that URL manually.

---

## Two Modes

### Replay Mode (recommended for live presentations)

- **No API calls, no internet required** once the app has loaded.
- Loads pre-computed results from Phases 3-5 (saved JSON/PNG files).
- All 5 case-study instances are available: True Positive, True Negative,
  False Positive, False Negative, and Borderline.
- Completely reliable — nothing can fail mid-presentation.

### Live Mode

- Runs the **real pipeline** on a new applicant in real time:
  model prediction → SHAP → DiCE counterfactuals → Groq LLM (3 calls).
- Two input options: random test-set row, or custom feature values.
- Counterfactual generation can take **15-30 seconds** (DiCE genetic search).
- LLM calls take **5-15 seconds** total (3 calls at ~3s each).
- If the LLM call fails (rate limit, no internet, expired key), the app
  displays a clear warning and still shows the SHAP and CF results.

---

## Pre-Presentation Checklist

1. **Test Live mode at least once beforehand** to confirm your Groq API key
   is still active. Keys expire or hit rate limits on free-tier plans.

2. Check that `.env` contains a valid key:
   ```
   GROQ_API_KEY=your_key_here
   ```

3. Make sure you're in the project root when you run `streamlit run app.py`
   (not inside the `src/` folder — relative paths will break).

4. For a presentation, prefer **Replay mode** — it is instant and cannot
   fail mid-demo. Show Live mode as a bonus if time allows.

---

## What Each Section Shows

| Section | Data source |
|---|---|
| Applicant Profile | Loaded from selected_instances.json (Replay) or entered by user (Live) |
| Model Prediction | Real LogisticRegression pipeline from Phase 2 |
| SHAP Waterfall | Saved PNG from Phase 3 (Replay) / computed live (Live) |
| SHAP Feature Table | Saved JSON from Phase 3 (Replay) / computed live (Live) |
| Counterfactuals | Saved JSON from Phase 4 (Replay) / DiCE live (Live) |
| LLM Explanations | Saved JSON from Phase 5 (Replay) / Groq API live (Live) |
| Research Finding | Master comparison table from Phase 6 (always from saved files) |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit: command not found` | Run `pip install streamlit` |
| `GROQ_API_KEY not found` | Check `.env` file exists in the project root |
| DiCE takes very long or hangs | This is normal for genetic search — wait up to 60s |
| Streamlit shows import errors | Run from the project root, not from `src/` |
