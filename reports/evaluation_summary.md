# Evaluation Summary

The three explanation approaches were evaluated using rule-based checks, LLM-based judging, and readability scores. Human evaluation has not yet been completed, so the results below are based only on the automated evaluation.
## Master Comparison Table

| Metric | shap_only | cf_only | combined | Best Arm |
|---|---|---|---|---|
| Avg Top-Feature Recall | 0.867 | 0.767 | **0.967** | combined |
| Contamination Rate | 0.0 | 0.0 | 0.0 | (tie) |
| Avg LLM Faithfulness (1-10) | 7.8 | **9.4** | 8.8 | cf_only |
| Avg LLM Understandability (1-10) | 9.0 | **9.8** | **9.8** | cf_only / combined |
| Avg LLM Actionability (1-10) | 2.0 | 6.4 | **9.0** | combined |
| Avg Flesch-Kincaid Grade (lower=easier) | 10.18 | 10.50 | **9.53** | combined |
| Avg CF Validity Rate | N/A | 1.0 | 1.0 | (tie) |
| Avg CF Feasibility Rate | N/A | 0.22 | 0.22 | (tie) |

## Key Finding: The Combined Arm Wins
The combined arm (SHAP + counterfactuals together) does NOT uniformly
dominate on every dimension. Here is the honest breakdown:

## Where Combined Wins Clearly

### 1. Feature Recall

**Combined: 0.967**

- Highest feature recall among the three approaches.
- SHAP-only: 0.867
- CF-only: 0.767

### 2. Actionability

**Combined: 9.0**

- Highest actionability score among the three approaches.
- SHAP-only: 2.0
- CF-only: 6.4
- SHAP-only was not designed to provide recommendations.
  
### 3. Readability

**Combined: 9.53**

- Combined achieved the lowest Flesch-Kincaid grade among the three approaches.
- A lower Flesch-Kincaid grade indicates easier reading.


## Where Combined Does NOT Win

### 4. Faithfulness

**CF-only: 9.4**

- Highest faithfulness score among the three approaches.
- Combined: 8.8
- SHAP-only: 7.8

### 5. Understandability

**CF-only and Combined: 9.8**

- Both approaches received the same highest score.
- SHAP-only: 9.0

## Contamination

**0% contamination across all 15 explanations.**

- SHAP-only did not introduce counterfactual recommendations.
- CF-only did not introduce SHAP-style explanations.


## Counterfactual Feasibility

- **24/24 (100%)** raw counterfactuals changed the model prediction.
- **5/24 (20.8%)** passed the predefined feasibility filter.
- This shows that **valid counterfactuals are not necessarily feasible recommendations**.

## Interpretation

**No single approach performed best across all metrics.**

- **Combined:** highest feature recall and actionability.
- **CF-only:** highest faithfulness.
- **CF-only and Combined:** highest understandability.
- **Human evaluation:** still pending.

## Limitations

1. **LLM judge:** possible evaluation bias.

2. **Small sample:** only 5 classification instances per arm.

3. **Human evaluation:** pending.


## Current Conclusion

No single approach was best on every metric.

- Combined:  strongest on feature recall and actionability.

- CF-only:  highest faithfulness.


- CF-only + Combined:  highest understandability.

Human evaluation is still required before drawing the final conclusion.