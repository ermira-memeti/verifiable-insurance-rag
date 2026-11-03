# Paper 3 - Final Human Evaluation Analysis

## Dataset
- 60 human-validated evidence cases.
- 120 completed answers: 60 G1 and 60 G2.
- Evidence sufficiency: 33 sufficient, 17 partial, 10 insufficient cases.
- Two final independent human reviewers scored all 120 answers.

## Inter-rater agreement
Observed agreement was 100% for every scored dimension. Quadratic weighted Cohen's kappa was 1.000 for correctness, faithfulness, completeness, citation support, and evidence handling. Cohen's kappa was 1.000 for unsupported-claim and contradiction flags.

## Overall human-scored quality
| Measure | G1 | G2 | Difference G2-G1 | Paired p |
|---|---:|---:|---:|---:|
| Correctness | 1.933 | 1.967 | +0.033 | 0.317 |
| Faithfulness | 1.983 | 2.000 | +0.017 | 0.317 |
| Completeness | 1.983 | 1.983 | +0.000 | 1.000 |
| Citation support | 1.983 | 1.983 | +0.000 | 1.000 |
| Evidence handling | 1.967 | 1.983 | +0.017 | 0.317 |

The human-scored metrics show a strong ceiling effect. None of the ordinal paired comparisons provides statistically persuasive evidence of a broad G2 advantage in this pilot.

## Evidence-state pattern
- **Sufficient evidence:** both methods received perfect human scores for all 33 cases.
- **Partial evidence:** human scores were identical between methods across all 17 cases.
- **Insufficient evidence:** G2 was slightly higher on correctness, faithfulness and evidence handling, but these differences were driven by one case and are not statistically stable.

## Formal abstention behavior
A separate structured-field analysis defines alignment as **formal abstention if and only if the human evidence label is INSUFFICIENT**.
- G1 alignment: **48/60 = 80.0%**.
- G2 alignment: **57/60 = 95.0%**.
- Discordant pairs: G2 better on 12 cases; G1 better on 3.
- Case-level exact McNemar p = **0.0352**.
- Question-level sensitivity (12 unique questions): Wilcoxon p = **0.1094**, sign-test p = **0.0703**, cluster bootstrap 95% CI for the G2-G1 difference = **[0.000, 0.283]**.

The structured abstention result is therefore promising but should be described as exploratory because the 60 conditions are nested within only 12 unique questions.

There is also a trade-off: G1 formally abstained on 12 of 17 PARTIAL cases, whereas G2 did not formally abstain on any PARTIAL case. Conversely, G2 did not set its formal abstention flag on 3 of 10 INSUFFICIENT cases. Both human reviewers nevertheless rated those three G2 answers as appropriately evidence-calibrated, showing that the structured abstention flag and semantic answer quality are not identical constructs.

## Recommended interpretation
The current pilot does **not** support a claim that G2 broadly improves answer correctness over G1. It does support the more nuanced conclusion that both methods are highly accurate when the supplied evidence is good, while the evidence-aware prompt changes how uncertainty and partial evidence are handled. The strongest detectable difference is in structured abstention calibration, not in overall answer correctness.

## Next research step
Before writing the final journal Results section, expand from the 12-question pilot to the full held-out benchmark or another independent test set. The pilot has too little score variance for robust correlation analysis between retrieval completeness and downstream answer quality. Preserve these pilot results as a pre-specified development/pilot analysis rather than treating them as the final confirmatory experiment.
