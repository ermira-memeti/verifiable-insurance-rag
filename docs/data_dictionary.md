# Data dictionary

## Core benchmark
- `data/core/questions.csv`: development and held-out questions, policy metadata, benchmark reference answers, and task descriptors.
- `data/core/passages.csv`: processed policy passages with stable passage IDs and source-page metadata.
- `data/core/evidence_map.csv`: question-to-evidence-unit mappings used by the frozen retrieval benchmark.
- `data/retrieval/retrieval_runs.csv`: frozen ranked passage outputs for the retrieval configurations.

## Frozen downstream evaluation
- `data/pilot/pilot_questions.csv`: 12-question purposive evaluation subset.
- `data/pilot/pilot_conditions.jsonl`: 60 controlled evidence scenarios supplied to G1 and G2.
- `data/evaluation/generation_outputs_g1.jsonl`: sanitized completed G1 outputs.
- `data/evaluation/generation_outputs_g2.jsonl`: sanitized completed G2 outputs.
- `data/evaluation/evidence_sufficiency_labels.csv`: de-identified human consensus labels for the 60 evidence scenarios.
- `data/evaluation/human_answer_scores_deidentified.csv`: de-identified consensus answer scores for all 120 completed responses.

## Results
- `results/tables/`: reported and supporting analysis tables.
- `results/figures/`: publication figures.
