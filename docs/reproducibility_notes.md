# Reproducibility notes

## Environment
The preserved local environment used Python 3.11.9. The exact installed OpenAI SDK build was not recorded; the project requirement was `openai>=2.0.0`. Hosted-model replay is not guaranteed to reproduce identical outputs.

## Frozen generation
- Model ID: `gpt-6-luna`
- API: OpenAI Responses API
- Prompt IDs: `G1_STANDARD_RAG_V1`, `G2_EVIDENCE_AWARE_RAG_V4`
- Final completed outputs: 60 G1 and 60 G2
- Unspecified generation parameters used provider defaults.

The published final outputs are provided in `data/evaluation/`; re-running generation is optional and may not reproduce the same responses.

## Reproduce analyses
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/analyze_final_results.py
python scripts/analyze_retrieval_to_answer.py
```

Generated reproductions are written to `reproduced_results/`.

## Optional re-generation
Copy `.env.example` to `.env`, add an API key, then run:
```bash
python scripts/generate_answers.py --methods G1 G2
python scripts/validate_generations.py
```
Do not overwrite the frozen outputs used in the manuscript.
