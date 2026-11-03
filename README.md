# From Evidence Retrieval to Verifiable Answers

This repository accompanies the manuscript **"From Evidence Retrieval to Verifiable Answers: Evaluating Evidence Sufficiency, Faithfulness, and Abstention in Insurance-Policy RAG"**.

## Study overview

The study evaluates how retrieval quality, semantic evidence sufficiency, and generation strategy interact in insurance-policy retrieval-augmented generation.

- 12 held-out FEMA insurance-policy questions
- 5 controlled evidence constructions per question
- 60 evidence scenarios
- 2 frozen prompting strategies (G1 and G2)
- 120 completed GPT-6 Luna responses
- human evaluation of evidence sufficiency and answer quality
- retrieval-to-generation and abstention analyses

## Repository contents

```text
config/                 Frozen generation configuration
prompts/                Full G1 and G2 prompt texts
scripts/                Scenario construction, generation, validation, and analysis

data/core/              Questions, passages, and evidence mappings
data/retrieval/         Frozen retrieval rankings
data/pilot/             Frozen 12-question / 60-scenario evaluation set
data/evaluation/        Sanitized model outputs and de-identified human labels

results/tables/         Final analysis tables
results/figures/        Publication figures
manifests/              Frozen selection and validation manifests
docs/                   Data dictionary, rubric, and reproducibility notes
audit/                  Sanitized non-final generation audit records
```

## Quick start

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python scripts/analyze_final_results.py
python scripts/analyze_retrieval_to_answer.py
```

The analyses use the frozen, de-identified evaluation data and do not require an OpenAI API key.

## Optional answer generation

Generation requires an OpenAI API key and may not reproduce the original hosted-model outputs exactly.

```bash
copy .env.example .env          # Windows Command Prompt
# or: cp .env.example .env      # macOS/Linux

python scripts/generate_answers.py --methods G1 G2
python scripts/validate_generations.py
```

The frozen outputs used in the manuscript are already included under `data/evaluation/`.

## Privacy and exclusions

The public package intentionally excludes:

- `.env` and API credentials;
- `.venv` and local environment binaries;
- reviewer identities and private reviewer workbooks;
- AI-reference reviewer files;
- calibration outputs and superseded scripts;
- confirmatory-experiment drafts that were not part of the reported study;
- local Windows/OneDrive paths.

See `DATA_USAGE.md` before publishing processed policy text.

## Citation

Citation metadata are provided in `CITATION.cff`. Add the final journal DOI and repository DOI after publication/archive deposition.

## License

Source code is released under the MIT License. Third-party and derived data are subject to the conditions described in `DATA_USAGE.md`.
