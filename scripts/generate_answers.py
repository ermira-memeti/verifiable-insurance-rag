#!/usr/bin/env python3
"""Generate Paper 3 pilot answers for G1 and G2 using frozen evidence inputs.

Features:
- OpenAI Responses API
- frozen local prompt files
- G1/G2 use the same model and API settings
- JSONL append-only outputs
- resume by generation_key
- retry with exponential backoff
- dry-run mode (no API calls)
- input/prompt/output SHA-256 provenance
- citation validation against supplied passage IDs

Run from 01_Implementation (recommended):
    python scripts/03_generate_pilot_answers.py --methods G1 G2

Test without API calls:
    python scripts/03_generate_pilot_answers.py --methods G1 G2 --dry-run --limit 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "generation_config.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")


def sanitize_value(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, list):
        return [sanitize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: sanitize_value(v) for k, v in value.items()}
    return value


def format_evidence(record: dict[str, Any]) -> str:
    blocks = []
    for idx, p in enumerate(record.get("condition_passages", []), start=1):
        p = sanitize_value(p)
        pid = p.get("passage_id", "")
        page = p.get("pdf_page", "")
        text = str(p.get("passage_text", "")).strip()
        blocks.append(f"Evidence {idx}\nPassage ID: {pid}\nPDF page: {page}\nText: {text}")
    return "\n\n".join(blocks)


def build_user_input(record: dict[str, Any]) -> str:
    return (
        f"Policy: {record.get('policy_name', record.get('policy_id', ''))}\n"
        f"Question: {record['question_text']}\n\n"
        f"SUPPLIED EVIDENCE\n{format_evidence(record)}"
    )


def strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def parse_json_response(text: str) -> tuple[dict[str, Any] | None, str | None]:
    cleaned = strip_code_fence(text)
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            return None, "JSON response is not an object"
        return parsed, None
    except json.JSONDecodeError as exc:
        # Conservative recovery: extract the outermost JSON object once.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(cleaned[start : end + 1])
                if isinstance(parsed, dict):
                    return parsed, f"Recovered JSON object after initial parse error: {exc}"
            except json.JSONDecodeError:
                pass
        return None, str(exc)


def validate_payload(method: str, payload: dict[str, Any] | None, allowed_ids: set[str]) -> list[str]:
    errors: list[str] = []
    if payload is None:
        return ["response_not_parseable_json"]
    required = ["answer", "citations", "abstained"]
    if method == "G2":
        required += ["evidence_sufficiency", "missing_information"]
    for field in required:
        if field not in payload:
            errors.append(f"missing_field:{field}")
    citations = payload.get("citations", [])
    if not isinstance(citations, list):
        errors.append("citations_not_list")
    else:
        invalid = [str(c) for c in citations if str(c) not in allowed_ids]
        if invalid:
            errors.append("invalid_citations:" + "|".join(invalid))
    if method == "G2" and payload.get("evidence_sufficiency") not in {"sufficient", "partial", "insufficient", None}:
        errors.append("invalid_evidence_sufficiency")
    return errors


def read_completed_keys(paths: list[Path]) -> set[str]:
    keys: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if row.get("status") == "completed" and row.get("generation_key"):
                    keys.add(row["generation_key"])
    return keys


def response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str):
        return text
    return str(text or "")


def call_api(client: Any, model: str, system_prompt: str, user_input: str) -> Any:
    return client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", choices=["G1", "G2"], default=["G1", "G2"])
    parser.add_argument("--model", default=None, help="Override OPENAI_MODEL/config model")
    parser.add_argument("--condition", default=None, help="Run only one condition_id, e.g. C5_INSUFFICIENT")
    parser.add_argument("--limit", type=int, default=None, help="Limit condition records after optional condition filtering")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Regenerate even if generation_key already completed")
    args = parser.parse_args()

    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")

    cfg = load_json(CONFIG_PATH)
    model = args.model or os.getenv("OPENAI_MODEL") or cfg["model"]
    input_path = ROOT / cfg["input_file"]
    records = load_jsonl(input_path)

    if args.condition is not None:
        records = [r for r in records if r.get("condition_id") == args.condition]
        if not records:
            available = sorted({str(r.get("condition_id")) for r in load_jsonl(input_path)})
            print(
                f"ERROR: No records found for condition {args.condition!r}. "
                f"Available conditions: {', '.join(available)}",
                file=sys.stderr,
            )
            return 2

    if args.limit is not None:
        records = records[: args.limit]

    prompts: dict[str, str] = {}
    for method in args.methods:
        prompt_path = ROOT / cfg["prompt_files"][method]
        prompts[method] = prompt_path.read_text(encoding="utf-8")

    output_paths = {m: ROOT / "generations" / "raw" / f"pilot_{m.lower()}_responses.jsonl" for m in args.methods}
    completed = set() if args.force else read_completed_keys(list(output_paths.values()))

    if args.dry_run:
        client = None
    else:
        if OpenAI is None:
            print("ERROR: openai package is not installed. Run: pip install -r requirements.txt", file=sys.stderr)
            return 2
        if not os.getenv("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY is not set. See .env.example", file=sys.stderr)
            return 2
        client = OpenAI()

    run_manifest = {
        "started_at": now_iso(),
        "model": model,
        "methods": args.methods,
        "input_file": str(input_path.relative_to(ROOT)),
        "input_sha256": sha256_text(input_path.read_text(encoding="utf-8")),
        "prompt_sha256": {m: sha256_text(prompts[m]) for m in args.methods},
        "dry_run": args.dry_run,
        "force": args.force,
        "condition_filter": args.condition,
        "condition_records_selected": len(records),
    }
    manifest_path = ROOT / "manifests" / f"generation_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    total = len(records) * len(args.methods)
    done = skipped = failed = 0
    for rec_index, raw_record in enumerate(records, start=1):
        record = sanitize_value(raw_record)
        user_input = build_user_input(record)
        allowed_ids = {str(p.get("passage_id")) for p in record.get("condition_passages", [])}

        for method in args.methods:
            generation_key = f"{record['question_id']}::{record['condition_id']}::{method}::{model}"
            if generation_key in completed and not args.force:
                skipped += 1
                print(f"SKIP {generation_key}")
                continue

            base = {
                "generation_key": generation_key,
                "pilot_version": record.get("pilot_version"),
                "question_id": record["question_id"],
                "policy_id": record.get("policy_id"),
                "policy_name": record.get("policy_name"),
                "condition_id": record["condition_id"],
                "generation_method": method,
                "model": model,
                "question_text": record["question_text"],
                "reference_answer": record.get("reference_answer"),
                "primary_concept": record.get("primary_concept"),
                "condition_passage_ids": record.get("condition_passage_ids", []),
                "condition_passages": record.get("condition_passages", []),
                "prompt_id": prompts[method].splitlines()[0].replace("PROMPT_ID:", "").strip(),
                "prompt_sha256": sha256_text(prompts[method]),
                "user_input_sha256": sha256_text(user_input),
                "created_at": now_iso(),
            }

            if args.dry_run:
                out = {
                    **base,
                    "status": "dry_run",
                    "api_response_id": None,
                    "attempt_count": 0,
                    "raw_response": None,
                    "parsed_response": None,
                    "parse_warning": None,
                    "validation_errors": [],
                    "output_sha256": None,
                }
                append_jsonl(output_paths[method], out)
                done += 1
                print(f"DRY  {generation_key}")
                continue

            last_error = None
            response = None
            attempts = int(cfg.get("retry_attempts", 3))
            for attempt in range(1, attempts + 1):
                try:
                    response = call_api(client, model, prompts[method], user_input)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    if attempt < attempts:
                        time.sleep(int(cfg.get("retry_base_seconds", 2)) * (2 ** (attempt - 1)))

            if response is None:
                failed += 1
                append_jsonl(output_paths[method], {**base, "status": "failed", "attempt_count": attempts, "error": last_error})
                print(f"FAIL {generation_key}: {last_error}", file=sys.stderr)
                continue

            raw_text = response_text(response)
            parsed, parse_warning = parse_json_response(raw_text)
            validation_errors = validate_payload(method, parsed, allowed_ids)
            status = "completed" if parsed is not None else "parse_failed"
            if status != "completed":
                failed += 1
            else:
                done += 1

            out = {
                **base,
                "status": status,
                "api_response_id": getattr(response, "id", None),
                "attempt_count": attempt,
                "raw_response": raw_text,
                "parsed_response": parsed,
                "parse_warning": parse_warning,
                "validation_errors": validation_errors,
                "output_sha256": sha256_text(raw_text),
            }
            append_jsonl(output_paths[method], out)
            print(f"DONE {done + skipped + failed}/{total} {generation_key} status={status}")

    print(json.dumps({"completed_or_dry": done, "skipped": skipped, "failed": failed, "total_planned": total}, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
