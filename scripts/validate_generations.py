#!/usr/bin/env python3
"""Validate G1/G2 pilot output files and export a compact CSV summary."""
from __future__ import annotations
import csv, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "generations" / "parsed" / "pilot_generation_validation.csv"


def load(path):
    if not path.exists(): return []
    rows=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip(): rows.append(json.loads(line))
    return rows


def main():
    rows=[]
    for method in ("G1","G2"):
        path=ROOT/"generations"/"raw"/f"pilot_{method.lower()}_responses.jsonl"
        for r in load(path):
            if r.get("status") == "dry_run":
                continue
            p=r.get("parsed_response") or {}
            rows.append({
                "generation_key":r.get("generation_key"),
                "question_id":r.get("question_id"),
                "condition_id":r.get("condition_id"),
                "method":method,
                "model":r.get("model"),
                "status":r.get("status"),
                "abstained":p.get("abstained"),
                "evidence_sufficiency":p.get("evidence_sufficiency", ""),
                "citation_count":len(p.get("citations",[])) if isinstance(p.get("citations",[]),list) else "",
                "validation_errors":";".join(r.get("validation_errors",[])),
                "api_response_id":r.get("api_response_id"),
                "output_sha256":r.get("output_sha256"),
            })
    OUT.parent.mkdir(parents=True,exist_ok=True)
    fields=list(rows[0].keys()) if rows else ["generation_key"]
    with OUT.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"Rows: {len(rows)}")
    print("Status:", dict(Counter(r["status"] for r in rows)))
    print("Methods:", dict(Counter(r["method"] for r in rows)))
    print("Conditions:", dict(Counter(r["condition_id"] for r in rows)))
    print(f"Wrote: {OUT}")
    return 0

if __name__=="__main__": raise SystemExit(main())
