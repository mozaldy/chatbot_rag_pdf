#!/usr/bin/env python3
"""
Lightweight RAG quality evaluator.

Runs a fixed query set against /api/chat and reports:
- response success rate
- citation presence and citation validity
- source count
- grounding token overlap (answer tokens found in retrieved source text)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, List

import requests


DEFAULT_CASES = [
    {"id": "q1", "question": "Summarize the key capabilities in the uploaded document."},
    {"id": "q2", "question": "What evidence mentions OCR or text extraction?"},
    {"id": "q3", "question": "What limitations or constraints are explicitly documented?"},
    {"id": "q4", "question": "Which models or tools are named, and what are they used for?"},
    {"id": "q5", "question": "Quote the section that discusses performance or throughput."},
]

CITATION_PATTERN = re.compile(r"\[(S\d+)\]")
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]{3,}")


@dataclass
class CaseResult:
    case_id: str
    question: str
    http_status: int
    ok: bool
    answer_chars: int
    source_count: int
    cited_ids: List[str]
    citations_present: bool
    citations_valid: bool
    grounding_overlap: float
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local RAG answer quality.")
    parser.add_argument("--api-base", default="http://localhost:8000/api", help="API base URL")
    parser.add_argument(
        "--cases",
        type=Path,
        default=None,
        help="Optional JSON file with [{'id': ..., 'question': ...}]",
    )
    parser.add_argument("--timeout", type=float, default=90.0, help="Per-request timeout seconds")
    parser.add_argument("--out", type=Path, default=None, help="Optional output JSON path")
    return parser.parse_args()


def load_cases(path: Path | None) -> List[dict[str, Any]]:
    if path is None:
        return DEFAULT_CASES
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Cases file must be a list of objects.")
    return payload


def grounding_overlap(answer: str, sources: Iterable[dict[str, Any]]) -> float:
    answer_terms = {tok.lower() for tok in TOKEN_PATTERN.findall(answer)}
    if not answer_terms:
        return 0.0
    source_blob = " ".join((src.get("text") or "") for src in sources).lower()
    hits = sum(1 for tok in answer_terms if tok in source_blob)
    return hits / float(len(answer_terms))


def run_case(api_base: str, case: dict[str, Any], timeout: float) -> CaseResult:
    case_id = str(case.get("id", "unknown"))
    question = str(case.get("question", "")).strip()
    if not question:
        return CaseResult(
            case_id=case_id,
            question=question,
            http_status=0,
            ok=False,
            answer_chars=0,
            source_count=0,
            cited_ids=[],
            citations_present=False,
            citations_valid=False,
            grounding_overlap=0.0,
            error="Empty question",
        )

    try:
        response = requests.post(
            f"{api_base}/chat",
            json={"messages": question},
            timeout=timeout,
        )
    except Exception as exc:
        return CaseResult(
            case_id=case_id,
            question=question,
            http_status=0,
            ok=False,
            answer_chars=0,
            source_count=0,
            cited_ids=[],
            citations_present=False,
            citations_valid=False,
            grounding_overlap=0.0,
            error=str(exc),
        )

    if response.status_code != 200:
        return CaseResult(
            case_id=case_id,
            question=question,
            http_status=response.status_code,
            ok=False,
            answer_chars=0,
            source_count=0,
            cited_ids=[],
            citations_present=False,
            citations_valid=False,
            grounding_overlap=0.0,
            error=response.text[:500],
        )

    payload = response.json()
    answer = str(payload.get("response", ""))
    sources = payload.get("sources", []) or []
    cited_ids = sorted(set(CITATION_PATTERN.findall(answer)))
    source_ids = {src.get("id") for src in sources}
    citations_valid = bool(cited_ids) and all(cited in source_ids for cited in cited_ids)

    return CaseResult(
        case_id=case_id,
        question=question,
        http_status=response.status_code,
        ok=True,
        answer_chars=len(answer),
        source_count=len(sources),
        cited_ids=cited_ids,
        citations_present=bool(cited_ids),
        citations_valid=citations_valid,
        grounding_overlap=grounding_overlap(answer, sources),
    )


def summarize(results: List[CaseResult]) -> dict[str, Any]:
    total = len(results) or 1
    ok_results = [r for r in results if r.ok]
    return {
        "cases_total": len(results),
        "success_rate": len(ok_results) / total,
        "citation_presence_rate": sum(r.citations_present for r in results) / total,
        "citation_valid_rate": sum(r.citations_valid for r in results) / total,
        "avg_source_count": sum(r.source_count for r in results) / total,
        "avg_grounding_overlap": sum(r.grounding_overlap for r in results) / total,
    }


def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases)
    results = [run_case(args.api_base, case, args.timeout) for case in cases]
    aggregate = summarize(results)

    print("\nRAG Evaluation")
    print("=" * 80)
    for result in results:
        status = "OK" if result.ok else "FAIL"
        print(
            f"{result.case_id:<8} {status:<5} "
            f"citations={result.cited_ids} "
            f"sources={result.source_count} "
            f"grounding={result.grounding_overlap:.2f}"
        )
        if result.error:
            print(f"  error: {result.error}")

    print("-" * 80)
    for key, value in aggregate.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")

    if args.out:
        payload = {
            "aggregate": aggregate,
            "results": [asdict(r) for r in results],
        }
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved report: {args.out}")


if __name__ == "__main__":
    main()
