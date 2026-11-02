"""Run deterministic structured-memory extraction evaluation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.memory_extractor import extract_memory_candidates


CASE_PATH = Path(__file__).resolve().parent / "memory_cases.json"
REPORT_PATH = Path(__file__).resolve().parent / "reports" / "memory_latest.json"


def _contains_value(candidates: list[Any], expected: str) -> bool:
    needle = expected.casefold()
    return any(needle in json.dumps(candidate.memory_value, ensure_ascii=False).casefold() for candidate in candidates)


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    candidates = extract_memory_candidates(str(case.get("text", "")))
    expected_keys = set(case.get("expected_keys") or [])
    actual_keys = {candidate.memory_key for candidate in candidates}
    key_tp = len(expected_keys & actual_keys)
    key_precision = key_tp / len(actual_keys) if actual_keys else (1.0 if not expected_keys else 0.0)
    key_recall = key_tp / len(expected_keys) if expected_keys else (1.0 if not actual_keys else 0.0)
    key_f1 = 2 * key_precision * key_recall / (key_precision + key_recall) if key_precision + key_recall else 0.0
    values = list(case.get("expected_values") or [])
    value_checks = {value: _contains_value(candidates, str(value)) for value in values}
    high_risk = bool(case.get("high_risk"))
    high_risk_safe = not high_risk or all(candidate.sensitivity != "high" or candidate.explicit for candidate in candidates)
    high_risk_not_auto_active = not high_risk or all(
        candidate.sensitivity == "high" for candidate in candidates if candidate.memory_key in {"allergies", "health_goal"}
    )
    temporal_ok = not case.get("temporal_scope") or any(
        candidate.temporal_scope == case["temporal_scope"] for candidate in candidates
    )
    checks = {
        "key_precision": key_precision >= 1.0,
        "key_recall": key_recall >= 1.0,
        "values": all(value_checks.values()),
        "high_risk_explicit": high_risk_safe,
        "high_risk_not_auto_active": high_risk_not_auto_active,
        "temporal_scope": temporal_ok,
    }
    return {
        "case_id": case.get("case_id"),
        "text": case.get("text"),
        "actual_keys": sorted(actual_keys),
        "expected_keys": sorted(expected_keys),
        "value_checks": value_checks,
        "key_precision": round(key_precision, 4),
        "key_recall": round(key_recall, 4),
        "key_f1": round(key_f1, 4),
        "checks": checks,
        "passed": all(checks.values()),
    }


def run(path: str | Path = CASE_PATH) -> dict[str, Any]:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    results = [evaluate_case(case) for case in cases]
    total = len(results) or 1
    high_risk_cases = [result for result, case in zip(results, cases) if case.get("high_risk")]
    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "case_count": len(results),
            "passed_count": sum(result["passed"] for result in results),
            "extraction_key_precision": round(sum(result["key_precision"] for result in results) / total, 4),
            "extraction_key_recall": round(sum(result["key_recall"] for result in results) / total, 4),
            "extraction_key_f1": round(sum(result["key_f1"] for result in results) / total, 4),
            "high_risk_safety_rate": round(sum(result["checks"]["high_risk_not_auto_active"] for result in high_risk_cases) / (len(high_risk_cases) or 1), 4),
            "negative_case_precision": round(sum(result["key_precision"] for result, case in zip(results, cases) if not case.get("expected_keys")) / (sum(not case.get("expected_keys") for case in cases) or 1), 4),
        },
        "passed": all(result["passed"] for result in results),
        "cases": results,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="运行结构化长期记忆确定性评测")
    parser.add_argument("--cases", type=Path, default=CASE_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    report = run(args.cases)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "metrics": report["metrics"], "report": str(args.report)}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
