"""Deterministic context-window evaluation and JSON report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import settings
from core.context_assembler import ContextAssembler
from core.context_budget import ContextBudget
from core.context_summary import ContextSummaryService

CASE_PATH = Path(__file__).resolve().parent / "context_cases.json"
REPORT_PATH = Path(__file__).resolve().parent / "reports" / "context_latest.json"


def _fallback_llm(**_: Any) -> Any:
    raise RuntimeError("deterministic context evaluation")


def _contains_fact(case: dict[str, Any], preparation: Any) -> bool:
    haystack = json.dumps(preparation.summary, ensure_ascii=False) + json.dumps(
        preparation.recent_messages, ensure_ascii=False
    )
    return all(str(fact).casefold() in haystack.casefold() for fact in case.get("expected_facts", []))


def run_context_evaluation(path: str | Path = CASE_PATH) -> dict[str, Any]:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    original = {
        "context_compaction_enabled": settings.context_compaction_enabled,
        "context_recent_turns": settings.context_recent_turns,
        "context_compaction_batch_turns": settings.context_compaction_batch_turns,
    }
    settings.context_compaction_enabled = True
    settings.context_recent_turns = 2
    settings.context_compaction_batch_turns = 2
    budget = ContextBudget(2048, 256, 128, 64, 0.75, 0.9)
    assembler = ContextAssembler(
        summary_service=ContextSummaryService(llm_factory=_fallback_llm),
        budget=budget,
    )
    results: list[dict[str, Any]] = []
    try:
        for case in cases:
            preparation = assembler.prepare(
                list(case.get("history", [])),
                {"role": "user", "content": case["current_message"]},
                structured_state={
                    "current_goal": case.get("grill_fields", {}).get("goal", ""),
                    "grill_context": {"fields": case.get("grill_fields", {})},
                },
                user_id="context-eval-user",
            )
            estimated = int(preparation.metadata["estimated_input_tokens"])
            budget_ok = estimated <= budget.input_budget
            summary_facts_ok = _contains_fact(case, preparation)
            state = preparation.packet.get("structured_state", {})
            actual_fields = ((state.get("grill_context") or {}).get("fields") or {}) if isinstance(state, dict) else {}
            grill_ok = all(actual_fields.get(key) == value for key, value in case.get("grill_fields", {}).items())
            compact_ok = bool(preparation.metadata.get("compacted")) == bool(case.get("expected_compaction"))
            checks = {
                "budget_adherence": budget_ok,
                "summary_fact_retention": summary_facts_ok,
                "grill_state_retention": grill_ok,
                "expected_compaction": compact_ok,
                "current_question_retained": preparation.recent_messages[-1]["content"] == case["current_message"],
            }
            results.append(
                {
                    "case_id": case["case_id"],
                    "checks": checks,
                    "passed": all(checks.values()),
                    "details": {
                        "estimated_input_tokens": estimated,
                        "budget_tokens": budget.input_budget,
                        "compacted": preparation.metadata.get("compacted", False),
                        "fallback": preparation.metadata.get("fallback", False),
                        "summary_version": preparation.metadata.get("summary_version", 0),
                    },
                }
            )
    finally:
        for key, value in original.items():
            setattr(settings, key, value)

    total = len(results) or 1
    metrics = {
        "case_count": len(results),
        "context_budget_adherence_rate": sum(item["checks"]["budget_adherence"] for item in results) / total,
        "context_compaction_success_rate": sum(item["checks"]["expected_compaction"] for item in results) / total,
        "summary_fact_retention_rate": sum(item["checks"]["summary_fact_retention"] for item in results) / total,
        "grill_state_retention_rate": sum(item["checks"]["grill_state_retention"] for item in results) / total,
        "context_overflow_rate": sum(not item["checks"]["budget_adherence"] for item in results) / total,
        "context_fallback_rate": sum(item["details"]["fallback"] for item in results) / total,
    }
    return {
        "schema_version": "1.0",
        "metrics": metrics,
        "passed": all(item["passed"] for item in results),
        "cases": results,
    }


def main() -> int:
    report = run_context_evaluation()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "metrics": report["metrics"]}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
