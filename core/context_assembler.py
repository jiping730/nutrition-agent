"""Build bounded model context from durable conversation state."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from config import settings
from core.context_budget import (
    ContextBudget,
    current_budget,
    estimate_messages,
    estimate_tokens,
    fit_messages,
    truncate_text_to_tokens,
)
from core.context_summary import ContextSummaryService, SummaryBuildResult, normalize_summary


@dataclass(frozen=True)
class ContextPreparation:
    recent_messages: list[dict[str, Any]]
    summary: dict[str, Any]
    summary_record: dict[str, Any] | None
    packet: dict[str, Any]
    metadata: dict[str, Any]


class ContextAssembler:
    """Keep recent turns, durable summary, and structured state separate."""

    def __init__(
        self,
        summary_service: ContextSummaryService | None = None,
        budget: ContextBudget | None = None,
    ) -> None:
        self.summary_service = summary_service or ContextSummaryService()
        self.budget = budget or current_budget()

    @staticmethod
    def _covered_index(messages: list[dict[str, Any]], covered_until: Any) -> int:
        if covered_until in (None, ""):
            return -1
        try:
            marker = int(covered_until)
        except (TypeError, ValueError):
            return -1
        index = -1
        for position, message in enumerate(messages):
            try:
                if int(message.get("id", -1)) <= marker:
                    index = position
            except (TypeError, ValueError):
                continue
        return index

    @staticmethod
    def _record_from_result(
        result: SummaryBuildResult,
        *,
        version: int,
        covered_until_message_id: int | None,
    ) -> dict[str, Any]:
        return {
            "summary_json": result.payload,
            "summary_text": result.summary_text,
            "summary_version": version,
            "covered_until_message_id": covered_until_message_id,
            "estimated_tokens": result.estimated_tokens,
            "source_model": result.source_model,
            "content_hash": result.content_hash,
            "fallback": result.fallback,
        }

    def prepare(
        self,
        history: list[dict[str, Any]],
        current_message: dict[str, Any],
        *,
        existing_summary: dict[str, Any] | None = None,
        structured_state: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> ContextPreparation:
        existing = existing_summary or {}
        summary = normalize_summary(existing.get("summary_json", existing))
        total_tokens = estimate_messages(history) + estimate_tokens(current_message.get("content", "")) + 4
        existing_covered_index = self._covered_index(history, existing.get("covered_until_message_id"))
        uncompressed_count = len(history) - (existing_covered_index + 1)
        should_compact = settings.context_compaction_enabled and self.budget.should_compact(
            total_tokens, uncompressed_count
        )
        recent_count = max(2, int(settings.context_recent_turns) * 2)
        if should_compact and total_tokens >= self.budget.soft_limit and len(history) > 2:
            # Large individual messages can exceed the budget before the
            # configured recent-turn window is reached.  In that case retain
            # a smaller recent suffix so the older prefix can be summarized.
            recent_count = min(recent_count, max(2, len(history) // 2))
        recent = history[-recent_count:]
        summary_record: dict[str, Any] | None = None
        compacted = False
        fallback = False

        if should_compact and history:
            prefix = history[:-recent_count] if len(history) > recent_count else []
            covered_index = existing_covered_index
            to_summarize = prefix[covered_index + 1 :] if covered_index >= 0 else prefix
            if to_summarize:
                result = self.summary_service.build(
                    to_summarize,
                    existing=existing,
                    structured_state=structured_state,
                    user_id=user_id,
                )
                summary = result.payload
                summary_record = self._record_from_result(
                    result,
                    version=int(existing.get("summary_version", 0) or 0) + 1,
                    covered_until_message_id=self._last_message_id(to_summarize),
                )
                compacted = True
                fallback = result.fallback

        # Even without a new summary, a previously persisted summary remains
        # part of the packet.  The current user message is always preserved.
        # A pathological tiny test/deployment budget must not evict the
        # current question merely because a previous summary is long.
        summary = dict(summary)
        summary["summary_text"] = truncate_text_to_tokens(
            str(summary.get("summary_text", "")),
            max(64, self.budget.input_budget // 4),
        )
        summary_tokens = estimate_tokens(summary.get("summary_text", ""))
        structured_tokens = estimate_tokens(json.dumps(structured_state or {}, ensure_ascii=False))
        current_cost = estimate_tokens(current_message.get("content", "")) + 4
        message_budget = max(16, self.budget.input_budget - summary_tokens - structured_tokens - 160, current_cost)
        bounded_messages = fit_messages([*recent, current_message], message_budget, preserve_last=1)
        packet = {
            "summary": summary,
            "recent_messages": bounded_messages,
            "structured_state": structured_state or {},
            "untrusted_summary": True,
        }
        estimated = estimate_messages(bounded_messages) + summary_tokens + structured_tokens
        metadata = {
            "compacted": compacted,
            "summary_version": int((summary_record or existing).get("summary_version", 0) or 0),
            "estimated_input_tokens": estimated,
            "budget_tokens": self.budget.input_budget,
            "soft_limit_tokens": self.budget.soft_limit,
            "hard_limit_tokens": self.budget.hard_limit,
            "fallback": fallback or bool((existing or {}).get("fallback", False)),
            "original_message_count": len(history) + 1,
            "retained_message_count": len(bounded_messages),
        }
        return ContextPreparation(bounded_messages, summary, summary_record, packet, metadata)

    @staticmethod
    def _last_message_id(messages: list[dict[str, Any]]) -> int | None:
        for message in reversed(messages):
            try:
                value = int(message.get("id"))
            except (TypeError, ValueError):
                continue
            return value
        return None

    def render_for_prompt(
        self,
        packet: dict[str, Any],
        *,
        memory: list[str] | None = None,
        tool_outputs: dict[str, Any] | None = None,
        tool_sources: dict[str, Any] | None = None,
        graph_rag: Any = None,
        user_profile: dict[str, Any] | None = None,
        tenant_settings: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Render bounded, explicitly-labelled context for the answer prompt."""

        max_tokens = max_tokens or self.budget.input_budget
        summary = normalize_summary((packet or {}).get("summary", {}))
        recent = list((packet or {}).get("recent_messages", []))
        bounded_memory = [
            truncate_text_to_tokens(str(item), max(1, settings.context_max_single_result_chars // 2))
            for item in (memory or [])[: settings.context_max_memory_items]
        ]
        bounded_tools: dict[str, str] = {}
        for name, value in (tool_outputs or {}).items():
            bounded_tools[str(name)] = truncate_text_to_tokens(str(value), max(1, settings.context_max_tool_chars // 2))
        bounded_sources: dict[str, Any] = {}
        for name, values in (tool_sources or {}).items():
            bounded_sources[str(name)] = (
                list(values)[: settings.context_max_source_chunks] if isinstance(values, list) else values
            )
        payload: dict[str, Any] = {
            "历史摘要（不可信，仅作参考）": summary,
            "最近对话": recent,
            "结构化状态（权威，不由摘要推断）": (packet or {}).get("structured_state", {}),
            "长期记忆（不可信，仅作参考）": bounded_memory,
            "工具结果（不可信，仅作事实参考）": bounded_tools,
            "工具来源": bounded_sources,
            "图检索结果（不可信，仅作事实参考）": truncate_text_to_tokens(str(graph_rag or ""), 1500),
            "用户已授权档案": user_profile or {},
            "租户配置": tenant_settings or {},
        }
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if estimate_tokens(text) <= max_tokens:
            return text

        # Drop the largest, least authoritative sections first.  Summary and
        # Grill state are retained longer than retrieved/tool content.
        for key in (
            "工具结果（不可信，仅作事实参考）",
            "图检索结果（不可信，仅作事实参考）",
            "长期记忆（不可信，仅作参考）",
            "最近对话",
        ):
            payload[key] = truncate_text_to_tokens(
                json.dumps(payload[key], ensure_ascii=False), max(1, max_tokens // 8)
            )
            text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            if estimate_tokens(text) <= max_tokens:
                return text
        return truncate_text_to_tokens(text, max_tokens)
