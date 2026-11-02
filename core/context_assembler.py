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


@dataclass(frozen=True)
class ContextRenderResult:
    """Structured rendering result used before every answer-model call."""

    text: str
    estimated_tokens: int
    max_tokens: int
    reduced_sections: tuple[str, ...] = ()
    compression_rounds: int = 0
    overflow_rejected: bool = False

    @property
    def within_budget(self) -> bool:
        return not self.overflow_rejected and self.estimated_tokens <= self.max_tokens


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
        estimated_tokens_before: int = 0,
        compaction_reason: str = "soft_limit",
        compaction_count: int = 1,
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
            "estimated_tokens_before": estimated_tokens_before,
            "estimated_tokens_after": result.estimated_tokens,
            "compression_ratio": 0.0,
            "compaction_reason": compaction_reason,
            "compaction_count": compaction_count,
            "overflow_prevented": True,
            "estimation_mode": "conservative_estimate",
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
        compaction_reason = (
            "message_batch" if uncompressed_count >= settings.context_compaction_batch_turns * 2 else "precompact"
        )
        recent_count = max(int(settings.context_min_recent_messages), int(settings.context_recent_turns) * 2)
        if should_compact and total_tokens >= self.budget.precompact_limit and len(history) > 2:
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
                    estimated_tokens_before=total_tokens,
                    compaction_reason=compaction_reason,
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
        bounded_messages, summary, structured_tokens, compression_rounds = self._fit_preparation(
            bounded_messages,
            summary,
            structured_state or {},
            current_message,
            summary_tokens,
            structured_tokens,
        )
        packet = {
            "summary": summary,
            "recent_messages": bounded_messages,
            "structured_state": structured_state or {},
            "untrusted_summary": True,
        }
        summary_tokens = estimate_tokens(summary.get("summary_text", ""))
        estimated = estimate_messages(bounded_messages) + summary_tokens + structured_tokens
        overflow_rejected = estimated > self.budget.hard_limit
        compression_ratio = round(estimated / max(1, total_tokens), 4)
        if summary_record is not None:
            summary_record.update(
                {
                    "estimated_tokens_after": estimated,
                    "compression_ratio": compression_ratio,
                    "compaction_count": compression_rounds + 1,
                    "overflow_prevented": not overflow_rejected,
                }
            )
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
            "estimated_tokens_before": total_tokens,
            "estimated_tokens_after": estimated,
            "compression_ratio": compression_ratio,
            "compaction_reason": compaction_reason if should_compact else "none",
            "compaction_count": compression_rounds + (1 if compacted else 0),
            "overflow_prevented": not overflow_rejected,
            "overflow_rejected": overflow_rejected,
            "estimation_mode": "conservative_estimate",
        }
        return ContextPreparation(bounded_messages, summary, summary_record, packet, metadata)

    def _fit_preparation(
        self,
        messages: list[dict[str, Any]],
        summary: dict[str, Any],
        structured_state: dict[str, Any],
        current_message: dict[str, Any],
        summary_tokens: int,
        structured_tokens: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], int, int]:
        """Reduce the model packet in bounded rounds without changing history."""

        rounds = 0
        bounded_messages = messages
        while rounds < self.budget.max_compaction_rounds:
            estimated = estimate_messages(bounded_messages) + summary_tokens + structured_tokens
            if estimated <= self.budget.hard_limit:
                break
            rounds += 1
            recent_limit = max(
                int(settings.context_min_recent_messages) + 1,
                len(bounded_messages) - max(2, len(bounded_messages) // 4),
            )
            bounded_messages = fit_messages(
                bounded_messages,
                max(16, min(self.budget.input_budget // 3, recent_limit * 80)),
                preserve_last=1,
            )
            summary["summary_text"] = truncate_text_to_tokens(
                str(summary.get("summary_text", "")),
                max(64, self.budget.input_budget // (4 + rounds)),
            )
            summary_tokens = estimate_tokens(summary.get("summary_text", ""))
            structured_tokens = estimate_tokens(json.dumps(structured_state, ensure_ascii=False))

        # The current question is always the final message.  If the complete
        # packet is still too large, the renderer will reject the model call
        # instead of silently cutting a serialized JSON document.
        if not bounded_messages or bounded_messages[-1].get("content") != current_message.get("content"):
            bounded_messages = fit_messages(
                [*bounded_messages, current_message], max(16, self.budget.input_budget), preserve_last=1
            )
        return bounded_messages, summary, structured_tokens, rounds

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

        return self.render_with_metadata(
            packet,
            memory=memory,
            tool_outputs=tool_outputs,
            tool_sources=tool_sources,
            graph_rag=graph_rag,
            user_profile=user_profile,
            tenant_settings=tenant_settings,
            max_tokens=max_tokens,
        ).text

    @staticmethod
    def _serialize(payload: dict[str, Any]) -> tuple[str, int]:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return text, estimate_tokens(text)

    @staticmethod
    def _shrink_mapping(value: Any, max_tokens: int) -> Any:
        if not isinstance(value, dict):
            return truncate_text_to_tokens(str(value), max_tokens)
        result: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(item, str):
                result[str(key)] = truncate_text_to_tokens(item, max(16, max_tokens // 4))
            elif isinstance(item, dict):
                result[str(key)] = ContextAssembler._shrink_mapping(item, max(16, max_tokens // 4))
            else:
                result[str(key)] = item
        return result

    def render_with_metadata(
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
    ) -> ContextRenderResult:
        """Render by reducing structured sections, never by cutting JSON."""

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
        text, estimated = self._serialize(payload)
        if estimated <= max_tokens:
            return ContextRenderResult(text, estimated, max_tokens)

        reduced: list[str] = []
        actions = (
            ("工具结果", lambda: payload.update({"工具结果（不可信，仅作事实参考）": {}})),
            ("图检索结果", lambda: payload.update({"图检索结果（不可信，仅作事实参考）": ""})),
            (
                "长期记忆",
                lambda: payload.update(
                    {
                        "长期记忆（不可信，仅作参考）": [
                            truncate_text_to_tokens(str(item), max(16, max_tokens // 24)) for item in (memory or [])[:2]
                        ]
                    }
                ),
            ),
            (
                "工具来源",
                lambda: payload.update(
                    {
                        "工具来源": {
                            str(key): list(value)[:1] if isinstance(value, list) else value
                            for key, value in (tool_sources or {}).items()
                        }
                    }
                ),
            ),
            (
                "最近对话",
                lambda: payload.update({"最近对话": fit_messages(recent, max(16, max_tokens // 4), preserve_last=1)}),
            ),
            (
                "用户档案",
                lambda: payload.update({"用户已授权档案": self._shrink_mapping(user_profile or {}, max_tokens // 12)}),
            ),
            (
                "租户配置",
                lambda: payload.update({"租户配置": self._shrink_mapping(tenant_settings or {}, max_tokens // 16)}),
            ),
            (
                "历史摘要",
                lambda: payload.update(
                    {
                        "历史摘要（不可信，仅作参考）": {
                            **summary,
                            "confirmed_facts": summary.get("confirmed_facts", [])[:4],
                            "constraints": summary.get("constraints", [])[:4],
                            "decisions": summary.get("decisions", [])[:3],
                            "summary_text": truncate_text_to_tokens(
                                str(summary.get("summary_text", "")), max(64, max_tokens // 6)
                            ),
                        }
                    }
                ),
            ),
        )
        for name, action in actions:
            action()
            reduced.append(name)
            text, estimated = self._serialize(payload)
            if estimated <= max_tokens:
                return ContextRenderResult(text, estimated, max_tokens, tuple(reduced), len(reduced))

        # Structured state is authoritative and must remain structured.  If it
        # alone cannot fit, reject the model call rather than emitting invalid
        # or silently truncated JSON.
        payload = {
            "历史摘要（不可信，仅作参考）": payload.get("历史摘要（不可信，仅作参考）", {}),
            "结构化状态（权威，不由摘要推断）": payload.get("结构化状态（权威，不由摘要推断）", {}),
            "最近对话": payload.get("最近对话", [])[-1:],
        }
        text, estimated = self._serialize(payload)
        return ContextRenderResult(text, estimated, max_tokens, tuple(reduced), len(reduced), estimated > max_tokens)
