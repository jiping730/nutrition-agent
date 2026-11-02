"""Token-budget helpers for conversation context.

The project intentionally does not require a provider-specific tokenizer.  A
conservative, deterministic estimator is sufficient for the small-platform
deployment and keeps the hard limit enforceable when the configured model
changes.  The original conversation messages are never modified by this
module; it only prepares a bounded model input.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

from config import settings


def _safety_factor(value: float | None = None) -> float:
    factor = settings.context_estimate_safety_factor if value is None else value
    return min(2.0, max(1.0, float(factor)))


def estimate_tokens(value: Any, *, safety_factor: float | None = None) -> int:
    """Estimate tokens conservatively for Chinese, English, and mixed text.

    This is deliberately labelled as an estimate.  It is not a provider
    tokenizer; the safety factor makes the application budget conservative
    when the configured model changes.
    """

    text = str(value or "")
    if not text:
        return 0
    raw = 0.0
    for char in text:
        category = unicodedata.category(char)
        if "CJK" in unicodedata.name(char, "") or 0x1F000 <= ord(char) <= 0x1FAFF:
            raw += 1.5
        elif char.isspace() or category.startswith("P"):
            raw += 0.35
        elif char.isascii() and (char.isalnum() or char in "_-/."):
            raw += 0.28
        else:
            raw += 0.75
    # A small safety factor prevents the heuristic from being optimistic at
    # the exact boundary.  It is still much cheaper than a tokenizer package.
    return max(1, math.ceil(raw * _safety_factor(safety_factor)))


def estimate_messages(messages: Iterable[dict[str, Any]]) -> int:
    return sum(estimate_tokens(message.get("content", "")) + 4 for message in messages)


def truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    """Keep the beginning of text while guaranteeing the estimated budget."""

    if max_tokens <= 0 or not text:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    # Token density varies by language, so binary search against the same
    # estimator instead of assuming a fixed characters-per-token ratio.
    low, high = 0, len(text)
    best = ""
    while low <= high:
        middle = (low + high) // 2
        candidate = text[:middle].rstrip()
        if estimate_tokens(candidate) <= max_tokens:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    if best and len(best) < len(text) and estimate_tokens(best + "…") <= max_tokens:
        return best + "…"
    return best


@dataclass(frozen=True)
class ContextBudgetDecision:
    """A serializable decision made by the conservative budget checker."""

    estimated_tokens: int
    budget_tokens: int
    soft_limit_tokens: int
    hard_limit_tokens: int
    reason: str
    estimation_mode: str = "conservative_estimate"

    @property
    def within_soft_limit(self) -> bool:
        return self.estimated_tokens < self.soft_limit_tokens

    @property
    def within_hard_limit(self) -> bool:
        return self.estimated_tokens <= self.hard_limit_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimated_tokens": self.estimated_tokens,
            "budget_tokens": self.budget_tokens,
            "soft_limit_tokens": self.soft_limit_tokens,
            "hard_limit_tokens": self.hard_limit_tokens,
            "reason": self.reason,
            "estimation_mode": self.estimation_mode,
            "within_soft_limit": self.within_soft_limit,
            "within_hard_limit": self.within_hard_limit,
        }


@dataclass(frozen=True)
class ContextBudget:
    window_tokens: int
    reserved_output_tokens: int
    reserved_system_tokens: int
    safety_margin_tokens: int
    soft_limit_ratio: float
    hard_limit_ratio: float
    estimate_safety_factor: float = 1.15
    precompact_ratio: float = 0.70
    max_compaction_rounds: int = 3

    @property
    def input_budget(self) -> int:
        return max(
            1,
            self.window_tokens - self.reserved_output_tokens - self.reserved_system_tokens - self.safety_margin_tokens,
        )

    @property
    def soft_limit(self) -> int:
        return max(1, min(self.input_budget, int(self.window_tokens * self.soft_limit_ratio)))

    @property
    def precompact_limit(self) -> int:
        return max(1, min(self.input_budget, int(self.window_tokens * self.precompact_ratio)))

    @property
    def hard_limit(self) -> int:
        return max(1, min(self.input_budget, int(self.window_tokens * self.hard_limit_ratio)))

    def should_compact(self, estimated_tokens: int, message_count: int = 0) -> bool:
        return (
            estimated_tokens >= self.precompact_limit
            or estimated_tokens >= self.soft_limit
            or message_count >= settings.context_compaction_batch_turns * 2
        )

    def within_hard_limit(self, estimated_tokens: int) -> bool:
        return estimated_tokens <= self.hard_limit

    def decide(self, estimated_tokens: int, reason: str = "preflight") -> ContextBudgetDecision:
        return ContextBudgetDecision(
            estimated_tokens=max(0, int(estimated_tokens)),
            budget_tokens=self.input_budget,
            soft_limit_tokens=self.precompact_limit,
            hard_limit_tokens=self.hard_limit,
            reason=reason,
        )


def current_budget() -> ContextBudget:
    return ContextBudget(
        window_tokens=max(256, int(settings.context_window_tokens)),
        reserved_output_tokens=max(0, int(settings.context_reserved_output_tokens)),
        reserved_system_tokens=max(0, int(settings.context_reserved_system_tokens)),
        safety_margin_tokens=max(0, int(settings.context_safety_margin_tokens)),
        estimate_safety_factor=min(2.0, max(1.0, float(settings.context_estimate_safety_factor))),
        precompact_ratio=min(0.99, max(0.1, float(settings.context_precompact_ratio))),
        soft_limit_ratio=min(0.99, max(0.1, float(settings.context_soft_limit_ratio))),
        hard_limit_ratio=min(0.99, max(0.2, float(settings.context_hard_limit_ratio))),
        max_compaction_rounds=max(1, int(settings.context_max_compaction_rounds)),
    )


def fit_messages(
    messages: list[dict[str, Any]],
    max_tokens: int,
    *,
    preserve_last: int = 1,
) -> list[dict[str, Any]]:
    """Return the newest complete-enough message suffix within a budget."""

    if not messages:
        return []
    selected: list[dict[str, Any]] = []
    used = 0
    for message in reversed(messages):
        content = str(message.get("content", ""))
        cost = estimate_tokens(content) + 4
        if not selected and preserve_last:
            if cost > max_tokens:
                content = truncate_text_to_tokens(content, max(1, max_tokens - 4))
                selected.append({**message, "content": content})
                break
        if used + cost > max_tokens:
            break
        selected.append(dict(message))
        used += cost
    selected.reverse()
    return selected
