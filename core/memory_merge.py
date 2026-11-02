"""Persistence and conflict handling for structured memory candidates."""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from uuid import uuid4

from config import settings
from core.memory_extractor import MemoryCandidate, is_collection_key
from core.persistence import _memory_content_hash, utc_now


def _normal(value: Any) -> Any:
    if isinstance(value, list):
        return sorted({_normal(item) for item in value if str(item).strip()}, key=str)
    if isinstance(value, dict):
        return {str(key): _normal(item) for key, item in sorted(value.items())}
    return str(value).strip().lower()


def content_hash(memory_key: str, memory_value: Any) -> str:
    return _memory_content_hash(memory_key, memory_value)


def _display_for(key: str, values: list[Any], fallback: str) -> str:
    labels = {
        "allergies": "过敏或不耐受",
        "disliked_foods": "忌口或不喜欢",
        "dietary_preferences": "饮食偏好",
        "cooking_equipment": "可用烹饪设备",
    }
    return f"{labels.get(key, key)}：{'、'.join(str(value) for value in values)}" if values else fallback


class MemoryMerger:
    """Merge a candidate into the user's structured memory namespace."""

    def __init__(self, store: Any):
        self.store = store

    def _expiry(self, candidate: MemoryCandidate):
        if candidate.temporal_scope == "today":
            return utc_now() + timedelta(days=1)
        if candidate.temporal_scope == "short_term":
            return utc_now() + timedelta(days=7)
        if settings.memory_default_ttl_days > 0:
            return utc_now() + timedelta(days=settings.memory_default_ttl_days)
        return None

    def persist(
        self,
        candidates: list[MemoryCandidate],
        *,
        user_id: str,
        tenant_id: str | None = None,
        source_conversation_id: str | None = None,
        source_message_id: int | None = None,
    ) -> list[dict[str, Any]]:
        if not candidates or not hasattr(self.store, "list_memory_items"):
            return []
        persisted: list[dict[str, Any]] = []
        max_items = max(1, int(settings.memory_candidate_max_items))
        for candidate in candidates[:max_items]:
            digest = content_hash(candidate.memory_key, candidate.memory_value)
            existing = self.store.list_memory_items(
                user_id,
                tenant_id=tenant_id,
                statuses={"active", "proposed"},
                include_all=False,
            )
            duplicate = next(
                (
                    item
                    for item in existing
                    if item["memory_key"] == candidate.memory_key and item.get("content_hash") == digest
                ),
                None,
            )
            if duplicate:
                persisted.append(duplicate)
                continue

            value = candidate.memory_value
            same_key = [item for item in existing if item["memory_key"] == candidate.memory_key]
            if is_collection_key(candidate.memory_key) and isinstance(candidate.memory_value, list):
                candidate_values = [str(item).strip() for item in candidate.memory_value if str(item).strip()]
                if same_key:
                    canonical = same_key[0]
                    old_values: list[str] = []
                    for item in same_key:
                        raw_values = item.get("memory_value")
                        if isinstance(raw_values, list):
                            old_values.extend(map(str, raw_values))
                        elif raw_values is not None:
                            old_values.append(str(raw_values))
                    normalized: dict[str, str] = {}
                    for item in [*old_values, *candidate_values]:
                        normalized.setdefault(_normal(item), item.strip())
                    merged_values = list(normalized.values())
                    updated = self.store.update_memory_item(
                        user_id,
                        canonical["id"],
                        memory_value=merged_values,
                        display_text=_display_for(candidate.memory_key, merged_values, candidate.display_text),
                        tenant_id=tenant_id,
                    )
                    if updated:
                        desired_status = (
                            "active"
                            if settings.memory_auto_confirm_low_risk and candidate.sensitivity == "low"
                            else "proposed"
                        )
                        if canonical.get("status") != "proposed" or desired_status == "proposed":
                            updated = self.store.update_memory_item_status(
                                user_id,
                                canonical["id"],
                                desired_status,
                                tenant_id=tenant_id,
                            ) or updated
                        persisted.append(updated)
                        for duplicate in same_key[1:]:
                            self.store.update_memory_item_status(
                                user_id,
                                duplicate["id"],
                                "superseded",
                                tenant_id=tenant_id,
                                superseded_by=canonical["id"],
                            )
                    continue

            item_id = str(uuid4())
            status = "active" if settings.memory_auto_confirm_low_risk and candidate.sensitivity == "low" else "proposed"
            for old in same_key:
                # Scalar values represent one current preference/goal.  Keep
                # the old value for auditability, but remove it from default
                # retrieval when the user expresses a newer value.
                if not is_collection_key(candidate.memory_key):
                    self.store.update_memory_item_status(
                        user_id,
                        old["id"],
                        "superseded",
                        tenant_id=tenant_id,
                        superseded_by=item_id,
                    )
            created = self.store.create_memory_item(
                item_id=item_id,
                user_id=user_id,
                tenant_id=tenant_id,
                memory_type=candidate.memory_type,
                memory_key=candidate.memory_key,
                memory_value=value,
                display_text=candidate.display_text,
                status=status,
                confidence=candidate.confidence,
                sensitivity=candidate.sensitivity,
                source_conversation_id=source_conversation_id,
                source_message_id=source_message_id,
                source_excerpt=candidate.source_excerpt,
                valid_from=utc_now(),
                expires_at=self._expiry(candidate),
                content_hash=content_hash(candidate.memory_key, value),
            )
            if created:
                persisted.append(created)
        return persisted
