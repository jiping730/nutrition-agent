"""Short-term and optional FAISS-backed long-term memory."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from typing import Any, List, Optional

from langchain_community.vectorstores import FAISS

from config import settings
from core.logging_config import configure_logging, log_event
from core.memory_extractor import extract_memory_candidates
from core.memory_merge import MemoryMerger
from rag.embeddings import get_embeddings

logger = configure_logging(settings.log_level)


class MemoryManager:
    def __init__(self, store: Any | None = None):
        self.store = store
        self.short_term: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self.long_term_path = os.path.join(settings.memory_path, "long_term")
        self.long_term = None
        try:
            self.long_term = self._load_long_term()
        except Exception as exc:
            log_event(logger, 30, "long_term_memory_disabled", error_type=type(exc).__name__)

        self.meta_file = os.path.join(settings.memory_path, "meta.json")
        self.governance_file = os.path.join(settings.memory_path, "governance.json")
        try:
            self._ensure_dir()
        except OSError as exc:
            log_event(logger, 30, "memory_meta_persistence_disabled", error_type=type(exc).__name__)

    def _ensure_dir(self):
        os.makedirs(settings.memory_path, exist_ok=True)
        if not os.path.exists(self.meta_file):
            with open(self.meta_file, "w", encoding="utf-8") as file:
                json.dump([], file)
        if self.store is None and not os.path.exists(self.governance_file):
            with open(self.governance_file, "w", encoding="utf-8") as file:
                json.dump({"consents": {}, "items": {}}, file)

    def set_store(self, store: Any) -> None:
        self.store = store

    def _load_governance(self) -> dict[str, Any]:
        try:
            with open(self.governance_file, "r", encoding="utf-8") as file:
                payload = json.load(file)
            if not isinstance(payload, dict):
                return {"consents": {}, "items": {}, "structured_items": {}}
            payload.setdefault("consents", {})
            payload.setdefault("items", {})
            payload.setdefault("structured_items", {})
            return payload
        except (FileNotFoundError, json.JSONDecodeError):
            return {"consents": {}, "items": {}, "structured_items": {}}

    def _save_governance(self, payload: dict[str, Any]) -> None:
        with open(self.governance_file, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)

    def has_consent(self, user_id: str | None, tenant_id: str | None = None) -> bool:
        if not user_id:
            return False
        if self.store is not None and hasattr(self.store, "get_memory_consent"):
            return bool(self.store.get_memory_consent(user_id, tenant_id))
        with self._lock:
            return bool(self._load_governance().get("consents", {}).get(user_id, False))

    def set_consent(self, user_id: str, granted: bool, tenant_id: str | None = None) -> bool:
        if not user_id or not user_id.strip():
            raise ValueError("user_id 不能为空")
        if self.store is not None and hasattr(self.store, "set_memory_consent"):
            result = bool(self.store.set_memory_consent(user_id, bool(granted), tenant_id))
            if not granted:
                self.short_term = [
                    item
                    for item in self.short_term
                    if not (item.get("user_id") == user_id and item.get("tenant_id") == tenant_id)
                ]
            return result
        with self._lock:
            payload = self._load_governance()
            payload["consents"][user_id] = bool(granted)
            if not granted:
                payload["items"].pop(user_id, None)
                self.short_term = [item for item in self.short_term if item.get("user_id") != user_id]
            self._save_governance(payload)
        return bool(granted)

    def export_user_memory(self, user_id: str, tenant_id: str | None = None) -> dict[str, Any]:
        if self.store is not None and hasattr(self.store, "export_memory"):
            return self.store.export_memory(user_id, tenant_id)
        with self._lock:
            payload = self._load_governance()
            return {
                "user_id": user_id,
                "consent": bool(payload["consents"].get(user_id, False)),
                "items": list(payload["items"].get(user_id, [])),
                "structured_items": list(payload["structured_items"].get(user_id, [])),
            }

    def list_memory_items(
        self,
        user_id: str,
        *,
        tenant_id: str | None = None,
        statuses: set[str] | None = None,
        include_all: bool = False,
    ) -> list[dict[str, Any]]:
        if not self.has_consent(user_id, tenant_id):
            return []
        if self.store is not None and hasattr(self.store, "list_memory_items"):
            return self.store.list_memory_items(
                user_id,
                tenant_id=tenant_id,
                statuses=statuses,
                include_all=include_all,
            )
        with self._lock:
            items = list(self._load_governance().get("structured_items", {}).get(user_id, []))
            if statuses:
                items = [item for item in items if item.get("status") in statuses]
            elif not include_all:
                items = [item for item in items if item.get("status") in {"active", "proposed"}]
            return items

    def update_memory_item_status(
        self,
        user_id: str,
        item_id: str,
        status: str,
        tenant_id: str | None = None,
        superseded_by: str | None = None,
    ) -> dict[str, Any] | None:
        if not self.has_consent(user_id, tenant_id):
            return None
        if self.store is not None and hasattr(self.store, "update_memory_item_status"):
            return self.store.update_memory_item_status(
                user_id, item_id, status, tenant_id=tenant_id, superseded_by=superseded_by
            )
        with self._lock:
            payload = self._load_governance()
            for item in payload.get("structured_items", {}).get(user_id, []):
                if item.get("id") == item_id:
                    item["status"] = status
                    if superseded_by:
                        item["superseded_by"] = superseded_by
                    self._save_governance(payload)
                    return item
            return None

    def update_memory_item(
        self,
        user_id: str,
        item_id: str,
        *,
        memory_value: Any | None = None,
        display_text: str | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, Any] | None:
        if not self.has_consent(user_id, tenant_id):
            return None
        if self.store is not None and hasattr(self.store, "update_memory_item"):
            return self.store.update_memory_item(
                user_id,
                item_id,
                memory_value=memory_value,
                display_text=display_text,
                tenant_id=tenant_id,
            )
        with self._lock:
            payload = self._load_governance()
            for item in payload.get("structured_items", {}).get(user_id, []):
                if item.get("id") == item_id:
                    if memory_value is not None:
                        item["memory_value"] = memory_value
                    if display_text is not None:
                        item["display_text"] = display_text
                    self._save_governance(payload)
                    return item
            return None

    def delete_memory_item(self, user_id: str, item_id: str, tenant_id: str | None = None) -> bool:
        if not self.has_consent(user_id, tenant_id):
            return False
        if self.store is not None and hasattr(self.store, "delete_memory_item"):
            return bool(self.store.delete_memory_item(user_id, item_id, tenant_id=tenant_id))
        with self._lock:
            payload = self._load_governance()
            items = payload.get("structured_items", {}).get(user_id, [])
            before = len(items)
            payload["structured_items"][user_id] = [item for item in items if item.get("id") != item_id]
            self._save_governance(payload)
            return len(payload["structured_items"][user_id]) < before

    def delete_user_memory(
        self, user_id: str, *, revoke_consent: bool = True, tenant_id: str | None = None
    ) -> int:
        if self.store is not None and hasattr(self.store, "delete_memory"):
            deleted = int(
                self.store.delete_memory(user_id, revoke_consent=revoke_consent, tenant_id=tenant_id)
            )
            self.short_term = [
                item
                for item in self.short_term
                if not (item.get("user_id") == user_id and item.get("tenant_id") == tenant_id)
            ]
            return deleted
        with self._lock:
            payload = self._load_governance()
            count = len(payload["items"].get(user_id, []))
            payload["items"].pop(user_id, None)
            payload.get("structured_items", {}).pop(user_id, None)
            if revoke_consent:
                payload["consents"][user_id] = False
            self._save_governance(payload)
            self.short_term = [item for item in self.short_term if item.get("user_id") != user_id]
            return count

    def _load_long_term(self):
        embeddings = get_embeddings()
        if os.path.exists(self.long_term_path):
            return FAISS.load_local(self.long_term_path, embeddings, allow_dangerous_deserialization=True)
        temp_texts = ["placeholder"]
        return FAISS.from_texts(temp_texts, embeddings)

    def save_long_term(self):
        if self.long_term is None:
            return
        os.makedirs(os.path.dirname(self.long_term_path), exist_ok=True)
        self.long_term.save_local(self.long_term_path)

    def _persist_structured_memory(
        self,
        safe_input: str,
        *,
        user_id: str,
        tenant_id: str | None = None,
        conversation_id: str | None = None,
        source_message_id: int | None = None,
    ) -> list[dict[str, Any]]:
        if not getattr(settings, "memory_extraction_enabled", True):
            return []
        candidates = extract_memory_candidates(safe_input)
        if not candidates:
            return []
        if self.store is not None and hasattr(self.store, "create_memory_item"):
            return MemoryMerger(self.store).persist(
                candidates,
                user_id=user_id,
                tenant_id=tenant_id,
                source_conversation_id=conversation_id,
                source_message_id=source_message_id,
            )

        with self._lock:
            payload = self._load_governance()
            structured = payload.setdefault("structured_items", {}).setdefault(str(user_id), [])
            for candidate in candidates[: max(1, settings.memory_candidate_max_items)]:
                from core.memory_merge import content_hash
                from uuid import uuid4

                structured.append(
                    {
                        "id": str(uuid4()),
                        "memory_type": candidate.memory_type,
                        "memory_key": candidate.memory_key,
                        "memory_value": candidate.memory_value,
                        "display_text": candidate.display_text,
                        "status": "active"
                        if settings.memory_auto_confirm_low_risk and candidate.sensitivity == "low"
                        else "proposed",
                        "confidence": candidate.confidence,
                        "sensitivity": candidate.sensitivity,
                        "source_conversation_id": conversation_id,
                        "source_message_id": source_message_id,
                        "source_excerpt": candidate.source_excerpt,
                        "content_hash": content_hash(candidate.memory_key, candidate.memory_value),
                        "created_at": datetime.now().isoformat(),
                    }
                )
            self._save_governance(payload)
            return structured[-len(candidates) :]

    def remember_user_facts(
        self,
        user_input: str,
        user_id: str | None = None,
        tenant_id: str | None = None,
        conversation_id: str | None = None,
        source_message_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Extract explicit user facts without archiving the assistant reply."""

        if not user_id or not self.has_consent(user_id, tenant_id):
            return []
        from core.safety_guard import SafetyGuard

        safe_input, _ = SafetyGuard.redact(user_input)
        try:
            return self._persist_structured_memory(
                safe_input,
                user_id=str(user_id),
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                source_message_id=source_message_id,
            )
        except Exception as exc:
            log_event(logger, 30, "structured_memory_write_failed", error_type=type(exc).__name__)
            return []

    def add_interaction(
        self,
        user_input: str,
        agent_output: str,
        user_id: Optional[str] = None,
        tenant_id: str | None = None,
        conversation_id: str | None = None,
    ):
        from core.safety_guard import SafetyGuard

        safe_input, _ = SafetyGuard.redact(user_input)
        safe_output, _ = SafetyGuard.redact(agent_output)
        self.short_term.append(
            {
                "input": safe_input,
                "output": safe_output,
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "tenant_id": tenant_id,
            }
        )
        if len(self.short_term) > getattr(settings, "short_term_size", 10):
            self.short_term.pop(0)

        if not self.has_consent(user_id, tenant_id):
            return

        if self.store is not None and hasattr(self.store, "save_memory"):
            try:
                self.store.save_memory(
                    user_id=str(user_id),
                    user_input=safe_input,
                    assistant_output=safe_output,
                    tenant_id=tenant_id,
                )
            except Exception as exc:
                log_event(logger, 30, "governed_memory_write_failed", error_type=type(exc).__name__)
        try:
            self._persist_structured_memory(
                safe_input,
                user_id=str(user_id),
                tenant_id=tenant_id,
                conversation_id=conversation_id,
            )
        except Exception as exc:
            log_event(logger, 30, "structured_memory_write_failed", error_type=type(exc).__name__)

    def retrieve_relevant_memories(
        self,
        query: str,
        user_id: Optional[str] = None,
        k: int = 5,
        tenant_id: str | None = None,
        current_goal: str | None = None,
        current_operation: str | None = None,
        request_intent: dict[str, Any] | None = None,
    ) -> List[str]:
        if user_id and not self.has_consent(user_id, tenant_id):
            return []
        intent = request_intent or {}
        task = current_goal or intent.get("task")
        operation = current_operation or intent.get("operation")
        if task == "meal_planner":
            allowed_keys = {"allergies", "disliked_foods", "dietary_preferences", "cooking_equipment", "health_goal"}
        elif task == "calorie_query":
            allowed_keys = {"health_goal", "calorie_tracking_preference", "dietary_preferences"}
        elif operation == "advice" or intent.get("goal"):
            allowed_keys = {"health_goal", "activity_level", "dietary_preferences", "allergies", "disliked_foods"}
        else:
            allowed_keys = None
        memories: list[str] = []
        try:
            items = self.list_memory_items(
                str(user_id),
                tenant_id=tenant_id,
                statuses={"active"},
                include_all=False,
            ) if user_id else []
            for item in items:
                if allowed_keys is not None and item.get("memory_key") not in allowed_keys:
                    continue
                memories.append(f"结构化记忆（不可信，仅作参考）：{item.get('display_text', '')}")
        except Exception as exc:
            log_event(logger, 30, "structured_memory_retrieval_failed", error_type=type(exc).__name__)
        return memories[: min(k, settings.memory_retrieval_top_k)]

    def save_meta(
        self,
        user_input: str,
        agent_output: str,
        quality: float,
        user_id: str | None = None,
        tenant_id: str | None = None,
    ):
        if user_id and not self.has_consent(user_id, tenant_id):
            return
        from core.safety_guard import SafetyGuard

        user_input, _ = SafetyGuard.redact(user_input)
        agent_output, _ = SafetyGuard.redact(agent_output)
        try:
            with open(self.meta_file, "r", encoding="utf-8") as file:
                meta = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            meta = []
        meta.append(
            {
                "input": user_input,
                "output": agent_output,
                "quality": quality,
                "timestamp": datetime.now().isoformat(),
            }
        )
        meta = meta[-100:]
        try:
            with open(self.meta_file, "w", encoding="utf-8") as file:
                json.dump(meta, file, indent=2, ensure_ascii=False)
        except OSError as exc:
            log_event(logger, 30, "memory_meta_write_failed", error_type=type(exc).__name__)

    def get_meta_stats(self):
        try:
            with open(self.meta_file, "r", encoding="utf-8") as file:
                meta = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"count": 0, "avg_quality": 0}
        avg_quality = sum(item["quality"] for item in meta) / len(meta) if meta else 0
        return {"count": len(meta), "avg_quality": avg_quality}
