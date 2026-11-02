from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from core.memory import MemoryManager
from core.memory_extractor import extract_memory_candidates
from core.memory_merge import MemoryMerger
from core.persistence import ConversationStore
from core.tenant import TenantService


@pytest.fixture
def database_path() -> Path:
    path = Path.cwd() / f".p2-7-memory-{uuid4().hex}.db"
    try:
        yield path
    finally:
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(f"{path}{suffix}")
            if candidate.exists():
                candidate.unlink()


def _store(database_path: Path) -> ConversationStore:
    return ConversationStore(database_url=f"sqlite:///{database_path}")


def test_extractor_only_reads_explicit_user_facts() -> None:
    candidates = extract_memory_candidates("我对花生过敏，不吃香菜，喜欢清淡饮食，想减肥，每天吃三顿")
    keys = {item.memory_key for item in candidates}
    assert {"allergies", "disliked_foods", "dietary_preferences", "health_goal", "meal_schedule"} <= keys
    assert all("建议" not in item.source_excerpt for item in candidates)


def test_consent_creates_raw_evidence_and_structured_items(database_path: Path) -> None:
    store = _store(database_path)
    manager = MemoryManager(store=store)
    manager.add_interaction("我对花生过敏", "建议你核对配料表", "user-a")
    assert store.export_memory("user-a")["items"] == []

    manager.set_consent("user-a", True)
    manager.add_interaction("我对花生过敏，喜欢清淡饮食", "建议你核对配料表", "user-a")
    exported = store.export_memory("user-a")
    assert len(exported["items"]) == 1
    assert {item["memory_key"] for item in exported["structured_items"]} == {"allergies", "dietary_preferences"}
    assert next(item for item in exported["structured_items"] if item["memory_key"] == "allergies")["status"] == "proposed"
    assert next(item for item in exported["structured_items"] if item["memory_key"] == "dietary_preferences")["status"] == "active"
    store.engine.dispose()


def test_user_facts_can_be_saved_during_grill_followup_without_raw_answer(database_path: Path) -> None:
    store = _store(database_path)
    manager = MemoryManager(store=store)
    manager.set_consent("user-a", True)
    manager.remember_user_facts("我想减肥", "user-a", conversation_id="conversation-1")
    items = store.list_memory_items("user-a")
    assert [item["memory_key"] for item in items] == ["health_goal"]
    assert store.list_memory("user-a") == []
    assert items[0]["source_conversation_id"] == "conversation-1"
    store.engine.dispose()


def test_collection_memories_merge_even_when_new_value_is_different(database_path: Path) -> None:
    store = _store(database_path)
    manager = MemoryManager(store=store)
    manager.set_consent("user-a", True)
    manager.add_interaction("我不吃香菜", "好", "user-a")
    manager.add_interaction("我不吃芹菜", "好", "user-a")
    items = store.list_memory_items("user-a", include_all=True)
    active = [item for item in items if item["status"] == "active"]
    assert len(active) == 1
    assert active[0]["memory_value"] == ["香菜", "芹菜"]
    store.engine.dispose()


def test_duplicate_and_conflicting_scalar_memory_are_merged(database_path: Path) -> None:
    store = _store(database_path)
    manager = MemoryManager(store=store)
    manager.set_consent("user-a", True)
    manager.add_interaction("我想减肥", "好", "user-a")
    manager.add_interaction("我的目标是增肌", "好", "user-a")
    items = store.list_memory_items("user-a", tenant_id=store.resolve_tenant_id("user-a"), include_all=True)
    goals = [item for item in items if item["memory_key"] == "health_goal"]
    assert len(goals) == 2
    assert {item["status"] for item in goals} == {"superseded", "proposed"}
    active = store.list_memory_items("user-a", statuses={"active"})
    assert all(item["memory_key"] != "health_goal" for item in active)
    store.engine.dispose()


def test_retrieval_is_intent_aware_and_does_not_inject_raw_answer(database_path: Path) -> None:
    store = _store(database_path)
    manager = MemoryManager(store=store)
    manager.set_consent("user-a", True)
    manager.add_interaction("我不吃香菜，喜欢清淡饮食", "你应该多喝水并安排晚餐", "user-a")
    meal_context = manager.retrieve_relevant_memories(
        "帮我安排餐食", user_id="user-a", current_goal="meal_planner", current_operation="plan"
    )
    assert any("香菜" in item for item in meal_context)
    assert any("清淡" in item for item in meal_context)
    assert all("多喝水" not in item for item in meal_context)
    calorie_context = manager.retrieve_relevant_memories(
        "查热量", user_id="user-a", current_goal="calorie_query", current_operation="lookup"
    )
    assert any("清淡" in item for item in calorie_context)
    store.engine.dispose()


def test_high_risk_confirmation_and_delete_stop_recall(database_path: Path) -> None:
    store = _store(database_path)
    manager = MemoryManager(store=store)
    manager.set_consent("user-a", True)
    manager.add_interaction("我对花生过敏", "知道了", "user-a")
    item = store.list_memory_items("user-a")[0]
    assert item["status"] == "proposed"
    assert manager.update_memory_item_status("user-a", item["id"], "active", item["tenant_id"])
    assert manager.retrieve_relevant_memories(
        "安排餐食", user_id="user-a", current_goal="meal_planner", current_operation="plan"
    )
    assert manager.delete_memory_item("user-a", item["id"], item["tenant_id"])
    assert manager.retrieve_relevant_memories(
        "安排餐食", user_id="user-a", current_goal="meal_planner", current_operation="plan"
    ) == []
    manager.set_consent("user-a", False, item["tenant_id"])
    assert store.export_memory("user-a", item["tenant_id"])["structured_items"] == []
    store.engine.dispose()


def test_memory_items_are_tenant_isolated(database_path: Path) -> None:
    store = _store(database_path)
    first = store.get_or_create_personal_tenant("user-a")
    second = TenantService(store).create(
        "user-a", "第二空间", "memory-second"
    )["id"]
    store.set_memory_consent("user-a", True, first)
    store.set_memory_consent("user-a", True, second)
    MemoryMerger(store).persist(
        extract_memory_candidates("我喜欢清淡饮食"), user_id="user-a", tenant_id=first
    )
    assert store.list_memory_items("user-a", tenant_id=first)
    assert store.list_memory_items("user-a", tenant_id=second) == []
    store.engine.dispose()


def test_short_term_expiry_is_recorded(database_path: Path) -> None:
    store = _store(database_path)
    store.set_memory_consent("user-a", True)
    candidate = extract_memory_candidates("这周暂时不吃辣")[0]
    created = MemoryMerger(store).persist([candidate], user_id="user-a")
    assert created[0]["expires_at"] is not None
    # The extractor deliberately keeps temporary scope explicit; persistence
    # uses that scope to set an expiry rather than silently making it permanent.
    assert candidate.temporal_scope == "short_term"
    store.engine.dispose()
