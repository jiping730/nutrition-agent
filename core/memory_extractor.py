"""Deterministic extraction of explicit user facts from one message.

The first structured-memory layer intentionally uses conservative rules.  It
is fast, auditable, and cannot turn an assistant recommendation into a user
fact.  An optional model extractor can be added later behind the setting
flag, but it must produce the same ``MemoryCandidate`` contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class MemoryCandidate:
    memory_type: str
    memory_key: str
    memory_value: Any
    display_text: str
    confidence: float = 0.95
    sensitivity: str = "low"
    explicit: bool = True
    temporal_scope: str = "persistent"
    source_excerpt: str = ""


_COLLECTION_KEYS = {"allergies", "disliked_foods", "dietary_preferences", "cooking_equipment"}
_HIGH_SENSITIVITY_KEYS = {"allergies", "health_goal", "medical_information"}


def _split_values(value: str) -> list[str]:
    value = re.sub(r"^(我|本人|自己|平时|目前|最近|暂时|家里|家中)的?", "", value.strip())
    value = re.sub(r"(这些|这种|相关的?)$", "", value).strip()
    parts = re.split(r"[、,，；;和及以及以及/]+", value)
    result: list[str] = []
    for part in parts:
        item = re.sub(r"^(我|本人|自己)的?", "", part.strip())
        item = re.sub(r"(食物|食品|饮食|习惯)$", "", item).strip(" ：:。！？!?\n\t")
        if item and len(item) <= 64 and item not in result:
            result.append(item)
    return result[:8]


def _temporal_scope(text: str) -> str:
    if "今天" in text:
        return "today"
    if any(marker in text for marker in ("暂时", "这周", "本周", "这几天")):
        return "short_term"
    return "persistent"


def _candidate(
    *, key: str, value: Any, display: str, text: str, confidence: float = 0.95
) -> MemoryCandidate:
    memory_type = {
        "allergies": "constraint",
        "disliked_foods": "constraint",
        "dietary_preferences": "preference",
        "health_goal": "goal",
        "calorie_tracking_preference": "preference",
        "cooking_equipment": "fact",
        "activity_level": "routine",
        "meal_schedule": "routine",
    }.get(key, "fact")
    return MemoryCandidate(
        memory_type=memory_type,
        memory_key=key,
        memory_value=value,
        display_text=display,
        confidence=confidence,
        sensitivity="high" if key in _HIGH_SENSITIVITY_KEYS else "low",
        temporal_scope=_temporal_scope(text),
        source_excerpt=text[:500],
    )


def extract_memory_candidates(user_text: str) -> list[MemoryCandidate]:
    """Extract only explicit, first-party statements from ``user_text``."""

    text = (user_text or "").strip()
    if not text or len(text) > 20_000:
        return []
    candidates: list[MemoryCandidate] = []

    if not re.search(r"(?:没有|无|没有任何).{0,8}(?:过敏|过敏原|不耐受)", text):
        for match in re.finditer(r"(?:我|本人)?(?:对|有)?\s*([^。！？!?；;，,]{1,80})(?:过敏|过敏原|不耐受)", text):
            values = _split_values(match.group(1))
            if values:
                candidates.append(
                    _candidate(
                        key="allergies",
                        value=values,
                        display=f"过敏或不耐受：{'、'.join(values)}",
                        text=text,
                    )
                )

    for match in re.finditer(
        r"(?:我|本人)?(?:不吃|不喜欢吃|不喜欢|忌口|不能吃|不想吃)\s*([^。！？!?；;，,]{1,80})",
        text,
    ):
        values = _split_values(match.group(1))
        if values and not any(item in {"饭", "东西", "这个", "那种"} for item in values):
            candidates.append(
                _candidate(
                    key="disliked_foods",
                    value=values,
                    display=f"忌口或不喜欢：{'、'.join(values)}",
                    text=text,
                )
            )

    for match in re.finditer(
        r"(?:我|本人)?(?<!不)(?:喜欢|偏好|倾向于|希望以)\s*([^。！？!?；;，,]{1,50}?)(?:饮食|吃法|搭配)?(?=[。！？!?；;，,]|$)",
        text,
    ):
        values = _split_values(match.group(1))
        values = [item for item in values if item not in {"以", "吃"}]
        if values:
            candidates.append(
                _candidate(
                    key="dietary_preferences",
                    value=values,
                    display=f"饮食偏好：{'、'.join(values)}",
                    text=text,
                )
            )

    goal_patterns = (
        (r"(?:想|希望|目标是|正在)\s*(减肥|减脂|控制体重|体重管理)", "weight_control"),
        (r"(?:想|希望|目标是|正在)\s*(增肌|增加肌肉)", "muscle_gain"),
        (r"(?:想|希望|目标是|正在)\s*(均衡饮食|健康饮食)", "balanced_diet"),
    )
    for pattern, value in goal_patterns:
        if re.search(pattern, text):
            label = {"weight_control": "减脂或控制体重", "muscle_gain": "增肌", "balanced_diet": "均衡饮食"}[value]
            candidates.append(_candidate(key="health_goal", value=value, display=f"健康目标：{label}", text=text))

    if re.search(r"(?:不想|不愿意|不需要|不喜欢)\s*(?:算|计算|记录|追踪)\s*(?:热量|卡路里)", text):
        candidates.append(
            _candidate(
                key="calorie_tracking_preference",
                value="avoid",
                display="热量记录偏好：不希望主动计算或记录",
                text=text,
            )
        )
    elif re.search(r"(?:愿意|希望|想要|需要)\s*(?:算|计算|记录|追踪)\s*(?:热量|卡路里)", text):
        candidates.append(
            _candidate(
                key="calorie_tracking_preference",
                value="welcomes",
                display="热量记录偏好：愿意计算或记录",
                text=text,
            )
        )

    equipment = re.search(r"(?:我|家里|家中)?(?:通常)?(?:只有|有|使用|用)\s*([^。！？!?；;，,]{1,50}?)(?:做饭|烹饪|煮饭|可以用)?(?:。|！|？|；|，|,|$)", text)
    if equipment and any(item in equipment.group(1) for item in ("电饭煲", "烤箱", "空气炸锅", "微波炉", "锅")):
        values = _split_values(equipment.group(1))
        if values:
            candidates.append(
                _candidate(
                    key="cooking_equipment",
                    value=values,
                    display=f"可用烹饪设备：{'、'.join(values)}",
                    text=text,
                )
            )

    exercise = re.search(r"(?:每周|一周)(?:大约|大概|通常)?\s*(?:运动|锻炼|健身)?\s*([0-9一二三四五六七八])\s*次(?:运动|锻炼|健身)?", text)
    if exercise:
        count = {"一": "1", "二": "2", "三": "3", "四": "4", "五": "5", "六": "6", "七": "7", "八": "8"}.get(exercise.group(1), exercise.group(1))
        value = f"每周运动{count}次"
        candidates.append(_candidate(key="activity_level", value=value, display=f"活动习惯：{value}", text=text))

    meals = re.search(r"(?:每天|一日)(?:大约|通常)?\s*(?:吃)?\s*([0-9一二三四五六七八])\s*顿", text)
    if meals:
        count = {"一": "1", "二": "2", "三": "3", "四": "4", "五": "5", "六": "6", "七": "7", "八": "8"}.get(meals.group(1), meals.group(1))
        value = f"每天{count}顿"
        candidates.append(_candidate(key="meal_schedule", value=value, display=f"用餐习惯：{value}", text=text))

    unique: list[MemoryCandidate] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        key = (candidate.memory_key, repr(candidate.memory_value))
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def is_collection_key(memory_key: str) -> bool:
    return memory_key in _COLLECTION_KEYS
