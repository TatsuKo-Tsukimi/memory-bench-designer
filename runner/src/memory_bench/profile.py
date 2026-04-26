"""Scenario-conditioned capability priorities."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


PRIORITY_WEIGHTS = {
    "high": 1.0,
    "medium": 0.5,
    "low": 0.1,
}


FAMILY_ALIASES = {
    "exploration": "exploration",
    "ranking": "ranking",
    "retrieval": "ranking",
    "retrieval_accuracy": "ranking",
    "adaptation": "adaptation",
    "learning": "adaptation",
    "cross_session_learning": "cross_session_learning",
    "maintenance": "maintenance",
    "update": "maintenance",
    "forgetting": "maintenance",
}


@dataclass
class ScenarioProfile:
    name: str = "unconditioned"
    priorities: Dict[str, float] = field(default_factory=dict)
    source_path: str | None = None

    @staticmethod
    def unconditioned() -> "ScenarioProfile":
        return ScenarioProfile(
            name="unconditioned",
            priorities={
                "exploration": 1.0,
                "ranking": 1.0,
                "adaptation": 1.0,
                "maintenance": 1.0,
            },
        )

    @staticmethod
    def from_yaml(path: str | Path) -> "ScenarioProfile":
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        node = raw.get("scenario_profile", raw)
        priorities = _normalize_priorities(node.get("priorities", {}))
        if not priorities:
            priorities = ScenarioProfile.unconditioned().priorities
        return ScenarioProfile(
            name=str(node.get("name", path.stem)),
            priorities=priorities,
            source_path=str(path),
        )

    def hash(self) -> str:
        canonical = json.dumps(
            dataclasses.asdict(self), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def weighted_axes(self) -> Dict[str, float]:
        return dict(self.priorities)

    def high_priority_axes(self) -> Dict[str, float]:
        high = {k: v for k, v in self.priorities.items() if v >= 1.0}
        return high or dict(self.priorities)


def _normalize_priorities(raw: Mapping[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in raw.items():
        axis = FAMILY_ALIASES.get(str(key), str(key))
        if isinstance(value, (int, float)):
            weight = float(value)
        else:
            weight = PRIORITY_WEIGHTS.get(str(value).lower())
            if weight is None:
                raise ValueError(
                    f"unknown scenario profile priority {value!r} for {key!r}; "
                    "expected high, medium, low, or a numeric weight"
                )
        out[axis] = weight
    return out
