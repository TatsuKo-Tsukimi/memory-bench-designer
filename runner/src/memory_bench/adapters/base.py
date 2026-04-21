"""Adapter base — the interface every memory strategy implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

from memory_bench.scenario.context import Query
from memory_bench.scenario.item_pool import Item


@dataclass
class Retrieval:
    query: Query
    adapter_name: str
    item_ids: List[str]
    scores: List[float]


class Adapter(ABC):
    name: str

    def __init__(self, name: str) -> None:
        self.name = name
        self._observed_order: List[str] = []
        self._obs_times: Dict[str, List[int]] = {}
        self._retr_times: Dict[str, List[int]] = {}
        self._content: Dict[str, str] = {}
        self._importance: Dict[str, float] = {}

    def observe(self, item: Item, global_step: int) -> None:
        if item.id not in self._content:
            self._observed_order.append(item.id)
            self._content[item.id] = item.content
            self._importance[item.id] = item.importance
            self._obs_times[item.id] = []
            self._retr_times[item.id] = []
        self._obs_times[item.id].append(global_step)
        self._on_observe(item, global_step)

    def record_retrieval(self, retrieval: Retrieval, global_step: int) -> None:
        for iid in retrieval.item_ids:
            self._retr_times.setdefault(iid, []).append(global_step)

    def record_feedback(self, retrieval: Retrieval, hits: List[bool]) -> None:
        self._on_feedback(retrieval, hits)

    def _on_observe(self, item: Item, global_step: int) -> None:
        """Hook for subclasses to build indexes incrementally."""

    def _on_feedback(self, retrieval: Retrieval, hits: List[bool]) -> None:
        """Hook for subclasses that learn from hit/miss signals."""

    @abstractmethod
    def retrieve(self, query: Query, global_step: int) -> Retrieval:
        raise NotImplementedError

    def known_item_ids(self) -> List[str]:
        return list(self._observed_order)

    def last_obs(self, item_id: str) -> int:
        times = self._obs_times.get(item_id)
        return times[-1] if times else -1

    def last_retr(self, item_id: str) -> int:
        times = self._retr_times.get(item_id)
        return times[-1] if times else -(10**9)
