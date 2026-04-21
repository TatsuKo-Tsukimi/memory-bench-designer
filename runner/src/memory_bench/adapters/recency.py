"""Recency adapter — the null baseline."""

from __future__ import annotations

from memory_bench.adapters.base import Adapter, Retrieval
from memory_bench.scenario.context import Query


class RecencyAdapter(Adapter):
    def __init__(self, name: str = "recency") -> None:
        super().__init__(name)

    def retrieve(self, query: Query, global_step: int) -> Retrieval:
        ranked = sorted(
            self.known_item_ids(),
            key=lambda iid: -self.last_obs(iid),
        )
        top = ranked[: query.top_k]
        scores = [float(self.last_obs(iid)) for iid in top]
        return Retrieval(query=query, adapter_name=self.name, item_ids=top, scores=scores)
