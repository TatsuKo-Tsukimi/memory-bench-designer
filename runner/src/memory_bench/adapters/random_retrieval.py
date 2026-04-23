"""Random retrieval adapter — baseline source for normalized scoring.

Returns top_k items sampled uniformly at random from the items the adapter
has observed. Used as a shadow adapter inside ``run_benchmark`` so every
metric gets an empirical baseline against a no-information policy. Given a
fixed ``seed``, results are reproducible.
"""

from __future__ import annotations

import random

from memory_bench.adapters.base import Adapter, Retrieval
from memory_bench.scenario.context import Query


class RandomRetrievalAdapter(Adapter):
    def __init__(self, name: str = "__random__", seed: int = 0) -> None:
        super().__init__(name)
        self._rng = random.Random(seed)

    def retrieve(self, query: Query, global_step: int) -> Retrieval:
        known = self.known_item_ids()
        if not known:
            return Retrieval(query=query, adapter_name=self.name, item_ids=[], scores=[])
        k = min(query.top_k, len(known))
        top = self._rng.sample(known, k)
        scores = [0.0] * len(top)
        return Retrieval(query=query, adapter_name=self.name, item_ids=top, scores=scores)
