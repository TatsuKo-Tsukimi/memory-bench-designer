"""Composite adapter — weighted blend of similarity, recency-decay, importance."""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from memory_bench.adapters.base import Adapter, Retrieval
from memory_bench.adapters.embedding import EmbeddingAdapter
from memory_bench.scenario.context import Query
from memory_bench.scenario.item_pool import Item


class CompositeAdapter(Adapter):
    """score = w_sim * embedding_cosine + w_decay * exp(-tau * age) + w_imp * importance.

    Defaults track the CrewAI-style recipe: 0.5 / 0.3 / 0.2.
    """

    def __init__(
        self,
        name: str = "composite",
        w_sim: float = 0.5,
        w_decay: float = 0.3,
        w_imp: float = 0.2,
        decay_tau: float = 0.05,
        embedding: EmbeddingAdapter | None = None,
    ) -> None:
        super().__init__(name)
        self.w_sim = w_sim
        self.w_decay = w_decay
        self.w_imp = w_imp
        self.decay_tau = decay_tau
        self._emb = embedding or EmbeddingAdapter(name=f"{name}._emb")

    def _on_observe(self, item: Item, global_step: int) -> None:
        self._emb.observe(item, global_step)

    def retrieve(self, query: Query, global_step: int) -> Retrieval:
        self._emb._flush_pending()
        if not self._emb._vectors:
            return Retrieval(query=query, adapter_name=self.name, item_ids=[], scores=[])

        model = self._emb._ensure_model()
        q_vec = model.encode(
            [query.text],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )[0]

        ids = list(self._emb._vectors.keys())
        matrix = np.stack([self._emb._vectors[i] for i in ids])
        sims = matrix @ q_vec

        scored: List[tuple[str, float]] = []
        for idx, iid in enumerate(ids):
            sim = float(sims[idx])
            age = global_step - self.last_obs(iid)
            decay = math.exp(-self.decay_tau * max(0, age))
            imp = self._importance.get(iid, 0.0)
            score = self.w_sim * sim + self.w_decay * decay + self.w_imp * imp
            scored.append((iid, score))

        scored.sort(key=lambda x: -x[1])
        top = scored[: query.top_k]
        return Retrieval(
            query=query,
            adapter_name=self.name,
            item_ids=[iid for iid, _ in top],
            scores=[s for _, s in top],
        )
