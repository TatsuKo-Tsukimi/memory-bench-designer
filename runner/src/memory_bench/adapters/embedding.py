"""Embedding adapter — dense cosine similarity via sentence-transformers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from memory_bench.adapters.base import Adapter, Retrieval
from memory_bench.scenario.context import Query
from memory_bench.scenario.item_pool import Item

_MODEL_CACHE: Dict[str, object] = {}


def _get_model(model_name: str):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    _MODEL_CACHE[model_name] = model
    return model


class EmbeddingAdapter(Adapter):
    """Cosine similarity over sentence-transformers embeddings."""

    def __init__(
        self,
        name: str = "embedding",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        super().__init__(name)
        self.model_name = model_name
        self._model = None
        self._vectors: Dict[str, np.ndarray] = {}
        self._pending_ids: List[str] = []
        self._pending_content: List[str] = []

    def config(self) -> Dict[str, Any]:
        cfg = super().config()
        cfg.update({"model_name": self.model_name})
        return cfg

    def _ensure_model(self):
        if self._model is None:
            self._model = _get_model(self.model_name)
        return self._model

    def _on_observe(self, item: Item, global_step: int) -> None:
        if item.id in self._vectors:
            return
        self._pending_ids.append(item.id)
        self._pending_content.append(item.content)

    def _flush_pending(self) -> None:
        if not self._pending_ids:
            return
        model = self._ensure_model()
        vecs = model.encode(
            self._pending_content,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        for iid, v in zip(self._pending_ids, vecs):
            self._vectors[iid] = v
        self._pending_ids.clear()
        self._pending_content.clear()

    def retrieve(self, query: Query, global_step: int) -> Retrieval:
        self._flush_pending()
        if not self._vectors:
            return Retrieval(query=query, adapter_name=self.name, item_ids=[], scores=[])

        model = self._ensure_model()
        q_vec = model.encode(
            [query.text],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )[0]

        ids = list(self._vectors.keys())
        matrix = np.stack([self._vectors[i] for i in ids])
        sims = matrix @ q_vec  # already normalized → cosine = dot
        top_idx = np.argsort(-sims)[: query.top_k]
        top_ids = [ids[i] for i in top_idx]
        top_scores = [float(sims[i]) for i in top_idx]
        return Retrieval(query=query, adapter_name=self.name, item_ids=top_ids, scores=top_scores)
