"""BM25 adapter — statistical IR baseline, pure-Python."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List

from memory_bench.adapters.base import Adapter, Retrieval
from memory_bench.scenario.context import Query
from memory_bench.scenario.item_pool import Item

_TOKEN = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in _TOKEN.findall(text)]


class BM25Adapter(Adapter):
    def __init__(self, name: str = "bm25", k1: float = 1.5, b: float = 0.75) -> None:
        super().__init__(name)
        self.k1 = k1
        self.b = b
        self._tf: Dict[str, Counter] = {}
        self._doc_len: Dict[str, int] = {}
        self._df: Counter = Counter()
        self._total_len: int = 0
        self._n_docs: int = 0

    def _on_observe(self, item: Item, global_step: int) -> None:
        if item.id in self._tf:
            return
        tokens = _tokenize(item.content)
        tf = Counter(tokens)
        self._tf[item.id] = tf
        self._doc_len[item.id] = len(tokens)
        self._total_len += len(tokens)
        self._n_docs += 1
        for term in tf.keys():
            self._df[term] += 1

    def retrieve(self, query: Query, global_step: int) -> Retrieval:
        q_tokens = _tokenize(query.text)
        if not q_tokens or self._n_docs == 0:
            return Retrieval(query=query, adapter_name=self.name, item_ids=[], scores=[])

        avgdl = self._total_len / self._n_docs if self._n_docs else 1.0
        scored: List[tuple[str, float]] = []
        for iid in self.known_item_ids():
            tf = self._tf.get(iid)
            if not tf:
                continue
            dl = self._doc_len[iid]
            score = 0.0
            for term in q_tokens:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                df = self._df.get(term, 0)
                if df == 0:
                    continue
                idf = math.log(1.0 + (self._n_docs - df + 0.5) / (df + 0.5))
                denom = f + self.k1 * (1 - self.b + self.b * dl / avgdl)
                score += idf * f * (self.k1 + 1) / denom
            if score > 0:
                scored.append((iid, score))

        scored.sort(key=lambda x: -x[1])
        top = scored[: query.top_k]
        return Retrieval(
            query=query,
            adapter_name=self.name,
            item_ids=[iid for iid, _ in top],
            scores=[s for _, s in top],
        )
