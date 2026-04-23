"""Metric base — accumulator that folds per-retrieval events into a scalar [0,1].

``MetricResult`` carries an optional ``baseline_score`` (the same metric's
value when a random-retrieval adapter is run against the same scenario) and
a ``normalized_score`` computed by the runner as
``(score - baseline) / (1 - baseline)`` clamped to ``[-1, 1]``. Every metric
is normalized uniformly — there is no opt-out. If the random baseline is
near-saturation (e.g. Coverage, where a uniform adapter sweeps ~97% of the
pool), adapters that don't beat random on that dimension show negative
normalized scores, which is the honest signal.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from memory_bench.adapters.base import Adapter, Retrieval
from memory_bench.scenario.item_pool import ItemPool


@dataclass
class MetricResult:
    family: str
    dimension: str
    score: float
    n_samples: int
    baseline_score: Optional[float] = None
    normalized_score: Optional[float] = None


class MetricAccumulator(ABC):
    family: str = ""
    dimension: str = ""

    def __init__(self) -> None:
        self._samples: List[float] = []

    @abstractmethod
    def observe(
        self,
        adapter: Adapter,
        retrieval: Retrieval,
        hits: List[bool],
        affinity_scores: List[float],
        pool: ItemPool,
        global_step: int,
    ) -> None:
        raise NotImplementedError

    def _push(self, x: float) -> None:
        self._samples.append(x)

    def score(self) -> float:
        if not self._samples:
            return 0.0
        return sum(self._samples) / len(self._samples)

    def result(self) -> MetricResult:
        return MetricResult(
            family=self.family,
            dimension=self.dimension,
            score=self.score(),
            n_samples=len(self._samples),
        )

    def reset(self) -> None:
        self._samples.clear()
