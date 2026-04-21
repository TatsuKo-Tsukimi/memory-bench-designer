"""Metric base — accumulator that folds per-retrieval events into a scalar [0,1]."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from memory_bench.adapters.base import Adapter, Retrieval
from memory_bench.scenario.item_pool import ItemPool


@dataclass
class MetricResult:
    family: str
    dimension: str
    score: float
    n_samples: int


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
