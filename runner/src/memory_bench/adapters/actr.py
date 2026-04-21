"""ACT-R adapter — cognitive activation (base-level + novelty + weak semantic nudge)."""

from __future__ import annotations

import math
import re
from typing import Dict, List

from memory_bench.adapters.base import Adapter, Retrieval
from memory_bench.scenario.context import Query

_TOKEN = re.compile(r"[A-Za-z0-9_]+")


def _tokens(text: str) -> set[str]:
    return {tok.lower() for tok in _TOKEN.findall(text)}


class ACTRAdapter(Adapter):
    """Novelty-first ACT-R.

    score(i, q) = base_activation(i) + alpha_nov * novelty(i) + beta_sem * jaccard(q, i)

    - base_activation: ln(1 + obs_count) * exp(-decay * age_since_last_obs)
    - novelty:         min(1.0, age_since_last_retrieval / cooldown)  (stale items bonus)
    - jaccard:         light semantic anchor on query text vs item content
    """

    def __init__(
        self,
        name: str = "actr",
        decay: float = 0.3,
        alpha_nov: float = 0.5,
        cooldown: int = 10,
        beta_sem: float = 0.25,
    ) -> None:
        super().__init__(name)
        self.decay = decay
        self.alpha_nov = alpha_nov
        self.cooldown = cooldown
        self.beta_sem = beta_sem
        self._token_cache: Dict[str, set[str]] = {}

    def _on_observe(self, item, global_step: int) -> None:
        if item.id not in self._token_cache:
            self._token_cache[item.id] = _tokens(item.content)

    def retrieve(self, query: Query, global_step: int) -> Retrieval:
        q_tokens = _tokens(query.text)
        scored: List[tuple[str, float]] = []
        for iid in self.known_item_ids():
            obs_times = self._obs_times.get(iid, [])
            if not obs_times:
                continue
            count = len(obs_times)
            age_obs = max(0, global_step - obs_times[-1])
            base = math.log(1 + count) * math.exp(-self.decay * age_obs)

            last_retr = self.last_retr(iid)
            age_retr = global_step - last_retr
            novelty = min(1.0, age_retr / self.cooldown) if last_retr > -(10**9) else 1.0

            item_tokens = self._token_cache.get(iid, set())
            if q_tokens and item_tokens:
                inter = len(q_tokens & item_tokens)
                union = len(q_tokens | item_tokens)
                jac = inter / union if union else 0.0
            else:
                jac = 0.0

            score = base + self.alpha_nov * novelty + self.beta_sem * jac
            scored.append((iid, score))

        scored.sort(key=lambda x: -x[1])
        top = scored[: query.top_k]
        return Retrieval(
            query=query,
            adapter_name=self.name,
            item_ids=[iid for iid, _ in top],
            scores=[s for _, s in top],
        )
