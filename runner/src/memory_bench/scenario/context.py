"""Query + ContextStream — drives the per-step retrieval demand."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List


@dataclass
class Query:
    session: int
    step: int
    theme: int
    text: str
    top_k: int = 10


@dataclass
class ContextStream:
    queries: List[Query]

    def at(self, session: int, step: int) -> Query:
        return self.queries[session * self._steps_per_session + step]

    _steps_per_session: int = 0


def build_context_stream(
    *,
    evolution_type: str,
    themes: List[List[str]],
    sessions: int,
    steps_per_session: int,
    drift_rate: float,
    top_k: int,
    rng: random.Random,
) -> ContextStream:
    """Generate the per-step theme + query text schedule.

    evolution_type:
      - "random":          uniform random theme per step
      - "narrow-band-drift": slow drift — adjacent themes preferred
      - "stable":          single theme fixed per session, rotates across sessions
      - "mode-shifts":     long stretches of one theme, occasional hard switch
    """
    n_themes = len(themes)
    queries: List[Query] = []

    if evolution_type == "random":
        for s in range(sessions):
            for t in range(steps_per_session):
                theme = rng.randrange(n_themes)
                queries.append(_mk_query(s, t, theme, themes, top_k, rng))

    elif evolution_type == "narrow-band-drift":
        current = rng.randrange(n_themes)
        for s in range(sessions):
            for t in range(steps_per_session):
                if rng.random() < drift_rate:
                    current = (current + rng.choice([-1, 1])) % n_themes
                queries.append(_mk_query(s, t, current, themes, top_k, rng))

    elif evolution_type == "stable":
        # One primary theme persists across all sessions; occasional off-theme queries.
        primary = rng.randrange(n_themes)
        for s in range(sessions):
            for t in range(steps_per_session):
                theme = primary if rng.random() < 0.85 else rng.randrange(n_themes)
                queries.append(_mk_query(s, t, theme, themes, top_k, rng))

    elif evolution_type == "mode-shifts":
        current = rng.randrange(n_themes)
        steps_in_mode = 0
        mode_len = max(3, steps_per_session // 2)
        for s in range(sessions):
            for t in range(steps_per_session):
                if steps_in_mode >= mode_len and rng.random() < drift_rate * 2:
                    current = rng.randrange(n_themes)
                    steps_in_mode = 0
                    mode_len = max(3, rng.randint(steps_per_session // 3, steps_per_session))
                queries.append(_mk_query(s, t, current, themes, top_k, rng))
                steps_in_mode += 1

    else:
        raise ValueError(f"unknown evolution_type: {evolution_type}")

    stream = ContextStream(queries=queries)
    stream._steps_per_session = steps_per_session
    return stream


def _mk_query(
    session: int,
    step: int,
    theme: int,
    themes: List[List[str]],
    top_k: int,
    rng: random.Random,
) -> Query:
    vocab = themes[theme]
    n_primary = rng.randint(2, 3)
    tokens = rng.sample(vocab, min(n_primary, len(vocab)))
    if rng.random() < 0.3 and len(themes) > 1:
        off_themes = [t for t in range(len(themes)) if t != theme]
        tokens.append(rng.choice(themes[rng.choice(off_themes)]))
    rng.shuffle(tokens)
    text = " ".join(tokens)
    return Query(session=session, step=step, theme=theme, text=text, top_k=top_k)
