"""Top-level scenario generator — YAML → (ItemPool, ContextStream)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from memory_bench.scenario.context import ContextStream, build_context_stream
from memory_bench.scenario.item_pool import Item, ItemPool


@dataclass
class ScenarioConfig:
    name: str
    pool_size: int
    sessions: int
    steps_per_session: int
    top_k: int
    archetypes: Dict[str, float]
    themes: List[Dict[str, Any]]
    context_evolution: Dict[str, Any]
    arrivals: Dict[str, Any]
    seed: int = 42

    @staticmethod
    def from_yaml(path: str | Path) -> "ScenarioConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return ScenarioConfig(
            name=raw.get("name", Path(path).stem),
            pool_size=int(raw["pool_size"]),
            sessions=int(raw["sessions"]),
            steps_per_session=int(raw["steps_per_session"]),
            top_k=int(raw.get("top_k", 10)),
            archetypes=dict(raw["archetypes"]),
            themes=list(raw["themes"]),
            context_evolution=dict(raw["context_evolution"]),
            arrivals=dict(raw.get("arrivals", {})),
            seed=int(raw.get("seed", 42)),
        )


def generate_scenario(cfg: ScenarioConfig) -> Tuple[ItemPool, ContextStream, ScenarioConfig]:
    rng = random.Random(cfg.seed)
    theme_vocabs = [list(t["vocab"]) for t in cfg.themes]
    n_themes = len(theme_vocabs)
    noise_vocab = _noise_vocab(theme_vocabs, rng)

    counts = _archetype_counts(cfg.pool_size, cfg.archetypes)

    pool = ItemPool()
    _gen_core(pool, counts["core"], theme_vocabs, rng)
    _gen_evolving(pool, counts["evolving"], theme_vocabs, cfg.sessions, rng)
    _gen_episode(
        pool,
        counts["episode"],
        theme_vocabs,
        cfg.sessions,
        cfg.steps_per_session,
        rng,
    )
    _gen_noise(
        pool,
        counts["noise"],
        noise_vocab,
        cfg.sessions,
        cfg.steps_per_session,
        rng,
    )

    stream = build_context_stream(
        evolution_type=cfg.context_evolution["type"],
        themes=theme_vocabs,
        sessions=cfg.sessions,
        steps_per_session=cfg.steps_per_session,
        drift_rate=float(cfg.context_evolution.get("drift_rate", 0.1)),
        top_k=cfg.top_k,
        rng=rng,
    )
    return pool, stream, cfg


def _archetype_counts(pool_size: int, ratios: Dict[str, float]) -> Dict[str, int]:
    names = ["core", "evolving", "episode", "noise"]
    counts = {n: int(round(pool_size * ratios.get(n, 0.0))) for n in names}
    diff = pool_size - sum(counts.values())
    if diff != 0:
        counts["episode"] += diff
    return counts


def _noise_vocab(theme_vocabs: List[List[str]], rng: random.Random) -> List[str]:
    tokens = set()
    for v in theme_vocabs:
        tokens.update(v)
    generic = [
        "meta",
        "summary",
        "note",
        "random",
        "chatter",
        "stuff",
        "thing",
        "lorem",
        "ipsum",
        "unrelated",
        "misc",
        "assorted",
        "tangent",
        "sidenote",
        "flavor",
        "filler",
    ]
    return [t for t in generic if t not in tokens]


def _broad_affinity(n_themes: int, primary: int, rng: random.Random) -> Dict[int, float]:
    aff = {primary: 1.0}
    for t in range(n_themes):
        if t == primary:
            continue
        aff[t] = round(rng.uniform(0.3, 0.7), 3)
    return aff


def _narrow_affinity(n_themes: int, primary: int, rng: random.Random) -> Dict[int, float]:
    aff = {primary: 1.0}
    for t in range(n_themes):
        if t == primary:
            continue
        aff[t] = round(rng.uniform(0.0, 0.15), 3)
    return aff


def _sample_content(vocab: List[str], rng: random.Random, min_n: int = 6, max_n: int = 14) -> str:
    n = rng.randint(min_n, min(max_n, len(vocab)))
    words = rng.sample(vocab, n) if n <= len(vocab) else rng.choices(vocab, k=n)
    return " ".join(words)


def _mixed_content(
    theme_vocabs: List[List[str]],
    primary: int,
    rng: random.Random,
    min_n: int,
    max_n: int,
    primary_ratio: float,
) -> str:
    """Content mixing primary theme with cross-theme pollution.

    primary_ratio = fraction of tokens from primary theme. Lower = more blur.
    """
    n = rng.randint(min_n, max_n)
    n_primary = max(1, int(n * primary_ratio))
    n_other = n - n_primary
    tokens = rng.choices(theme_vocabs[primary], k=n_primary)
    others = [t for t in range(len(theme_vocabs)) if t != primary]
    for _ in range(n_other):
        t = rng.choice(others)
        tokens.append(rng.choice(theme_vocabs[t]))
    rng.shuffle(tokens)
    return " ".join(tokens)


def _gen_core(pool: ItemPool, n: int, theme_vocabs: List[List[str]], rng: random.Random) -> None:
    n_themes = len(theme_vocabs)
    for i in range(n):
        primary = i % n_themes
        content = _mixed_content(theme_vocabs, primary, rng, 10, 16, primary_ratio=0.4)
        item = Item(
            id=f"core_{i:04d}",
            archetype="core",
            content=content,
            primary_theme=primary,
            theme_affinity=_broad_affinity(n_themes, primary, rng),
            importance=round(rng.uniform(0.7, 1.0), 3),
            created_session=0,
            created_step=0,
        )
        pool.add(item)


def _gen_evolving(
    pool: ItemPool, n: int, theme_vocabs: List[List[str]], sessions: int, rng: random.Random
) -> None:
    n_themes = len(theme_vocabs)
    pairs = n // 2
    mid_session = max(1, sessions // 2)
    for i in range(pairs):
        primary = rng.randrange(n_themes)
        original = Item(
            id=f"evolving_{i:04d}_v1",
            archetype="evolving",
            content=_mixed_content(theme_vocabs, primary, rng, 8, 12, primary_ratio=0.7),
            primary_theme=primary,
            theme_affinity=_narrow_affinity(n_themes, primary, rng),
            importance=round(rng.uniform(0.5, 0.9), 3),
            created_session=0,
            created_step=rng.randint(0, 3),
        )
        successor = Item(
            id=f"evolving_{i:04d}_v2",
            archetype="evolving",
            content="updated: "
            + _mixed_content(theme_vocabs, primary, rng, 8, 12, primary_ratio=0.7),
            primary_theme=primary,
            theme_affinity=_narrow_affinity(n_themes, primary, rng),
            importance=round(rng.uniform(0.6, 1.0), 3),
            created_session=mid_session,
            created_step=rng.randint(0, 3),
        )
        pool.add(original)
        pool.add(successor)
        pool.link_supersedes(original.id, successor.id, at_session=mid_session)


def _gen_episode(
    pool: ItemPool,
    n: int,
    theme_vocabs: List[List[str]],
    sessions: int,
    steps_per_session: int,
    rng: random.Random,
) -> None:
    n_themes = len(theme_vocabs)
    for i in range(n):
        primary = rng.randrange(n_themes)
        session = rng.randrange(sessions)
        step = rng.randrange(steps_per_session)
        item = Item(
            id=f"episode_{i:04d}",
            archetype="episode",
            content=_mixed_content(theme_vocabs, primary, rng, 7, 11, primary_ratio=0.7),
            primary_theme=primary,
            theme_affinity=_narrow_affinity(n_themes, primary, rng),
            importance=round(rng.uniform(0.3, 0.7), 3),
            created_session=session,
            created_step=step,
        )
        pool.add(item)


def _gen_noise(
    pool: ItemPool,
    n: int,
    noise_vocab: List[str],
    sessions: int,
    steps_per_session: int,
    rng: random.Random,
) -> None:
    for i in range(n):
        session = rng.randrange(sessions)
        step = rng.randrange(steps_per_session)
        item = Item(
            id=f"noise_{i:04d}",
            archetype="noise",
            content=_sample_content(noise_vocab, rng, 5, 10),
            primary_theme=None,
            theme_affinity={},
            importance=round(rng.uniform(0.1, 0.3), 3),
            created_session=session,
            created_step=step,
        )
        pool.add(item)
