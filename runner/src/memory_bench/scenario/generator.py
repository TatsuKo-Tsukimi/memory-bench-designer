"""Top-level scenario generator — YAML → (ItemPool, ContextStream).

v0.2 splits the former single ``ScenarioConfig`` into two:

* :class:`ProtocolConfig` — evaluation physics (seed, pool_size, sessions,
  steps_per_session, top_k). Frozen for cross-run comparability. Equivalent
  to autoresearch's ``prepare.py`` constants.
* :class:`ScenarioConfig` — content (themes, archetypes, context_evolution,
  arrivals). User-customizable.

Pre-v0.2 single-file YAMLs still load via :func:`load_legacy_yaml` with a
``DeprecationWarning``.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from memory_bench.scenario.context import ContextStream, build_context_stream
from memory_bench.scenario.item_pool import Item, ItemPool


_PROTOCOL_FIELDS = ("seed", "pool_size", "sessions", "steps_per_session", "top_k")
_SCENARIO_FIELDS = ("archetypes", "themes", "context_evolution", "arrivals")


def detect_yaml_kind(path: str | Path) -> str:
    """Classify a YAML file as 'protocol', 'scenario', or 'legacy'.

    'legacy' = pre-v0.2 single-file with both protocol and content fields.
    'protocol' = only protocol fields.
    'scenario' = only content fields.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    has_p = any(k in raw for k in _PROTOCOL_FIELDS)
    has_s = any(k in raw for k in _SCENARIO_FIELDS)
    if has_p and has_s:
        return "legacy"
    if has_p:
        return "protocol"
    return "scenario"


@dataclass
class ProtocolConfig:
    seed: int = 42
    pool_size: int = 200
    sessions: int = 10
    steps_per_session: int = 40
    top_k: int = 10

    @staticmethod
    def from_yaml(path: str | Path) -> "ProtocolConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        content_keys = [k for k in _SCENARIO_FIELDS if k in raw]
        if content_keys:
            raise ValueError(
                f"{path} looks like a scenario or legacy YAML (has content keys "
                f"{content_keys}). Pass it to --scenario or load with "
                f"load_legacy_yaml() instead."
            )
        return ProtocolConfig(
            seed=int(raw.get("seed", 42)),
            pool_size=int(raw["pool_size"]),
            sessions=int(raw["sessions"]),
            steps_per_session=int(raw["steps_per_session"]),
            top_k=int(raw.get("top_k", 10)),
        )


@dataclass
class ScenarioConfig:
    name: str
    archetypes: Dict[str, float]
    themes: List[Dict[str, Any]]
    context_evolution: Dict[str, Any]
    arrivals: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_yaml(path: str | Path) -> "ScenarioConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        protocol_keys = [k for k in _PROTOCOL_FIELDS if k in raw]
        if protocol_keys:
            raise ValueError(
                f"{path} looks like a protocol or legacy YAML (has protocol keys "
                f"{protocol_keys}). Pass it to --protocol or load with "
                f"load_legacy_yaml() instead."
            )
        return ScenarioConfig(
            name=raw.get("name", Path(path).stem),
            archetypes=dict(raw["archetypes"]),
            themes=list(raw["themes"]),
            context_evolution=dict(raw["context_evolution"]),
            arrivals=dict(raw.get("arrivals", {})),
        )


def load_legacy_yaml(path: str | Path) -> Tuple[ProtocolConfig, ScenarioConfig]:
    """Load a pre-v0.2 single-file scenario YAML.

    Emits a :class:`DeprecationWarning`. The file must contain both protocol
    and content fields (the pre-v0.2 layout). Returns the split pair.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    has_protocol = any(k in raw for k in _PROTOCOL_FIELDS)
    has_content = any(k in raw for k in _SCENARIO_FIELDS)
    if not (has_protocol and has_content):
        raise ValueError(
            f"{path} is not a legacy single-file YAML (expected both protocol "
            f"and scenario fields). Use ProtocolConfig.from_yaml or "
            f"ScenarioConfig.from_yaml directly."
        )
    warnings.warn(
        f"Loading legacy single-file scenario YAML from {path}. Split into "
        f"protocols/ + scenarios/ before v0.3 — see README for the new layout.",
        DeprecationWarning,
        stacklevel=2,
    )
    protocol = ProtocolConfig(
        seed=int(raw.get("seed", 42)),
        pool_size=int(raw["pool_size"]),
        sessions=int(raw["sessions"]),
        steps_per_session=int(raw["steps_per_session"]),
        top_k=int(raw.get("top_k", 10)),
    )
    scenario = ScenarioConfig(
        name=raw.get("name", path.stem),
        archetypes=dict(raw["archetypes"]),
        themes=list(raw["themes"]),
        context_evolution=dict(raw["context_evolution"]),
        arrivals=dict(raw.get("arrivals", {})),
    )
    return protocol, scenario


def generate_scenario(
    protocol: ProtocolConfig, scenario: ScenarioConfig
) -> Tuple[ItemPool, ContextStream]:
    rng = random.Random(protocol.seed)
    theme_vocabs = [list(t["vocab"]) for t in scenario.themes]
    n_themes = len(theme_vocabs)
    noise_vocab = _noise_vocab(theme_vocabs, rng)

    counts = _archetype_counts(protocol.pool_size, scenario.archetypes)

    pool = ItemPool()
    _gen_core(pool, counts["core"], theme_vocabs, rng)
    _gen_evolving(pool, counts["evolving"], theme_vocabs, protocol.sessions, rng)
    _gen_episode(
        pool,
        counts["episode"],
        theme_vocabs,
        protocol.sessions,
        protocol.steps_per_session,
        scenario.arrivals,
        rng,
    )
    _gen_noise(
        pool,
        counts["noise"],
        noise_vocab,
        protocol.sessions,
        protocol.steps_per_session,
        scenario.arrivals,
        rng,
    )

    stream = build_context_stream(
        evolution_type=scenario.context_evolution["type"],
        themes=theme_vocabs,
        sessions=protocol.sessions,
        steps_per_session=protocol.steps_per_session,
        drift_rate=float(scenario.context_evolution.get("drift_rate", 0.1)),
        top_k=protocol.top_k,
        rng=rng,
    )
    return pool, stream


def _archetype_counts(pool_size: int, ratios: Dict[str, float]) -> Dict[str, int]:
    names = ["core", "evolving", "episode", "noise"]
    counts = {n: int(round(pool_size * ratios.get(n, 0.0))) for n in names}
    diff = pool_size - sum(counts.values())
    if diff != 0:
        counts["episode"] += diff
    if counts["evolving"] % 2:
        for donor in ("episode", "noise", "core"):
            if counts[donor] > 0:
                counts["evolving"] += 1
                counts[donor] -= 1
                break
        else:
            counts["evolving"] -= 1
            counts["episode"] += 1
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
    if not vocab:
        vocab = ["placeholder"]
    low = min(min_n, len(vocab))
    high = max(low, min(max_n, len(vocab)))
    n = rng.randint(low, high)
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
    if others:
        for _ in range(n_other):
            t = rng.choice(others)
            tokens.append(rng.choice(theme_vocabs[t]))
    else:
        tokens.extend(rng.choices(theme_vocabs[primary], k=n_other))
    rng.shuffle(tokens)
    return " ".join(tokens)


def _arrival_slot(
    archetype: str,
    arrivals: Dict[str, Any],
    sessions: int,
    steps_per_session: int,
    rng: random.Random,
) -> Tuple[int, int]:
    mode = arrivals.get(archetype, "streaming")
    if isinstance(mode, dict):
        session = int(mode.get("session", 0))
        step = int(mode.get("step", 0))
        return (
            max(0, min(sessions - 1, session)),
            max(0, min(steps_per_session - 1, step)),
        )
    if mode == "streaming":
        return rng.randrange(sessions), rng.randrange(steps_per_session)
    if mode == "initial":
        return 0, 0
    if mode == "session-start":
        return rng.randrange(sessions), 0
    raise ValueError(
        f"unknown arrival mode for {archetype}: {mode!r}; "
        "expected streaming, initial, session-start, or {session, step}"
    )


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
    arrivals: Dict[str, Any],
    rng: random.Random,
) -> None:
    n_themes = len(theme_vocabs)
    for i in range(n):
        primary = rng.randrange(n_themes)
        session, step = _arrival_slot("episode", arrivals, sessions, steps_per_session, rng)
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
    arrivals: Dict[str, Any],
    rng: random.Random,
) -> None:
    for i in range(n):
        session, step = _arrival_slot("noise", arrivals, sessions, steps_per_session, rng)
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
