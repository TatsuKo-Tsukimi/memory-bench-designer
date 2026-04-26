import random

from memory_bench.scenario.generator import (
    ProtocolConfig,
    ScenarioConfig,
    _archetype_counts,
    _mixed_content,
    _sample_content,
    generate_scenario,
)


def test_mixed_content_supports_single_theme_without_crash():
    rng = random.Random(123)
    text = _mixed_content([['alpha', 'beta', 'gamma']], primary=0, rng=rng, min_n=6, max_n=6, primary_ratio=0.4)
    tokens = text.split()
    assert len(tokens) == 6
    assert set(tokens).issubset({'alpha', 'beta', 'gamma'})


def test_sample_content_handles_empty_vocab():
    rng = random.Random(123)
    text = _sample_content([], rng, min_n=5, max_n=10)
    tokens = text.split()
    assert tokens
    assert set(tokens) == {'placeholder'}


def test_evolving_count_is_even_without_losing_pool_items():
    counts = _archetype_counts(
        50, {"core": 0.3, "evolving": 0.3, "episode": 0.3, "noise": 0.1}
    )
    assert counts["evolving"] % 2 == 0
    assert sum(counts.values()) == 50


def test_arrival_modes_control_episode_and_noise_slots():
    protocol = ProtocolConfig(seed=3, pool_size=20, sessions=3, steps_per_session=5, top_k=3)
    scenario = ScenarioConfig(
        name="arrivals",
        archetypes={"core": 0.1, "evolving": 0.2, "episode": 0.4, "noise": 0.3},
        themes=[
            {"name": "a", "vocab": ["alpha", "beta", "gamma"]},
            {"name": "b", "vocab": ["one", "two", "three"]},
        ],
        context_evolution={"type": "stable", "drift_rate": 0.1},
        arrivals={"episode": "initial", "noise": {"session": 2, "step": 4}},
    )
    pool, _ = generate_scenario(protocol, scenario)

    episodes = [item for item in pool.items.values() if item.archetype == "episode"]
    noise = [item for item in pool.items.values() if item.archetype == "noise"]

    assert episodes
    assert noise
    assert {(item.created_session, item.created_step) for item in episodes} == {(0, 0)}
    assert {(item.created_session, item.created_step) for item in noise} == {(2, 4)}
    assert len(pool) == protocol.pool_size
