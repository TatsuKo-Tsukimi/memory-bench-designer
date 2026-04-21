import random

from memory_bench.scenario.generator import _mixed_content, _sample_content


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
