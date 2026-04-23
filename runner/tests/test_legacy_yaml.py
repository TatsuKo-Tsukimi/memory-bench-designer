"""Tests for pre-v0.2 single-file scenario YAML backward compatibility."""

from __future__ import annotations

import textwrap
import warnings

import pytest

from memory_bench.scenario.generator import (
    ProtocolConfig,
    ScenarioConfig,
    load_legacy_yaml,
)


LEGACY_YAML = textwrap.dedent(
    """
    name: legacy-example
    seed: 13
    pool_size: 50
    sessions: 2
    steps_per_session: 4
    top_k: 3

    archetypes:
      core: 0.3
      evolving: 0.3
      episode: 0.3
      noise: 0.1

    themes:
      - name: one
        vocab: [a, b, c, d]
      - name: two
        vocab: [e, f, g, h]

    context_evolution:
      type: stable
      drift_rate: 0.1

    arrivals:
      episode: streaming
      noise: streaming
    """
).strip()


def test_load_legacy_yaml_emits_deprecation_warning(tmp_path):
    p = tmp_path / "legacy.yaml"
    p.write_text(LEGACY_YAML, encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        protocol, scenario = load_legacy_yaml(p)
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert isinstance(protocol, ProtocolConfig)
    assert isinstance(scenario, ScenarioConfig)


def test_legacy_values_match_expected(tmp_path):
    p = tmp_path / "legacy.yaml"
    p.write_text(LEGACY_YAML, encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        protocol, scenario = load_legacy_yaml(p)
    assert protocol.seed == 13
    assert protocol.pool_size == 50
    assert protocol.sessions == 2
    assert protocol.steps_per_session == 4
    assert protocol.top_k == 3
    assert scenario.name == "legacy-example"
    assert scenario.archetypes["core"] == 0.3
    assert len(scenario.themes) == 2
    assert scenario.context_evolution["type"] == "stable"


def test_scenario_from_yaml_rejects_legacy_file(tmp_path):
    p = tmp_path / "legacy.yaml"
    p.write_text(LEGACY_YAML, encoding="utf-8")
    with pytest.raises(ValueError, match="protocol"):
        ScenarioConfig.from_yaml(p)


def test_protocol_from_yaml_rejects_legacy_file(tmp_path):
    p = tmp_path / "legacy.yaml"
    p.write_text(LEGACY_YAML, encoding="utf-8")
    with pytest.raises(ValueError, match="content"):
        ProtocolConfig.from_yaml(p)


def test_load_legacy_yaml_rejects_split_file(tmp_path):
    """If the user points load_legacy_yaml at a split-format file that only
    has content keys (no protocol keys), raise — nothing to merge.
    """
    p = tmp_path / "content_only.yaml"
    p.write_text(
        textwrap.dedent(
            """
            name: content-only
            archetypes: {core: 0.4, evolving: 0.2, episode: 0.3, noise: 0.1}
            themes:
              - name: x
                vocab: [a, b]
            context_evolution: {type: stable, drift_rate: 0.1}
            """
        ).strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="legacy"):
        load_legacy_yaml(p)
