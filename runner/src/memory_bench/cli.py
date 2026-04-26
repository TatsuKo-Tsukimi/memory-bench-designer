"""memory-bench CLI."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import click

from memory_bench.adapters.actr import ACTRAdapter
from memory_bench.adapters.bm25 import BM25Adapter
from memory_bench.adapters.recency import RecencyAdapter
from memory_bench.metrics.adaptation import CrossSessionLearning, Personalization
from memory_bench.metrics.exploration import Coverage, NoveltyGuarantee
from memory_bench.metrics.maintenance import ForgettingQuality, UpdateCoherence
from memory_bench.metrics.ranking import FrequencyGain, RelevanceAtK
from memory_bench.profile import ScenarioProfile
from memory_bench.report import write_reports
from memory_bench.runner import run_benchmark, run_benchmark_suite
from memory_bench.scenario.generator import (
    ProtocolConfig,
    ScenarioConfig,
    detect_yaml_kind,
    load_legacy_yaml,
)
from memory_bench.trace import TraceConfig, replay_trace


def _build_adapters(include_embedding: bool, include_composite: bool):
    adapters = [RecencyAdapter(), BM25Adapter(), ACTRAdapter()]
    if include_embedding:
        try:
            from memory_bench.adapters.embedding import EmbeddingAdapter

            adapters.append(EmbeddingAdapter())
        except ImportError:
            click.echo(
                "[warn] embedding adapter requested but sentence-transformers not installed; "
                "skipping. Install with `pip install memory-bench[embedding]`.",
                err=True,
            )
    if include_composite:
        try:
            from memory_bench.adapters.composite import CompositeAdapter

            adapters.append(CompositeAdapter())
        except ImportError:
            click.echo(
                "[warn] composite adapter requires sentence-transformers; skipping.",
                err=True,
            )
    return adapters


@click.group()
def cli() -> None:
    """memory-bench: agent memory benchmark runner."""


@cli.command()
@click.option(
    "--protocol",
    "protocol_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to protocol YAML (evaluation physics: seed, pool_size, sessions, "
    "steps_per_session, top_k). If omitted, --scenario is auto-classified: "
    "content-only files run under ProtocolConfig() defaults, legacy single-file "
    "formats are loaded with a deprecation warning.",
)
@click.option(
    "--scenario",
    "scenario_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to scenario YAML (content: archetypes, themes, context_evolution, "
    "arrivals).",
)
@click.option(
    "--out",
    "out_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option(
    "--embedding/--no-embedding",
    default=False,
    help="Include the sentence-transformers embedding adapter.",
)
@click.option(
    "--composite/--no-composite",
    default=False,
    help="Include the composite multi-signal adapter.",
)
@click.option(
    "--seeds",
    default=None,
    help="Comma-separated seeds for multi-run capability profiles, e.g. 1,2,3,4,5.",
)
@click.option(
    "--profile",
    "profile_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Optional scenario profile YAML with capability priorities.",
)
def run(
    protocol_path: Path | None,
    scenario_path: Path,
    out_dir: Path,
    embedding: bool,
    composite: bool,
    seeds: str | None,
    profile_path: Path | None,
) -> None:
    """Run a scenario end-to-end."""
    if protocol_path is None:
        kind = detect_yaml_kind(scenario_path)
        if kind == "legacy":
            click.echo(
                "[warn] --protocol not provided; loading --scenario as legacy "
                "single-file YAML (deprecated, will be removed in v0.3).",
                err=True,
            )
            protocol, scenario = load_legacy_yaml(scenario_path)
        elif kind == "scenario":
            click.echo(
                "[info] --protocol not provided; using default ProtocolConfig "
                "(seed=42, pool_size=200, sessions=10, steps_per_session=40, top_k=10).",
                err=True,
            )
            protocol = ProtocolConfig()
            scenario = ScenarioConfig.from_yaml(scenario_path)
        else:  # "protocol"
            raise click.UsageError(
                f"{scenario_path} is a protocol YAML, not a scenario. "
                "Pass it via --protocol and provide a scenario YAML via --scenario."
            )
    else:
        try:
            protocol = ProtocolConfig.from_yaml(protocol_path)
            scenario = ScenarioConfig.from_yaml(scenario_path)
        except ValueError as e:
            raise click.UsageError(str(e)) from e

    seed_values = _parse_seeds(seeds, protocol.seed)
    if len(seed_values) == 1:
        protocol = replace(protocol, seed=seed_values[0])
    profile = ScenarioProfile.from_yaml(profile_path) if profile_path else None
    adapters = _build_adapters(embedding, composite)
    metric_factories = [
        NoveltyGuarantee,
        Coverage,
        RelevanceAtK,
        FrequencyGain,
        Personalization,
        CrossSessionLearning,
        UpdateCoherence,
        ForgettingQuality,
    ]
    if len(seed_values) == 1 and profile is None:
        click.echo(
            f"Running {scenario.name}: {len(adapters)} adapters x {len(metric_factories)} metrics"
        )
        results = run_benchmark(protocol, scenario, adapters, metric_factories)
    else:
        click.echo(
            f"Running {scenario.name}: {len(adapters)} adapters x "
            f"{len(metric_factories)} metrics x {len(seed_values)} seeds"
        )
        results = run_benchmark_suite(
            protocol=protocol,
            scenario=scenario,
            adapter_factory=lambda: _build_adapters(embedding, composite),
            metric_factories=metric_factories,
            seeds=seed_values,
            profile=profile,
        )
    write_reports(results, out_dir)
    click.echo(f"Wrote {out_dir / 'results.md'}")
    if hasattr(results, "protocol_hash"):
        click.echo(
            f"protocol_hash={results.protocol_hash}  scenario_hash={results.scenario_hash}"
        )
    else:
        click.echo(
            f"protocol_template_hash={results.protocol_template_hash}  "
            f"scenario_hash={results.scenario_hash}  profile_hash={results.profile_hash}"
        )


def _parse_seeds(raw: str | None, default_seed: int) -> list[int]:
    if raw is None or not raw.strip():
        return [default_seed]
    out = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            try:
                out.append(int(part))
            except ValueError as e:
                raise click.UsageError(f"invalid seed value: {part!r}") from e
    if not out:
        raise click.UsageError("--seeds must contain at least one integer")
    return out


@cli.command()
@click.option(
    "--trace",
    "trace_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to trace replay YAML.",
)
@click.option(
    "--out",
    "out_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option(
    "--profile",
    "profile_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Optional scenario profile YAML with capability priorities.",
)
@click.option(
    "--embedding/--no-embedding",
    default=False,
    help="Include the sentence-transformers embedding adapter.",
)
@click.option(
    "--composite/--no-composite",
    default=False,
    help="Include the composite multi-signal adapter.",
)
def replay(
    trace_path: Path,
    out_dir: Path,
    profile_path: Path | None,
    embedding: bool,
    composite: bool,
) -> None:
    """Replay a real agent-memory trace against adapters."""
    trace = TraceConfig.from_yaml(trace_path)
    profile = ScenarioProfile.from_yaml(profile_path) if profile_path else None
    adapters = _build_adapters(embedding, composite)
    click.echo(f"Replaying {trace.name}: {len(adapters)} adapters")
    results = replay_trace(
        trace=trace,
        adapter_factory=lambda: _build_adapters(embedding, composite),
        profile=profile,
    )
    write_reports(results, out_dir)
    click.echo(f"Wrote {out_dir / 'results.md'}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
