"""memory-bench CLI."""

from __future__ import annotations

from pathlib import Path

import click

from memory_bench.adapters.actr import ACTRAdapter
from memory_bench.adapters.bm25 import BM25Adapter
from memory_bench.adapters.recency import RecencyAdapter
from memory_bench.metrics.adaptation import CrossSessionLearning, Personalization
from memory_bench.metrics.exploration import Coverage, NoveltyGuarantee
from memory_bench.metrics.maintenance import ForgettingQuality, UpdateCoherence
from memory_bench.metrics.ranking import FrequencyGain, RelevanceAtK
from memory_bench.report import write_reports
from memory_bench.runner import run_benchmark
from memory_bench.scenario.generator import ScenarioConfig


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
    "--scenario",
    "scenario_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
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
def run(scenario_path: Path, out_dir: Path, embedding: bool, composite: bool) -> None:
    """Run a scenario end-to-end."""
    cfg = ScenarioConfig.from_yaml(scenario_path)
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
    click.echo(f"Running {cfg.name}: {len(adapters)} adapters × {len(metric_factories)} metrics")
    results = run_benchmark(cfg, adapters, metric_factories)
    write_reports(results, out_dir)
    click.echo(f"Wrote {out_dir / 'results.md'}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
