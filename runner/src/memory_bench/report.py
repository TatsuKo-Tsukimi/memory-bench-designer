"""Markdown + JSON report generator.

v0.2 emits two per-dimension leaderboards (raw and normalized-vs-random
baseline) plus ``protocol_hash`` / ``scenario_hash`` in the header. Two
result files are comparable only when their ``protocol_hash`` matches;
the ``normalized_score`` column is what you compare across scenarios with
the same protocol. Every metric is normalized uniformly — there is no
self-normalized opt-out.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from memory_bench.metrics.base import MetricResult
from memory_bench.runner import BenchmarkResults


def render_markdown(results: BenchmarkResults) -> str:
    lines: List[str] = []
    lines.append(f"# {results.scenario_name}")
    lines.append("")

    lines.append("## Run identity")
    lines.append("")
    lines.append(f"- protocol_hash: `{results.protocol_hash}`")
    lines.append(f"- scenario_hash: `{results.scenario_hash}`")
    lines.append("")
    lines.append(
        "> Two result files are comparable only when `protocol_hash` matches. "
        "Use the **Normalized** table when comparing across scenarios."
    )
    lines.append("")

    lines.append("## Protocol")
    lines.append("")
    pm = results.protocol_meta
    lines.append(f"- pool_size: **{pm['pool_size']}**")
    lines.append(f"- sessions × steps: **{pm['sessions']} × {pm['steps_per_session']}**")
    lines.append(f"- top_k: **{pm['top_k']}**")
    lines.append(f"- seed: {pm['seed']}")
    lines.append("")

    lines.append("## Scenario content")
    lines.append("")
    sm = results.scenario_meta
    lines.append(f"- archetypes: {sm['archetypes']}")
    lines.append(f"- context_evolution: {sm['context_evolution']}")
    lines.append(f"- n_themes: {sm['n_themes']}")
    lines.append("")

    adapters = list(results.per_adapter.keys())
    dims = _unique_dimensions(results)

    lines.append("## Per-dimension leaderboard — raw")
    lines.append("")
    lines.append(_render_table(adapters, dims, results, use_normalized=False))
    lines.append("")

    lines.append("## Per-dimension leaderboard — normalized (vs random baseline)")
    lines.append("")
    lines.append(
        "Normalized = (raw − baseline) / (1 − baseline), clamped to [−1, 1]. "
        "0 = matches random, 1 = perfect, negative = worse than random. "
        "When the random baseline is near-saturation (common on coverage, where "
        "a uniform adapter touches ~97% of the pool), adapters that concentrate "
        "clamp to −1 — an honest signal that they explore narrower than random."
    )
    lines.append("")
    lines.append(_render_table(adapters, dims, results, use_normalized=True))
    lines.append("")

    lines.append("## Baseline scores (random adapter)")
    lines.append("")
    lines.append("| dimension | baseline score |")
    lines.append("|---|---|")
    for d in dims:
        b = results.baseline_scores.get(d)
        lines.append(f"| {d} | {b:.3f} |" if b is not None else f"| {d} | — |")
    lines.append("")

    lines.append("## Per-family averages — normalized")
    lines.append("")
    families = _unique_families(results)
    header2 = "| adapter | " + " | ".join(families) + " |"
    sep2 = "|" + "|".join(["---"] * (len(families) + 1)) + "|"
    lines.append(header2)
    lines.append(sep2)
    for adapter in adapters:
        fam_avgs = _family_avgs_for(results, adapter, use_normalized=True)
        cells = " | ".join(f"{fam_avgs.get(f, 0.0):+.3f}" for f in families)
        lines.append(f"| {adapter} | {cells} |")

    lines.append("")
    lines.append("## Family winners (by normalized family-avg)")
    lines.append("")
    lines.append("| family | winner | normalized family-avg |")
    lines.append("|---|---|---|")
    for family, winner, score in _family_winners(results, use_normalized=True):
        lines.append(f"| {family} | **{winner}** | {score:+.3f} |")

    lines.append("")
    lines.append("## Reading guide")
    lines.append("")
    lines.append(
        "- **Raw** is comparable only within this run. 0.90 on ForgettingQuality "
        "can just mean \"10% noise scenario\" — the baseline scores table shows "
        "what a random adapter got."
    )
    lines.append(
        "- **Normalized** is comparable across any scenarios that share the "
        "same `protocol_hash`. Positive means the adapter extracts signal the "
        "random baseline cannot; negative means worse than random on that "
        "dimension."
    )
    lines.append(
        "- **Family winners** use normalized averages. Within a family, the "
        "winner reflects capability above baseline, not absolute score."
    )
    lines.append(
        "- **Don't aggregate across families.** They measure orthogonal "
        "capabilities — tradeoffs are the point."
    )
    return "\n".join(lines) + "\n"


def _render_table(
    adapters: List[str],
    dims: List[str],
    results: BenchmarkResults,
    use_normalized: bool,
) -> str:
    header = "| adapter | " + " | ".join(dims) + " |"
    sep = "|" + "|".join(["---"] * (len(dims) + 1)) + "|"
    out = [header, sep]
    for adapter in adapters:
        row_map: Dict[str, MetricResult] = {
            r.dimension: r for r in results.per_adapter[adapter]
        }
        cells = []
        for d in dims:
            r = row_map.get(d)
            cells.append(_format_cell(r, use_normalized))
        out.append(f"| {adapter} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def _format_cell(r: Optional[MetricResult], use_normalized: bool) -> str:
    if r is None:
        return "—"
    if not use_normalized:
        return f"{r.score:.3f}"
    if r.normalized_score is None:
        return f"{r.score:.3f}"
    return f"{r.normalized_score:+.3f}"


def _unique_families(results: BenchmarkResults) -> List[str]:
    seen: List[str] = []
    for adapter_results in results.per_adapter.values():
        for r in adapter_results:
            if r.family not in seen:
                seen.append(r.family)
    return seen


def _unique_dimensions(results: BenchmarkResults) -> List[str]:
    seen: List[str] = []
    for adapter_results in results.per_adapter.values():
        for r in adapter_results:
            if r.dimension not in seen:
                seen.append(r.dimension)
    return seen


def _score_for_avg(r: MetricResult, use_normalized: bool) -> float:
    if use_normalized and r.normalized_score is not None:
        return r.normalized_score
    return r.score


def _family_avgs_for(
    results: BenchmarkResults, adapter: str, use_normalized: bool
) -> Dict[str, float]:
    by_family: Dict[str, List[float]] = {}
    for r in results.per_adapter[adapter]:
        by_family.setdefault(r.family, []).append(_score_for_avg(r, use_normalized))
    return {f: sum(vs) / len(vs) for f, vs in by_family.items() if vs}


def _family_winners(results: BenchmarkResults, use_normalized: bool):
    by_family: Dict[str, Dict[str, List[float]]] = {}
    for adapter, adapter_results in results.per_adapter.items():
        for r in adapter_results:
            by_family.setdefault(r.family, {}).setdefault(adapter, []).append(
                _score_for_avg(r, use_normalized)
            )
    for family, adapter_scores in by_family.items():
        avgs = {a: sum(vs) / len(vs) for a, vs in adapter_scores.items() if vs}
        if not avgs:
            continue
        winner = max(avgs.items(), key=lambda kv: kv[1])
        yield family, winner[0], winner[1]


def write_reports(results: BenchmarkResults, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(results.as_dict(), indent=2), encoding="utf-8"
    )
    (out_dir / "results.md").write_text(render_markdown(results), encoding="utf-8")
