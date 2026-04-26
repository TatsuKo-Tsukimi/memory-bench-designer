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
from memory_bench.runner import BenchmarkResults, BenchmarkSuiteResults
from memory_bench.trace import TraceReplayResults


def render_markdown(
    results: BenchmarkResults | BenchmarkSuiteResults | TraceReplayResults,
) -> str:
    if isinstance(results, BenchmarkSuiteResults):
        return render_suite_markdown(results)
    if isinstance(results, TraceReplayResults):
        return render_trace_markdown(results)

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
    lines.append(f"- sessions x steps: **{pm['sessions']} x {pm['steps_per_session']}**")
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

    lines.append("## Per-dimension leaderboard - raw")
    lines.append("")
    lines.append(_render_table(adapters, dims, results, use_normalized=False))
    lines.append("")

    lines.append("## Per-dimension leaderboard - normalized (vs random baseline)")
    lines.append("")
    lines.append(
        "Normalized = (raw - baseline) / (1 - baseline), clamped to [-1, 1]. "
        "0 = matches random, 1 = perfect, negative = worse than random. "
        "When the random baseline is near-saturation (common on coverage, where "
        "a uniform adapter touches ~97% of the pool), adapters that concentrate "
        "clamp to -1: an honest signal that they explore narrower than random."
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
        lines.append(f"| {d} | {b:.3f} |" if b is not None else f"| {d} | - |")
    lines.append("")

    lines.append("## Per-family averages - normalized")
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
        "can just mean \"10% noise scenario\"; the baseline scores table shows "
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
        "capabilities; tradeoffs are the point."
    )
    return "\n".join(lines) + "\n"


def render_trace_markdown(results: TraceReplayResults) -> str:
    lines: List[str] = []
    lines.append(f"# {results.trace_name}")
    lines.append("")
    lines.append(
        "> Trace replay evaluates real agent-memory behavior as a scenario-conditioned "
        "capability profile."
    )
    lines.append("")
    lines.append("## Trace identity")
    lines.append("")
    lines.append(f"- domain: **{results.domain}**")
    lines.append(f"- profile: **{results.profile_meta['name']}**")
    lines.append("")

    lines.append("## Capability profile")
    lines.append("")
    lines.append(_render_stat_table(results.per_adapter))
    lines.append("")

    lines.append("## Scenario fit")
    lines.append("")
    lines.append("| adapter | fit mean |")
    lines.append("|---|---:|")
    for adapter, stat in _sorted_stats(results.scenario_fit):
        lines.append(f"| {adapter} | {stat.mean:+.3f} |")
    lines.append("")

    lines.append("## Pareto frontier")
    lines.append("")
    lines.append("- frontier: " + ", ".join(f"`{a}`" for a in results.pareto_frontier))
    if results.dominated:
        for adapter, dominator in results.dominated.items():
            lines.append(f"- dominated: `{adapter}` is dominated by `{dominator}`")
    else:
        lines.append("- dominated: none under the active profile axes")
    lines.append("")

    lines.append("## Query checks")
    lines.append("")
    for adapter, rows in results.query_results.items():
        lines.append(f"### {adapter}")
        lines.append("")
        lines.append("| query | retrieved |")
        lines.append("|---|---|")
        for row in rows:
            lines.append(f"| {row['query_id']} | {', '.join(row['retrieved']) or '-'} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def render_suite_markdown(results: BenchmarkSuiteResults) -> str:
    lines: List[str] = []
    lines.append(f"# {results.scenario_name}")
    lines.append("")
    lines.append(
        "> Memory systems are evaluated as scenario-conditioned capability "
        "profiles, not as a single universal leaderboard."
    )
    lines.append("")

    lines.append("## Run identity")
    lines.append("")
    lines.append(f"- protocol_template_hash: `{results.protocol_template_hash}`")
    lines.append(f"- scenario_hash: `{results.scenario_hash}`")
    lines.append(f"- profile_hash: `{results.profile_hash}`")
    lines.append(f"- seeds: {', '.join(str(s) for s in results.seeds)}")
    lines.append("")

    lines.append("## Scenario profile")
    lines.append("")
    lines.append(f"- name: **{results.profile_meta['name']}**")
    priorities = results.profile_meta.get("priorities", {})
    if results.profile_meta["name"] == "unconditioned":
        lines.append("- mode: unconditioned profile (all capability axes equally weighted)")
    else:
        lines.append(f"- priorities: {priorities}")
    lines.append("")

    lines.append("## Capability profile")
    lines.append("")
    lines.append(
        "Scores are normalized-vs-random means across seeds. `+/-` is an "
        "approximate 95% confidence interval."
    )
    lines.append("")
    lines.append(_render_stat_table(results.family_stats))
    lines.append("")

    lines.append("## Scenario fit")
    lines.append("")
    lines.append(
        "Scenario fit is a weighted capability score for this profile. It is a "
        "conditional recommendation signal, not a global ranking."
    )
    lines.append("")
    lines.append("| adapter | fit mean | 95% CI |")
    lines.append("|---|---:|---:|")
    for adapter, stat in _sorted_stats(results.scenario_fit):
        lines.append(f"| {adapter} | {stat.mean:+.3f} | +/- {stat.ci95:.3f} |")
    lines.append("")

    lines.append("## Pareto frontier")
    lines.append("")
    if results.pareto_frontier:
        lines.append("- frontier: " + ", ".join(f"`{a}`" for a in results.pareto_frontier))
    else:
        lines.append("- frontier: none")
    if results.dominated:
        for adapter, dominator in results.dominated.items():
            lines.append(f"- dominated: `{adapter}` is dominated by `{dominator}`")
    else:
        lines.append("- dominated: none under the active high-priority axes")
    lines.append("")

    lines.append("## Stable strengths")
    lines.append("")
    lines.append("| capability axis | most stable winner | win rate |")
    lines.append("|---|---|---:|")
    for axis, stability in results.winner_stability.items():
        rates = stability["winner_rates"]
        winner, rate = max(rates.items(), key=lambda kv: kv[1])
        lines.append(f"| {axis} | {winner} | {rate:.2f} |")
    lines.append("")

    lines.append("## Per-dimension detail")
    lines.append("")
    lines.append(_render_stat_table(results.metric_stats))
    lines.append("")

    lines.append("## Reading guide")
    lines.append("")
    lines.append(
        "- **Capability profile** shows where each adapter is strong or weak across seeds."
    )
    lines.append(
        "- **Scenario fit** applies this scenario's priorities; it should not be read as a universal winner."
    )
    lines.append(
        "- **Pareto frontier** keeps adapters with meaningful tradeoffs and marks only clearly dominated choices."
    )
    return "\n".join(lines) + "\n"


def _render_stat_table(adapter_stats) -> str:
    axes: List[str] = []
    for stats in adapter_stats.values():
        for axis in stats:
            if axis not in axes:
                axes.append(axis)
    header = "| adapter | " + " | ".join(axes) + " |"
    sep = "|" + "|".join(["---"] * (len(axes) + 1)) + "|"
    out = [header, sep]
    for adapter in adapter_stats:
        cells = []
        for axis in axes:
            stat = adapter_stats[adapter].get(axis)
            if stat is None:
                cells.append("-")
            else:
                cells.append(f"{stat.mean:+.3f} +/- {stat.ci95:.3f}")
        out.append(f"| {adapter} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def _sorted_stats(stats):
    return sorted(stats.items(), key=lambda kv: kv[1].mean, reverse=True)


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
        return "-"
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


def write_reports(
    results: BenchmarkResults | BenchmarkSuiteResults | TraceReplayResults,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(results.as_dict(), indent=2), encoding="utf-8"
    )
    (out_dir / "results.md").write_text(render_markdown(results), encoding="utf-8")
