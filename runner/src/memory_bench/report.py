"""Markdown + JSON report generator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from memory_bench.runner import BenchmarkResults


def render_markdown(results: BenchmarkResults) -> str:
    lines: List[str] = []
    lines.append(f"# {results.scenario_name}")
    lines.append("")
    lines.append("## Scenario")
    lines.append("")
    meta = results.scenario_meta
    lines.append(f"- pool_size: **{meta['pool_size']}**")
    lines.append(f"- sessions: **{meta['sessions']}** × steps_per_session: **{meta['steps_per_session']}**")
    lines.append(f"- archetypes: {meta['archetypes']}")
    lines.append(f"- context_evolution: {meta['context_evolution']}")
    lines.append(f"- seed: {meta['seed']}")
    lines.append("")

    adapters = list(results.per_adapter.keys())
    dims = _unique_dimensions(results)

    lines.append("## Per-dimension leaderboard")
    lines.append("")
    header = "| adapter | " + " | ".join(dims) + " |"
    sep = "|" + "|".join(["---"] * (len(dims) + 1)) + "|"
    lines.append(header)
    lines.append(sep)

    for adapter in adapters:
        results_map = {r.dimension: r.score for r in results.per_adapter[adapter]}
        row_scores = [results_map.get(d, 0.0) for d in dims]
        cells = " | ".join(f"{s:.3f}" for s in row_scores)
        lines.append(f"| {adapter} | {cells} |")

    lines.append("")
    lines.append("## Per-family averages")
    lines.append("")
    families = _unique_families(results)
    header2 = "| adapter | " + " | ".join(families) + " |"
    sep2 = "|" + "|".join(["---"] * (len(families) + 1)) + "|"
    lines.append(header2)
    lines.append(sep2)
    for adapter in adapters:
        fam_avgs = _family_avgs_for(results, adapter)
        cells = " | ".join(f"{fam_avgs.get(f, 0.0):.3f}" for f in families)
        lines.append(f"| {adapter} | {cells} |")

    lines.append("")
    lines.append("## Family winners")
    lines.append("")
    lines.append("| family | winner | family-avg score |")
    lines.append("|---|---|---|")
    for family, winner, score in _family_winners(results):
        lines.append(f"| {family} | **{winner}** | {score:.3f} |")

    lines.append("")
    lines.append("## Reading guide")
    lines.append("")
    lines.append(
        "Higher is better across all dimensions. The family winner is the adapter "
        "with the highest average score within that family's dimensions — reflecting "
        "overall capability in that area. Don't aggregate across families: they "
        "measure orthogonal capabilities."
    )
    return "\n".join(lines) + "\n"


def _unique_families(results: BenchmarkResults) -> List[str]:
    seen: List[str] = []
    for adapter_results in results.per_adapter.values():
        for r in adapter_results:
            if r.family not in seen:
                seen.append(r.family)
    return seen


def _family_avgs_for(results: BenchmarkResults, adapter: str) -> Dict[str, float]:
    by_family: Dict[str, List[float]] = {}
    for r in results.per_adapter[adapter]:
        by_family.setdefault(r.family, []).append(r.score)
    return {f: sum(vs) / len(vs) for f, vs in by_family.items() if vs}


def _unique_dimensions(results: BenchmarkResults) -> List[str]:
    seen: List[str] = []
    for adapter_results in results.per_adapter.values():
        for r in adapter_results:
            if r.dimension not in seen:
                seen.append(r.dimension)
    return seen


def _family_winners(results: BenchmarkResults):
    by_family: Dict[str, Dict[str, List[float]]] = {}
    for adapter, adapter_results in results.per_adapter.items():
        for r in adapter_results:
            by_family.setdefault(r.family, {}).setdefault(adapter, []).append(r.score)
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
