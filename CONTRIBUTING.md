# Contributing to memory-bench-designer

Thanks for the interest. Here's how to land a change.

## Scope of welcome contributions

- **More adapters.** Any memory strategy with a clean interface: contextual bandits, graph-walk retrieval, hybrid retrievers, vendor wrappers. See `runner/src/memory_bench/adapters/base.py` for the interface.
- **More metrics.** Dimensions that extend or refine the 4-family taxonomy. Mechanical preferred; if LLM-judged, must be clearly marked and have a mechanical fallback.
- **More canonical scenarios.** Add a `configs/scenario-<name>.yaml` and a matching `skill/examples/<name>-walkthrough.md`. Should represent a genuinely distinct use case; if the capability profile doesn't meaningfully differ from an existing canonical pattern, the scenario isn't pulling its weight.
- **Alternative embedding backends.** Currently only sentence-transformers/all-MiniLM-L6-v2. Adding BAAI/bge-*, OpenAI embeddings, etc. as switchable backends would be useful.
- **Skill improvements.** Better elicitation rules, tighter interpretation templates, shorter default flows.

## Scope of things we'll likely close

- **Vendor-specific benchmarks.** We benchmark algorithmic primitives, not specific services (Mem0, Zep, Letta). If you want Zep-specific numbers, wrap Zep as an adapter; don't add Zep-specific metrics.
- **LLM-judged-only dimensions.** Everything mechanical-first. If a dimension *requires* an LLM to score it, it's a v0.3+ conversation.
- **Web UI.** Claude Code renders markdown natively. A web dashboard is out of scope.

## Adding an adapter

```python
# runner/src/memory_bench/adapters/<your_adapter>.py
from .base import Adapter, Retrieval
from memory_bench.scenario.context import Query
from memory_bench.scenario.item_pool import Item

class YourAdapter(Adapter):
    def __init__(self) -> None:
        super().__init__("your_adapter")

    def _on_observe(self, item: Item, global_step: int) -> None:
        # called when a new item enters the pool
        ...

    def retrieve(self, query: Query, global_step: int) -> Retrieval:
        # return a ranked Retrieval object
        ...

    def _on_feedback(self, retrieval: Retrieval, hits: list[bool]) -> None:
        # optional: called after retrieval with hit/miss feedback
        ...
```

Wire it into `runner/src/memory_bench/cli.py` behind a flag. Add a profile row to `skill/references/adapter-profiles.md` after running it against all three canonical scenarios.

## Adding a scenario

1. `runner/configs/scenarios/<name>.yaml` - the DSL file
2. `skill/examples/<name>-walkthrough.md` - a worked conversation demonstrating how a user would arrive at this scenario
3. Run it against all 5 canonical adapters; record the family winners and magnitudes
4. Update the README's "thesis, in one table" with a column if the new scenario surfaces a distinct capability profile

## Code style

- Python 3.11+
- `from __future__ import annotations` at the top of every runner module
- Dataclasses over dicts for structured data
- Docstrings only when the WHY is non-obvious; skip them for obvious type-annotated functions
- Pure Python in hot paths when possible; numpy only in the embedding adapter

## Testing

Run the unit tests, then verify all three canonical scenarios and compare against `skill/references/adapter-profiles.md`:

```bash
cd runner
pytest

for s in game-ai npc-cognition coding-agent; do
  memory-bench run --scenario configs/scenarios/$s.yaml \
    --out results/$s/ --embedding --composite
done
```

If your change causes the family winners in `skill/references/adapter-profiles.md` to shift, that's a red flag worth discussing in your PR unless the shift is the explicit goal of the change.

## PR checklist

- [ ] Change runs end-to-end against all three canonical scenarios
- [ ] README updated if the change affects the thesis table
- [ ] `skill/references/adapter-profiles.md` updated if the change affects per-adapter scores
- [ ] New scenarios include a matching walkthrough
- [ ] No new non-optional dependencies (keep optional-extras for heavy deps like sentence-transformers)

## License

By contributing, you agree your contributions are licensed under the same MIT license.
