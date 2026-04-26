# memory-bench runner

Deterministic benchmark runner. Takes a `scenario.yaml`, generates an item pool + streaming session schedule, runs N adapters, outputs per-dimension scores.

## Install

```bash
cd runner
pip install -e .
# optional: for embedding adapter
pip install -e ".[embedding]"
```

## Run

```bash
memory-bench run --scenario configs/scenarios/game-ai.yaml --out results/game-ai/
```

Use `--protocol configs/protocols/default.yaml` when you want to pin the
evaluation harness explicitly. Without `--protocol`, the CLI uses the same
default protocol values.

## What it measures

8 dimensions across 4 families (2 per family, pulled from MemLife-Bench's validated 13):

| Family | Dimensions |
|---|---|
| Exploration | Novelty Guarantee, Coverage |
| Ranking | Relevance @ K, Frequency Gain |
| Adaptation | Personalization, Cross-Session Learning |
| Maintenance | Update Coherence, Forgetting Quality |

## Adapters in v0.2

- **Recency** - pure recency-ordered (null baseline)
- **BM25** - statistical retrieval
- **Embedding-cosine** - `sentence-transformers/all-MiniLM-L6-v2` *(optional extra)*
- **ACT-R** - cognitive activation (base-level + novelty + weak semantic nudge)
- **Composite** - weighted multi-signal (0.5 similarity + 0.3 decay + 0.2 importance)
