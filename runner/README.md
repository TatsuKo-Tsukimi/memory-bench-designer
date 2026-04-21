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
memory-bench run --scenario configs/scenario-game-ai.yaml --out results/
```

## What it measures

8 dimensions across 4 families (2 per family, pulled from MemLife-Bench's validated 13):

| Family | Dimensions |
|---|---|
| Exploration | Novelty Guarantee, Coverage |
| Ranking | Recency Decay, Frequency Gain |
| Adaptation | Personalization, Cross-Session Learning |
| Maintenance | Update Coherence, Forgetting Quality |

## Adapters in v0.1

- **Recency** — pure recency-ordered (null baseline)
- **BM25** — statistical retrieval
- **Embedding-cosine** — `sentence-transformers/all-MiniLM-L6-v2` *(optional extra)*
- **ACT-R** — cognitive activation (base-level + spreading activation)
- **Composite** — weighted multi-signal (0.5 similarity + 0.3 decay + 0.2 importance)
