# memory-bench runner

Deterministic benchmark runner. Takes a scenario or trace, runs memory adapters,
and reports a scenario-conditioned capability profile rather than a universal
leaderboard.

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

For stable capability profiles across seeds:

```bash
memory-bench run \
  --scenario configs/scenarios/coding-agent.yaml \
  --profile configs/scenario_profiles/coding-agent.yaml \
  --seeds 1,2,3,4,5 \
  --out results/coding-agent-profile/
```

For real trace replay:

```bash
memory-bench replay \
  --trace configs/traces/coding-architecture-update.yaml \
  --profile configs/trace_profiles/coding-agent.yaml \
  --out results/trace-coding-update/
```

## What it measures

8 dimensions across 4 families (2 per family, pulled from MemLife-Bench's validated 13):

| Family | Dimensions |
|---|---|
| Exploration | Novelty Guarantee, Coverage |
| Ranking | Relevance @ K, Frequency Gain |
| Adaptation | Personalization, Cross-Session Learning |
| Maintenance | Update Coherence, Forgetting Quality |

With `--seeds`, each metric is aggregated as mean / standard deviation /
approximate 95% CI. Reports also include scenario fit, stable strengths, and
a Pareto frontier so tradeoffs are visible instead of hidden behind one score.

## Trace replay

Trace YAMLs use a framework-neutral event/query format:

```yaml
trace_version: 1
name: coding-agent-update-drift
domain: coding
events:
  - id: e1
    session: s1
    role: user
    content: "Auth uses JWT refresh tokens."
  - id: e2
    session: s2
    role: user
    content: "Auth now uses opaque session IDs, not JWT."
    supersedes: [e1]
queries:
  - id: q1
    after: e2
    text: "What auth mechanism should I preserve?"
    expected_include: [e2]
    expected_exclude: [e1]
```

This is the import boundary for Hermes, OpenClaw, LLM-wiki, and provider
snapshots; adapters do not need to integrate with those runtimes directly.

## Adapters in v0.3

- **Recency** - pure recency-ordered (null baseline)
- **BM25** - statistical retrieval
- **Embedding-cosine** - `sentence-transformers/all-MiniLM-L6-v2` *(optional extra)*
- **ACT-R** - cognitive activation (base-level + novelty + weak semantic nudge)
- **Composite** - weighted multi-signal (0.5 similarity + 0.3 decay + 0.2 importance)
