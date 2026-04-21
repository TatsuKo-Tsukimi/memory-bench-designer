# memory-bench-designer

A Claude Code skill + Python benchmark runner that helps you **design a memory benchmark for your own agent scenario** — instead of forcing your use case into someone else's fixed benchmark.

> There is no best agent memory strategy. Only tradeoffs. Which tradeoffs matter depends on your scenario.

## What this gives you

- A conversational **skill** (`skill/`) that walks you from *"I'm building an agent that remembers X"* to a concrete `scenario.yaml` through multi-turn elicitation.
- A deterministic Python **runner** (`runner/`) that takes your scenario and benchmarks 5 memory strategies (Recency, BM25, ACT-R, Embedding, Composite) across 8 dimensions covering Exploration / Ranking / Adaptation / Maintenance.
- A markdown report telling you, for *your* scenario: which strategy to prototype with, what tradeoffs you're accepting, what to tune.

## Why it exists

Every current memory benchmark (LoCoMo, LongMemEval, MemoryAgentBench, MemoryBench, AgentMemoryBench, FiFA) is a static artifact with a fixed scenario. Adjacent parametric tools (LongMemEval, RULER, Letta Evals) aren't memory-lifecycle-driven. Commercial memory vendors (Mem0, Zep, Letta, Supermemory, Cognee) ship no configuration interface. If your scenario doesn't match theirs, their leaderboard tells you nothing.

This project closes that gap by applying four validated methodologies — Bloom's pipeline (Anthropic), EvalGen's grade-before-criteria, AdaTest's inner/outer loop, CheckList's explicit capability matrix — to the agent-memory domain.

## The thesis, in one table

We ran the same five strategies against three canonical scenarios. Scores 0–1, higher is better. Family-winner shown.

| Family | Game AI<br>(noise + drift) | NPC Cognition<br>(stable) | Coding Agent<br>(mode-shifts) |
|---|---|---|---|
| Exploration | ACT-R **0.54** | ACT-R **0.65** | ACT-R **0.55** |
| Ranking | **BM25 0.70** | Embedding 0.74 | Embedding 0.74 |
| Adaptation | Embedding 0.47 | Embedding 0.49 | Embedding 0.48 |
| Maintenance | Composite 0.65 | Composite 0.69 | Composite 0.68 |

Two things to notice. First, the **Ranking winner flips** between BM25 and Embedding depending on how narrow-band the drift is — picking Embedding universally would cost you performance in tight-vocab game scenarios. Second, the **magnitudes differ by up to 0.20** even when the winner is stable — ACT-R is a much stronger Exploration choice in NPC scenarios than in game-AI. Both signals matter when picking a strategy. Full breakdown and interpretation in `skill/references/adapter-profiles.md`.

## Quickstart

### Runner only

```bash
cd runner
pip install -e ".[embedding]"

memory-bench run \
  --scenario configs/scenario-game-ai.yaml \
  --out results/game-ai/ \
  --embedding --composite
```

Output lands in `results/game-ai/results.md` and `results/game-ai/results.json`.

### Skill (Claude Code)

```bash
cp -r skill ~/.claude/skills/memory-bench-designer
```

Then from Claude Code: *"I want to design a memory benchmark for a <your use case>."* The skill activates and walks you through the 4-stage flow. Three fully-worked examples in `skill/examples/`.

## Repo layout

```
memory-bench-designer/
├── skill/                Claude Code skill (conversational front-end)
│   ├── SKILL.md
│   ├── references/       4×8 taxonomy, adapter profiles, use-case patterns, elicitation rules
│   ├── examples/         3 full walkthroughs (game-AI, NPC, coding)
│   └── templates/        scenario.yaml, weights.yaml skeletons
├── runner/               Python benchmark runner
│   ├── src/memory_bench/
│   │   ├── scenario/     DSL → item pool + context stream
│   │   ├── adapters/     5 memory strategies
│   │   ├── metrics/      8 dimensions across 4 families
│   │   ├── runner.py     orchestrator
│   │   └── report.py     markdown output
│   ├── configs/          3 scenario configs (game-ai, npc-cognition, coding-agent)
│   └── pyproject.toml
├── LICENSE               MIT
├── CONTRIBUTING.md
└── README.md
```

## What the skill does

1. **Understanding** — multi-turn elicitation (3–5 turns). Asks for examples, not criteria (EvalGen).
2. **Ideation** — writes a `scenario.yaml` matched to your description.
3. **Rollout** — runs the benchmark CLI (1–5 min).
4. **Judgment** — interprets the report for *your* use case: winner, tradeoffs, what to prototype with, what to try if production regresses.

Then offers a refinement loop (AdaTest inner/outer).

## What v0.1 deliberately skips

- No LLM-as-judge metrics. Everything is mechanical / deterministic. Reproducible given a seed.
- No benchmarks of commercial services (Mem0, Zep, Letta). Only algorithmic primitives.
- No PyPI release yet. Install via `git clone + pip install -e`.
- No CI/CD, no auto-docs, no web UI.

## Prior art we explicitly borrowed from

| Source | What we took |
|---|---|
| [Bloom](https://anthropic.com) (Anthropic, 2025) | 4-stage pipeline, seed.yaml pattern, Python + variation_dimensions |
| [EvalGen](https://arxiv.org/abs/2404.12272) (Shankar et al., UIST 2024) | Grade-before-criteria, ≤16-example cap, parallel criteria editing |
| [AdaTest](https://arxiv.org/abs/2202.07205) (Microsoft, ACL 2022) | Inner/outer loop, top-N of hundreds, always-visible ranking |
| [CheckList](https://arxiv.org/abs/2005.04118) (Ribeiro et al., ACL 2020) | Explicit 2D capability × test-type matrix |
| MemLife-Bench (prior user work) | Item archetypes, affinity-driven feedback, 4-family structure |

We are not reimplementing any of these — we are applying their patterns to a domain (agent memory lifecycle) that none of them have covered.

## License

MIT. See [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Welcome contributions: more adapters, more metrics, more canonical scenarios, alternative embedding backends.
