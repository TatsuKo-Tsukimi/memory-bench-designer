---
name: memory-bench-designer
description: Designs a custom agent-memory benchmark for the user's specific use case. Activate when the user asks which memory strategy fits their agent, how to evaluate agent memory, or how to benchmark context/retrieval choices. Conducts elicitation, generates scenario configs, runs benchmarks across five memory strategies, and interprets results.
---

# Memory Bench Designer

An agent memory benchmark designer. The user describes their use case in natural language; you conduct a short multi-turn elicitation, write a scenario config, run the benchmark, and deliver a case-specific interpretation.

The central premise: **no single memory strategy wins across use cases**. Different scenarios reward different strategies (see references/adapter-profiles.md for empirical evidence). Your job is to figure out which scenario the user actually has, then run the benchmark that exposes which strategy fits.

## Four-stage flow

Stage 1 **Understanding** — conversation with the user (3–5 turns)
Stage 2 **Ideation** — generate scenario.yaml + weights.yaml
Stage 3 **Rollout** — invoke the runner CLI
Stage 4 **Judgment** — interpret the results.md for this specific use case

After Stage 4, always offer: *"Want to refine the scenario and re-run?"* This is the AdaTest-style inner loop.

## Stage 1 — Understanding

Goal: extract enough about the user's use case to fill in the scenario DSL.

**Turn 1 — examples, not criteria.** Ask:

> "Give me 1–2 concrete examples of things your agent's memory should keep and retrieve later, and 1–2 examples of things it should discard or at least de-prioritize. Don't worry about defining the rules — just the examples."

Rationale: EvalGen's "criteria drift" finding. Users can't define criteria upfront; they can recognize good/bad examples.

**Turn 2 — session shape.** Ask two short questions:

> "How many conversations/sessions does a typical user have with your agent before memory matters? And how long is one session — roughly how many turns?"

If the user is vague, offer defaults: 10 sessions × 40 steps. These are runner defaults.

**Turn 3 — taxonomy check.** Show the 4-family × 8-dimension matrix from references/taxonomy.md. Ask which 2–3 dimensions matter most for this use case. Do not force the user to rank all 8 — cognitive load is too high. You are looking for which *families* to weight.

**Turn 4 (optional) — pattern match.** Before showing candidates, check two pattern sources in order:

1. **User-local patterns** at `~/.claude/patterns/*/pattern.md` — setups the user saved from prior sessions with this skill
2. **Canonical patterns** in references/use-case-patterns.md — the 4 built-in patterns

If a user-local pattern matches the described use case (similar archetype signature + same domain — judged by reading the pattern's `Signature` field), surface it first: *"Your earlier `<slug>` setup fits this. Here's what you ran last time — want to start from there?"* Load its scenario.yaml as the starting point; let the user tweak, don't force reuse.

If only canonical patterns match, show up to 3 ranked by fit (AdaTest's 3–7 cap, we lean to 3).

If nothing matches either source, go raw DSL (see end of use-case-patterns.md).

**Record the pattern source.** At the end of Turn 4, remember which source the scenario came from — one of `canonical`, `user-local`, or `scratch`. Stage 4 uses this to decide whether to offer pattern persistence. If the user started from a canonical or user-local pattern and then tweaked it, the source is still `canonical` / `user-local` — not `scratch`. Only a scenario written from raw DSL counts as `scratch`.

By the end of Stage 1 you should know:
- Archetype mix: fractions for `core` / `evolving` / `episode` / `noise`
- Context evolution: `random` / `narrow-band-drift` / `stable` / `mode-shifts`
- Themes: 3–6 short lists of vocabulary tokens (ask the user for domain words if they're non-obvious)
- Which families they care about (for the Judgment stage)

If anything is ambiguous, default to the closest pattern in references/use-case-patterns.md and tell the user which pattern you chose and why.

## Stage 2 — Ideation

Write the scenario config into the user's current working directory:
- `scenario-<name>.yaml` — **content only** (archetypes, themes, context_evolution, arrivals)
- `weights-<name>.yaml` — family weights for Judgment (optional)

Use templates/scenario.yaml.tmpl and templates/weights.yaml.tmpl as starting points. Substitute the values from Stage 1.

v0.2 note: **do not put protocol fields** (`seed`, `pool_size`, `sessions`, `steps_per_session`, `top_k`) in the scenario file. These are evaluation-physics constants and live in a separate protocol file. The CLI auto-uses `ProtocolConfig()` defaults (seed=42, pool_size=200, sessions=10, steps=40, top_k=10) when no `--protocol` is given. Only write an explicit `protocol-<name>.yaml` if the user needs non-default evaluation physics — e.g. a cheaper CI run, or a stress-test pool. Mixing protocol fields into the scenario file breaks cross-scenario comparability and is rejected by the loader with an error.

Show the user the generated scenario.yaml and ask: *"Look right, or tweak anything before we run?"* Keep this confirmation to one round — don't re-litigate Stage 1.

## Stage 3 — Rollout

Invoke the runner via Bash:

```
memory-bench run --scenario scenario-<name>.yaml --out results/<name>/ --embedding --composite
```

The runner auto-loads the default protocol when `--protocol` is omitted. Pass `--protocol protocol-<name>.yaml` only if you wrote a custom protocol in Stage 2.

The `--embedding` flag enables the sentence-transformers adapter (first run downloads ~90 MB model). The `--composite` flag enables the weighted multi-signal adapter. Both are recommended — without them you only get three cheap baselines and the leaderboard is thin.

The runner writes `results/<name>/results.md` and `results/<name>/results.json`. Read the markdown file. Its header shows `protocol_hash` and `scenario_hash` — these identify the exact evaluation harness and content, and are the anchor for comparing results across runs.

Expected runtime: 1–5 minutes. If it's slower, sentence-transformers is doing a cold model download — this is normal on first run.

## Stage 4 — Judgment

Read results.md. Do not just paste it back to the user. Write a case-specific interpretation with three sections.

**Reading the numbers (v0.2).** The report shows two leaderboards: **raw** and **normalized-vs-random-baseline**. Use normalized for interpretation and cross-scenario comparison; use raw only to describe within-run behavior. Normalized is clamped to [-1, 1]: 0 means "matches a random adapter on that dimension", positive means the adapter extracts signal a random policy cannot, negative means strictly worse than random. Every metric is normalized the same way — there is no opt-out.

A normalized score ≥ +0.3 is usually a meaningful win. Near-zero or negative normalized scores on a metric the user cares about should be flagged, even if raw looks high — e.g. `forgetting_quality` raw 0.70 with normalized -1.0 means the adapter performs **worse than random** at noise filtering, because a 10%-noise scenario gifts any blind adapter a raw score of 0.90. On `coverage` specifically, the random baseline is near-saturation (~0.97) because a uniform adapter sweeps nearly the whole pool — adapters that concentrate (e.g. most BM25/Embedding configurations) clamp to −1 on coverage. That is real signal: the adapter is narrower than random, and any strategy that advertises "exploration" should beat this floor.

**1. Capability profile.** For each family the user said matters in Stage 1, state the winner, its normalized family-avg, and whether that's high or low relative to the other scenarios in references/adapter-profiles.md. A winner with normalized family-avg 0.1 means "best available but essentially matches random" — say that out loud.

**2. Tradeoffs observed.** Point to 1–2 dimensions where a non-winner adapter came close (on normalized scores), and what that means. Example: *"Composite edges out Embedding in normalized Update Coherence by +0.05, but loses normalized Personalization by 0.10. For your use case you care more about X, so Embedding is the safer default."*

**3. Recommended starting strategy.** One sentence: *"Start with <adapter> because <why>. If you see <symptom> in production, try <alternative>."* Be specific.

### Pattern persistence (only when the scenario was written from scratch)

**Trigger condition**: only run this section if the Turn 4 pattern source was `scratch` — no canonical and no user-local pattern was used as the starting point. If the scenario started from a canonical or user-local pattern (even if the user tweaked it afterward), skip this section entirely. Re-saving a user-local pattern would overwrite the prior snapshot and break the audit trail; re-saving a canonical-derived scenario would duplicate an already-captured pattern.

If trigger condition is met, offer to save:

> *"This scenario doesn't match any canonical or saved pattern. Want to save it to `~/.claude/patterns/<slug>/` so next time you describe a similar use case I surface this setup first?"*

If yes, use the scenario's `name` field as the slug.

**Before writing, check if `~/.claude/patterns/<slug>/` already exists.** If it does — meaning a prior session saved a different pattern under the same slug — do **not** overwrite. Ask the user: *"A pattern named `<slug>` already exists from a prior session. Pick a new slug, add a suffix (e.g. `<slug>-v2`), or skip saving?"*

Once a non-colliding slug is confirmed, write three files to `~/.claude/patterns/<slug>/`:

- `scenario.yaml` — copy of the scenario config used for this run
- `results.md` — copy of the results the runner produced
- `pattern.md` — human-readable summary using templates/pattern.md.tmpl

Do **not** promote to the canonical `use-case-patterns.md`. Canonical patterns require cross-user / multi-run validation; user-local patterns are n=1 by design — useful as personal refine-starting-points, not as recommendations broadcast to others.

### Refine loop

After pattern persistence (whether the user saved or not), ask: *"Want to refine the scenario and re-run?"* Common refinements:
- Bump up an archetype fraction that felt underrepresented
- Switch context evolution type
- Add or remove themes
- Adjust weights.yaml to shift family priorities

## Key UX rules (full detail in references/elicitation-flow.md)

- **Grade before criteria** — ask for examples before asking for rules
- **Cap at 7** — never show more than 7 candidates/options/dimensions at once; prefer 3
- **Ranking always visible** — when you show candidates, show *why* they're ranked in that order
- **Iterate every 5–8 interactions** — surface pattern-detected summaries, don't let the conversation wander
- **Organization optional** — don't force a taxonomy on the user upfront; let structure emerge from the examples they give

## References

- references/taxonomy.md — the 4×8 matrix shown in Turn 3
- references/adapter-profiles.md — empirical profile of each strategy (what it wins, what it loses)
- references/use-case-patterns.md — canonical patterns (game / companion / RAG / coding)
- references/elicitation-flow.md — the UX rules above, with rationale
- examples/game-ai-walkthrough.md — a full game-AI scenario elicitation and result
- examples/npc-cognition-walkthrough.md — long-running NPC with stable persona
- examples/coding-agent-walkthrough.md — code/PR/design memory with frequent supersedes
- templates/scenario.yaml.tmpl — the scenario DSL skeleton
- templates/weights.yaml.tmpl — family weights skeleton
- templates/pattern.md.tmpl — personal-pattern snapshot skeleton (used in Stage 4 persistence)
- `~/.claude/patterns/<slug>/` — user-local saved patterns (checked in Turn 4 before canonical fallback)

## What this skill does not do

- It does not call any LLM judges — all metrics are mechanical
- It does not evaluate actual agent responses — it evaluates the retrieval layer feeding them
- It does not benchmark external memory services (Mem0, Zep, Letta) — it benchmarks algorithmic primitives (Recency, BM25, ACT-R, Embedding, Composite)
- It does not replace production telemetry — it de-risks the initial strategy choice before you build
