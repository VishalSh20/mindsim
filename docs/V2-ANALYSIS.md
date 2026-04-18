# V2 Proposal Analysis

An assessment of the v2 suggestion docs (`ARCHITECTURE-v2-suggestions.md`, `AGENTIC-ARCHITECTURE-v2-suggestions.md`, `MECHANICS-v2-suggestions.md`, `ISSUES-v2.md`) against the current v1 codebase. Purpose: separate the load-bearing ideas from the marketing, and surface the contradictions and ambiguities that will bite if v2 is implemented literally.

---

## 1. What v1 Actually Is (Ground Truth)

Before judging v2 fixes, the v1 baseline — verified against code, not docs:

| Claim in v2 docs | Status in v1 code | Evidence |
|---|---|---|
| "Forces are summed linearly" | True | `forces.py:345-349` — `total_utility += force_arr` over all 7 forces |
| "Probability weighting is missing" | True | No `gamma`, no weighting function anywhere in `engine/` |
| "Agents are stateless" | True | `AGENT_DTYPE` in `population.py:20-42` has no phase / tenure / awareness_strength / history |
| "Simulation is single-shot" | True | `simulate.py:28-147` has no round loop |
| "Social proof is global" | True | `forces.py:66,278` uses scalar `product_adoption_rate` |
| "Reference price is one scalar" | **Partly false** | Per-agent ref price *is* implemented via `by_archetype` + `competitor_awareness_frac` blend (`forces.py:72-89`, `population.py:230`) |
| "`by_archetype` calibration unused" | **Now false** | Commit 8d07602 added archetype overrides to the prompt (`prompts.py:126-161`) |
| "`price_sensitivity` / `novelty_weight` are dead" | **Now false** | Both are read in `forces.py:171,198` |
| "Category maturity unused in forces" | True | Zero matches for `maturity` in `engine/` |
| "Events regenerate population" | True | `events.py:84-88` calls `simulate()` from scratch |
| "Interventions have no cost/time" | True | `InterventionResult` has `lift_pp`, `mechanism`, no cost or timeline |

**Implication:** ISSUES-v2.md is mostly accurate as a snapshot, but Gaps 14, 15, and 19 are already partially closed by recent commits. v2's narrative of "v1 is impoverished across the board" is now a little stale — roughly 20% of the gap list has moved.

---

## 2. The Core Proposition, Stripped Down

Across the three v2 docs, four ideas do the real work. Everything else is scaffolding:

1. **Product → feature matrix.** Instead of `perceived_benefit=0.65`, a product is 4-8 features, each with `{score, certainty, visibility, time_to_realize, category}`. Competitors get the same structure.
2. **Agent state machine.** Six phases (`unaware → aware → considering → trialing → adopted → locked_in | churned`) plus memory (`tenure`, `investment_depth`, `trial_history`). Drives a multi-round simulation.
3. **Local social proof via clusters.** 10-20 network clusters replace the global `adoption_rate` scalar. Social proof is computed per cluster.
4. **Structured force interactions.** Anchoring feeds *into* prospect value; loss × discounting multiply for delayed-benefit products; everything else adds.

Ideas 1 and 2 are the big unlocks — 10 of the 19 gaps collapse once those are in place. Ideas 3 and 4 are refinements that only become meaningful after 1 and 2 ship.

---

## 3. Strong Points

### 3.1 The "boundaries not core" principle survives

The temptation in an agentic redesign is to put LLM calls in the hot path. v2 explicitly refuses: Phase 2 stays pure NumPy, and the justification (Chen et al. on counterfactuals, non-reproducibility, can't differentiate for sensitivity analysis) is correct. This is the single most important architectural decision in the doc and it's defended well.

### 3.2 Per-agent reference price from competitor tiers (MECHANICS-v2)

The three-tier competitor knowledge model (primary / secondary / unaware) is the best-engineered idea in the v2 bundle. It solves Gap 19 (decorative reference price components), Gap 2 (ghost competitors), and partially Gap 8 (situational status quo) with a single mechanism, and it *derives* rather than stipulates the agent-specific reference price:

```
agent_ref_price = primary.price × 0.6 + mean(secondary.price) × 0.3 + category_default × 0.1
```

An innovator who knows 4 tools gets a calibrated reference; a laggard who knows only the free incumbent treats any paid product as expensive. This emerges from the knowledge structure, which is exactly the right kind of mechanism.

### 3.3 Triangulation rule for pros/cons (AGENTIC-v2, Agent 11)

Requiring both simulation evidence *and* VoC evidence before a claim counts as a pro/con is a real epistemic discipline. Single-source claims go to a "nuances" bucket. This is much harder to hallucinate than a bare LLM summary and is the one place in v2 where the agentic scaffolding genuinely buys something the math-only pipeline can't do.

### 3.4 Emergence as validation (MECHANICS-v2, closing section)

"If you get Rogers' adoption curve without coding Rogers' adoption curve, your behavioral parameters are calibrated correctly." This is the right validation criterion for a multi-round behavioral model — and it's falsifiable, which matters. The v1 pipeline has no equivalent test.

### 3.5 Google Trends as objective grounding

Converting `category_maturity` from an LLM guess into a function of Trends interest-over-time is a genuine improvement. Same for FOMO intensity from search-volume delta. These turn two of the most slippery parameters into something that a second analyst could reproduce.

---

## 4. Inner Contradictions

### 4.1 Two incompatible fixes for Gap 7 (anchoring double-count)

- **ARCHITECTURE-v2 Layer 5** says: anchoring modifies the *input* to prospect value — `effective_price = price × (1 − anchoring_effect)`, then prospect runs on the adjusted price.
- **MECHANICS-v2** says: reference price is derived per-agent from their competitor knowledge tiers, and agents compare against their own reference.

These are not the same mechanism. The first preserves a separate "anchoring" concept that modifies a single global price. The second dissolves anchoring into the structure of the prospect computation itself — there is no longer a distinct "anchor force" because every agent's reference is already personal. Both can't be true simultaneously. The second is cleaner; the first should be dropped if MECHANICS-v2 is adopted.

### 4.2 Feature category `cost` breaks prospect theory

Layer 1 assigns every feature a `category ∈ {core_value, cost, switching_cost_reducer, social_signal}` and then proposes:

```
agent_gain = sum(feature_score × agent_weight × certainty)
```

If `credit_economics` has `category: cost` and `score: 0.4`, summing it into gain collapses the K&T gain/loss asymmetry. The whole point of `λ ≈ 2.25` is that losses hit ~2.25× harder than gains of the same magnitude; once cost features are absorbed into a unified utility sum, that asymmetry has to be reconstructed somewhere else, and v2 never says where. Either `cost` features must be routed to a separate loss aggregator, or the feature matrix needs explicit gain-side and loss-side vectors. The docs don't resolve this.

### 4.3 "LLMs interpret, math decides" vs. Agent 7

Agent 7 (Per-Archetype Reaction, AGENTIC-v2) emits `feature_weights{dim: weight}` — the *exact vector that drives the prospect computation in Phase 2*. That is not interpretation; it is the decision coefficient. The principle at the top of AGENTIC-v2 ("agents at the boundaries, math in the core") is violated the moment feature weights come from LLM role-play rather than from Rogers archetype distributions or a principled mapping. The existing `archetypes.yaml` distribution approach is a stronger design; Agent 7 should either be demoted to a sanity-check or its output should be merged into the YAML-driven distributions with an explicit weighting rule.

### 4.4 Clusters come from… where?

Layer 4 proposes 10-20 clusters with cross-links. Layer 6 Call 3 asks the LLM "what clusters matter?". AGENTIC-v2 has no dedicated cluster-discovery agent and no tool for it. Clusters are not in research output, not in Trends data, not in the VoC stack. The result: clusters will be whatever the calibrator LLM imagines them to be, which defeats the point of grounding social structure in data. Either add a cluster-discovery agent (Reddit subreddit graph, org-size segmentation, geography from Trends) or reduce Layer 4 from "network topology" to "segmentation" and be honest about it.

### 4.5 Sensitivity analysis as specified no longer works

Current sensitivity perturbs each parameter ±30% independently, on a single-shot simulation. In a multi-round state-machine model:

- Timing parameters (decay rates, consideration thresholds, trial duration) have **path-dependent** effects that ±30% won't surface linearly.
- Parameter *interactions* dominate (Gap 17 in v2's own list), and a local method can't capture them.
- Running sensitivity at 10 rounds × 1000 agents × 13 params × 2 perturbations is no longer negligible cost.

v2 claims Gap 17 is addressed by "Phase 2 sensitivity + Agent 13 interaction testing" but doesn't specify the method. This is a non-trivial piece of math (global sensitivity? Sobol indices? Morris screening?) that's hand-waved.

### 4.6 Events accumulation (Gap 13) isn't actually solved

v2 says Gap 13 is addressed by "Phase 2 state persistence". But agent state lives in-process during a single simulation run. For events to accumulate across CLI invocations (the actual user pain — "competitor launches" then later "competitor gets bad press"), state has to be serialized and reloaded. Neither ARCHITECTURE-v2 nor AGENTIC-v2 describes persistence between runs. This is a solvable gap but the current wording pretends it's solved by multi-round mechanics, which it isn't.

---

## 5. Ambiguities

### 5.1 `time_to_realize: "immediate" | "delayed"` vs continuous `time_to_value`

ARCHITECTURE-v2 Layer 1 shows `time_to_realize` as a categorical string. v1's hyperbolic discounting uses a continuous `time_to_value` in years. ARCHITECTURE-v2 Layer 5 uses a threshold: `if time_to_value > threshold: multiply, else: add`. Three different treatments in the same doc. The binary categorical is probably intended as LLM-facing shorthand that gets mapped to a continuous value, but this is never stated.

### 5.2 How `certainty` plugs into probability weighting

ISSUES-v2 Gap 6 says Tversky–Kahneman's w(p) = p^γ / (p^γ + (1-p)^γ)^(1/γ) with γ=0.61 is unimplemented. ARCHITECTURE-v2 says Gap 6 is solved "in Layer 5 (add to prospect computation)" — no formula. The LLM-facing parameter closest to a probability is `benefit_certainty` (or per-feature `certainty`). Is certainty treated as the probability p of receiving the feature's score? That would be defensible but has never been stated, and it's not the same thing as the probability weighting in the original K&T paper (which is about lotteries with explicit probabilities, not subjective confidence in a scalar benefit). This needs to be nailed down.

### 5.3 Feature-matrix competitor modeling + heterogeneous weights → incomparable utilities?

If two agents have different `feature_weights`, their `agent_gain` scales are not on the same unit. That's fine for within-agent decisions (each agent has its own logistic). But intervention ranking, sensitivity aggregation, and segment comparison all rely on comparing utilities across agents. Needs either normalization (which loses information) or a shift to ranking-based aggregation. Not addressed.

### 5.4 "Still vectorizable" for clusters + state machine + per-feature forces

The "still pure NumPy" claim is technically true but understated. Per-feature forces change `forces` from shape `(N,)` to `(N, F)`. Cluster-local adoption requires a gather over cluster membership. State transitions require masked updates per phase. The 5-10× slowdown estimate is plausible only if someone writes the vectorized state-machine carefully; a naive implementation could easily be 50-100× slower and drag simulation into multi-second territory where it currently claims <100ms.

### 5.5 Validator (Agent 9) conflicts with novel-category calibration

The Validator enforces parameter bounds "hard-coded from RESEARCH.md". Fine for well-studied categories. But the simulator's own pitch is that it handles products in new categories where published λ and γ values may not apply. Hard bounds + novel category = Validator failures + re-prompt loop that silently steers the LLM back into the research-established range. This is a feature ↔ reliability trade-off that's never acknowledged.

### 5.6 Cost and rate-limit realism of the tool stack

AGENTIC-v2 budgets `~$0.08 per run` on gpt-4o-mini with "~20 tool calls" including G2/Capterra/Reddit/HN/App Store scrapes. G2 and Capterra are actively hostile to scraping (Cloudflare + aggressive detection). Reddit API has rate limits and auth requirements post-2023. gpt-4o-mini handling 14 structured-schema calls with strict validation reliably is itself questionable — Validator re-runs will push the call count higher. The $0.08 figure is probably 3-5× low in practice, and the scraping stack is the single biggest operational risk not discussed.

### 5.7 VoC sample bias

Agent 5 grounds pros/cons in Reddit, HN, G2, App Store reviews. That is a sample heavily weighted toward Innovators and Early Adopters. Late Majority and Laggards — the 50% of the population that's hardest to simulate and most consequential for "crossing the chasm" — don't post reviews and aren't on HN. The triangulation rule in §3.3 inherits this bias: claims that only Laggards would hold will systematically fail triangulation and end up in "nuances." The doc presents VoC as ground truth when it's a skewed sample.

---

## 6. What Survives If You Strip Aggressively

If budget is limited and you cut v2 to its minimum viable form, the ordering should be:

1. **Feature matrix + per-agent feature weights** (ARCHITECTURE-v2 Layers 1 + 2). Unlocks 5 gaps. Biggest ROI per unit work.
2. **Force restructuring: anchoring-into-prospect + loss×discount multiplicative** (Layer 5). Pure formula change, no new state. Fixes Gaps 5, 7 at ~1 day of work.
3. **Tiered competitor knowledge + per-agent reference price** (MECHANICS-v2). Plugs into (1) and replaces the anchoring force entirely — adopt this instead of Layer 5's anchoring fix.
4. **Multi-round state machine** (Layer 3 + MECHANICS-v2 phase transitions). Expensive. Fixes Gaps 3, 8, 9, 13. Don't start until 1-3 are stable.
5. **Cluster topology** (Layer 4). Only after (4). Requires a cluster-discovery mechanism that v2 doesn't currently specify.
6. **Agentic decomposition** (AGENTIC-v2's 14 agents). The *most speculative* piece and the one most likely to produce hallucinated structure. Deliver the math layer first; retrofit agents only where they demonstrably beat a single well-prompted calibration call.

The triangulated pros/cons work (Agent 11) is orthogonal to the math redesign and can ship independently of any of the above.

---

## 7. Bottom Line

**v2 is directionally right and locally under-specified.** The four load-bearing ideas (feature matrix, state machine, local social proof, structured force interactions) are correct responses to real v1 gaps. The per-agent reference price derivation in MECHANICS-v2 is the strongest single piece of design. The triangulated pros/cons is the strongest single piece of agentic scaffolding.

But the three docs are not internally consistent: Gap 7 has two incompatible fixes; the feature `cost` category silently breaks prospect theory's gain/loss asymmetry; Agent 7 violates the doc's own "LLMs don't decide" principle; clusters appear without a discovery mechanism; sensitivity analysis needs redesigning and isn't; event persistence is declared solved when it isn't; tool-stack cost and scraping realism are optimistic.

A v2 implementation that takes the docs literally will ship a system whose math is cleaner but whose agentic layer is harder to defend than the current single-prompt calibrator. The right sequencing is math first (items 1-5 in §6), agents last — and the agentic layer should earn each of its 14 agents by demonstrating it beats a simpler alternative, not by being listed in a data-flow diagram.
