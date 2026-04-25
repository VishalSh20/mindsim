# ARCHITECTURE-v2: Proposed Redesign

Redesign proposal addressing the 19 gaps identified in ISSUES-v2.md. Core philosophy unchanged: **LLMs interpret language, math computes decisions.** The math just operates on vectors instead of scalars, over time instead of a snapshot, and through networks instead of global averages.

---

## The Core Design Question

**Should agents be LLM calls or math?**

No. LLM-per-agent is a trap:

- 1,000 LLM calls per round x multiple rounds = slow, expensive, non-reproducible
- Chen et al. 2023 (already cited in RESEARCH.md) showed LLMs get demand curves wrong under counterfactual reasoning
- You lose the ability to run sensitivity analysis (you can't differentiate through an LLM call)
- You lose reproducibility — same prompt, different day, different result

The current system is right that LLMs are interpreters, math is the engine. The problem isn't that the math exists — it's that the math operates on impoverished representations.

**The fix is: keep math agents, but give them richer inputs and state.**

---

## Layer 1: Product as a Feature Matrix, Not a Scalar

Instead of `perceived_benefit = 0.65`, a product becomes:

```yaml
Product:
  features:
    - name: "code_quality"
      score: 0.85
      category: "core_value"
      time_to_realize: "immediate"
      certainty: 0.7
      visibility: 0.3        # can others see this feature?

    - name: "credit_economics"
      score: 0.4
      category: "cost"
      time_to_realize: "immediate"
      certainty: 0.95
      visibility: 0.1

    - name: "ecosystem_integrations"
      score: 0.6
      category: "switching_cost_reducer"
      time_to_realize: "delayed"
      certainty: 0.5
      visibility: 0.6
```

Each feature carries its own certainty, visibility, and time-to-value. This means:
- Hyperbolic discounting applies per-feature (immediate features aren't penalized)
- Social proof weights per-feature (visible features get more social proof)
- Prospect value is computed per-feature then aggregated with agent-specific weights

**The LLM's calibration job becomes:** "Given this product and market, what are the 4-8 features that matter, and what are their scores?" This is a much more natural task for an LLM than "compress everything into one number."

**Competitors become feature matrices too.** Now you can do real competitive analysis — "Product A wins on features 1,3,5; Competitor B wins on features 2,4" — and different agents care about different features.

---

## Layer 2: Agent Feature Weights (Why People Differ)

Each agent gets a **weight vector** over feature categories, derived from their archetype + personality + income:

```yaml
Agent (budget-conscious early majority):
  feature_weights:
    core_value: 0.3
    cost: 0.6          # price-sensitive
    switching_cost: 0.4
    social_signal: 0.1

Agent (status-seeking innovator):
  feature_weights:
    core_value: 0.5
    cost: 0.1          # doesn't care about price
    switching_cost: 0.1
    social_signal: 0.7  # wants the shiny thing
```

Now prospect value becomes:

```
agent_gain = sum(feature_score[i] x agent_weight[i] x feature_certainty[i])
```

Different agents literally value the same product differently **on specific dimensions**, not just by a personality multiplier on a single number. The Claude Code vs cheaper-but-worse-tool comparison now works naturally — budget agents prefer Tool B, quality agents prefer Tool A, and the simulation tells you **which segment you're winning and losing**.

---

## Layer 3: Agent State Machine (Memory + Trajectory)

This is the big architectural change. Instead of stateless agents, each agent carries state:

```yaml
AgentState:
  phase: unaware | aware | considering | trialing | adopted | churned | locked_in

  awareness_strength: 0.0-1.0      # graded, not binary
  awareness_decay_rate: 0.02/round

  current_solution: "competitor_X" | "nothing" | "this_product"
  tenure_with_current: 14           # months — drives real status quo bias
  investment_depth: 0.6             # data, integrations, habits built

  trial_history:
    - product: "this_product"
      outcome: "positive"
      rounds_ago: 3

  social_exposure:                  # who in their network adopted
    cluster_adoption_rate: 0.15     # their local cluster, not global
    saw_influencer_endorse: true
```

**Phase transitions replace binary adopt/not-adopt:**

```
unaware -> aware         (marketing, word-of-mouth, visibility)
aware -> considering     (awareness_strength > threshold, attention allocation)
considering -> trialing  (if free trial exists, prospect value > friction)
trialing -> adopted      (trial experience + consistency pressure from Cialdini)
adopted -> churned       (value decay, better competitor, habit failure)
adopted -> locked_in     (high investment_depth, high switching cost)
```

This solves:
- **Gap 3** (memory) — agents remember trial experiences
- **Gap 8** (situational status quo) — `tenure_with_current x investment_depth` drives status quo bias, not just personality
- **Gap 9** (attention) — `awareness_strength` decays and competes
- **Gap 13** (event accumulation) — events modify agent state, not just parameters

**Simulation becomes multi-round:**
- Round 1: Awareness spreads, some agents enter consideration
- Round 2-3: Considerers evaluate, some trial
- Round 4+: Trials convert or churn, social proof builds, network effects kick in
- Round N: Equilibrium or continued growth

Each round is still pure NumPy — you're just running the force computation on a richer state vector per round, and updating states between rounds.

---

## Layer 4: Network Topology (Not Global Averages)

Instead of a global `adoption_rate` that every agent sees:

```yaml
Network:
  clusters:
    - id: "tech_twitter"
      size: 150
      internal_visibility: 0.8
      cross_cluster_links: ["startup_founders", "dev_teams"]

    - id: "enterprise_IT"
      size: 300
      internal_visibility: 0.3
      cross_cluster_links: ["dev_teams"]
```

Each agent belongs to 1-2 clusters. Social proof is computed **locally**:

```
local_adoption = adoption_rate_in_my_cluster
social_proof = social_proof_need x log(1 + local_adoption x cluster_visibility) x (1 - certainty)
```

This gives you:
- **Critical mass thresholds** — Slack is useless until your team cluster hits ~40% adoption
- **Cascade dynamics** — adoption spreads cluster-to-cluster, not uniformly
- **Influencer effects** — an adoption in a high-connectivity node matters more than in an isolated node

You don't need a full graph. 10-20 clusters with cross-links captures 80% of the network effect. Still vectorizable — it's just a sparse adjacency matrix multiplication per round.

---

## Layer 5: Force Interactions (Not Linear Sum)

Replace the linear sum with a structured interaction model:

```python
# Current (wrong):
total_utility = prospect + anchoring + status_quo + social + fomo + discount + identity

# Proposed:
# Step 1: Anchoring modifies the INPUT to prospect value (not additive)
effective_price = price * (1 - anchoring_effect)  # anchor reduces perceived price
prospect = compute_prospect(effective_price, ...)  # then prospect uses adjusted price

# Step 2: Loss aversion x Discounting multiply for delayed-benefit products
if time_to_value > threshold:
    prospect_adjusted = prospect * discount_multiplier  # multiplicative
else:
    prospect_adjusted = prospect + discount_penalty      # additive for immediate

# Step 3: Social proof scales with uncertainty (already correct)
social = social_proof_need * log(1 + local_adoption) * (1 - certainty)

# Step 4: Everything else adds
total_utility = prospect_adjusted + social + fomo + status_quo + identity
```

The key insight: **some forces modify other forces' inputs, some forces multiply, and only the remaining forces add.** The research literature tells you which is which — it's not a modeling choice, it's what the papers actually say.

---

## Layer 6: LLM Role (Richer Calibration, Same Boundaries)

The LLM's job expands but stays in the interpretation layer:

**Current LLM calibration (1 call):**
> "Here's a product. Give me 13 numbers."

**Proposed LLM calibration (2-3 calls):**

**Call 1 — Feature Extraction:**
> "Here's a product and its market research. What are the 4-8 features that matter to buyers? For each: score, certainty, visibility, time-to-realize."

**Call 2 — Competitive Mapping:**
> "Here are the features you identified. Here are the competitors from research. Score each competitor on the same features. Where does this product win and lose?"

**Call 3 — Market Context:**
> "Given this category maturity, growth rate, and competitive landscape, calibrate: network structure (what clusters matter?), awareness channels, and intervention levers specific to this market."

The LLM still never touches the decision math. But it's doing a **more natural task** — describing products in terms of features is how humans actually think about products, not "give me a benefit_certainty between 0 and 1."

---

## Layer 7: Interventions as Simulated Scenarios, Not Parameter Hacks

Instead of "multiply adoption_rate by 3x":

```yaml
Intervention: "Launch free trial"
  Mechanism:
    - Adds phase transition: considering -> trialing (with no price barrier)
    - After trial: consistency pressure (Cialdini) adds +0.15 to prospect value
    - Trial-to-paid conversion modeled explicitly over 3 rounds
  Cost: $X per trial user (server costs, support)
  Timeline: effect begins round 2, peaks round 4

Intervention: "Slack integration"
  Mechanism:
    - Adds feature: {name: "slack_integration", score: 0.7, category: "switching_cost_reducer"}
    - Reduces switching_cost for agents in "enterprise_IT" cluster
    - Increases social_visibility in "dev_teams" cluster
  Cost: 2 engineering months
  Timeline: effect begins round 5
```

Now interventions have costs, timelines, specific mechanisms, and can be combined. The LLM generates the intervention definition; the math engine simulates it.

---

## What the Pipeline Looks Like

```
User text
  -> Understand (1 LLM call — same as now)
  -> Research (Tavily — same as now)
  -> Calibrate (2-3 LLM calls — feature extraction + competitive mapping + market context)
  -> Generate Population (NumPy — same, but agents have state + cluster assignment)
  -> Simulate (multi-round loop, still pure NumPy):
      for round in 1..N:
        - compute per-feature forces for each agent
        - apply force interactions (multiplicative where specified)
        - update agent phases (state transitions)
        - propagate adoption through network clusters
        - check for equilibrium
  -> Analyze (sensitivity + LLM audit — same structure, richer data)
```

**Performance:** The multi-round loop adds ~5-10x to simulation time. But the base is <100ms, so 10 rounds x 100ms = ~1 second. Still negligible compared to LLM calls.

---

## What We Explicitly Would NOT Do

- **LLM per agent** — non-reproducible, slow, can't differentiate, fails on counterfactuals
- **Full graph simulation** — 1000-node graph is overkill; 10-20 clusters with cross-links captures the dynamics
- **Continuous time** — discrete rounds are fine; the behavioral economics literature works in discrete decisions anyway
- **Bayesian updating of parameters** — tempting but makes the model opaque; better to run sensitivity analysis on fixed parameters
- **Agent "personalities" via LLM prompts** — the archetype system with richer numeric traits is more rigorous than "imagine you're a cautious 45-year-old IT manager"

---

## Summary Table

| Dimension | Current (v1) | Proposed (v2) |
|-----------|-------------|---------------|
| Product representation | 6 scalars | 4-8 feature vectors with per-feature metadata |
| Competitor modeling | 1 reference price | Feature matrix per competitor |
| Agent state | Stateless, binary aware/not | State machine with 6 phases, memory, tenure |
| Social structure | Global average | 10-20 clusters with cross-links |
| Force interactions | Linear sum | Anchoring->prospect (input), loss x discount (multiply), rest add |
| Simulation | Single-shot | Multi-round with state transitions |
| Interventions | Parameter hacks | Simulated scenarios with cost/timeline |
| LLM role | 1 call, 13 numbers | 2-3 calls, feature extraction + competitive mapping |
| Computation | Pure NumPy | Still pure NumPy, just richer arrays |

---

## Gaps Addressed

| Gap (from ISSUES-v2) | Addressed By |
|-----------------------|-------------|
| Gap 1: Product is one number | Layer 1 (feature matrix) |
| Gap 2: Competitors are ghosts | Layer 1 + Layer 6 Call 2 (competitive mapping) |
| Gap 3: No memory | Layer 3 (agent state machine) |
| Gap 4: No network effects | Layer 4 (cluster topology) |
| Gap 5: Forces should multiply | Layer 5 (structured interactions) |
| Gap 6: Probability weighting missing | Layer 5 (add to prospect computation) |
| Gap 7: Anchoring double-counts | Layer 5 (anchoring modifies prospect input) |
| Gap 8: Status quo is dispositional | Layer 3 (tenure + investment_depth) |
| Gap 9: No attention model | Layer 3 (graded awareness_strength with decay) |
| Gap 10: Category maturity unused | Layer 6 Call 3 (market context calibration) |
| Gap 11: No Weber's Law | Layer 1 (per-feature scoring enables log-scaled price perception) |
| Gap 12: Interventions have no cost/time | Layer 7 (cost + timeline per intervention) |
| Gap 13: Events can't accumulate | Layer 3 (events modify persistent agent state) |
