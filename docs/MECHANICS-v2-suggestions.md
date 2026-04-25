# MECHANICS-v2: Competitor Awareness & Phase Transitions

Detailed design for the two mechanics that ARCHITECTURE-v2-suggestions.md left hand-wavy: how agents learn about competitors, and how agents move through adoption phases in the multi-round simulation.

---

## Competitor Awareness

### The Problem With "Awareness of All Competitors"

If you have 5 competitors and 1,000 agents, full per-competitor awareness means 5,000 awareness states. That's fine computationally, but it's wrong **behaviorally**. Real people don't have equal-depth knowledge of every competitor. They know:

- 1-2 competitors well (the one they use, the one everyone talks about)
- 1-2 by name only ("I've heard of it")
- The rest not at all

### Three-Tier Competitor Knowledge Model

Each agent has tiered knowledge of the competitive landscape:

```yaml
Agent.competitor_knowledge:
  primary:    "VS Code"              # the one they use or know deeply
  secondary:  ["Cursor"]             # heard of, know roughly what it does
  unaware_of: ["Windsurf", "Zed"]   # doesn't know these exist
```

**How tiers are assigned at population generation:**

- Market share drives probability of being the `primary` — if VS Code has 70% market share, 70% of agents have it as primary
- `social_visibility x cluster_membership` drives secondary awareness — agents in "tech_twitter" cluster are more likely to have heard of Cursor than agents in "enterprise_IT"
- Archetype modulates breadth — Innovators know 3-4 tools, Laggards know 1

**What each tier gives the agent:**

| Tier | Feature scores visible? | Price known? | Reference price contribution |
|------|------------------------|-------------|------------------------------|
| Primary | Full feature matrix | Yes | Weighted at 0.6 |
| Secondary | Category-level ("it's an AI code tool") | Roughly ("around $20") | Weighted at 0.3 |
| Unaware | Nothing | No | 0 |

### Agent-Specific Reference Price

The reference price emerges naturally from the knowledge structure:

```
agent_ref_price = (
    primary_competitor.price x 0.6
    + mean(secondary_competitor.prices) x 0.3
    + category_default_price x 0.1  # "what I'd expect to pay"
)
```

An Innovator who knows 4 tools has a well-calibrated reference price. A Laggard who only knows the dominant free tool has `ref_price ~ $0`, making any paid product feel expensive. This emerges from the knowledge structure — you don't have to hardcode it.

### Tiered Feature Comparison

When an agent evaluates the product's gain side, they compare per-feature — but only against competitors they actually know:

```
For each feature:
  if agent knows primary competitor:
    relative_score = your_feature_score - primary.feature_score
  else:
    relative_score = your_feature_score - category_average

  agent_gain += agent_weight[feature] x relative_score
```

An agent who uses VS Code (free, good, but no AI) compares the AI coding tool against VS Code. An agent who uses Cursor compares against Cursor. **Same product, different competitive context per agent.**

### How Competitor Awareness Changes Over Rounds

- **Round 1:** Initial awareness distribution (from population generation)
- **Between rounds:** Social exposure can promote competitors from unaware -> secondary
  - "3 people in my cluster started using Cursor" -> I now know Cursor exists
  - Marketing events ("Cursor launches Super Bowl ad") -> mass secondary awareness
- **Rare:** Secondary -> primary only happens through deep engagement (trial, extended research). Most agents never promote.

---

## Phase Transitions: How Agents Move Through Adoption

### The State Machine

```
unaware --> aware --> considering --> trialing --> adopted --> locked_in
                                        |            |
                                        +-- quit     +-- churned
```

Each transition has a **trigger condition** and a **probability gate**. The trigger determines eligibility; the probability determines whether it actually happens this round.

---

### UNAWARE -> AWARE

This is NOT a decision — the agent doesn't choose to become aware. Awareness happens *to* them.

**Triggers (any one):**
- Marketing reach: random draw against `marketing_intensity x channel_match`
- Network exposure: someone in agent's cluster adopted (word of mouth)
- Category event: major news in the category ("AI tools are everywhere")

**Probability per round:**

```
P(become_aware) = 1 - (1 - base_marketing_reach)
                    x (1 - cluster_adoption_rate x word_of_mouth_rate)
                    x (1 - category_buzz_factor)
```

**Result:**

```
awareness_strength = initial_strength  (0.3-0.7 depending on channel)
# Saw an ad = 0.3, friend told them = 0.6, saw viral tweet = 0.5
```

Between rounds, `awareness_strength` decays by `awareness_decay_rate` per round. If it drops below a threshold (say 0.05), the agent reverts to unaware. **This models forgetting** — you saw an ad 6 months ago but can't remember the product name.

---

### AWARE -> CONSIDERING

The agent knows the product exists but hasn't evaluated it. The transition to "considering" means they're actively thinking about it.

**Trigger condition:**

```
awareness_strength > consideration_threshold
AND agent has cognitive bandwidth (not overwhelmed by other decisions)
```

**What raises awareness_strength above threshold:**
- Repeated exposure (each exposure adds +0.1 to strength)
- Social signal ("my boss mentioned it" = +0.2)
- Pain event ("my current tool broke" = +0.3)
- NOT just time passing — mere awareness decays

**Probability gate:**

```
P(start_considering) = awareness_strength x relevance_to_agent

relevance = max(agent_weight[i] x product_feature[i].score) for top features
# If the product's best feature matches what the agent cares about, they consider
```

**Category maturity modulates the threshold** (solves Gap 10 from ISSUES-v2):
- Nascent category: `consideration_threshold = 0.6` (harder to take seriously)
- Growing category: `consideration_threshold = 0.35` (easier, there's buzz)
- Saturated category: `consideration_threshold = 0.5` (agent already has a solution, needs a reason)

---

### CONSIDERING -> TRIALING

This is where the **7-force model fires for the first time**. The agent evaluates the product.

**Trigger condition:**

```
Agent is in "considering" phase
AND has been considering for >= 1 round (not impulsive, except Innovators)
```

**The full force computation runs:**

```
total_utility = compute_forces(agent, product, competitors_agent_knows)
```

**Decision:**

```
if free_trial_available:
    # Lower bar — trial is low-commitment
    P(trial) = logistic(total_utility x temperature, offset=+0.1)
    # The +0.1 offset means marginal agents will try something free
    # that they wouldn't pay for — this is the free trial mechanism
else:
    # Must commit to purchase — full evaluation
    P(trial) = 0  # skip to considering -> adopted directly
```

**Key:** For products with no free trial, agents jump directly from considering -> adopted (or not). The trial phase only exists when the product offers it. This is why free trials are powerful — they add an intermediate low-commitment step.

---

### TRIALING -> ADOPTED (or QUIT)

The agent has tried the product. Now they evaluate actual experience vs expectations.

**Trigger condition:**

```
Agent has been trialing for trial_duration rounds
```

**Evaluation:**

```
experienced_value = perceived_benefit x experience_noise(mean=1.0, std=0.15)
# Some agents have better-than-expected experience, some worse

consistency_pressure = 0.15  # Cialdini — "I already use it, I should keep using it"

sunk_cost = trial_rounds x time_investment x 0.1
# The more they've invested in the trial, the harder to quit

revised_utility = original_utility + consistency_pressure + sunk_cost
                  + (experienced_value - expected_value) x adjustment_weight
```

**Decision:**

```
P(adopt) = logistic(revised_utility x temperature)
```

**Agent memory matters here.** The trial experience is stored:

```
agent.trial_history.append({
    product: "this_product",
    outcome: "positive" if adopted else "negative",
    experienced_value: experienced_value,
    rounds_ago: 0
})
```

If the agent doesn't adopt after trial, they go to **quit** state, but they remember the experience. A future event ("price drops 50%") can bring them back to considering, and their revised evaluation includes "I tried it and it was decent but not worth $20 — at $10 maybe."

---

### ADOPTED -> LOCKED_IN

Over time, adopted agents build investment:

```
Per round while adopted:
  investment_depth += usage_intensity x integration_depth x 0.05

  if investment_depth > lock_in_threshold (e.g., 0.7):
    phase = locked_in
    # This agent is now very unlikely to churn
    # Their status_quo_bias for THIS product is now high
    # They've become a defender of the status quo, not a challenger
```

---

### ADOPTED -> CHURNED

```
Churn triggers (checked each round):
  - Value decay: if experienced_value < churn_threshold for N consecutive rounds
  - Better competitor: if a competitor enters with higher utility (recompute forces)
  - Price increase event: recompute prospect value with new price
  - Habit failure: for behavior-change products, random dropout rate

P(churn) = base_churn_rate x (1 - investment_depth) x (1 - consistency_pressure)
# Deep investment and consistency both protect against churn
```

---

## Example: Full Simulation Loop

```
Round 1:
  [900 unaware] [80 aware] [20 considering] [0 trialing] [0 adopted]

  - 50 agents become aware (marketing + category buzz)
  - 15 aware agents start considering (repeated exposure crossed threshold)
  - 12 considering agents compute forces -> 8 start trial, 2 adopt directly

Round 2:
  [850 unaware] [95 aware] [23 considering] [8 trialing] [2 adopted]

  - 40 more become aware (marketing continues)
  - 10 aware agents' awareness decayed below threshold -> back to unaware
  - 8 trialing agents evaluate experience -> 5 adopt, 2 continue trial, 1 quits
  - Cluster "tech_twitter" now has 4% adoption -> word-of-mouth kicks in

Round 3:
  [820 unaware] [105 aware] [30 considering] [2 trialing] [7 adopted]

  - Word-of-mouth from adopted agents in "tech_twitter" pushes 15 cluster-mates to aware
  - Social proof force is now non-zero for agents in that cluster
  - FOMO starts registering for high-fomo agents who see cluster adoption

  ...

Round 10:
  [400 unaware] [150 aware] [80 considering] [20 trialing] [320 adopted] [30 locked_in]

  - Adoption curve is S-shaped (emerged naturally from mechanics)
  - "The chasm" is visible between round 5-7 (early majority requires social proof
    threshold that takes time to build)
  - Equilibrium approaching — new awareness roughly equals churn
```

---

## The Key Insight

**No round is special.** The same force computation + state transition rules run every round. The S-curve, the chasm, the network cascade — they all **emerge** from the mechanics rather than being hardcoded. That's the validation that the model is working: if you get Rogers' adoption curve without coding Rogers' adoption curve, your behavioral parameters are calibrated correctly.
