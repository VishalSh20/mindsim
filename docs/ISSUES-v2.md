# ISSUES-v2: Mental Model Gaps & Conceptual Crudeness Audit

Comprehensive audit of where the simulation's **conceptual framework** — not just implementation — is too crude to capture how humans actually decide. Builds on ISSUES.md (implementation bugs) with architectural and theoretical gaps.

---

## Gap 1: Everything About the Product Is One Number

`perceived_benefit = 0.65` — that's it. A meditation app and an AI coding tool with the same score are identical to the model. There's no feature vector, no "great at X but terrible at Y."

**Example:** Two AI coding tools — Tool A has better code quality but worse credit economics, Tool B has worse code quality but subsidizes credits more. The model assigns each a single `perceived_benefit` number and can't distinguish them.

**What's needed:** A feature vector per product with per-agent feature weights. The gain side becomes:
```
agent_gain = sum(feature_score[i] × agent_weight[i]) for each feature
```
where an Innovator weights "cutting-edge capability" at 0.8 and "price" at 0.2, while a Late Majority agent weights them inversely.

**Impact:** Every force downstream inherits this oversimplification. This is the single biggest source of crudeness.

---

## Gap 2: Competitors Are Ghosts

The research stage finds competitors (names, prices), but after calibration they collapse into a single `reference_price` scalar. There's no "Competitor A is better on quality, Competitor B is cheaper, Competitor C has network lock-in."

**What's missing:**
- Feature-level competitive comparison
- Market share tracking
- Competitive response modeling

**Impact:** The simulation literally cannot answer "what happens if a stronger competitor enters" — only "what happens if the reference price changes."

---

## Gap 3: No Memory, No Momentum

Agents are stateless. They don't remember trying the product, don't accumulate switching costs, don't build habits. Each simulation round (and each event) generates a **fresh population**.

**Can't model:**
- "Users who tried the free trial are 3x more likely to convert"
- "Adoption snowballs as teams reach critical mass"
- "Churn after month 3 when novelty wears off"

**Impact:** The model gives you a snapshot, never a trajectory.

---

## Gap 4: Network Effects Don't Exist

Social proof is `log(1 + adoption_rate × visibility)` — a scalar formula, not a network simulation. Slack is worthless if your 3 teammates don't use it, but game-changing if they do.

**What's missing:**
- Clusters, teams, influencer nodes
- Critical mass thresholds
- Multi-sided market dynamics

**Impact:** Every agent evaluates independently against a global average. Products whose value depends on who else uses them (most B2B tools, communication products) are fundamentally mismodeled.

---

## Gap 5: Loss Aversion x Discounting Should Multiply, But They Add

RESEARCH.md explicitly says these forces **multiply** for upfront-cost + delayed-benefit products (habit trackers, enterprise tools, education). The code **sums all 7 forces linearly** in `compute_decisions()`.

**Why it matters:** Products where you pay now and benefit later are systematically under-penalized. The free trial recommendation is less emphatic than it should be.

**Affected products:** ~30-40% of all products — anything habit-based, education, enterprise software, fitness apps.

---

## Gap 6: Probability Weighting Is Completely Missing

Tversky & Kahneman 1992 (cited in RESEARCH.md with γ=0.61) describes how people overweight small probabilities and underweight large ones. It's referenced, parameterized, and **never implemented**.

**What this means:** "There's a 5% chance this makes me 10x more productive" should be irrationally attractive (overweighted small probability), but the model treats it linearly. Affects any product with uncertain benefits — most SaaS products.

---

## Gap 7: Anchoring Double-Counts With Prospect Value

When a product is cheaper than the reference price, **both** Force 1 (prospect value — price looks small) and Force 2 (anchoring — price below reference) reward independently.

**The problem:** In Kahneman & Tversky's actual framework, anchoring works **through** the loss function (shrinking the perceived loss), not as a separate additive bonus.

**Impact:** Underpriced products get an inflated boost — roughly 10-20 percentage points of adoption inflation for products significantly below reference price.

---

## Gap 8: Status Quo Bias Is a Personality Trait, Not a Situation

An Innovator always has `status_quo_bias = 0.15`, a Laggard always `0.85`. But status quo bias should depend on **how invested you are in the current solution**.

**The contradiction:** A Laggard who's never used any tool in this category should have *low* status quo bias (there's no status quo to defend). A Laggard who's used Excel for 20 years should have *high* status quo bias.

**What's needed:** Status quo bias = f(archetype_disposition, investment_in_current_solution, time_with_current_tool).

---

## Gap 9: No Attention or Salience Model

Awareness is binary — you're aware or you're not. Real attention is graded, competes with other products, decays over time, and spikes with marketing events.

**What's missing:**
- "Top-of-mind" vs "vaguely heard about it 3 months ago"
- Attention decay over time
- Competition for mindshare
- Marketing event spikes

**Impact:** The awareness gate is a light switch when it should be a dimmer.

---

## Gap 10: Category Maturity Is Computed and Thrown Away

The research stage classifies the category as nascent/growing/mainstream/saturated. This should fundamentally change the adoption curve — nascent categories face "does this category even make sense?" skepticism that mature categories don't.

**Current state:** Classified, stored, **never used in any force computation**.

**Should affect:** Social proof (nascent categories have less signal), status quo bias (no established behavior to defend in nascent categories), FOMO (higher in growing categories), identity signaling (stronger in nascent categories for early adopters).

---

## Gap 11: No Weber's Law on Price

Going from $5 to $10 (100% increase) feels massive. Going from $100 to $105 (5% increase) feels negligible. But the model treats price as a linear fraction of income.

**What's needed:** Logarithmic price perception — diminishing sensitivity to price changes at higher price points. Weber's Law: `perceived_change = actual_change / reference_point`.

---

## Gap 12: Interventions Have No Cost and No Time

"Social proof push" gives 3x adoption rate for free, instantly. Real interventions cost money, take months, and interact with each other.

**What's missing:**
- Intervention cost modeling (CAC, ad spend, content creation)
- Implementation timeline
- Interference between combined interventions (free trial + price cut might be subadditive)
- ROI calculation (lift per dollar, not just raw lift)

**Impact:** Intervention ranking is by raw lift, which is misleading — a 5pp lift that costs $10K is better than a 10pp lift that costs $500K.

---

## Gap 13: Events Can't Accumulate

Each event regenerates the population from scratch. So "Competitor A launches" followed by "Competitor A gets bad press" doesn't model agents who switched to A and now regret it.

**Impact:** Events are isolated shocks, not a storyline. Sequential events can't build on each other.

---

## Additional Gaps (From Deep Code Audit)

### Gap 14: price_sensitivity and novelty_weight Are Dead Parameters
- Sampled from archetype distributions in `archetypes.yaml`
- **Never used** in force computation in `forces.py`
- Appear in diagnostic dumps, potentially misleading users

### Gap 15: No Calibration for Archetype-Specific Product Perception
- `by_archetype` field exists on `CalibratedParam`
- `CALIBRATE_PROMPT` never asks the LLM for per-archetype overrides
- Architecture supports it, calibration doesn't use it

### Gap 16: LLM Confidence Scores Are Unvalidated
- LLM assigns confidence (0-1) for each parameter
- No validation against published ranges
- If LLM sets `loss_aversion_lambda = 0.5` (below published minimum of 1.0), no warning fires

### Gap 17: Sensitivity Analysis Is Local-Only
- Tests +/-30% perturbations independently
- Never tests parameter interactions (what if both benefit_certainty AND present_bias_beta are low?)
- Misses nonlinear and interaction effects entirely

### Gap 18: discovered_competitors Never Populated
- Research stage only appends to `verified_competitors`
- `discovered_competitors` field exists but is always empty

### Gap 19: Reference Price Components Are Decorative
- LLM generates `reference_price.components` (e.g., "Free tools, $0, weight 0.4")
- These are stored for display but never used in computation
- Documentation without computation

---

## Root Causes

Most gaps trace back to **three architectural choices**:

1. **Products are scalars, not vectors** — no feature-level modeling means no real competitive analysis (Gaps 1, 2, 11)
2. **Agents are stateless** — no memory means no trajectories, no network effects, no accumulated switching costs (Gaps 3, 4, 8, 9, 13)
3. **Forces are linearly additive** — the research says they interact multiplicatively in specific combinations, but the engine just sums them (Gaps 5, 6, 7)

Fix those three and about 10 of the 19 gaps collapse.

---

## Priority Order for Fixes

| Priority | Gap | Effort | Impact |
|----------|-----|--------|--------|
| P0 | Gap 5: Multiplicative force interactions | Low | High — formula change only |
| P0 | Gap 7: Anchoring double-counting | Low | High — restructure two forces |
| P1 | Gap 1: Product feature vector | High | Critical — unlocks competitive modeling |
| P1 | Gap 3: Agent state/memory | High | Critical — unlocks trajectories |
| P1 | Gap 6: Probability weighting | Medium | High — formula already in RESEARCH.md |
| P2 | Gap 2: Competitive differentiation | High | High — depends on Gap 1 |
| P2 | Gap 4: Network effects | High | High — depends on Gap 3 |
| P2 | Gap 8: Situational status quo bias | Medium | Medium |
| P2 | Gap 10: Use category maturity | Low | Medium — data already exists |
| P3 | Gap 9: Graded attention model | Medium | Medium |
| P3 | Gap 11: Weber's Law pricing | Low | Low-Medium |
| P3 | Gap 12: Intervention costs/timing | Medium | Medium |
| P3 | Gap 13: Event accumulation | High | Medium — depends on Gap 3 |
| P3 | Gap 14-19: Implementation cleanups | Low each | Low each |
