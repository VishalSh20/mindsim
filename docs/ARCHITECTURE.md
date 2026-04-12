# mindsim

## Summary

mindsim is a behavioral economics simulation engine that predicts how psychologically diverse populations react to products, pricing, and market events. You describe a product in plain English. The engine researches the market, generates 1,000 agents with published psychological profiles, runs each through a 7-force decision model grounded in 28 peer-reviewed sources, and produces a mechanism map showing *why* people adopt or reject — not just whether they will.

The core insight: the useful question isn't "will my product succeed?" but "which cognitive forces drive or block adoption, and what interventions shift them?" The output is a force decomposition — named behavioral mechanisms (loss aversion, anchoring, status quo bias, FOMO, social proof, present bias, identity signaling) with quantified effect sizes, confidence bands, and ranked interventions.

The engine handles market dynamics through an event system. Instead of simulating time, you inject events — "a competitor launches at 10x your price," "the category goes mainstream," "a recession hits." Each event is interpreted into per-force parameter adjustments by an LLM that sees the full simulation state, then the math reruns instantly. Events are stackable, producing scenario trees.

**What makes it different from "ask ChatGPT about my product":** Every prediction is decomposed into named mechanisms from published research. Every parameter has a basis and confidence score. Every assumption is challengeable. The user sees *which bias* is blocking adoption and *which intervention* breaks it, with effect sizes from real behavioral economics, not LLM vibes.

**Technical shape:** CLI-first. Adaptive pipeline (not agents, not rigid steps). 3 LLM calls + 1 per event. 2-8 Tavily API calls (confidence-gated). 1,000-agent simulation in <100ms via vectorized NumPy. Total runtime: 10-16 seconds for a full research-backed run. Override reruns in ~4 seconds. Local LLM support via Ollama for zero-cost operation.

---

## Architecture

### Design Principles

**Pipeline, not agents.** Each stage has a typed input/output contract. LLMs are called for specific reasoning tasks (parse product, calibrate parameters, interpret events, write reports), not for autonomous decision-making. The simulation itself is pure math.

**Adaptive, not rigid.** The research stage plans its own queries with stopping conditions. Simple products (known category, listed competitors) cost 2-3 API calls. Novel products (new category, ambiguous positioning) cost 7-8. Complexity of input drives complexity of research.

**Transparent, not black-box.** Every parameter has a `basis` (why this number), `confidence` (how sure), and `sensitivity` (what happens if it's wrong). Users can challenge any assumption and rerun.

**Theory-anchored, not data-fit.** Agent decision logic implements published behavioral economics models (prospect theory, anchoring, status quo bias) with published parameters — not patterns learned from data. This means the model works on novel products where no historical data exists.

---

### System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   USER INPUT                                                        │
│   Natural language product description                              │
│   + optional: --income-range, --geo, --event, --compare,            │
│               --interactive, --skip-research, --override            │
│                                                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1: UNDERSTAND                                    1 LLM call  │
│                                                                     │
│  Input:  raw user text                                              │
│  Output: ProductProfile (structured, NO numerical parameters)       │
│          + ResearchPlan (queries with confidence thresholds)         │
│                                                                     │
│  The LLM extracts what the user explicitly said, flags what's       │
│  missing, and generates a prioritized research plan. It does NOT    │
│  assign any simulation parameters — those come after research.      │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ ProductProfile                                              │    │
│  │  product.name, product.price, product.price_model           │    │
│  │  competitors[].name, competitors[].price (if user stated)   │    │
│  │  target_audience (stated or inferred with flag)              │    │
│  │  value_proposition (user's exact claim)                      │    │
│  │  missing_information[] (gaps to fill via research)           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ ResearchPlan                                                │    │
│  │  priority_queries[] — each with:                            │    │
│  │    query, goal, confidence_threshold, fallback_assumption   │    │
│  │  optional_queries[] — each with:                            │    │
│  │    query, goal, trigger_condition                           │    │
│  └─────────────────────────────────────────────────────────────┘    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2: RESEARCH                          Adaptive: 2-8 Tavily   │
│                                             + 1 Google Trends       │
│  Input:  ResearchPlan                                0 LLM calls   │
│  Output: MarketContext (verified data + confidence flags)           │
│                                                                     │
│  Executes the research plan adaptively:                             │
│                                                                     │
│  Phase 1 — Priority queries (always run)                            │
│    For each query:                                                  │
│      1. Run Tavily search (basic = 1 credit)                       │
│      2. Evaluate confidence via heuristics (keyword match,          │
│         source consistency, source authority — NO LLM call)         │
│      3. If confidence >= threshold → integrate result               │
│      4. If confidence < threshold AND budget remains →              │
│         refine query (rule-based, not LLM), run advanced            │
│         search (2 credits), re-evaluate                             │
│      5. If still low → use fallback assumption, flag as             │
│         low confidence                                              │
│                                                                     │
│  Phase 2 — Google Trends (always, free)                             │
│    Category growth rate + current interest level                    │
│    Used to classify category maturity:                              │
│    nascent (<20 interest) | growing (20-60, positive slope)         │
│    | mainstream (60+, flat) | declining (negative slope)            │
│                                                                     │
│  Phase 3 — Optional queries (conditional)                           │
│    Only fire if trigger conditions met from Phase 1 results         │
│    Example: "pricing psychology" query only if competitor            │
│    pricing data was ambiguous                                       │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ MarketContext                                               │    │
│  │  verified_competitors[]: name, confirmed_price, has_free_   │    │
│  │    tier, market_position, source_url                        │    │
│  │  discovered_competitors[]: found by landscape search,       │    │
│  │    not listed by user                                       │    │
│  │  category_penetration: float + confidence                   │    │
│  │  category_growth: from Google Trends                        │    │
│  │  category_maturity: nascent|growing|mainstream|saturated     │    │
│  │  low_confidence_flags[]: which goals couldn't be met        │    │
│  │  research_cost: total Tavily credits spent                  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  Skippable with --skip-research (uses fallback assumptions).        │
│  Budget cap with --budget N (default: 10 credits).                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3: CALIBRATE                                    1 LLM call   │
│                                                                     │
│  Input:  ProductProfile + MarketContext (together, first time)       │
│  Output: SimulationConfig (ALL numerical parameters)                │
│                                                                     │
│  One LLM call that sees the raw product description AND the         │
│  research results together, assigning every parameter fresh.        │
│  No premature anchoring — the LLM has never seen numbers for        │
│  this product before this call.                                     │
│                                                                     │
│  The prompt includes behavioral science defaults for grounding:     │
│    λ = 2.25 (Kahneman & Tversky 1992)                              │
│    β = 0.5 for behavior-change, 0.85 for consumption               │
│    (Augenblick et al. 2015)                                        │
│    status_quo_bias: 0.3 easy-switch, 0.7 locked-in                 │
│    social_visibility: 0.2 private, 0.7 visible                     │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ SimulationConfig                                            │    │
│  │                                                             │    │
│  │ simulation_params:                                          │    │
│  │   reference_price:                                          │    │
│  │     value: 8.50                                             │    │
│  │     components:                                             │    │
│  │       - source: "Free tools", weight: 0.40                  │    │
│  │       - source: "Copilot $10", weight: 0.35                 │    │
│  │       - source: "Cursor $20", weight: 0.25                  │    │
│  │     confidence: 0.85                                        │    │
│  │                                                             │    │
│  │   [each parameter has: value, basis, confidence]            │    │
│  │   category_penetration: 0.18                                │    │
│  │   benefit_certainty: 0.55                                   │    │
│  │   time_to_value: 0.25                                       │    │
│  │   requires_behavior_change: 0.35                            │    │
│  │   switching_cost: 0.50                                      │    │
│  │   social_visibility: 0.60                                   │    │
│  │   identity_signal: 0.50                                     │    │
│  │   present_bias_β: 0.75                                      │    │
│  │   fomo_intensity: 0.55                                      │    │
│  │                                                             │    │
│  │   awareness_by_archetype:                                   │    │
│  │     innovator: 0.90                                         │    │
│  │     early_adopter: 0.65                                     │    │
│  │     early_majority: 0.30                                    │    │
│  │     late_majority: 0.10                                     │    │
│  │     laggard: 0.02                                           │    │
│  │                                                             │    │
│  │ population_config:                                          │    │
│  │   income_distribution: {mean_log: 11.2, sigma: 0.6}        │    │
│  │   market_segment: tech_professional                         │    │
│  │                                                             │    │
│  │ assumptions_made[]:                                         │    │
│  │   - parameter: category_penetration                         │    │
│  │     value: 0.18                                             │    │
│  │     basis: "Tavily: '15-20% of devs use paid AI tools'"    │    │
│  │     confidence: 0.6                                         │    │
│  │     sensitivity: "high"                                     │    │
│  └─────────────────────────────────────────────────────────────┘    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 4: SIMULATE                              0 LLM calls        │
│                                                 Pure NumPy, <100ms │
│                                                                     │
│  Input:  SimulationConfig                                           │
│  Output: SimulationResult                                           │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                                                              │   │
│  │  4a. GENERATE POPULATION (1,000 agents)                      │   │
│  │                                                              │   │
│  │  5 archetypes from Rogers (1962):                            │   │
│  │    Innovator (2.5%) | Early Adopter (13.5%)                  │   │
│  │    Early Majority (34%) | Late Majority (34%)                │   │
│  │    Laggard (16%)                                             │   │
│  │                                                              │   │
│  │  Each agent samples from archetype distributions:            │   │
│  │    loss_aversion_λ — Innovator: N(1.5, 0.3)                 │   │
│  │                       Early Majority: N(2.25, 0.5)           │   │
│  │                       Laggard: N(3.5, 0.6)                   │   │
│  │    + 20% of population draws from near-zero subgroup         │   │
│  │      N(1.1, 0.2) per Gächter et al. 2022                    │   │
│  │                                                              │   │
│  │    status_quo_bias, social_proof_need, novelty_weight,       │   │
│  │    price_sensitivity — all archetype-distributed             │   │
│  │                                                              │   │
│  │  Income sampled independently from lognormal                 │   │
│  │    (matched to target market segment)                        │   │
│  │                                                              │   │
│  │  Personality → economics correlations:                       │   │
│  │    Neuroticism ↔ loss_aversion (r≈0.3, Lauriola 2001)       │   │
│  │    Openness ↔ novelty_weight (r≈0.3, Nicholson 2005)        │   │
│  │    Agreeableness ↔ social_proof_need (r≈0.4)                │   │
│  │                                                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                               │                                     │
│                               ▼                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                                                              │   │
│  │  4b. AWARENESS FILTER                                        │   │
│  │                                                              │   │
│  │  Each agent has an awareness probability by archetype.       │   │
│  │  Agents below threshold don't enter the decision pipeline.   │   │
│  │                                                              │   │
│  │  In a "nascent" category with low awareness:                 │   │
│  │    Only innovators + some early adopters are aware            │   │
│  │    → addressable market is ~16% of population                │   │
│  │  In a "mainstream" category:                                 │   │
│  │    Most agents are aware → 70%+ addressable                  │   │
│  │                                                              │   │
│  │  Agents who ARE aware also have per-competitor awareness.    │   │
│  │  Innovators know ~90% of competitors.                        │   │
│  │  Late majority knows ~20% (only the dominant player).        │   │
│  │  This affects each agent's reference price computation.      │   │
│  │                                                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                               │                                     │
│                               ▼                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                                                              │   │
│  │  4c. COMPUTE 7 FORCES (vectorized, all agents at once)       │   │
│  │                                                              │   │
│  │  For each aware agent, compute all forces simultaneously:    │   │
│  │                                                              │   │
│  │  FORCE 1: PROSPECT VALUE                                     │   │
│  │    Source: Kahneman & Tversky 1979, 1992                     │   │
│  │    gain = perceived_benefit ^ α (α=0.88)                     │   │
│  │    loss = λ × (monthly_cost / income)^β (β=0.88)             │   │
│  │    prospect = gain - loss                                    │   │
│  │    λ per agent (1.0-3.5), from archetype + Gächter mixture   │   │
│  │                                                              │   │
│  │  FORCE 2: REFERENCE PRICE ANCHORING                          │   │
│  │    Source: Tversky & Kahneman 1974, Mazumdar 2005            │   │
│  │    ref = awareness-weighted average of known competitor       │   │
│  │          prices (different per archetype — innovators know    │   │
│  │          more competitors than laggards)                     │   │
│  │    anchor_effect = (ref - price) / max(ref, 1) × 0.3        │   │
│  │    Positive = deal, Negative = overpriced                    │   │
│  │                                                              │   │
│  │  FORCE 3: STATUS QUO BIAS                                    │   │
│  │    Source: Samuelson & Zeckhauser 1988                        │   │
│  │    -(agent.status_quo_bias × category_penetration ×           │   │
│  │      (0.5 + 0.5 × switching_cost))                           │   │
│  │    Always negative. Stronger when current solution is         │   │
│  │    entrenched and switching cost is high.                     │   │
│  │                                                              │   │
│  │  FORCE 4: SOCIAL PROOF                                       │   │
│  │    Source: Cialdini 1984, Salganik 2006                       │   │
│  │    agent.social_proof_need × log(1 + adoption × visibility)  │   │
│  │      × (1 - benefit_certainty)                               │   │
│  │    Matters MORE when benefit is uncertain.                    │   │
│  │    In single-round mode, adoption = category_penetration      │   │
│  │    of THIS product (initially low for new products).          │   │
│  │                                                              │   │
│  │  FORCE 5: ANTICIPATED REGRET (FOMO)                          │   │
│  │    Source: Loomes & Sugden 1982, Zeelenberg 1999              │   │
│  │    fomo_intensity × social_visibility ×                       │   │
│  │      category_growth_signal × agent.social_proof_need         │   │
│  │    Amplified when: category is growing, product is visible,   │   │
│  │    agent is socially sensitive. Distinct from loss aversion   │   │
│  │    (price pain) — this is opportunity-miss pain.              │   │
│  │                                                              │   │
│  │  FORCE 6: HYPERBOLIC DISCOUNTING                             │   │
│  │    Source: Laibson 1997, Augenblick 2015                      │   │
│  │    -(time_to_value × (1 - β) × perceived_benefit × 0.3)      │   │
│  │    β = 0.5 for behavior-change products (Augenblick 2015)    │   │
│  │    β = 0.85 for consumption products                          │   │
│  │    Penalizes products where the payoff is weeks away.         │   │
│  │                                                              │   │
│  │  FORCE 7: IDENTITY SIGNALING                                 │   │
│  │    Source: Veblen 1899, Berger & Heath 2007                   │   │
│  │    agent.openness × identity_signal × social_visibility ×0.2 │   │
│  │    Stronger for identity-relevant categories (tech, fashion)  │   │
│  │    and for high-openness agents.                              │   │
│  │                                                              │   │
│  │  KEY INTERACTIONS (not separate forces, but how forces        │   │
│  │  combine non-linearly):                                      │   │
│  │    • Loss aversion × Discounting: multiplicative for          │   │
│  │      products with upfront cost + delayed benefit             │   │
│  │    • Social proof × Uncertainty: social proof weight          │   │
│  │      scales with (1 - benefit_certainty)                     │   │
│  │    • Anchoring × Loss aversion: high reference price          │   │
│  │      reduces the input to the loss function                   │   │
│  │                                                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                               │                                     │
│                               ▼                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                                                              │   │
│  │  4d. DECISION + SEGMENTATION                                 │   │
│  │                                                              │   │
│  │  total_utility = sum of 7 forces per agent                   │   │
│  │  adopt_probability = logistic(total_utility × temperature)   │   │
│  │  decision = random() < adopt_probability                     │   │
│  │                                                              │   │
│  │  Segmentation:                                               │   │
│  │    By archetype: innovator/early_adopter/etc adoption rates  │   │
│  │    By income: high/mid/low bracket adoption rates            │   │
│  │    By decision proximity:                                    │   │
│  │      LOCKED IN (prob > 0.7): would adopt regardless          │   │
│  │      CONVERTIBLE (prob 0.3-0.7): within intervention reach   │   │
│  │      UNREACHABLE (prob < 0.3): no realistic intervention     │   │
│  │                                                              │   │
│  │  Force decomposition is WEIGHTED toward boundary agents      │   │
│  │    (prob near 0.5) — these are the agents whose forces       │   │
│  │    actually matter for strategy.                             │   │
│  │    weight = 1 - (2 × |prob - 0.5|)²                          │   │
│  │                                                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                               │                                     │
│                               ▼                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                                                              │   │
│  │  4e. EVENT PROCESSING (if events provided)          1 LLM    │   │
│  │                                                    per event │   │
│  │  The event interpreter receives FULL simulation context:     │   │
│  │    - current force decomposition                             │   │
│  │    - current reference price + components                    │   │
│  │    - current adoption rate                                   │   │
│  │    - category maturity                                       │   │
│  │    - top barrier and top driver                              │   │
│  │                                                              │   │
│  │  Returns per-force adjustments with mechanisms:              │   │
│  │    anchoring: {                                              │   │
│  │      new_reference_price: 73.0,                              │   │
│  │      magnitude: +0.13,                                       │   │
│  │      mechanism: "Claude Code at $200 enters reference set.   │   │
│  │        Awareness-weighted ref moves from $8.50 to ~$73.      │   │
│  │        Product now perceived as significantly below ref."    │   │
│  │    }                                                         │   │
│  │    status_quo: {                                             │   │
│  │      magnitude: +0.06,                                       │   │
│  │      mechanism: "Category legitimacy increases. 'Not using   │   │
│  │        AI tools' shifts from normal to falling behind."      │   │
│  │    }                                                         │   │
│  │    ... (all 7 forces)                                        │   │
│  │                                                              │   │
│  │  Also returns:                                               │   │
│  │    segment_effects: per income bracket reasoning              │   │
│  │    second_order_effects: things the simulation can't model   │   │
│  │      but the user should know about                          │   │
│  │                                                              │   │
│  │  Adjustments are applied to params → simulation reruns       │   │
│  │  → before/after comparison generated                         │   │
│  │                                                              │   │
│  │  Events are stackable:                                       │   │
│  │    base → +event1 → +event2 → each modifies the state       │   │
│  │                                                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ SimulationResult                                            │    │
│  │  total_adoption: 0.22 (of all agents)                       │    │
│  │  aware_adoption: 0.41 (of aware agents only)                │    │
│  │  force_decomposition: {force_name: weighted_avg} × 7        │    │
│  │  segments:                                                  │    │
│  │    by_archetype: {innovator: 0.92, early_maj: 0.24, ...}   │    │
│  │    by_income: {high: 0.30, mid: 0.24, low: 0.14}           │    │
│  │    by_proximity: {locked: 0.14, convertible: 0.29,          │    │
│  │                    unreachable: 0.57}                       │    │
│  │  event_results[]: before/after per event with per-force     │    │
│  │    reasoning                                                │    │
│  │  agent_details: full per-agent data for deep dives          │    │
│  └─────────────────────────────────────────────────────────────┘    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 5: ANALYZE                               recomputation      │
│                                                 + 1 LLM call       │
│                                                                     │
│  Input:  SimulationResult + CalibrationResult (with assumptions)    │
│  Output: AnalysisReport                                             │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                                                              │   │
│  │  5a. SENSITIVITY ANALYSIS (no LLM — pure recomputation)      │   │
│  │                                                              │   │
│  │  For every parameter with confidence < 0.8:                  │   │
│  │    Rerun simulation with param +30% and -30%                 │   │
│  │    Record adoption swing                                     │   │
│  │    Sort by swing magnitude (biggest uncertainty first)       │   │
│  │                                                              │   │
│  │  Example:                                                    │   │
│  │    category_penetration ±30% → adoption swings 13pp ⚠       │   │
│  │    switching_cost ±30% → adoption swings 9pp                 │   │
│  │    social_visibility ±30% → adoption swings 5pp              │   │
│  │                                                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                               │                                     │
│                               ▼                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                                                              │   │
│  │  5b. CONFIDENCE BAND (no LLM — computed from sensitivity)    │   │
│  │                                                              │   │
│  │  Total uncertainty = RSS of all parameter swings              │   │
│  │  Confidence band = central ± total_uncertainty/2             │   │
│  │                                                              │   │
│  │  Example: 22% (confidence band: 16% – 29%)                  │   │
│  │  Driven by: category_penetration, switching_cost             │   │
│  │                                                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                               │                                     │
│                               ▼                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                                                              │   │
│  │  5c. INTERVENTION RANKING (no LLM)                           │   │
│  │                                                              │   │
│  │  For each candidate intervention:                            │   │
│  │    - Free trial: reduces loss_aversion input by 60%          │   │
│  │    - Price cut 20%: reduces price directly                   │   │
│  │    - Annual discount: exploits hyperbolic discounting        │   │
│  │    - Social proof push: multiplies social_proof force by 2x  │   │
│  │    - Freemium tier: shifts reference price down              │   │
│  │                                                              │   │
│  │  Rerun simulation for each → rank by adoption lift           │   │
│  │                                                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                               │                                     │
│                               ▼                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                                                              │   │
│  │  5d. BEHAVIORAL AUDIT (1 LLM call)                           │   │
│  │                                                              │   │
│  │  LLM receives: force decomposition, sensitivity analysis,    │   │
│  │  confidence bands, intervention rankings, segment data,      │   │
│  │  event results, and the original assumptions list.           │   │
│  │                                                              │   │
│  │  Generates 3-5 paragraph natural language report:            │   │
│  │    1. Headline finding (adoption + biggest force)            │   │
│  │    2. Convertible pool analysis (who's within reach)         │   │
│  │    3. #1 intervention with predicted impact                  │   │
│  │    4. Event analysis (if events were injected)               │   │
│  │    5. Key uncertainty + what the user should validate        │   │
│  │                                                              │   │
│  │  Rules: name specific mechanisms, give specific numbers,     │   │
│  │  flag discovered competitors, lead with non-obvious insights │   │
│  │                                                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ AnalysisReport                                              │    │
│  │  text: behavioral audit (markdown)                          │    │
│  │  sensitivity: {param: {low, base, high, swing}} per param   │    │
│  │  confidence_band: {central, low, high, drivers[]}           │    │
│  │  interventions: ranked list with predicted lift              │    │
│  │  assumptions: challengeable list with IDs (A1, A2, ...)     │    │
│  └─────────────────────────────────────────────────────────────┘    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 6: OUTPUT + ITERATE                                          │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                                                              │   │
│  │  6a. CLI OUTPUT (Rich library)                               │   │
│  │                                                              │   │
│  │  ┌────────────────────────────────────────────────────────┐  │   │
│  │  │ MINDSIM BEHAVIORAL AUDIT                               │  │   │
│  │  │                                                        │  │   │
│  │  │ ADOPTION: 22% (confidence band: 16%–29%)               │  │   │
│  │  │ Convertible pool: 29% │ Locked: 14% │ Unreachable: 57% │  │   │
│  │  │                                                        │  │   │
│  │  │ FORCE DECOMPOSITION (convertible pool)                 │  │   │
│  │  │ prospect    ▓▓▓▓▓▓▓▓░░  -0.21  ■■■ high confidence    │  │   │
│  │  │ anchoring   ░░░▓▓▓▓░░░  +0.11  ■■■ high               │  │   │
│  │  │ status_quo  ▓▓▓▓▓▓▓▓▓▓  -0.27  ■■░ medium             │  │   │
│  │  │ social      ░░▓░░░░░░░  +0.04  ■■░ medium              │  │   │
│  │  │ fomo        ░░░▓▓░░░░░  +0.07  ■■■ high                │  │   │
│  │  │ discount    ░░░▓▓▓░░░░  -0.09  ■■■ high                │  │   │
│  │  │ identity    ░░▓░░░░░░░  +0.03  ■░░ low                 │  │   │
│  │  │                                                        │  │   │
│  │  │ SENSITIVITY                                            │  │   │
│  │  │ category_penetration  16% ──[22%]── 29%  swing: 13pp ⚠│  │   │
│  │  │ switching_cost        18% ──[22%]── 27%  swing: 9pp    │  │   │
│  │  │                                                        │  │   │
│  │  │ EVENT: Claude Code at $200/mo                          │  │   │
│  │  │ Adoption: 22% → 34% (+12pp)                            │  │   │
│  │  │ anchoring:  +0.11 → +0.24  "$20 now feels cheap"      │  │   │
│  │  │ status_quo: -0.27 → -0.21  "not using AI = outdated"  │  │   │
│  │  │ BY SEGMENT: high -4pp | mid +18pp | low +14pp          │  │   │
│  │  │                                                        │  │   │
│  │  │ INTERVENTIONS                                          │  │   │
│  │  │ #1 Free trial:    +11pp (22%→33%)                      │  │   │
│  │  │ #2 Annual billing: +5pp (22%→27%)                      │  │   │
│  │  │ #3 Social proof:   +4pp (if adoption > 15%)            │  │   │
│  │  │                                                        │  │   │
│  │  │ ASSUMPTIONS (challenge with --override)                │  │   │
│  │  │ A1 category_penetration  0.18  ■■░  "Tavily: 15-20%"  │  │   │
│  │  │ A2 switching_cost        0.50  ■■░  "free alternatives"│  │   │
│  │  │ A3 social_visibility     0.60  ■■░  "dev tools visible"│  │   │
│  │  │                                                        │  │   │
│  │  │ [behavioral audit text paragraphs]                     │  │   │
│  │  └────────────────────────────────────────────────────────┘  │   │
│  │                                                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                                                              │   │
│  │  6b. INTERACTIVE MODE (--interactive / -i)                   │   │
│  │                                                              │   │
│  │  After output, enters a command loop:                        │   │
│  │                                                              │   │
│  │  > override A1=0.30 A2=0.3                                   │   │
│  │    Re-calibrates with overridden params (skip understand     │   │
│  │    + research) → re-simulates → re-analyzes. ~4 seconds.    │   │
│  │                                                              │   │
│  │  > event "Product Hunt launch with 500 upvotes"              │   │
│  │    Interprets event with full context → shows before/after.  │   │
│  │                                                              │   │
│  │  > deep A1                                                   │   │
│  │    Shows full evidence chain for assumption A1:              │   │
│  │    - Tavily search results that informed it                  │   │
│  │    - Why 0.18 and not higher/lower                           │   │
│  │    - Sensitivity impact if wrong                             │   │
│  │                                                              │   │
│  │  > agent 472                                                 │   │
│  │    Shows full decision trace for agent #472:                 │   │
│  │    - Archetype, income, personality params                   │   │
│  │    - Each force value and why                                │   │
│  │    - Final utility and adoption probability                  │   │
│  │    - "Would convert if: free trial offered"                  │   │
│  │                                                              │   │
│  │  > compare "$15/mo with free tier"                           │   │
│  │    Runs a parallel simulation with modified product →        │   │
│  │    shows side-by-side force comparison.                      │   │
│  │                                                              │   │
│  │  > export results.json                                       │   │
│  │    Dumps full SimulationResult + AnalysisReport as JSON.     │   │
│  │                                                              │   │
│  │  > quit                                                      │   │
│  │                                                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

### LLM Strategy

```
┌─────────────────────────────────────────────────────────┐
│  LLM CLIENT                                             │
│                                                         │
│  Priority: Local first (Ollama) → API fallback          │
│                                                         │
│  Local: Qwen3-8B or Llama-4-8B via Ollama               │
│    Good enough for structured JSON extraction            │
│    No API cost. Works offline.                           │
│                                                         │
│  API: Claude Sonnet via LiteLLM (or any provider)       │
│    Better reasoning for complex event interpretation    │
│    Fallback when local unavailable                      │
│                                                         │
│  User override: --model "claude-sonnet-4-20250514"      │
│                                                         │
│  All prompts centralized in llm/prompts.py              │
│  All calls use temperature 0.1-0.2 (structured output)  │
│  except analyze (0.4, slightly creative for prose)      │
└─────────────────────────────────────────────────────────┘
```

---

### Call Budget Summary

| Stage | LLM | Tavily | Trends | Time |
|---|---|---|---|---|
| Understand | 1 | 0 | 0 | ~2s |
| Research | 0 | 2-8 (adaptive) | 1 | ~3-8s |
| Calibrate | 1 | 0 | 0 | ~3s |
| Simulate | 0 | 0 | 0 | <0.1s |
| Event (each) | 1 | 0 | 0 | ~3s |
| Analyze | 1 | 0 | 0 | ~3s |
| **Total** | **3 + 1/event** | **2-8** | **1** | **~12-16s** |

Override rerun (skip understand + research): 1 LLM call, ~4s.
With `--skip-research`: 3 LLM calls, ~8s total.

---

### File Structure

```
mindsim/
├── cli.py                          # entry point, arg parsing, interactive loop
├── pipeline/
│   ├── understand.py               # Stage 1: extract + plan research
│   ├── research.py                 # Stage 2: adaptive Tavily + Trends
│   ├── calibrate.py                # Stage 3: all params, one pass
│   ├── simulate.py                 # Stage 4: orchestrate engine
│   ├── events.py                   # Stage 4e: full-context event interp
│   └── analyze.py                  # Stage 5: sensitivity + confidence + report
├── engine/
│   ├── population.py               # archetype-based agent generation
│   ├── forces.py                   # 7 forces, vectorized NumPy
│   ├── sensitivity.py              # ±30% rerun for uncertain params
│   ├── interventions.py            # candidate interventions + rerun
│   └── archetypes.py               # Rogers archetypes + param distributions
├── models/
│   ├── product.py                  # ProductProfile dataclass
│   ├── market.py                   # MarketContext dataclass
│   ├── config.py                   # SimulationConfig dataclass
│   └── results.py                  # SimulationResult + AnalysisReport
├── llm/
│   ├── client.py                   # Ollama / LiteLLM abstraction
│   └── prompts.py                  # all system prompts, centralized
├── tools/
│   ├── tavily_client.py            # Tavily search wrapper
│   └── trends.py                   # Google Trends (pytrends) wrapper
├── config/
│   ├── defaults.yaml               # fallback params when research skipped
│   └── archetypes.yaml             # archetype definitions + distributions
├── docs/
│   ├── RESEARCH.md                 # 28 sources, full bibliography
│   └── ARCHITECTURE.md             # this document
├── requirements.txt                # numpy, rich, tavily-python, pytrends,
│                                   # ollama, litellm, pydantic
└── README.md
```

---

### Build Schedule

**Day 1: Core Engine**
- `engine/population.py` — archetype generation, NumPy arrays
- `engine/forces.py` — all 7 forces, vectorized
- `engine/archetypes.py` — Rogers params + distributions
- `models/` — all dataclasses
- Test: hardcoded SimulationConfig → forces + adoption rate

**Day 2: Pipeline Stages 1 + 3**
- `pipeline/understand.py` — product parsing LLM call
- `pipeline/calibrate.py` — parameter assignment LLM call
- `llm/client.py` — Ollama + LiteLLM abstraction
- `llm/prompts.py` — all prompts
- Test: natural language → SimulationConfig → simulation → numbers

**Day 3: MVP Ship (CLI output)**
- `cli.py` — argument parsing, pipeline orchestration
- `pipeline/analyze.py` — basic report generation (LLM call)
- Rich-formatted terminal output
- `--skip-research` mode (no Tavily, uses defaults)
- **Ship: `pip install mindsim` works, produces behavioral audit**

**Day 4: Research Integration**
- `pipeline/research.py` — adaptive Tavily + Google Trends
- `tools/tavily_client.py` + `tools/trends.py`
- Confidence evaluation heuristics
- Connect research → calibrate pipeline
- Test: full pipeline with real Tavily calls

**Day 5: Events + Sensitivity**
- `pipeline/events.py` — full-context event interpreter
- `engine/sensitivity.py` — ±30% rerun on low-confidence params
- `engine/interventions.py` — candidate interventions + ranking
- Confidence bands in output
- Assumptions table in output

**Day 6: Interactive Mode + Polish**
- Interactive command loop (override, event, deep, agent, compare)
- `deep` command — full evidence chain for any assumption
- `agent` command — individual agent decision trace
- `compare` command — side-by-side simulation
- `export` command — JSON dump

**Day 7: Demo + Launch**
- 5 pre-built demos (Cursor/Claude, Notion pricing, a novel product, a physical product, a freemium conversion)
- 60-second demo recording
- README with architecture diagram, screenshots, science citations
- Launch posts for HN, Reddit, X

---

### Known Limitations (v1)

1. **Single-round only.** No multi-round time dynamics. Events replace temporal simulation.
2. **Independent agents.** No intra-round social influence. Social proof uses prior adoption rate, not real-time cascade.
3. **WEIRD populations.** Behavioral parameters from Western experimental studies. Non-US markets may differ.
4. **Lab-to-market transfer.** All effect sizes from controlled experiments. Real markets have more noise. Treat outputs as directional insights, not point predictions.
5. **LLM interpretation uncertainty.** Event interpretation and calibration involve LLM judgment calls. Different models or prompts may produce different results.
6. **No supply-side modeling.** Assumes the product is available and functional. No production constraints or competitive response dynamics.
7. **Research depth limited by free tier.** Tavily free tier (1,000 credits/month) supports ~125-500 runs depending on research complexity.

---

### Research Foundation

28 peer-reviewed sources across 7 behavioral mechanisms. Full bibliography with per-source usage mapping in `docs/RESEARCH.md`. Every parameter in the simulation traces to a specific published finding. Key sources:

- **Decision math:** Kahneman & Tversky 1979/1992 (prospect theory), Tom et al. 2007 + Gächter et al. 2022 (λ distributions), Samuelson & Zeckhauser 1988 (status quo), Loomes & Sugden 1982 (regret theory)
- **Population structure:** Rogers 1962 (adoption archetypes), Costa & McCrae 1992 (Big Five norms), Lauriola & Levin 2001, Nicholson et al. 2005 (personality-economics correlations)
- **Architecture validation:** MIT AgentTorch / AAMAS 2025 (archetype scaling pattern), Horton 2023 (LLM behavioral validity), Chen et al. 2023 (why not pure LLM agents)