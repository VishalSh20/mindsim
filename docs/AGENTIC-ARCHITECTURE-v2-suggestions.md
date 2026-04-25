# AGENTIC-ARCHITECTURE-v2: Complete Agentic Design

Integrates ARCHITECTURE-v2-suggestions.md (layered redesign), ISSUES-v2.md (19 gaps), MECHANICS-v2-suggestions.md (competitor awareness + phase transitions), and conversation decisions on hybrid agentic pipeline. Adds Google Trends, scraping tools, and rich output artifacts (pros/cons, segment narratives, KPIs, reasoning trails, research overview).

---

## Design Philosophy

Three principles:

1. **Agents at the boundaries, math in the core.** LLMs handle language->structure (front-end) and number->narrative (back-end). Simulation stays pure NumPy.
2. **Every agent has one job.** gpt-4o-mini can't juggle multi-purpose prompts. Narrow schema per agent -> debuggable chain.
3. **Every agent emits rationale, not just results.** Rationale becomes the audit trail — "parameter=0.65 because 4/5 G2 reviews cited switching friction," not a black-box number.

---

## The 14-Agent System

- **Phase 1 (9 agents)** = calibration front-end
- **Phase 2 (0 agents)** = pure NumPy simulation
- **Phase 3 (5 agents)** = analysis back-end

---

## Phase 1: Research & Calibration

### Agent 1 — Product Understanding
- **In:** user text
- **Out:** `ProductProfile{category, value_prop, target_user_hypothesis, pricing_model, requires_behavior_change}`
- **Tools:** none
- **Why separate:** clean structured starting point for everything downstream

### Agent 2 — Research Orchestrator
- **In:** ProductProfile
- **Out:** `ResearchPlan{market_questions[], competitor_hints[], voc_targets[], trends_queries[]}`
- **Tools:** none (planner)
- **Why separate:** planning is distinct from execution; this agent allocates the research budget

### Agent 3 — Market Context (with tools)
- **In:** ResearchPlan.market + .trends
- **Tools:** Tavily (industry reports), **Google Trends API**
- **Out:** `MarketContext{size, growth_rate, maturity[nascent|growing|mainstream|saturated], trend_momentum, seasonality, geographic_heat, sources[]}`
- **Budget cap:** 4 Tavily + 3 Trends calls
- **Why Google Trends here:** makes `category_maturity` an **objective** signal (rising interest -> growing, flat -> mainstream) instead of an LLM guess. Fixes Gap 10.

### Agent 4 — Competitor Discovery + Deep-Dive (with tools)
- **In:** ProductProfile + ResearchPlan
- **Tools:** Tavily (discovery), **Scraper** (pricing pages, G2/Capterra reviews, feature pages), **Google Trends** (relative popularity across competitors)
- **Out:** `CompetitorMatrix[] = {name, price_tiers[], features[], feature_scores[], market_share_hint, trend_momentum, review_sentiment_summary, sources[]}`
- **Budget cap:** 2 discovery + 2 per competitor x max 4 competitors = 10 calls
- **Fixes:** Gap 2 (competitors are ghosts). Each competitor becomes a full feature matrix, not a name + price.

### Agent 5 — Voice of Customer (with tools) — **NEW, critical**
- **In:** ProductProfile + CompetitorMatrix
- **Tools:** Reddit API, Hacker News API, G2/Capterra scraper, App Store review scraper, Tavily (blog critiques)
- **Out:** `VoCReport{pain_points[], delight_points[], feature_sentiment{feature: {positive%, negative%, quotes[]}}, complaint_themes[], unmet_needs[]}`
- **Budget cap:** 8-10 calls
- **Why critical:** this is the ONLY agent that grounds sentiment in actual customer voice. Without it, pros/cons are LLM hallucinations. Quotes get carried through to the final report as evidence.

### Agent 6 — Dimension Definition
- **In:** ProductProfile + MarketContext + CompetitorMatrix + VoCReport
- **Out:** `FeatureDimensions[] = {name, category[core_value|cost|switching_cost|social_signal], score_for_product, certainty, visibility, time_to_realize, rationale, evidence_refs[]}`
- **Tools:** none
- **Why separate:** the critical creative task. Focused attention + full research context = right dimensions. Fixes Gap 1.

### Agent 7 — Per-Archetype Reaction (x5 parallel)
- **In:** FeatureDimensions + CompetitorMatrix + VoCReport + one archetype spec
- **Tools:** none (pure role-play)
- **Out per archetype:** `{archetype, feature_weights{dim: weight}, awareness_probability, reaction_narrative, anticipated_objections[], anticipated_motivations[], competitors_known_by_tier{primary, secondary, unaware}}`
- **Parallelize:** 5 independent calls, no dependencies
- **Cache:** generic archetype personality ("Laggards value stability") cached once; only product-specific reaction is fresh. Fixes Gap 15.

### Agent 8 — Parameter Calibrator
- **In:** everything above
- **Out:** full `SimulationParams` for NumPy engine + per-archetype overrides + **parameter_reasoning_trail** (per-param justification with evidence_refs linking back to research)
- **Tools:** none
- **Constraint:** parameters must fall within published research bounds (hard-coded from RESEARCH.md)

### Agent 9 — Validator
- **In:** full calibration bundle + all rationale trails
- **Out:** `ValidationReport{pass, issues[], confidence_score}`
- **Tools:** none
- **Checks:** parameter bounds, cross-agent consistency (e.g., maturity="saturated" but category_penetration=0.15 -> flag), sparse-rationale detection
- **On failure:** orchestrator re-prompts the specific failing agent with the issue

---

## Phase 2: Deterministic Core (no agents)

Everything from ARCHITECTURE-v2-suggestions.md:
- Population generator (with feature weights, cluster assignment, initial states, tiered competitor knowledge)
- Multi-round state machine
- Force computation with multiplicative interactions (Layer 5)
- Network propagation (cluster-local social proof)
- Sensitivity analysis

**Output:** `SimulationResult` with per-agent full trace (state history, force contributions, decision points), per-cluster dynamics, per-segment force dominance. **This is what currently gets dumped but never analyzed** — Phase 3 mines it.

---

## Phase 3: Analysis & Reporting

### Agent 10 — Segment Analyst
- **In:** SimulationResult
- **Out:** `SegmentReport{segments: [{archetype, size, adoption_rate, dominant_drivers[force, magnitude, why], dominant_blockers[], time_to_50%, representative_agents[3-5 trajectory narratives]}], segment_differences}`
- **Answers:** "what section of people like it"

### Agent 11 — Pros/Cons Synthesizer
- **In:** SegmentReport + VoCReport + ForceDominance
- **Out:** `ProsConsReport{pros: [{statement, evidence{simulation, voc}, segments[], mechanism}], cons: [...], nuances: "Pro for Late Majority is Con for Innovators (seen as cheap, not premium)"}`
- **Rule:** pros/cons require **triangulation** — both simulation data AND VoC data must support. Single-source claims go in "nuances."
- **Answers:** "pros, cons, why people like/dislike"

### Agent 12 — KPI Generator (could be pure Python — "agent" frame is for consistency)
- **In:** SimulationResult
- **Out:** `KPIDashboard` containing:
  - **Adoption:** total %, TAM penetration, time-to-50%, chasm crossing round
  - **Force dominance:** top contributing force, top blocking force, force variance across segments
  - **Convertible pool:** agents at P(adopt) in [0.4, 0.6] — the "almost" population (key strategic target)
  - **Cascade:** cluster-to-cluster spread timing, critical mass thresholds crossed
  - **Sensitivity:** top 3 parameters with highest swing
  - **Quality:** validation score, parameter confidence distribution

### Agent 13 — Intervention Designer
- **In:** all above; can invoke Phase 2 core to simulate interventions
- **Out:** ranked `InterventionList[{name, mechanism[param changes], targeted_at[segments, clusters], estimated_cost, timeline, simulated_lift, cost_per_adoption_lift, rationale}]`
- **Grounding:** each intervention is justified by (a) sensitivity analysis showing which forces move the needle, (b) VoC data showing what customers actually want, (c) simulation test result showing actual lift. Fixes Gap 12.

### Agent 14 — Narrative Audit
- **In:** everything
- **Out:** final human-readable report, sections:
  1. Executive summary (1 paragraph)
  2. **Research overview** — what was researched, key findings, sources, confidence (from Agents 3/4/5)
  3. **Parameter reasoning** — why each param has the value it does (from Agent 8's trail)
  4. Adoption simulation — curve, chasm, per-segment story
  5. **Pros & cons** (from Agent 11)
  6. **KPI dashboard** (from Agent 12)
  7. Recommended interventions (from Agent 13)
  8. Limitations & caveats (from Agent 9 + known system limits)

---

## Tool Integration

### Google Trends API — specific uses

| Use | Where | Purpose |
|-----|-------|---------|
| Interest over time | Agent 3 | Objective signal for category maturity |
| Related queries | Agent 4 | Discover competitors users also research |
| Geographic distribution | Agent 3 | Regional market sizing, cluster targeting |
| Breakout terms | Agent 4 | Emerging competitors not yet well-known |
| Search volume delta | Agent 3 | **FOMO intensity signal** — spiking interest -> higher FOMO parameter |

### Scraping Stack — recommended targets

| Target | API/Method | Agent | Value |
|--------|-----------|-------|-------|
| Reddit | Official API (free) | 5 | Pain points, unvarnished opinion |
| Hacker News | Firebase API (free) | 5 | Launch sentiment, technical critique |
| G2 / Capterra | HTML scraping | 4, 5 | Structured per-feature ratings |
| App Store / Play Store | `app-store-scraper` library | 5 | Bulk review data |
| Competitor pricing pages | BeautifulSoup extraction | 4 | Authoritative pricing structure |
| Product Hunt | Public API | 4 | Emerging competitor discovery |

**Safety:** scrapers return **structured extractions**, never raw HTML to the LLM. Prevents prompt injection and token waste.

---

## Caching Strategy

Critical for both cost and consistency:

1. **Archetype personality** — generic Rogers reactions don't change per product. Cache once.
2. **Competitor feature matrix** — VS Code's features don't change when you simulate a new product. Cache per competitor, 30-day TTL, refresh on demand.
3. **Market context** — per category (e.g., "AI coding tools"), 7-day TTL.
4. **Google Trends** — aggressive caching (data updates daily).
5. **LLM responses** — keyed on `(agent_id, input_hash)`. Same input -> same output. **Critical for reproducibility.**

A rerun of the same product ~ free. A variant test reuses ~60% of cached research.

---

## Output Artifacts (Explicit Mapping to User Needs)

| User Requirement | Artifact | Produced By |
|------------------|----------|-------------|
| Adoption metrics | KPIDashboard | Agent 12 |
| Pros of product | ProsConsReport.pros (triangulated) | Agent 11 |
| Cons of product | ProsConsReport.cons (triangulated) | Agent 11 |
| Who likes it (segments) | SegmentReport.segments | Agent 10 |
| Why they like/dislike | ProsConsReport + VoC quotes | Agents 11 + 5 |
| Reasoning behind metrics | parameter_reasoning_trail | Agent 8 -> Agent 14 |
| Research overview | Research section of final report | Agents 3/4/5 -> Agent 14 |
| KPIs over metrics | KPIDashboard | Agent 12 |

---

## Data Flow

```
User text
  |
  v
[1 Product Understanding] --> ProductProfile
  |
  v
[2 Research Orchestrator] --> ResearchPlan
  |
  +--> [3 Market Context]        (Tavily + Google Trends)
  +--> [4 Competitor Deep-Dive]  (Tavily + Scraper + Google Trends)
  +--> [5 Voice of Customer]     (Scraper + Tavily)
                                 |
                                 v
                    [6 Dimension Definition]
                                 |
                                 v
        [7 x5 Per-Archetype Reaction]  <-- parallel
                                 |
                                 v
                   [8 Parameter Calibrator] --> SimulationParams + reasoning_trail
                                 |
                                 v
                       [9 Validator] ------+
                                 |         | (on fail, re-prompt specific agent)
                                 v         |
           DETERMINISTIC NumPy CORE <------+
           Population -> Simulation -> Sensitivity
                                 |
                                 v
                    SimulationResult (rich dump — no longer wasted)
                                 |
         +----------+------------+--------------+
         v          v            v              v
  [10 Segment] [11 Pros/Cons] [12 KPI]  [13 Interventions]
                                             |  (invokes sim core
                                             |   to test each)
                                             v
                                [14 Narrative Audit] --> Final Report
```

---

## Cost Estimate (gpt-4o-mini)

| Phase | LLM calls | Tool calls | Est cost |
|-------|-----------|-----------|----------|
| Phase 1 | ~14 (9 + 5 archetypes parallel) | ~20 (Tavily + Trends + scraping) | ~$0.06 |
| Phase 2 | 0 | 0 | $0 |
| Phase 3 | 5 + ~3 sim re-runs for interventions | 0 | ~$0.02 |
| **Total per run** | **~20 LLM calls** | **~20 tool calls** | **~$0.08** |

Cached re-run: ~$0.01. Cost is not a constraint.

---

## Error Handling & Degradation

**Per-agent criticality:**
- **Critical** (pipeline stops): Agents 1, 2, 6, 8, 9
- **Degradable** (pipeline continues with note): Agents 3, 4, 5, 7, 10-13

If VoC (5) fails -> pipeline proceeds with VoC=empty -> pros/cons get weaker evidence -> final report notes the degradation explicitly. Transparency over silent failure.

**Validator triggers re-runs** on calibration drift (param outside published bounds).

**Tool budgets** prevent runaway Tavily/scraping costs.

---

## What This Gives You That v1 and Even Base-v2 Don't

1. **Grounded pros/cons** — triangulated across simulation + actual customer voice with quoted evidence
2. **Segment narratives** — "Innovators adopt because X; Late Majority blocks because Y" with representative trajectories
3. **Parameter reasoning trail** — every number explained, linked to evidence
4. **Research overview in the output** — what was searched, from where, confidence level (currently invisible)
5. **KPIs from the currently-wasted dump** — convertible pool, chasm timing, cascade spread, force variance
6. **Rationale-backed interventions** — "Early Majority is 18% below critical mass in tech_twitter cluster; Slack integration targets that cluster at $X with +8pp expected over 3 months"
7. **Google Trends grounding** — category maturity and FOMO intensity become objective signals
8. **Real competitor differentiation** — scraped feature matrices + per-archetype competitor knowledge tiers from MECHANICS-v2
9. **Cache -> cheap iteration** — testing variants of the same product is nearly free

---

## Gaps Addressed (cross-reference to ISSUES-v2)

| Gap | Addressed By |
|-----|-------------|
| Gap 1: Product is one number | Agent 6 (feature dimensions) |
| Gap 2: Competitors are ghosts | Agent 4 (scraped feature matrices) |
| Gap 3: No memory | Phase 2 state machine |
| Gap 4: No network effects | Phase 2 cluster topology |
| Gap 5: Forces should multiply | Phase 2 force interactions |
| Gap 6: Probability weighting missing | Phase 2 force computation |
| Gap 7: Anchoring double-counts | Phase 2 force restructuring |
| Gap 8: Status quo dispositional | Phase 2 state machine |
| Gap 9: No attention model | Phase 2 graded awareness |
| Gap 10: Category maturity unused | Agent 3 (Google Trends validation) + Phase 2 |
| Gap 11: No Weber's Law | Agent 6 (per-feature scoring) |
| Gap 12: Interventions no cost/time | Agent 13 (cost + timeline + sim test) |
| Gap 13: Events can't accumulate | Phase 2 state persistence |
| Gap 14: Dead parameters | Agent 8 (calibrator uses all fields) |
| Gap 15: No archetype-specific calibration | Agent 7 x5 (per-archetype reactions) |
| Gap 16: LLM confidence unvalidated | Agent 9 (validator) |
| Gap 17: Sensitivity local-only | Phase 2 sensitivity + Agent 13 interaction testing |
| Gap 18: discovered_competitors empty | Agent 4 populates |
| Gap 19: Reference price decorative | Agent 8 uses agent-specific reference (per MECHANICS-v2) |
