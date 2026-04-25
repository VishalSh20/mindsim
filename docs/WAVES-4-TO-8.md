# v2-middle Roadmap: Waves 4 through 8.5

Status at time of writing: Waves 0-3 shipped on branch `v2-middle/wave-0`,
HEAD at `7681079` (Wave 3 [3/3]). Wave 3.5 (revert cooldown + archetype
threshold multipliers) was rolled back after end-to-end testing. See
§Open-from-Wave-3 below for the compressed-Rogers-spread issue that
remains and how later waves may address it.

This document is the single source of truth for remaining work. The
overall plan lives in `~/.claude/plans/based-on-these-devise-modular-finch.md`
— this file is its public-facing, repo-versioned equivalent scoped to
the work that hasn't shipped yet.

---

## Open from Wave 3

Two known rough edges carried forward:

1. **Rogers adoption spread is too compressed in multi-round.** For a
   strong product, current output shows innovator 96% / laggard 71%
   (25pp gap) vs the published literature's ~60-80pp. The multi-round
   revert-to-AWARE mechanism gives every aware agent roughly N
   independent adoption draws over N rounds, which collapses the
   archetype curve. Wave 3.5 attempted to fix this with cooldown + per-
   archetype thresholds but was reverted for producing its own
   over-corrections. Will be revisited inside Wave 4 with a more
   conservative approach (see §Wave 4 §Refining the spread).

2. **ProximitySegment display (Locked/Convertible/Unreachable) is now
   computed on the decision-phase holdouts** rather than the full
   population, so a successful run shows "Locked: 0%" when what the
   user really wants to see is "how many eventually adopted". Fixed
   cosmetically in Wave 8's report overhaul.

3. **Interventions section stays disabled** until Wave 7 rewrites it
   as phase mechanisms.

---

## Wave 4 — Clusters + local social proof

**Goal:** replace the global `product_adoption_rate` scalar that Wave 3
uses as a word-of-mouth proxy with **cluster-local** adoption rates, so
social proof is computed in terms of an agent's actual network rather
than a population-wide average. Unlocks "Slack-style" dynamics where a
product is useless below cluster critical mass and indispensable above
it.

### In scope

- **`engine/clusters.py` (new)** — cluster assignment logic and
  cluster-local adoption computation.
- **`Cluster` Pydantic model** on `MarketContext` (already scaffolded
  in Wave 0 as a list of `dict` placeholders; promote to typed model).
- **Cluster assignment in population generation** — deterministic
  fallback scheme: archetype × income tercile produces 6 clusters. Real
  scraped communities arrive in Wave 5.
- **Cluster-local social proof** in `forces.py` — replace the `spn *
  log(1 + product_adoption × social_visibility)` with a per-cluster
  rate: each agent's social proof depends on adoption **within their
  cluster**, weighted by the cluster's visibility.
- **`RoundSnapshot.cluster_adoption`** — populated each round (already
  a field from Wave 3; gets filled in now).
- **Cluster influence on WOM in `promote_unaware_to_aware`** — agents
  are more likely to become aware if their cluster already has
  adopters, replacing the current global scalar.

### Out of scope

- Scraping-driven cluster membership (Wave 5).
- Full graph simulation — v2-middle stays at ≤ 6 clusters with cross-
  links, not per-agent network topology.
- Refinement of the Rogers spread issue above may be addressed here as
  a side-effect of cluster-local social proof being weaker at low
  cluster penetration (which delays laggards naturally).

### Files created / modified

**New**
- `backend/src/mindsim/engine/clusters.py` — `Cluster` dataclass,
  `assign_clusters(agents, archetype_ids, incomes)`,
  `compute_cluster_adoption_rates(agents)`, `cluster_visibility_for(cluster_id)`.
- `backend/tests/test_clusters.py` — assignment determinism, rate
  aggregation correctness, critical-mass emergence.

**Modified**
- `backend/src/mindsim/engine/population.py` — call `assign_clusters`
  in `generate_population` and write `cluster_id` for each agent.
- `backend/src/mindsim/engine/forces.py` — social proof formula uses
  cluster-local rate; reads `agents["cluster_id"]` and a pre-computed
  `cluster_rates` array passed through.
- `backend/src/mindsim/engine/state_machine.py` — `promote_unaware_to_aware`
  uses cluster-local rate for the WOM term.
- `backend/src/mindsim/pipeline/simulate.py` — compute cluster rates
  once per round, pass through to force computation.
- `backend/src/mindsim/models/market.py` — typed `Cluster` model;
  attach to `MarketContext.clusters`.

### Exit criteria

- Two-cluster topology test: product with all adoption in cluster A
  produces stronger social proof for cluster-A agents than cluster-B
  agents.
- Cluster-local rates shown in `RoundSnapshot.cluster_adoption` for
  each round.
- Existing 220 tests still pass; no regression in single-cluster
  fallback path.

### Refining the Rogers spread (opportunistic during Wave 4)

The compressed-spread issue from Wave 3 partially eases once cluster-
local social proof kicks in: laggards sit in low-adoption clusters
early (by assignment), so their social proof stays near zero even as
innovator clusters heat up. This naturally delays late-archetype
consideration. If after Wave 4 testing the spread is still ≤ 30pp on
strong products, a targeted revisit of the Wave 3.5 ideas (cooldown
only, no archetype-threshold mult) is scheduled as a Wave 4.5 patch.
The goal there is minimally-invasive: the failure mode of full 3.5 was
that we over-constrained two mechanisms at once.

### Risks

- **Cluster assignment determinism with small populations** — 1000
  agents split across 6 clusters gives ~170 each; statistical noise
  matters. Tests run with 2000+ agents where possible.
- **Cluster visibility calibration** — pulling a number out of the air
  for each cluster. Defer to scraped data in Wave 5; use sensible
  defaults in Wave 4 (tech_twitter 0.8, enterprise_IT 0.3, etc.).

---

## Wave 5 — Scrapers live + VoC + A2/A3 agents + tiered competitors

**Goal:** replace all scalar-only research with grounded data.
Scrapers fetch real customer voice; A2 synthesises market context from
Tavily + scraped docs; A3 analyses VoC with bias disclosure; each agent
gets a *tiered* view of competitors rather than a single reference
price.

### In scope

- **Real HTTP scrapers** — the Wave 0 stubs in `scrape/reddit.py`,
  `scrape/hackernews.py`, `scrape/pricing.py` become live.
  - Reddit via public JSON endpoints (`reddit.com/r/X/search.json`, no
    auth). Rate limiter enforces 1 req/s.
  - Hacker News via Algolia (`hn.algolia.com/api/v1/search`).
  - Pricing pages via `httpx` + `beautifulsoup4` with per-domain
    extractors (registry of known pricing URL patterns).
- **A3 VoC Analyst** — new agent, reads scraped documents → emits
  `VoCReport{pain_points, delight_points, feature_sentiment, bias_note}`.
  The `bias_note` is set whenever the corpus is Reddit/HN-dominant to
  disclose "weighted toward innovators/early adopters".
- **A2 Research Synthesizer** — new agent, consumes Tavily + scraped
  docs → produces `MarketContext` + `CompetitorMatrix` with real
  `feature_scores` per competitor (Wave 2 emitted placeholders).
- **Tiered competitor knowledge** per MECHANICS-v2:
  - Each agent has primary / secondary / unaware competitors.
  - Primary drawn by market-share weight (scraped), secondary by
    cluster visibility × archetype breadth.
  - Per-agent reference price formula:
    `0.6 × primary.price + 0.3 × mean(secondary.price) + 0.1 × category_default`.
- **Maturity heuristic replaces pytrends** — `engine/maturity.py`
  gains `classify_from_signals(n_competitors, review_volume,
  category_age_hint)` returning `nascent|growing|mainstream|saturated`.
- **Remove `pytrends` dep** from `pyproject.toml`; delete `tools/trends.py`.
- **Stable evidence IDs** — A3 VoC emits every quote with a
  deterministic `quote_id = "voc:<hash>"`; A2 research synthesizer
  emits every source snippet with `source_id = "src:<hash>"`. IDs are
  the anchor Wave 8.5 uses to build the provenance trail, so this is
  a cheap addition here rather than a retrofit later. No new files —
  just ID discipline baked into the A2/A3 output contracts.

### Out of scope

- G2/Capterra/AppStore scraping — deferred to v2.2. Documented as
  known limitation in the final report.
- LLM-driven cluster discovery — clusters stay at the fallback scheme
  from Wave 4 unless a subreddit-driven discovery agent is explicitly
  scoped.

### Files created / modified

**New**
- `backend/src/mindsim/pipeline/voc.py` — A3 Voice-of-Customer analyst
  stage.
- `backend/src/mindsim/pipeline/research_synthesizer.py` — A2 stage.
- `backend/src/mindsim/llm/prompts.py` — `VOC_PROMPT`, `SYNTHESIZER_PROMPT`.
- `backend/src/mindsim/engine/competitor_tiers.py` — per-agent tier
  assignment + reference-price derivation.
- `backend/tests/test_scrape_live.py` — HTTP fixtures (requests_mock
  or httpx respx), rate-limiter enforcement, robots.txt honour,
  budget enforcement.
- `backend/tests/test_voc_agent.py`, `test_synthesizer_agent.py`,
  `test_competitor_tiers.py`.

**Modified**
- `backend/src/mindsim/scrape/reddit.py`, `hackernews.py`, `pricing.py`
  — stubs become real HTTP (currently return empty / None).
- `backend/src/mindsim/pipeline/research.py` — emit scraped docs
  alongside Tavily results; drop pytrends.
- `backend/src/mindsim/engine/population.py` — tier assignment at
  population gen; stamp per-agent primary/secondary competitor IDs.
- `backend/src/mindsim/engine/forces.py` — agent_ref_price reads from
  tier-derived per-agent values rather than the scalar reference.
- `backend/src/mindsim/models/market.py` — `VerifiedCompetitor.feature_scores`,
  `VerifiedCompetitor.market_share` populated.
- `backend/pyproject.toml` — remove `pytrends`.
- `backend/src/mindsim/tools/trends.py` — deleted.

### Exit criteria

- Canned HTTP fixtures for Reddit / HN produce a non-empty VoC corpus
  on a test product.
- Maturity classification agrees with manual judgment on 3 canonical
  products (AI coding tool → growing; habit tracker → mainstream;
  brain-computer interface consumer device → nascent).
- Tiered competitor assignment: distribution of primary competitors
  matches market-share weights within ±5pp on 2000-agent runs.
- Agent-specific reference price produces measurable adoption
  differences between innovators (who know multiple competitors) and
  laggards (who anchor to dominant / free option).
- Every VoC quote and research source emitted by A2/A3 carries a
  stable, deterministic ID that round-trips through `MarketContext`
  and remains resolvable downstream.
- All existing tests pass; `pytrends` no longer appears in `uv.lock`.

### Risks

- **Scraping fragility.** Reddit JSON format is stable but rate-limited
  aggressively. Graceful degradation: if a scrape fails, A2/A3 proceed
  with empty corpus and the final report flags reduced confidence.
- **VoC sample bias** — Reddit/HN skew toward innovators/early
  adopters. The `bias_note` on `VoCReport` is a transparency mitigation,
  not an elimination. Narrative author in Wave 8 must surface it when
  pros/cons lean on VoC.
- **Cost creep** — scraping + two new agents pushes the per-run LLM
  count up to the full 7. Budget test in the manifest must catch if
  a run exceeds 2× the baseline.

---

## Wave 6 — Events + session state

**Goal:** events stop regenerating the population. An agent who adopted
last round and now sees "competitor launches at half price" retains
their state, may churn, and carries their trajectory into subsequent
rounds. Enables multi-event narratives.

### In scope

- **`--session <path>` CLI flag** — serialises agent array (`.npz`)
  plus run manifest (JSON) at end of run; loads at start of next run.
- **Event pipeline rewrite** — `pipeline/events.py` mutates loaded
  state rather than calling `simulate()` from scratch. Per-event
  sub-loop runs 2-3 additional rounds on the persisted population.
- **A6 Event Interpreter prompt** expanded — receives per-segment
  trajectory summary (what agents were doing pre-event) rather than a
  snapshot, so it can reason about "this event affects agents who have
  already adopted differently from agents still considering".
- **Session manifest** — each session file tracks `run_id`, parent
  `run_id`, event sequence, accumulated LLM calls. Cross-session
  auditability.

### Out of scope

- Multi-product simulations in a single session (v2.x).
- Branching / alternate-history scenarios.

### Files created / modified

**New**
- `backend/src/mindsim/pipeline/session.py` — `save_session`,
  `load_session`, dtype-compatible `.npz` serialisation with schema
  version.
- `backend/tests/test_session_roundtrip.py`, `test_events_accumulate.py`.

**Modified**
- `backend/src/mindsim/cli.py` — `--session`, `--session-out` flags.
- `backend/src/mindsim/pipeline/events.py` — mutate-in-place path;
  rounds within an event respect state machine.
- `backend/src/mindsim/llm/prompts.py` — `EVENT_PROMPT` extended.

### Exit criteria

- Canonical test: "Product A launches → Product A gets bad press"
  produces FEWER final adopters than the reverse order, because the
  second event arrives after some agents have invested in A and are
  now locked-in.
- Session round-trip is byte-compatible: saving then loading
  reproduces exactly the same agent array (traits + state).
- Schema version in session file — mismatched versions refuse to load
  with a clear error.

### Risks

- **NumPy dtype stability across versions.** Session files are only
  portable across matching AGENT_DTYPE layouts. Version the dtype and
  reject stale files.
- **State-at-event-time ambiguity** — when does the event "apply"?
  Between rounds is the clean answer; document it.

---

## Wave 7 — Interventions v2 (phase mechanisms, cost + timeline)

**Goal:** delete the Wave 3-disabled `DEPRECATED_V1_INTERVENTIONS` and
replace them with explicit phase mechanisms. Free trial is a
`considering → trialing` edge with `price=0` for trial rounds, not a
parameter hack. Every intervention carries cost and timeline. Ranking
by **cost-per-adoption-lift**, not raw lift.

### In scope

- **Trial phase wiring** — TRIALING phase (already in `Phase` enum
  since Wave 3) gets its transitions:
  - `considering → trialing` when a free trial is active.
  - `trialing → adopted` or `trialing → quit` based on `experienced_value`
    vs expectations, plus Cialdini consistency pressure (+0.15 to
    prospect value) and sunk-cost (`trial_rounds × 0.1`).
  - `trial_outcome` field (already in dtype) gets written; future
    churn can reference it.
- **Intervention schema rewrite** — `InterventionSpec` gains
  `cost_usd`, `timeline_rounds`, `mechanism_type` fields.
- **Five rewritten interventions:**
  1. **Free trial** — phase mechanism, not price=0 hack. Cost = server
     costs × trial users × duration.
  2. **Price cut** — modifies `price` in `SimulationParams`; cost =
     revenue per user × user count × rounds.
  3. **Annual discount** — cost-free but modifies `present_bias_beta`
     upward; timeline = 1 round.
  4. **Social proof push** — modifies marketing_reach upward for N
     rounds; cost = ad spend.
  5. **Freemium tier** — changes product_matrix to add a zero-price
     feature option; persists in the simulation state.
- **Cost-per-lift ranking** — `InterventionResult` gains `cost_usd`,
  `timeline_rounds`, `cost_per_adoption_pp`. Primary ranking key is
  cost_per_adoption_pp; secondary key is total lift.
- **Pairwise combination test** — top-2 single interventions tested
  together (4 total: A alone, B alone, A+B, baseline). Reports
  sub-additive / super-additive behaviour.

### Out of scope

- Adaptive interventions that react mid-simulation (v2.x).
- External-data-driven cost estimation — costs come from the LLM with
  a range, not live pricing APIs.

### Files created / modified

**Modified**
- `backend/src/mindsim/engine/interventions.py` — full rewrite; old
  specs in `DEPRECATED_V1_INTERVENTIONS` deleted.
- `backend/src/mindsim/engine/state_machine.py` — trial-phase
  transitions activate when trial intervention is armed.
- `backend/src/mindsim/models/results.py` — extended
  `InterventionResult` schema.
- `backend/src/mindsim/pipeline/analyze.py` — ranking logic.
- `backend/src/mindsim/cli.py` — display block for the INTERVENTIONS
  section returns (was removed in Wave 3.1).

**New**
- `backend/tests/test_interventions_v2.py` — free-trial positive lift
  for delayed-benefit products (fixing the Wave 1/2 regression),
  cost-per-lift ranking correctness, pairwise sub-additivity detection.

### Exit criteria

- Free trial on a delayed-benefit product (habit tracker archetype)
  produces POSITIVE adoption lift, not the negative lift seen in Wave
  1/2. This is the headline fix.
- Intervention ranking by cost-per-lift surfaces cheap-but-high-lift
  options above expensive-but-higher-lift ones.
- `DEPRECATED_V1_INTERVENTIONS` list is gone.

### Risks

- **Cost estimates are calibrated by the LLM** and will be noisy.
  Document the uncertainty; expose a range rather than a point value.
- **Trial-phase transitions interact with Wave 3's cooldown mechanism**
  — an agent who trials and quits shouldn't immediately re-enter
  TRIALING. Explicit cooldown for quit state.

---

## Wave 8 — KPI dashboard, triangulated pros/cons, final-report overhaul

**Goal:** the SimulationResult and scraped VoC corpus stop being under-
consumed. KPIs are mined in pure Python; pros/cons are enforced-
triangulated in code (not just prompted); the narrative author's prompt
is expanded to surface all new artifacts; the CLI terminal output and
the dumped report get a publication-ready overhaul.

### In scope

- **`pipeline/kpi.py` (new)** — pure-Python KPI extraction from
  `SimulationResult`:
  - Adoption summary (total, aware, time-to-50%, chasm round).
  - Force dominance (top driver, top blocker, segment variance).
  - **Convertible pool count** (the strategic lever).
  - Cascade metrics (first cluster to cross critical mass, spread
    rounds — requires Wave 4 cluster data).
  - Top-3 sensitivity parameters.
  - Validation score from A5 (Wave 2).
- **`pipeline/segments.py` (new)** — per-archetype dominant drivers /
  blockers / time-to-50% / representative 3-5 trajectory narratives
  (using `RoundSnapshot` data from Wave 3).
- **Triangulated pros/cons enforced in code.** Each `ProsConsItem`
  (scaffolded in Wave 0) requires both `simulation_evidence` AND
  `voc_evidence`. Single-source claims go to `polarity="nuance"`.
  The narrative author receives triangulated items only.
- **A7 Narrative Author prompt expanded** — receives the KPI,
  segments, pros/cons, and run manifest. The final report has new
  sections: research overview, parameter reasoning trail, pros & cons
  with quoted evidence, KPI dashboard, rationale-backed interventions,
  limitations.
- **Manifest surfaced in terminal** — stage timings, LLM call counts,
  cache hit ratio, cost visible at end of run (as envisioned in §7.1
  of the overall plan).
- **ProximitySegment semantics corrected** for multi-round — relabel
  or replace with "initial / in-flight / decided / holdout" breakdown
  that makes sense given agents have moved through phases.

### Out of scope

- Frontend / web UI. Project remains CLI-only until a separate effort.
- Interactive deep-dive command loop changes beyond what's already
  there.

### Files created / modified

**New**
- `backend/src/mindsim/pipeline/kpi.py`
- `backend/src/mindsim/pipeline/segments.py`
- `backend/src/mindsim/pipeline/pros_cons.py` — triangulation enforcer.
- `backend/tests/test_kpi.py`, `test_segments.py`, `test_pros_cons.py`.

**Modified**
- `backend/src/mindsim/pipeline/analyze.py` — orchestrates KPI,
  segments, pros/cons alongside existing sensitivity + interventions.
- `backend/src/mindsim/llm/prompts.py` — expanded `ANALYZE_PROMPT` for A7.
- `backend/src/mindsim/cli.py` — terminal display overhaul; new
  sections, manifest footer, corrected proximity breakdown.

### Exit criteria

- Pros/cons with only simulation evidence (no VoC support) are routed
  to the `"nuance"` bucket and labelled as such in the final report.
- KPI dashboard populates for a standard run without missing fields.
- Warm-cache rerun of a canonical product completes in <5 s
  (cache from Wave 0 now exercised across the full pipeline).
- Final report contains all 8 sections and reads as a unified audit
  document, not a collection of tables.

### Risks

- **A7 prompt length.** The report sections multiply the payload.
  Keep each artifact compact; the narrative author summarises rather
  than quoting verbatim.
- **Triangulation rule may be too strict** for new categories with no
  scraped VoC yet. Flag such products explicitly in the final report's
  "limitations" section rather than silently degrading.

---

## Wave 8.5 — Provenance & Moat Surface

**Goal:** turn the evidence data produced in Waves 2 / 5 / 8 into a
first-class output surface. Every number in the final report traces
to resolvable evidence; every pros/cons claim declares its evidence
strength; a Validator agent enforces completeness before the narrator
runs. This is the wave that makes the triangulation design visible
(not just internally enforced) and makes the provenance chain
navigable end-to-end — the two pillars the CLI / forthcoming React
frontend rely on to separate this product from a generic LLM-written
SWOT.

### In scope

- **`models/evidence.py` (new)** — unified `Evidence` record and
  `EvidenceStore`.
  - `Evidence{id, text, source_type: voc|research|trends|default|llm_judgment,
    source_url: str | None, archetype_attribution: str | None, confidence: float}`.
  - `EvidenceStore{by_id: dict[str, Evidence]}` attached to
    `SimulationResult`. Populated from the stable IDs Wave 5 emits.
  - Compression rule: only keep quotes / sources actually referenced
    by at least one downstream artifact (params, features,
    ProsConsItems). Unreferenced evidence is dropped before
    serialisation so the store doesn't bloat the dump.
- **Extend `Assumption`** (`models/config.py`) with:
  - `evidence_refs: list[str]` — links into `EvidenceStore`.
  - `source_type: Literal["voc", "research", "default", "llm_judgment"]`.
  - `published_bounds: tuple[float, float] | None` — validator
    target range, populated from RESEARCH.md for parameters with
    published ranges.
- **Extend `CalibratedParam`** with optional
  `evidence_refs: list[str]` (empty default preserves every v1 and
  Wave-0..8 code path; nothing else needs to change).
- **Extend `ProsConsItem`** (`models/results.py`) with
  `evidence_strength: {n_sim: int, n_voc: int, class:
  "triangulated" | "nuance" | "weak"}`. Classification is mechanical:
  `n_sim >= 1 AND n_voc >= 1` → triangulated; exactly one side → nuance;
  zero → weak (should not reach the narrator; validator blocks).
- **A8 Calibrator prompt updated** to emit `evidence_refs` alongside
  `basis` for every parameter it sets. The existing `basis` string
  stays — this is additive.
- **A9 Validator agent (new, `pipeline/validator.py`)** — runs after
  calibrate, before simulate. Emits
  `ValidationReport{pass, issues[], confidence_score}`.
  - Param values within `published_bounds`.
  - Every `evidence_refs` entry resolves in `EvidenceStore`.
  - Sparse-rationale detection — `basis < 10 chars` OR zero
    evidence refs on a non-default parameter.
  - Cross-agent consistency — e.g., `maturity="saturated"` paired
    with `category_penetration < 0.2` flags.
  - On failure: orchestrator re-prompts the specific upstream agent
    with the issue. Hard cap: 2 retries per agent, then the run
    continues with the issue recorded on the final report.
- **`pipeline/provenance.py` (new)** — pure Python assembly of
  `ParameterReasoningTrail` from `CalibratedParam` +
  `Assumption` + `EvidenceStore`. One entry per headline parameter,
  structured as `{param, value, basis, evidence: [resolved Evidence
  records]}`. No LLM call.
- **A7 Narrative Author prompt** gains two explicitly-named sections
  in its output:
  - **"Evidence & Reasoning"** — one sentence per headline number,
    each citing its evidence_refs.
  - **"Triangulated Claims"** — pros/cons grouped by
    `evidence_strength.class`. Nuances get a dedicated sub-heading
    ("One-sided signal — treat as directional, not load-bearing").
- **CLI display** — each KPI line in the terminal output gains a
  compact `[cite: q7, src3]` footer; new `--show-evidence` flag
  prints the fully-expanded `EvidenceStore`.

### Out of scope

- React / web frontend rendering. Wave 8.5 produces structured JSON
  and terminal output; the frontend effort consumes them.
- Automated evidence quality scoring beyond counts
  (source-reliability weighting, cross-quote agreement scoring).
  Deferred to v2.1.
- Retroactive evidence for parameters calibrated from defaults (when
  `--skip-research` is on). These stay labelled as
  `source_type="default"` with no refs, and the report surfaces that.

### Files created / modified

**New**

- `backend/src/mindsim/models/evidence.py` — `Evidence`,
  `EvidenceStore`.
- `backend/src/mindsim/pipeline/provenance.py` —
  `ParameterReasoningTrail` assembly.
- `backend/src/mindsim/pipeline/validator.py` — A9 agent + validator
  rule set.
- `backend/tests/test_evidence_roundtrip.py`,
  `test_provenance_trail.py`, `test_validator.py`.

**Modified**

- `backend/src/mindsim/models/config.py` — `Assumption.evidence_refs`,
  `Assumption.source_type`, `Assumption.published_bounds`,
  `CalibratedParam.evidence_refs`.
- `backend/src/mindsim/models/results.py` —
  `ProsConsItem.evidence_strength`, `SimulationResult.evidence_store`.
- `backend/src/mindsim/llm/prompts.py` — `CALIBRATE_PROMPT` updated
  to emit `evidence_refs`; new `VALIDATOR_PROMPT`; `ANALYZE_PROMPT`
  (A7) extended with the two named sections.
- `backend/src/mindsim/pipeline/analyze.py` — orchestrates
  provenance + validator alongside existing KPI / segments /
  pros-cons pipeline.
- `backend/src/mindsim/pipeline/calibrate.py` — populates
  `published_bounds` from a RESEARCH.md-derived constants table.
- `backend/src/mindsim/cli.py` — evidence citations on KPI lines;
  `--show-evidence` flag; validation issues rendered in a warnings
  banner when present.

### Exit criteria

- Every parameter in the final report has at least one resolvable
  `evidence_ref` OR is explicitly marked
  `source_type="default"` ("no research — prior-only value").
- Validator catches out-of-bounds parameters on a deliberately
  broken fixture and blocks narrative generation until the
  calibrator retry succeeds or the 2-retry cap is hit.
- `ProsConsItem.evidence_strength.class` is computed for every item;
  claims classified `"nuance"` render in a dedicated subsection in
  the final report, not mixed with triangulated claims.
- At least one ProsConsItem on a canonical-product run reaches
  `"triangulated"` — proves the end-to-end chain fires (VoC quote →
  stable ID → EvidenceStore → ProsConsItem → narrative).
- Final report contains distinct `"Evidence & Reasoning"` and
  `"Triangulated Claims"` sections with resolved citations.
- `EvidenceStore` after compression contains only referenced
  records; dump size is bounded.
- All existing tests still pass.

### Risks

- **Validator re-prompt loop.** Hard cap at 2 retries per agent,
  then continue with the issue recorded. Prevents runaway cost /
  latency on a stubborn calibration.
- **Evidence bloat.** The compression rule (drop unreferenced
  evidence at serialisation) is the mitigation; test it explicitly
  with a high-volume VoC fixture.
- **Default parameter paths** (skip-research, nascent categories
  with no VoC) are evidence-less by construction. Design choice:
  label them transparently rather than fabricate refs. A7 must
  surface this as a limitations-section bullet when it's load-
  bearing for a headline number.
- **Prompt-injection surface.** Scraped quotes carried verbatim
  through the pipeline into the narrative author's prompt could
  contain adversarial instructions. Mitigation: A3 already extracts
  structured fields rather than raw HTML (per AGENTIC-ARCHITECTURE
  safety note); add a strip step on `Evidence.text` before
  serialisation and clamp to 200 chars per quote.

---

## Wave-by-wave dependency graph

```
Wave 3 (shipped) ──┐
                   ├──► Wave 4 (clusters)
                   │     │
                   │     ▼
                   ├──► Wave 5 (scrapers + A2/A3 + tiered competitors
                   │     │       + stable quote_id / source_id)
                   │     │  needs Wave 4 clusters for community mapping
                   │     ▼
                   ├──► Wave 6 (session state + accumulating events)
                   │     │  orthogonal to W4/W5; just needs state machine from W3
                   │     ▼
                   ├──► Wave 7 (interventions v2)
                   │     │  needs Wave 3 phase machinery; Wave 6 adds richness
                   │     ▼
                   ├──► Wave 8 (KPI + pros/cons + final report)
                   │     │  needs: Wave 5 VoC (for triangulation),
                   │     │         Wave 4 cluster data (for cascade KPIs),
                   │     │         Wave 7 intervention cost/timeline (for ranking)
                   │     ▼
                   └──► Wave 8.5 (provenance + moat surface)
                         needs: Wave 5 stable IDs (for EvidenceStore),
                                Wave 8 ProsConsItem + KPI pipeline,
                                (A9 Validator gates re-entry into calibrate)
```

Waves 4 and 6 can in principle run in parallel (disjoint file sets).
Waves 5, 7, 8, 8.5 each have hard dependencies on earlier waves as
shown.

---

## Testing discipline across waves

Every wave ships:

1. **Unit tests** for each new module, with numerical assertions (not
   sign-only). This is a conscious regression on ISSUES.md §Test gaps.
2. **Integration tests** that exercise the happy path end-to-end for
   the features the wave added, even if they rely on fixtures for
   external HTTP.
3. **Regression guard** — the full previous test suite still passes
   with no skips or xfails added.
4. **Manual smoke** — user runs the CLI end-to-end on a canonical
   product fixture (`AI coding assistant` for most waves; `AI tutor`
   when status-quo mechanics matter; `useless app` as the zero-adoption
   sanity check).

Numerical test anchors are allowed to drift between waves as mechanics
evolve, but the drift MUST be explained in the commit message.

---

## One-line per-wave summary

| Wave | Headline |
|------|----------|
| 4 | Social proof becomes cluster-local; Slack-like critical-mass dynamics emerge. |
| 5 | Scrapers + real VoC + per-agent tiered competitor knowledge replace scalar research; stable evidence IDs anchor the provenance chain. |
| 6 | Events stop regenerating the population; trajectories accumulate across CLI invocations. |
| 7 | Free trial becomes a phase transition, not a parameter hack. Interventions carry cost + timeline. |
| 8 | KPIs, triangulated pros/cons, unified final report consume everything the simulation has been producing. |
| 8.5 | Provenance chain is navigable end-to-end; A9 Validator gates narrative; triangulated moat is a named, first-class output surface. |
