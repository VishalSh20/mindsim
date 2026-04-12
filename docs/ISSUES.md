# mindsim — Pipeline Analysis & Known Issues

Comprehensive audit of implementation vs ARCHITECTURE.md and RESEARCH.md.
Issues are categorized by severity: **FIXED** (patched in this pass), **HIGH** (formula/model wrong), **MEDIUM** (deviations from docs), **LOW** (minor).

---

## FIXED in this pass

### FIX-1: Hyperbolic discounting was scalar, not per-agent
**File:** `engine/forces.py:206-211`
**Problem:** Force 6 computed a single float and broadcast it to all agents via `np.full()`. Frederick et al. 2002 (cited in RESEARCH.md) explicitly says "discount rates vary enormously across individuals" and justifies per-agent variation.
**Fix:** Added per-agent present-bias modulation based on openness. High-openness agents (novelty seekers) get weaker present bias (β closer to 1.0), low-openness agents get stronger present bias. Produces ±15% variation around the calibrated β.

### FIX-2: Free trial intervention didn't reduce price
**File:** `engine/interventions.py:31-36`
**Problem:** The "Free trial" intervention only boosted `perceived_benefit` by 1.3× and `benefit_certainty` by 1.4×, but never touched price. RESEARCH.md says free trials "break the multiplicative penalty by removing the immediate cost."
**Fix:** Free trial now sets `price=0` during simulation, which eliminates the loss component of prospect value entirely. Also still boosts benefit_certainty by 1.4× (users experience value before paying).

### FIX-3: research.py crash when category_growth is None
**File:** `pipeline/research.py:94`
**Problem:** `f"(growth: {context.category_growth:.2f})"` would raise `TypeError` if `get_category_trends()` returned no `growth_rate` key.
**Fix:** Conditional formatting with "N/A" fallback.

### FIX-4: calibrate.py always showed price as "/mo"
**File:** `pipeline/calibrate.py:77`
**Problem:** Price always formatted as `$X/mo` regardless of `billing_period`. A product at "$120/year" would be sent to the LLM as "$120/mo", causing massively wrong calibration.
**Fix:** Now reads `billing_period` and formats as `/mo`, `/yr`, or `one-time` accordingly.

### FIX-5: category_penetration displayed as raw float to LLM
**File:** `pipeline/calibrate.py:105`
**Problem:** Penetration of 0.18 displayed as "Category penetration: 0.18" — ambiguous to the LLM (could be read as 0.18% or 18%).
**Fix:** Now formats as "18.0%".

### FIX-6: reference_price excluded from sensitivity analysis
**File:** `engine/sensitivity.py`
**Problem:** `reference_price` was not in `SENSITIVE_PARAMS` despite having default confidence of 0.3 and being a key driver of the anchoring force.
**Fix:** Added special-case handling for `ReferencePriceParam` in the sensitivity loop.

### FIX-7: Confidence score inflation from `.com` in authority domains
**File:** `pipeline/research.py:137`
**Problem:** `".com"` in the authority domains list meant virtually every URL scored as "authoritative", inflating confidence scores for all search results.
**Fix:** Replaced with specific authoritative domains (`techcrunch.com`, `crunchbase.com`, `g2.com`, etc.) and changed `"pricing"` to `"/pricing"` (path match, not substring).

### FIX-8: Intervention price handling was brittle
**File:** `engine/interventions.py:99-111`
**Problem:** Price modification was a hardcoded `if intervention.name == "Price cut 20%"` check. Adding new price-related interventions required editing the dispatch logic.
**Fix:** Interventions now use `price_zero: True` and `price_factor: 0.8` as generic modification keys that the application logic handles cleanly.

---

## HIGH severity — formula/model discrepancies vs research docs

### HIGH-1: Loss aversion × Hyperbolic discounting should multiply, not add
**File:** `engine/forces.py` (compute_decisions, line 248)
**RESEARCH.md says:** "These multiply, not add, for products with upfront cost + delayed benefit (habit trackers, education tools, enterprise software)."
**Implementation:** All 7 forces are summed: `total_utility += np.nan_to_num(force_arr, nan=0.0)`. There is no multiplicative interaction.
**Impact:** Products with upfront cost + delayed benefit (the exact category RESEARCH.md calls out) are not penalized as heavily as the model intends. This affects intervention recommendations — the simulation may under-recommend free trials for habit-change products.
**Suggested fix:** For products where `requires_behavior_change > 0.5` and `time_to_value > 0.3`, compute `prospect_value * hyperbolic_discounting` as a joint term instead of adding them.

### HIGH-2: Probability weighting function (γ=0.61) completely unimplemented
**Source:** Tversky & Kahneman 1992 (cited in RESEARCH.md)
**RESEARCH.md says:** "the probability weighting function for how agents perceive uncertain benefits — people overweight small probabilities and underweight large ones."
**Implementation:** Not present anywhere in the codebase. The `benefit_certainty` parameter approximates this but is not the same mechanism.
**Impact:** Agents evaluate uncertain benefits linearly rather than with the documented S-shaped probability weighting.

### HIGH-3: Prospect value loss function diverges from canonical K&T formula
**File:** `engine/forces.py:108-131`
**RESEARCH.md says:** `v(x) = -λ|x|^β` applied to monetary cost.
**Implementation:** Constructs a synthetic 0-1 `loss_input` from three blended components (relative price ratio, absolute log-scaled pain, income relief), then applies `λ * loss_input^β`.
**Specific sub-issues:**
- The `relative_loss = (price_ratio - 1) / price_ratio` formula is a custom construction not in any cited paper
- Income relief (40% cap, $15,000 threshold) has no cited source
- Gain side ignores income entirely, creating an asymmetry not in the original K&T formulation
**Note:** This may be an intentional engineering decision to keep forces on comparable scales. But it should be documented if so.

### HIGH-4: Anchoring × Loss aversion double-counts for underpriced products
**File:** `engine/forces.py`
**RESEARCH.md says:** "A high reference price reduces perceived loss, effectively lowering the agent's experienced λ."
**Implementation:** When `price < reference_price`, two things happen simultaneously:
1. In Force 1 (prospect value): `relative_loss` goes to 0 (price looks cheap vs reference)
2. In Force 2 (anchoring): anchor force becomes positive (price below reference = good)
Both forces independently benefit underpriced products, creating a double-counting effect. The architecture says anchoring should work *through* the loss function (lower effective λ), not as a separate additive force.

---

## MEDIUM severity — undocumented deviations

### MED-1: Loss aversion 80% component uses per-archetype distributions
**File:** `engine/population.py:127`
**RESEARCH.md says:** "80% drawn from normal(2.25, 0.5)"
**Implementation:** Draws from per-archetype means: innovator=1.5, early_adopter=1.8, early_majority=2.25, late_majority=2.8, laggard=3.5. Only early_majority matches the documented distribution.
**Assessment:** This is arguably better design (each archetype has its own loss aversion profile per Rogers), but contradicts the literal statement in RESEARCH.md.

### MED-2: `price_sensitivity` and `novelty_weight` are sampled but never used
**Files:** `engine/population.py` (generated), `engine/forces.py` (not referenced)
**Problem:** These fields exist in `AGENT_DTYPE`, are sampled from archetype distributions, but no force computation reads them. They are dead data that consume memory and could mislead users inspecting the diagnostic dump.

### MED-3: `product_adoption_rate` not in CALIBRATE_PROMPT
**File:** `pipeline/calibrate.py:201`, `llm/prompts.py`
**Problem:** The calibrate parser tries to extract `product_adoption_rate` from LLM output, but the prompt's JSON schema does not include this field. The LLM will never return it, so it always defaults to 0.05.

### MED-4: `discovered_competitors` never populated
**File:** `pipeline/research.py:165`
**Problem:** Research only appends to `verified_competitors`, never `discovered_competitors`. The field exists on `MarketContext` but is always empty.

### MED-5: Population config defaults disagree with ARCHITECTURE.md
**File:** `pipeline/calibrate.py:170-171`
**Problem:** Code defaults `income_mean_log=11.0, income_sigma=0.7`. ARCHITECTURE.md example shows `income_mean_log=11.2, income_sigma=0.6`. These produce different income distributions (median $59,874 vs $73,130).

### MED-6: "saturated" vs "declining" terminology mismatch
**Files:** `models/market.py:27` says "saturated", `ARCHITECTURE.md` says "declining"
**Impact:** Mostly cosmetic, but could confuse the LLM if it receives "saturated" when architecture docs say "declining".

### MED-7: `_should_trigger` only recognizes 3 hardcoded patterns
**File:** `pipeline/research.py:216-231`
**Problem:** Any trigger condition from the LLM that doesn't match "ambiguous pricing", "no competitors", or "low confidence" silently falls through to `return False`, disabling the optional query.

---

## LOW severity — minor issues

### LOW-1: All temperatures, defaults, and thresholds are hardcoded
**Files:** `pipeline/understand.py:42`, `pipeline/calibrate.py:53`, `pipeline/research.py` throughout
**Problem:** Values like temperature=0.15, confidence_threshold=0.7, and all fallback parameter defaults are hardcoded in Python rather than loaded from `config/defaults.yaml`. The YAML file exists but is never referenced by the calibrate stage.

### LOW-2: CompetitorInfo fields `has_free_tier`, `market_position`, `source_url` never populated in understand stage
**File:** `pipeline/understand.py:47-52`
**Impact:** Minor — these are populated later by research if applicable.

### LOW-3: EventResult.segment_effects type mismatch
**File:** `models/results.py:72`
**Problem:** Typed as `dict[str, float]` but the EVENT_PROMPT asks the LLM for descriptive strings, not floats.

### LOW-4: Sensitivity analysis results are order-dependent
**File:** `engine/sensitivity.py:63`
**Problem:** Uses a single RNG that advances sequentially through `SENSITIVE_PARAMS`. Reordering the list changes results. Not a bug per se, but surprising.

### LOW-5: Event processing regenerates the entire population
**File:** `pipeline/events.py:84`
**Problem:** Events call `simulate()` which generates a fresh population. Agent-level state from the previous simulation is lost (agents who adopted won't necessarily adopt again). Events can't model "agents who already adopted won't un-adopt."

### LOW-6: Confidence band fallback is a hardcoded ±5pp
**File:** `pipeline/analyze.py:119-123`
**Problem:** When all params have confidence ≥ 0.80, the fallback returns a fixed ±5pp band not derived from any data.

---

## Test gaps worth noting

1. **No test verifies actual numeric output of any force formula** — all tests check signs or relative ordering only. A bug changing `ALPHA = 0.88` to `ALPHA = 0.5` would pass all tests.
2. **`compute_weighted_forces` boundary weighting is untested** — the test only asserts the key exists, not that boundary agents are weighted more.
3. **Personality-economics correlations are untested** — the neuroticism→loss_aversion and agreeableness→social_proof correlation code could be removed and all tests would pass.
4. **End-to-end adoption test has an 85pp acceptance range** (10-95%) — essentially untestable as a regression check.

---

## How to use the diagnostic dump

Run with `--dump` to write `mindsim_dump.json`:
```bash
mindsim --dump --skip-research "A $20/mo AI coding assistant"
```

The dump file contains:
- **Stage 1 (understand):** Parsed product profile, research plan queries
- **Stage 2 (research):** Verified competitors, confidence flags, trends data
- **Stage 3 (calibrate):** Every parameter with value/basis/confidence, assumptions table
- **Stage 4 (simulate):** Per-agent data for all 1,000 agents — archetype, traits, all 7 force values, total utility, adoption probability, and decision
- **Stage 5 (analyze):** Sensitivity swings, confidence band, intervention rankings

To spot-check specific agents, look at `stages.4_simulate.agents[N].forces` and verify each force makes sense given the agent's traits and the calibrated parameters.
