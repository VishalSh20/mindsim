# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is mindsim

Behavioral economics simulation engine. Takes a plain-English product description, researches the market, generates 1,000 psychologically diverse agents (Rogers adoption curve archetypes), runs each through a 7-force decision model grounded in 28 peer-reviewed sources, and outputs a behavioral audit showing which cognitive mechanisms drive or block adoption.

Currently **backend CLI only** (no frontend/web UI yet).

## Build & Run

```bash
cd backend
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .

# Copy .env.example to .env and set API keys
cp ../.env.example ../.env

# Quick run (no Tavily needed)
mindsim --skip-research "A $20/mo AI coding assistant"

# Full run with market research
mindsim "A $20/mo AI coding assistant"

# With events and interactive mode
mindsim --skip-research --event "Claude Code launches at $200/mo" -i "A $20/mo AI tool"
```

**Environment variables:** `OPENAI_API_KEY` (required for cloud LLM), `TAVILY_API_KEY` (required unless `--skip-research`), `MINDSIM_MODEL` (optional, default `auto` which picks OpenAI if key present, else local Ollama).

## Tests

```bash
cd backend
uv run python -m pytest tests/ -v

# Single test file
uv run python -m pytest tests/test_forces.py -v

# Single test
uv run python -m pytest tests/test_forces.py::test_name -v
```

## Architecture

**Pipeline, not agents.** Five sequential stages with typed input/output contracts:

```
User text -> Understand (1 LLM call) -> Research (2-8 Tavily, 0 LLM)
  -> Calibrate (1 LLM call) -> Simulate (pure NumPy, <100ms)
  -> Analyze (sensitivity + 1 LLM call for audit text)
```

Events add 1 LLM call each. Total: 3 LLM calls + 1 per event for a full run.

### Key design decisions

- **LLMs are interpreters, not deciders.** LLM calls handle natural language understanding (parse product, calibrate params, interpret events, write reports). The actual agent decision math is pure NumPy — no LLM in the hot path.
- **Simulation is vectorized.** All 1,000 agents computed simultaneously via NumPy structured arrays. No Python loops over agents. `AGENT_DTYPE` in `engine/population.py` defines the structured array schema.
- **Every parameter has provenance.** `CalibratedParam` (in `models/config.py`) wraps every numerical parameter with `value`, `basis` (why this number), and `confidence` (0-1). Parameters are challengeable via `--override`.
- **Research is adaptive and confidence-gated.** The research stage plans queries with confidence thresholds and fallback assumptions. Simple products cost 2-3 Tavily calls; novel products cost 7-8.

### Source layout (`backend/src/mindsim/`)

| Directory | Purpose |
|-----------|---------|
| `pipeline/` | 5 stages: `understand.py`, `research.py`, `calibrate.py`, `simulate.py`, `analyze.py` + `events.py` |
| `engine/` | Pure math: `forces.py` (7-force computation), `population.py` (agent generation), `archetypes.py` (Rogers curve), `sensitivity.py`, `interventions.py` |
| `models/` | Pydantic models: `config.py` (SimulationConfig/Params), `product.py`, `market.py`, `results.py` |
| `llm/` | `client.py` (OpenAI/Ollama abstraction), `prompts.py` (all LLM prompts) |
| `tools/` | External APIs: `tavily_client.py`, `trends.py` (Google Trends) |
| `config/` | YAML: `archetypes.yaml` (Rogers archetype distributions), `defaults.yaml` (skip-research fallbacks) |

### The 7 Forces (in `engine/forces.py`)

All computed in `compute_forces()` as vectorized NumPy operations on the full agent population:

1. **Prospect Value** — Kahneman & Tversky value function (gain^alpha - lambda*loss^beta), per-agent loss aversion
2. **Anchoring** — Price vs reference price (competitor-awareness-weighted)
3. **Status Quo Bias** — Blends adoption friction (new behavior) and switching friction (existing tool) by category penetration
4. **Social Proof** — Scales with uncertainty: `social_proof_need * log(1 + adoption * visibility) * (1 - benefit_certainty)`
5. **FOMO** — Distinct from loss aversion: opportunity-miss pain, scales with social visibility and category growth
6. **Hyperbolic Discounting** — Present bias penalty on delayed benefits, domain-specific beta (0.5 for habit-change, 0.85 for consumption)
7. **Identity Signaling** — Openness * identity_signal * social_visibility

Forces sum to total utility -> logistic function (temperature=3.0) -> stochastic adoption decision.

### Agent population

5 archetypes from Rogers (1962) with published shares: Innovator (2.5%), Early Adopter (13.5%), Early Majority (34%), Late Majority (34%), Laggard (16%). Each archetype has distinct distributions for loss aversion, status quo bias, social proof need, FOMO susceptibility, openness, etc. defined in `config/archetypes.yaml`. Loss aversion uses a mixture model: 80% archetype distribution + 20% near-zero subgroup (Gachter et al. 2022).

### Critical interactions between forces

These are load-bearing in the simulation and affect intervention recommendations:
- **Loss aversion x Hyperbolic discounting** multiply for upfront-cost + delayed-benefit products (why free trials are consistently recommended)
- **Social proof x Benefit uncertainty** — social proof matters MORE when benefit is uncertain
- **Anchoring x Loss aversion** — high reference price (expensive competitor) reduces perceived loss

## LLM client

`llm/client.py` auto-detects: env var `MINDSIM_MODEL` > OpenAI API key > local Ollama. Default cloud model is `gpt-4.1`. Default Ollama model is `qwen3:8b`. Prefix Ollama models with `ollama/` (e.g., `ollama/qwen3:8b`). `complete_json()` handles markdown fence stripping and `<think>` tag removal for reasoning models.

## Diagnostic dump

Run with `--dump` to write full pipeline state to `mindsim_dump.json` for debugging:
```bash
mindsim --dump --skip-research "A $20/mo AI coding assistant"
mindsim --dump --dump-path my_debug.json "product description"
```
Contains per-agent data (all 1,000 agents with traits, force values, probability, decision), research output, calibrated parameters with provenance, sensitivity analysis, and intervention results.

## Known issues

See `docs/ISSUES.md` for a comprehensive audit of implementation vs ARCHITECTURE.md and RESEARCH.md, including formula discrepancies, dead parameters, and test gaps.

## Research foundation

28 peer-reviewed sources across 7 behavioral mechanisms. See `docs/RESEARCH.md` for the full bibliography with specific parameters extracted from each source. See `docs/ARCHITECTURE.md` for detailed system design.
