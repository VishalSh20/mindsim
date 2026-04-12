# mindsim

**Behavioral economics simulation engine** — predict how psychologically diverse populations react to products, pricing, and market events.

Describe a product in plain English. mindsim researches the market, generates 1,000 agents with published psychological profiles, runs each through a 7-force decision model grounded in 28 peer-reviewed sources, and outputs a behavioral audit showing which cognitive mechanisms drive or block adoption.

## Quick Start

```bash
cd backend
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .

# Copy and fill in your API keys
cp ../.env.example ../.env

# Run with skip-research (no Tavily key needed)
mindsim --skip-research "A $20/mo AI coding assistant that helps developers write code 2x faster"

# Full run with market research
mindsim "A $20/mo AI coding assistant that helps developers write code 2x faster"

# With an event
mindsim --skip-research --event "Claude Code launches at $200/mo" "A $20/mo AI coding tool"

# Interactive mode
mindsim --skip-research -i "Notion competitor at $15/mo"
```

## The 7 Forces

| Force | Source | Direction |
|---|---|---|
| **Prospect Value** | Kahneman & Tversky 1979 | ± (gain vs loss) |
| **Anchoring** | Tversky & Kahneman 1974 | ± (deal vs overpriced) |
| **Status Quo Bias** | Samuelson & Zeckhauser 1988 | Always negative |
| **Social Proof** | Cialdini 1984, Salganik 2006 | Positive (scales with uncertainty) |
| **FOMO** | Loomes & Sugden 1982 | Positive (opportunity-miss pain) |
| **Hyperbolic Discounting** | Laibson 1997 | Negative (delayed benefit penalty) |
| **Identity Signaling** | Veblen 1899, Berger & Heath 2007 | Positive (status signal) |

## Architecture

```
User text → Understand (LLM) → Research (Tavily) → Calibrate (LLM)
  → Simulate (pure NumPy, <100ms) → Analyze (sensitivity + LLM audit)
```

- **3 LLM calls** + 1 per event
- **2-8 Tavily API calls** (confidence-gated)
- **1,000 agents** in <100ms via vectorized NumPy
- **Total runtime:** 10-16s full run, ~4s override rerun

## CLI Options

```
mindsim "product description"          # Full pipeline
  --skip-research                      # Skip Tavily, use defaults
  --event "market event"               # Inject events (stackable)
  --override A1=0.30                   # Override assumptions
  --interactive / -i                   # Interactive mode
  --model "claude-sonnet-4-20250514"           # LLM override
  --budget 10                          # Max Tavily credits
  --verbose / -v                       # Debug logging
```

## Interactive Commands

```
override A1=0.30 A2=0.5   — Re-simulate with overridden params
event "description"        — Process a market event
deep A1                    — Full evidence chain for assumption
agent 472                  — Decision trace for agent #472
export results.json        — Export full results as JSON
quit                       — Exit
```

## Tests

```bash
cd backend
uv run python -m pytest tests/ -v
```

## Research Foundation

28 peer-reviewed sources across 7 behavioral mechanisms. See [docs/RESEARCH.md](docs/RESEARCH.md) for the full bibliography. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the system design.
