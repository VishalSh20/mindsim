"""mindsim CLI — entry point, Rich output, interactive mode.

Usage:
  mindsim "A $20/mo AI coding assistant that helps developers write code faster"
  mindsim --skip-research "A habit tracking app for $5/mo"
  mindsim --interactive "Notion competitor at $15/mo"
  mindsim --event "a competitor launches at 10x the price" "My $20 SaaS tool"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import numpy as np
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

console = Console()


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(verbose=args.verbose)

    from mindsim.llm.client import LLMClient
    from mindsim.models.config import SimulationConfig
    from mindsim.models.market import MarketContext
    from mindsim.models.results import AnalysisReport, SimulationResult
    from mindsim.pipeline.analyze import analyze
    from mindsim.pipeline.calibrate import calibrate
    from mindsim.pipeline.events import process_event
    from mindsim.pipeline.research import research
    from mindsim.pipeline.simulate import simulate
    from mindsim.pipeline.understand import understand

    product_text = args.product_description

    console.print()
    console.print(
        Panel(
            f"[bold cyan]MINDSIM[/] — Behavioral Economics Simulation Engine",
            subtitle="[dim]7 forces × 1,000 agents × published behavioral science[/]",
            border_style="cyan",
        )
    )
    console.print()

    llm = LLMClient(model=args.model)
    rng = np.random.default_rng(42)

    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Stage 1: Understand
        task = progress.add_task("[cyan]Understanding product...", total=None)
        profile = understand(product_text, llm)
        progress.update(task, completed=True, description="[green]✓ Product understood")

        # Stage 2: Research
        task = progress.add_task("[cyan]Researching market...", total=None)
        market = research(
            profile,
            skip=args.skip_research,
            budget=args.budget,
        )
        progress.update(task, completed=True, description="[green]✓ Research complete")

        # Stage 3: Calibrate
        task = progress.add_task("[cyan]Calibrating parameters...", total=None)
        config = calibrate(profile, market, llm)

        # Apply CLI overrides
        if args.override:
            config = _apply_overrides(config, args.override)

        progress.update(task, completed=True, description="[green]✓ Parameters calibrated")

        # Stage 4: Simulate
        task = progress.add_task("[cyan]Running simulation (1,000 agents)...", total=None)
        sim_result = simulate(config, rng=rng)
        progress.update(task, completed=True, description="[green]✓ Simulation complete")

        # Stage 4e: Events
        event_results = []
        if args.event:
            for event_text in args.event:
                task = progress.add_task(
                    f"[cyan]Processing event: {event_text[:40]}...", total=None
                )
                sim_result, event_result = process_event(
                    event_text, sim_result, config, llm, rng
                )
                event_results.append(event_result)
                progress.update(task, completed=True, description=f"[green]✓ Event processed")

        # Stage 5: Analyze
        task = progress.add_task("[cyan]Analyzing results...", total=None)
        report = analyze(sim_result, config, llm, rng)
        progress.update(task, completed=True, description="[green]✓ Analysis complete")

    elapsed = time.time() - start_time

    # Display results
    console.print()
    _display_results(sim_result, report, config, elapsed)

    # Interactive mode
    if args.interactive:
        _interactive_loop(sim_result, config, report, llm, rng)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="mindsim",
        description="Behavioral economics simulation engine",
    )
    parser.add_argument(
        "product_description",
        help="Natural language product description",
    )
    parser.add_argument(
        "--event", "-e",
        action="append",
        help="Market event to simulate (can specify multiple)",
    )
    parser.add_argument(
        "--override", "-o",
        action="append",
        help="Override assumption: A1=0.30",
    )
    parser.add_argument(
        "--skip-research",
        action="store_true",
        help="Skip Tavily research, use behavioral science defaults",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=10,
        help="Max Tavily credits per run (default: 10)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model override (e.g., 'claude-sonnet-4-20250514', 'ollama/qwen3:8b')",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Enter interactive mode after initial run",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def setup_logging(verbose: bool = False):
    """Configure logging with Rich handler."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, show_time=False)],
    )


def _display_results(
    sim_result: "SimulationResult",
    report: "AnalysisReport",
    config: "SimulationConfig",
    elapsed: float,
):
    """Display Rich-formatted simulation results."""
    # ── HEADER ──
    cb = report.confidence_band
    header = Text()
    header.append("ADOPTION: ", style="bold")
    header.append(f"{sim_result.total_adoption*100:.0f}%", style="bold green")
    header.append(f" (confidence band: {cb.low*100:.0f}%–{cb.high*100:.0f}%)", style="dim")
    console.print(Panel(header, title="[bold cyan]MINDSIM BEHAVIORAL AUDIT[/]", border_style="cyan"))

    # Display awareness and aware-adoption as context
    awareness_rate = sim_result.n_aware / max(sim_result.n_agents, 1)
    console.print(
        f"  Awareness: [cyan]{awareness_rate*100:.0f}%[/] │ "
        f"Adoption among aware: [cyan]{sim_result.aware_adoption*100:.0f}%[/] │ "
        f"Agents: {sim_result.n_agents}"
    )

    # ── PROXIMITY ──
    prox = sim_result.by_proximity
    prox_text = (
        f"Convertible pool: [yellow]{prox.convertible*100:.0f}%[/] │ "
        f"Locked: [green]{prox.locked*100:.0f}%[/] │ "
        f"Unreachable: [red]{prox.unreachable*100:.0f}%[/]"
    )
    console.print(prox_text)
    console.print()

    # ── FORCE DECOMPOSITION ──
    console.print("[bold]FORCE DECOMPOSITION[/] (convertible pool)")
    forces = sim_result.force_decomposition.as_dict()

    force_table = Table(show_header=False, box=None, padding=(0, 1))
    force_table.add_column("Force", width=22)
    force_table.add_column("Bar", width=22)
    force_table.add_column("Value", width=8, justify="right")
    force_table.add_column("Confidence", width=14)

    for force_name, value in forces.items():
        bar = _force_bar(value)
        # Find confidence for this force's primary param
        confidence = _get_force_confidence(force_name, config)
        conf_indicator = _confidence_indicator(confidence)
        display_name = force_name.replace("_", " ")
        force_table.add_row(display_name, bar, f"{value:+.2f}", conf_indicator)

    console.print(force_table)
    console.print()

    # ── SENSITIVITY ──
    if report.sensitivity:
        console.print("[bold]SENSITIVITY[/]")
        for s in report.sensitivity[:5]:
            warn = " ⚠" if s.swing > 10 else ""
            console.print(
                f"  {s.parameter:25s} "
                f"{s.low_adoption*100:.0f}% ──[{s.base_adoption*100:.0f}%]── "
                f"{s.high_adoption*100:.0f}%  "
                f"swing: {s.swing:.0f}pp{warn}"
            )
        console.print()

    # ── EVENTS ──
    for er in sim_result.event_results:
        console.print(f"[bold]EVENT:[/] {er.event_text}")
        delta_color = "green" if er.adoption_delta > 0 else "red"
        console.print(
            f"  Adoption: {er.adoption_before*100:.0f}% → "
            f"{er.adoption_after*100:.0f}% "
            f"([{delta_color}]{er.adoption_delta*100:+.0f}pp[/])"
        )
        for adj in er.force_adjustments:
            console.print(f"  {adj.force}: {adj.magnitude:+.2f}  \"{adj.mechanism[:60]}\"")
        console.print()

    # ── INTERVENTIONS ──
    if report.interventions:
        console.print("[bold]INTERVENTIONS[/]")
        for i, intv in enumerate(report.interventions[:5], 1):
            console.print(
                f"  #{i} {intv.name:20s} "
                f"[green]{intv.lift_pp:+.0f}pp[/] "
                f"({intv.adoption_before*100:.0f}%→{intv.adoption_after*100:.0f}%)"
            )
        console.print()

    # ── ASSUMPTIONS ──
    if report.assumptions:
        console.print("[bold]ASSUMPTIONS[/] (challenge with --override)")
        for a in report.assumptions:
            conf = _confidence_indicator(a.get("confidence", 0.5))
            console.print(
                f"  {a['id']:3s} {a['parameter']:25s} "
                f"{a['value']:6.2f}  {conf}  \"{a['basis'][:40]}\""
            )
        console.print()

    # ── SEGMENTS ──
    seg_table = Table(title="Adoption by Archetype", border_style="dim")
    seg_table.add_column("Archetype", style="cyan")
    seg_table.add_column("Rate", justify="right")
    seg_table.add_column("Count", justify="right")
    for seg in sim_result.by_archetype:
        rate_style = "green" if seg.adoption_rate > 0.5 else "yellow" if seg.adoption_rate > 0.2 else "red"
        seg_table.add_row(
            seg.name,
            f"[{rate_style}]{seg.adoption_rate*100:.1f}%[/]",
            f"{seg.count}/{seg.total}",
        )
    console.print(seg_table)
    console.print()

    # ── AUDIT TEXT ──
    if report.text:
        console.print(Panel(report.text, title="[bold]Behavioral Audit[/]", border_style="blue"))
        console.print()

    console.print(f"[dim]Completed in {elapsed:.1f}s • {sim_result.n_agents} agents[/]")


def _force_bar(value: float, width: int = 20) -> str:
    """Create a colored force bar visualization."""
    # Normalize value to [-1, 1] range for display
    clamped = max(-1.0, min(1.0, value * 3))  # scale up for visibility
    center = width // 2

    if clamped >= 0:
        fill = int(clamped * center)
        bar = "░" * center + "▓" * fill + "░" * (center - fill)
    else:
        fill = int(-clamped * center)
        bar = "░" * (center - fill) + "▓" * fill + "░" * center

    return bar


def _confidence_indicator(confidence: float) -> str:
    """Create ■■■ / ■■░ / ■░░ confidence indicator."""
    if confidence >= 0.7:
        return "[green]■■■[/] high"
    elif confidence >= 0.4:
        return "[yellow]■■░[/] medium"
    else:
        return "[red]■░░[/] low"


def _get_force_confidence(force_name: str, config: "SimulationConfig") -> float:
    """Get the confidence score for the primary parameter of a force."""
    param_map = {
        "prospect_value": "perceived_benefit",
        "anchoring": None,  # reference price
        "status_quo": "switching_cost",
        "social_proof": "benefit_certainty",
        "fomo": "fomo_intensity",
        "hyperbolic_discounting": "present_bias_beta",
        "identity_signaling": "identity_signal",
    }
    param_name = param_map.get(force_name)
    if param_name is None:
        return config.simulation_params.reference_price.confidence
    cparam = getattr(config.simulation_params, param_name, None)
    if cparam and hasattr(cparam, "confidence"):
        return cparam.confidence
    return 0.5


def _apply_overrides(config: "SimulationConfig", overrides: list[str]) -> "SimulationConfig":
    """Apply CLI overrides like A1=0.30."""
    modified = config.model_copy(deep=True)

    for override in overrides:
        if "=" not in override:
            console.print(f"[yellow]Warning: invalid override '{override}', expected format A1=0.30[/]")
            continue

        key, value_str = override.split("=", 1)
        try:
            new_value = float(value_str)
        except ValueError:
            console.print(f"[yellow]Warning: '{value_str}' is not a valid number[/]")
            continue

        # Find the assumption
        key = key.strip().upper()
        for assumption in modified.assumptions:
            if assumption.id == key:
                param_name = assumption.parameter
                cparam = getattr(modified.simulation_params, param_name, None)
                if cparam and hasattr(cparam, "value"):
                    cparam.value = new_value
                    assumption.value = new_value
                    console.print(f"[green]Override: {param_name} = {new_value}[/]")
                break
        else:
            # Try as direct param name
            cparam = getattr(modified.simulation_params, key.lower(), None)
            if cparam and hasattr(cparam, "value"):
                cparam.value = new_value
                console.print(f"[green]Override: {key.lower()} = {new_value}[/]")
            else:
                console.print(f"[yellow]Warning: unknown assumption/param '{key}'[/]")

    return modified


def _interactive_loop(
    sim_result: "SimulationResult",
    config: "SimulationConfig",
    report: "AnalysisReport",
    llm: "LLMClient",
    rng: np.random.Generator,
):
    """Interactive command loop."""
    from mindsim.pipeline.analyze import analyze
    from mindsim.pipeline.events import process_event
    from mindsim.pipeline.simulate import simulate

    console.print()
    console.print("[bold cyan]Interactive Mode[/] — type 'help' for commands")
    console.print()

    current_result = sim_result
    current_config = config
    current_report = report

    while True:
        try:
            cmd = console.input("[bold cyan]mindsim>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not cmd:
            continue

        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if command == "quit" or command == "exit" or command == "q":
            break

        elif command == "help":
            console.print("""
[bold]Commands:[/]
  override A1=0.30 A2=0.5  — Re-simulate with overridden params
  event "description"       — Process a market event
  deep A1                   — Full evidence chain for assumption
  agent 472                 — Decision trace for agent #472
  compare "description"     — Side-by-side with modified product
  export results.json       — Export full results as JSON
  quit                      — Exit interactive mode
""")

        elif command == "override":
            overrides = arg.split()
            current_config = _apply_overrides(current_config, overrides)
            current_result = simulate(current_config, rng=rng)
            current_report = analyze(current_result, current_config, llm, rng)
            _display_results(current_result, current_report, current_config, 0)

        elif command == "event":
            event_text = arg.strip('"\'')
            current_result, event_result = process_event(
                event_text, current_result, current_config, llm, rng
            )
            console.print(
                f"\nAdoption: {event_result.adoption_before*100:.0f}% → "
                f"{event_result.adoption_after*100:.0f}% "
                f"({event_result.adoption_delta*100:+.0f}pp)\n"
            )
            for adj in event_result.force_adjustments:
                console.print(f"  {adj.force}: {adj.magnitude:+.2f}  \"{adj.mechanism}\"")

        elif command == "deep":
            _deep_dive(arg.strip(), current_config, current_report)

        elif command == "agent":
            try:
                agent_id = int(arg.strip())
                _agent_trace(agent_id, current_result)
            except ValueError:
                console.print("[red]Usage: agent <number>[/]")

        elif command == "export":
            filename = arg.strip() or "mindsim_results.json"
            _export_results(current_result, current_report, current_config, filename)

        else:
            console.print(f"[red]Unknown command: {command}. Type 'help' for commands.[/]")


def _deep_dive(assumption_id: str, config: "SimulationConfig", report: "AnalysisReport"):
    """Show full evidence chain for an assumption."""
    assumption_id = assumption_id.upper()

    for a in report.assumptions:
        if a.get("id") == assumption_id:
            console.print(Panel(
                f"[bold]{a['parameter']}[/] = {a['value']}\n\n"
                f"[bold]Basis:[/] {a['basis']}\n"
                f"[bold]Confidence:[/] {_confidence_indicator(a['confidence'])}\n"
                f"[bold]Sensitivity:[/] {a.get('sensitivity', 'unknown')}",
                title=f"[bold]Assumption {assumption_id}[/]",
                border_style="blue",
            ))

            # Find sensitivity data
            for s in report.sensitivity:
                if s.parameter == a["parameter"]:
                    console.print(
                        f"\n  If {s.parameter} is 30% lower: adoption = {s.low_adoption*100:.0f}%"
                        f"\n  If {s.parameter} is 30% higher: adoption = {s.high_adoption*100:.0f}%"
                        f"\n  Total swing: {s.swing:.0f}pp"
                    )
            return

    console.print(f"[yellow]Assumption {assumption_id} not found[/]")


def _agent_trace(agent_id: int, sim_result: "SimulationResult"):
    """Show full decision trace for a specific agent."""
    agents = sim_result._agents
    forces = sim_result._agent_forces
    probs = sim_result._agent_probs
    decisions = sim_result._agent_decisions

    if agents is None or forces is None:
        console.print("[red]Agent data not available[/]")
        return

    if agent_id < 0 or agent_id >= len(agents):
        console.print(f"[red]Agent {agent_id} out of range (0-{len(agents)-1})[/]")
        return

    agent = agents[agent_id]
    archetype_names = ["innovator", "early_adopter", "early_majority", "late_majority", "laggard"]
    archetype = archetype_names[agent["archetype_id"]]

    console.print(Panel(
        f"[bold]Agent #{agent_id}[/]\n"
        f"  Archetype: {archetype}\n"
        f"  Loss aversion λ: {agent['loss_aversion_lambda']:.2f}\n"
        f"  Status quo bias: {agent['status_quo_bias']:.2f}\n"
        f"  Social proof need: {agent['social_proof_need']:.2f}\n"
        f"  FOMO susceptibility: {agent['fomo_susceptibility']:.2f}\n"
        f"  Openness: {agent['openness']:.2f}\n"
        f"  Income: ${agent['income']:,.0f}\n"
        f"  Aware: {agent['aware']}\n",
        title=f"Agent Profile",
        border_style="cyan",
    ))

    if agent["aware"]:
        console.print("[bold]Force values:[/]")
        for fname, farr in forces.items():
            val = farr[agent_id]
            if not np.isnan(val):
                console.print(f"  {fname:25s} {val:+.4f}")

        prob = probs[agent_id]
        decided = decisions[agent_id]
        console.print(f"\n  Total utility → Adoption probability: {prob:.3f}")
        console.print(f"  Decision: {'[green]ADOPT[/]' if decided else '[red]REJECT[/]'}")


def _export_results(
    sim_result: "SimulationResult",
    report: "AnalysisReport",
    config: "SimulationConfig",
    filename: str,
):
    """Export results as JSON."""
    data = {
        "simulation": sim_result.model_dump(exclude={"_agent_forces", "_agent_probs", "_agent_decisions", "_agents"}),
        "analysis": report.model_dump(),
        "config": config.model_dump(),
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2, default=str)

    console.print(f"[green]Results exported to {filename}[/]")


if __name__ == "__main__":
    main()
