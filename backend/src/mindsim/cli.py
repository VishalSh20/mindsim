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

    from mindsim.diagnostics import DiagnosticDump
    from mindsim.llm.client import LLMClient
    from mindsim.models.config import SimulationConfig
    from mindsim.models.market import MarketContext
    from mindsim.models.results import AnalysisReport, SimulationResult
    from mindsim.pipeline.analyze import analyze
    from mindsim.pipeline.calibrate import calibrate
    from mindsim.pipeline.events import attach_session, process_event
    from mindsim.pipeline.research import research
    from mindsim.pipeline.research_synthesizer import synthesize_market
    from mindsim.pipeline.session import (
        SessionVersionError,
        append_event_to_history,
        load_session,
        new_run_id,
        save_session,
    )
    from mindsim.pipeline.simulate import (
        run_rounds,
        simulate,
        snapshot_state,
        summarize_state,
    )
    from mindsim.pipeline.understand import understand
    from mindsim.pipeline.voc import analyze_voc

    product_text = args.product_description
    if not args.session and not product_text:
        console.print("[red]Error: product_description is required unless --session is given[/]")
        sys.exit(2)
    dump = DiagnosticDump(path=args.dump_path) if args.dump else None

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

    # Wave 6 session lineage. parent_run_id is the loaded session's run_id;
    # event_history accumulates across the lineage so the saved file holds
    # the full chain of "what happened to this population".
    parent_run_id: str | None = None
    event_history: list = []
    run_id = new_run_id()
    market = None  # populated either by research stage or skipped on session-load

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        if args.session:
            # ── Session-resume path: skip 1-4, load persisted agents + config. ──
            task = progress.add_task(f"[cyan]Loading session {args.session}...", total=None)
            try:
                state = load_session(args.session)
            except SessionVersionError as exc:
                console.print(f"[red]Session refused: {exc}[/]")
                sys.exit(2)
            config = state.config
            parent_run_id = state.run_id
            event_history = list(state.event_history)
            progress.update(
                task,
                completed=True,
                description=f"[green]✓ Session loaded (parent={parent_run_id}, n={state.n_agents})",
            )

            if args.override:
                config = _apply_overrides(config, args.override)

            task = progress.add_task("[cyan]Snapshotting state...", total=None)
            sim_result = snapshot_state(state.agents, config.simulation_params, rng)
            attach_session(sim_result, state.agents)
            progress.update(task, completed=True, description="[green]✓ State snapshot ready")
        else:
            # ── Fresh-run path: full pipeline. ──
            task = progress.add_task("[cyan]Understanding product...", total=None)
            profile = understand(product_text, llm)
            if dump:
                dump.record_understand(profile)
            progress.update(task, completed=True, description="[green]✓ Product understood")

            task = progress.add_task("[cyan]Researching market...", total=None)
            market = research(
                profile,
                skip=args.skip_research,
                budget=args.budget,
                enable_scrapers=not args.skip_research,
            )
            if dump:
                dump.record_research(market)
            progress.update(task, completed=True, description="[green]✓ Research complete")

            if not args.skip_research:
                task = progress.add_task("[cyan]Mining voice-of-customer...", total=None)
                voc_report = analyze_voc(market.scraped_docs_cache, profile, llm)
                market.voc_report = voc_report
                progress.update(task, completed=True, description="[green]✓ VoC analysis complete")

                task = progress.add_task("[cyan]Synthesizing competitor matrix...", total=None)
                synthesize_market(
                    context=market,
                    tavily_results=market.tavily_results_cache,
                    scraped_pricing_docs=[
                        d for d in market.scraped_docs_cache if d.source == "pricing_page"
                    ],
                    voc=voc_report,
                    profile=profile,
                    llm_client=llm,
                )
                progress.update(task, completed=True, description="[green]✓ Synthesizer complete")

            task = progress.add_task("[cyan]Calibrating parameters...", total=None)
            config = calibrate(profile, market, llm)
            if args.override:
                config = _apply_overrides(config, args.override)
            if dump:
                dump.record_calibration(config)
            progress.update(task, completed=True, description="[green]✓ Parameters calibrated")

            task = progress.add_task("[cyan]Running simulation (1,000 agents)...", total=None)
            sim_result = simulate(config, rng=rng, market=market)
            if dump:
                dump.record_simulation(sim_result, config)
            progress.update(task, completed=True, description="[green]✓ Simulation complete")

        # Stage 4e: Events (shared by both paths)
        event_results = []
        if args.event:
            for event_text in args.event:
                task = progress.add_task(
                    f"[cyan]Processing event: {event_text[:40]}...", total=None
                )
                adoption_before = sim_result.total_adoption
                sim_result, event_result = process_event(
                    event_text, sim_result, config, llm, rng
                )
                event_history = append_event_to_history(
                    event_history,
                    event_text,
                    adoption_before,
                    sim_result.total_adoption,
                )
                event_results.append(event_result)
                if dump:
                    dump.record_event(event_text, event_result)
                progress.update(task, completed=True, description="[green]✓ Event processed")

        # Stage 5: Analyze
        task = progress.add_task("[cyan]Analyzing results...", total=None)
        report = analyze(sim_result, config, llm, rng, market=market)
        if dump:
            dump.record_analysis(report)
        progress.update(task, completed=True, description="[green]✓ Analysis complete")

        if args.session_out:
            if sim_result._agents is None:
                console.print("[yellow]No agent array to persist; --session-out skipped[/]")
            else:
                task = progress.add_task("[cyan]Saving session...", total=None)
                saved = save_session(
                    path=args.session_out,
                    agents=sim_result._agents,
                    config=config,
                    run_id=run_id,
                    parent_run_id=parent_run_id,
                    event_history=event_history,
                )
                progress.update(
                    task,
                    completed=True,
                    description=f"[green]✓ Session saved: {saved}",
                )

    elapsed = time.time() - start_time

    # Display results
    console.print()
    _display_results(sim_result, report, config, elapsed)

    # Wave 8.5 — fully expanded EvidenceStore on demand.
    if args.show_evidence:
        _display_evidence_store(sim_result)

    # Write diagnostic dump
    if dump:
        dump.write()
        console.print(f"[dim]Diagnostic dump written to {args.dump_path}[/]")

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
        nargs="?",
        default=None,
        help="Natural language product description (omit when using --session)",
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
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Write diagnostic dump (research, params, per-agent data) to mindsim_dump.json",
    )
    parser.add_argument(
        "--dump-path",
        default="mindsim_dump.json",
        help="Path for diagnostic dump file (default: mindsim_dump.json)",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Load a previously-saved session (.npz). Skips understand/research/calibrate/simulate "
             "and applies any --event flags directly to the persisted population.",
    )
    parser.add_argument(
        "--session-out",
        default=None,
        help="Save the post-event session state to this path (.npz).",
    )
    parser.add_argument(
        "--show-evidence",
        action="store_true",
        help="Print the full EvidenceStore (Wave 8.5) at end of run.",
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

    # ── PROXIMITY (Wave 8 relabel) ──
    # Multi-round semantics: under Wave 3+ most decideable agents have
    # already settled, so the legacy "Locked / Convertible / Unreachable"
    # labels misled the reader (Locked: 0% on a successful run). The new
    # phrasing reflects what the buckets actually represent in the final
    # round's holdouts: the convertible pool is the strategic lever.
    prox = sim_result.by_proximity
    prox_text = (
        f"Holdout pool — Decision-ready: [green]{prox.locked*100:.0f}%[/] │ "
        f"Convertible: [yellow]{prox.convertible*100:.0f}%[/] │ "
        f"Stalled: [red]{prox.unreachable*100:.0f}%[/]"
    )
    console.print(prox_text)
    if report.kpi:
        console.print(
            f"  ↳ {report.kpi.convertible_pool} agents sit at "
            f"P(adopt) ∈ [0.40, 0.60] — flippable with small interventions"
        )
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

    # ── INTERVENTIONS (Wave 7) ──
    # Ranked by cost-per-adoption-pp ascending. Combo row (if any) lives
    # at the end and gets its own subsection.
    if report.interventions:
        import math as _math
        singles = [i for i in report.interventions if i.combined_with is None]
        combos = [i for i in report.interventions if i.combined_with is not None]

        console.print("[bold]INTERVENTIONS[/] (ranked by cost-per-adoption-pp)")
        intv_table = Table(show_header=True, box=None, padding=(0, 1))
        intv_table.add_column("#", width=3)
        intv_table.add_column("Name", width=22)
        intv_table.add_column("Lift", width=8, justify="right")
        intv_table.add_column("Rounds", width=6, justify="right")
        intv_table.add_column("Cost", width=14, justify="right")
        intv_table.add_column("$/pp", width=10, justify="right")
        for i, intv in enumerate(singles[:5], 1):
            lift_color = "green" if intv.lift_pp > 0 else "red"
            cost_str = (
                f"${intv.cost_usd_low/1000:.0f}–${intv.cost_usd_high/1000:.0f}k"
                if intv.cost_usd_high > 0
                else "free"
            )
            cpp_str = (
                f"${intv.cost_per_adoption_pp/1000:.1f}k"
                if _math.isfinite(intv.cost_per_adoption_pp)
                else "—"
            )
            intv_table.add_row(
                str(i),
                intv.name,
                f"[{lift_color}]{intv.lift_pp:+.1f}pp[/]",
                str(intv.timeline_rounds),
                cost_str,
                cpp_str,
            )
        console.print(intv_table)

        if combos:
            console.print()
            console.print("[bold]PAIRWISE COMBO[/] (top-2 applied together)")
            for combo in combos:
                tag_color = {
                    "super_additive": "green",
                    "sub_additive": "yellow",
                    "additive": "cyan",
                }.get(combo.additivity or "additive", "cyan")
                console.print(
                    f"  {combo.name}: [{tag_color}]{combo.lift_pp:+.1f}pp[/] "
                    f"([{tag_color}]{combo.additivity}[/])"
                )
                console.print(f"    {combo.mechanism}")
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

    # ── KPI DASHBOARD (Wave 8) ──
    if report.kpi:
        kpi = report.kpi
        kpi_table = Table(show_header=False, box=None, padding=(0, 1))
        kpi_table.add_column("Metric", width=28)
        kpi_table.add_column("Value", style="cyan")
        ttf = kpi.adoption.time_to_50pct
        ttf_str = f"round {ttf}" if ttf is not None else "not reached"
        chasm_str = (
            f"round {kpi.adoption.chasm_round}"
            if kpi.adoption.chasm_round is not None
            else "no stall detected"
        )
        kpi_table.add_row("Time to 50% adoption", ttf_str)
        kpi_table.add_row("Chasm round", chasm_str)
        kpi_table.add_row("Top driver", kpi.force_dominance.top_driver or "—")
        kpi_table.add_row("Top blocker", kpi.force_dominance.top_blocker or "—")
        kpi_table.add_row("Convertible pool", f"{kpi.convertible_pool} agents")
        if kpi.cascade and kpi.cascade.first_cluster_crossed_critical_mass is not None:
            kpi_table.add_row(
                "First cluster to critical mass",
                f"cluster {kpi.cascade.first_cluster_crossed_critical_mass}",
            )
        if kpi.top_sensitivity_params:
            kpi_table.add_row(
                "Top sensitivity params",
                ", ".join(kpi.top_sensitivity_params),
            )
        console.print("[bold]KPI DASHBOARD[/]")
        console.print(kpi_table)
        console.print()

    # ── TRIANGULATED PROS / CONS (Wave 8) ──
    if report.pros_cons:
        triangulated = [p for p in report.pros_cons if p.polarity in ("pro", "con")]
        nuances = [p for p in report.pros_cons if p.polarity == "nuance"]
        if triangulated or nuances:
            console.print("[bold]TRIANGULATED PROS / CONS[/]")
            if triangulated:
                for item in triangulated:
                    color = "green" if item.polarity == "pro" else "red"
                    console.print(f"  [{color}][{item.polarity.upper()}][/] {item.statement}")
                    console.print(f"    [dim]{item.mechanism}[/]")
            if nuances:
                console.print(
                    "  [yellow]One-sided signal — directional only:[/]"
                )
                for item in nuances[:5]:  # cap so the panel stays readable
                    console.print(f"    [dim]• {item.statement}[/]")
            console.print()

    # ── SEGMENT NARRATIVES (Wave 8) ──
    if report.segments and report.segments.segments:
        console.print("[bold]PER-SEGMENT NARRATIVES[/]")
        for seg in report.segments.segments:
            rate_color = (
                "green" if seg.adoption_rate > 0.5
                else "yellow" if seg.adoption_rate > 0.2 else "red"
            )
            console.print(
                f"  [cyan]{seg.archetype}[/] "
                f"[{rate_color}]{seg.adoption_rate*100:.1f}%[/] "
                f"({seg.count}/{seg.total}) — "
                f"driver: {seg.dominant_driver} "
                f"({seg.dominant_driver_value:+.2f}), "
                f"blocker: {seg.dominant_blocker} "
                f"({seg.dominant_blocker_value:+.2f})"
            )
            for traj in seg.representative_trajectories[:2]:
                console.print(f"    [dim]· {traj}[/]")
        console.print()

    # ── AUDIT TEXT ──
    if report.text:
        console.print(Panel(report.text, title="[bold]Behavioral Audit[/]", border_style="blue"))
        console.print()

    console.print(f"[dim]Completed in {elapsed:.1f}s • {sim_result.n_agents} agents[/]")


def _display_evidence_store(sim_result):
    """Wave 8.5 — print the resolvable EvidenceStore."""
    store = getattr(sim_result, "evidence_store", None)
    if store is None or not store.by_id:
        console.print("[dim]No EvidenceStore attached (skip-research run?).[/]")
        return
    console.print()
    console.print(f"[bold]EVIDENCE STORE[/] ({len(store)} records, post-compression)")
    for eid, ev in store.by_id.items():
        type_color = {
            "voc": "magenta", "research": "cyan",
            "trends": "yellow", "default": "dim", "llm_judgment": "blue",
        }.get(ev.source_type, "white")
        console.print(
            f"  [{type_color}]{eid}[/] ({ev.source_type}, conf {ev.confidence:.2f})"
        )
        if ev.source_url:
            console.print(f"    [dim]{ev.source_url}[/]")
        if ev.text:
            console.print(f"    \"{ev.text[:120]}{'…' if len(ev.text) > 120 else ''}\"")
    console.print()


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
