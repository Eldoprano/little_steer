"""
Label Evolution System — CLI entry point.

Commands:
  run       Run the evolution loop
  visualize Show current taxonomy state and stats
  list      List all runs
  inspect   Show detected split strategy and sample sub-texts for a model
"""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import SubText, balance_index, build_index, discover_models, load_sub_texts, match_models
from labeler import build_history_block, call_labeler, operation_to_taxonomy_op, validate_operation, build_system_prompt, build_user_prompt
from model_profiles import get_profile
from sampling import processed_pool, sample_next_lazy, unprocessed_pool
from state import RUNS_DIR, RunConfig, RunState, list_runs
from taxonomy import TaxonomyOperation

console = Console()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


@click.group()
def cli():
    """Label Evolution System for AI safety research."""
    pass


# ── run ───────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--models", multiple=True, help="Model name substrings to include (required for new run)")
@click.option("--variant", default="both", type=click.Choice(["heretic", "normal", "both"], case_sensitive=False),
              show_default=True, help="heretic=only heretic files, normal=exclude heretic, both=all")
@click.option("--dataset", multiple=True, help="Dataset name substrings (e.g. clear_harm, strong_reject). Default: all except lima.")
@click.option("--include-benign", is_flag=True, default=False, help="Include benign/lima dataset files")
@click.option("--labeler", default=None, help="OpenAI model string, e.g. gpt-4o-mini (required for new run)")
@click.option("--steps", default=100, type=int, show_default=True, help="Number of steps to run")
@click.option("--max-labels", default=20, type=int, show_default=True, help="Max active labels")
@click.option("--seed", "seed_file", default=None, type=click.Path(exists=True), help="Seed taxonomy JSON")
@click.option("--run-id", default=None, help="Resume an existing run by ID")
@click.option("--sampling-seed", default=42, type=int, show_default=True,
              help="RNG seed for reproducible sampling (same seed + same data = same sequence)")
@click.option("--no-history", is_flag=True, default=False, help="Disable recent-changes/graveyard block in prompts")
@click.option("--max-recent-ops", default=3, type=int, show_default=True, help="Recent ops to show in prompt")
@click.option("--max-graveyard", default=5, type=int, show_default=True, help="Max retired labels to show in prompt")
@click.option("--name", default=None, help="Custom run name/ID for a new run (ignored when resuming with --run-id)")
@click.option("--no-viz", is_flag=True, default=False, help="Skip the visualizer prompt after the run finishes")
@click.option("--balance-datasets", is_flag=True, default=False,
              help="Cap each dataset file to the size of the smallest one (equal sampling probability across datasets)")
@click.option("--revisit-rate", default=0.10, type=float, show_default=True,
              help="Fraction of steps that revisit already-seen records (0.0–1.0)")
def run(models, variant, dataset, include_benign, labeler, steps, max_labels, seed_file,
        run_id, sampling_seed, no_history, max_recent_ops, max_graveyard, name, no_viz,
        balance_datasets, revisit_rate):
    """Run the label evolution loop."""
    if run_id:
        # ── Resume Mode ───────────────────────────────────────────────────────
        path = RunState.get_run_path(run_id)
        if not path.exists():
            console.print(f"[red]Run '{run_id}' not found. Cannot resume.[/red]")
            sys.exit(1)
        state = RunState.load(path)
        state_path = path
        config = state.config

        # Allow overrides from CLI
        if models:
            available = discover_models()
            matched = match_models(list(models), available)
            matched = _filter_variant(matched, variant)
            matched = _filter_dataset(matched, list(dataset), include_benign)
            config.models = matched
        else:
            matched = config.models

        if labeler:
            config.labeler = labeler
        
        # Note: max_labels and sampling_seed could also be overridden here if needed,
        # but usually we want to stick to the original config for consistency.
        # We'll stick to what's in config unless explicitly specified (but Click 
        # always provides defaults, so it's tricky without checking param source).
        # For now, we trust the loaded config.
    else:
        # ── New Run Mode ──────────────────────────────────────────────────────
        if not models:
            console.print("[red]Error: --models is required for a new run.[/red]")
            sys.exit(1)
        if not labeler:
            console.print("[red]Error: --labeler is required for a new run.[/red]")
            sys.exit(1)

        available = discover_models()
        matched = match_models(list(models), available)
        matched = _filter_variant(matched, variant)
        matched = _filter_dataset(matched, list(dataset), include_benign)
        if not matched:
            console.print(f"[red]No models matched from: {list(models)} (variant={variant})[/red]")
            console.print(f"Available: {available}")
            sys.exit(1)

        config = RunConfig(
            models=matched,
            labeler=labeler,
            max_labels=max_labels,
            seed_file=seed_file,
            sampling_seed=sampling_seed,
        )
        state, state_path = RunState.load_or_create(name, config)

    # ── Initialization ────────────────────────────────────────────────────────

    is_resume = state.stats["steps_completed"] > 0
    if is_resume:
        console.print(f"[yellow]Resuming run {state.run_id} from step {state.stats['steps_completed']}[/yellow]")
    else:
        console.print(f"[green]New run: {state.run_id}[/green]")
    
    console.print(f"[bold]Models:[/bold] {matched}")
    console.print(f"[dim]State file: {state_path}[/dim]")
    console.print(f"[dim]Labeler: {config.labeler}[/dim]")
    console.print(f"[dim]Sampling seed: {config.sampling_seed}[/dim]")

    if seed_file:
        state.load_seed(seed_file)
        console.print(f"Loaded {len(state.taxonomy.active)} labels from seed")
    
    if not is_resume:
        state.snapshot(0)
        state.save(state_path)

    # Build lazy index
    console.print(f"Building index from {len(matched)} model file(s)...")
    index = build_index(matched)
    if not index:
        console.print("[red]No records found. Check model names and data directory.[/red]")
        sys.exit(1)
    console.print(f"Indexed {len(index)} records")

    rng = random.Random(config.sampling_seed + state.stats["steps_completed"])

    if balance_datasets:
        pre_balance = len(index)
        index = balance_index(index, rng)
        by_stem_counts = {}
        for ref in index.values():
            by_stem_counts[ref.stem] = by_stem_counts.get(ref.stem, 0) + 1
        min_n = min(by_stem_counts.values()) if by_stem_counts else 0
        console.print(f"[dim]Balanced: {pre_balance} → {len(index)} records ({min_n} per dataset)[/dim]")

    # Append-only response backup — survives crashes, one line per API call
    response_log = state_path.parent / "responses.jsonl"

    # ── Main Loop ─────────────────────────────────────────────────────────────

    start_step = state.stats["steps_completed"]
    try:
        for i in range(steps):
            step = start_step + i + 1

            try:
                sub_text, prev_sub_text, is_revisit = sample_next_lazy(index, state, rng=rng, revisit_fraction=revisit_rate)
            except RuntimeError as e:
                console.print(f"[red]Sampling error: {e}[/red]")
                break

            history_block = ""
            if not no_history:
                history_block = build_history_block(
                    state.history["operations"],
                    state.taxonomy.graveyard,
                    max_recent_ops=max_recent_ops,
                    max_graveyard=max_graveyard,
                )

            response = None
            for _attempt in range(3):
                try:
                    response = call_labeler(config.labeler, sub_text, state.taxonomy, step, config.max_labels,
                                            prev_sub_text=prev_sub_text,
                                            history_block=history_block,
                                            response_log=response_log)
                    # Retry if completely empty — no labels AND no ops (unusable response)
                    if not response.labels and not response.operations:
                        if _attempt < 2:
                            logging.debug("Step %d attempt %d: empty response, retrying", step, _attempt + 1)
                            continue
                    break
                except Exception as e:
                    if _attempt < 2:
                        logging.warning("Step %d attempt %d failed: %s — retrying", step, _attempt + 1, e)
                        continue
                    console.print(f"[red]Step {step} error: {e}[/red]")
                    state.stats["errors"] += 1
                    state.stats["steps_completed"] += 1
                    state.save(state_path)
                    response = None
                    break

            if response is None:
                continue

            # Snapshot the taxonomy BEFORE applying any operations (what the model saw)
            taxonomy_at_step = [
                {"label_id": e.label_id, "name": e.name, "description": e.description,
                 "usage_count": e.usage_count, "created_at_step": e.created_at_step}
                for e in state.taxonomy.active.values()
            ]

            # Process operations
            triggered_change = False
            applied_ops: list[TaxonomyOperation] = []
            rejected_ops_log: list[dict] = []
            rejection_hints: list[str] = []

            for raw_op in response.operations:
                tax_op = operation_to_taxonomy_op(raw_op, sub_text, step, response.justification)
                valid, reason = validate_operation(tax_op, state.taxonomy, config.max_labels)
                if valid:
                    state.taxonomy.apply_operation(tax_op)
                    state.record_operation(tax_op, triggered_change=True)
                    applied_ops.append(tax_op)
                    triggered_change = True
                else:
                    state.stats["total_invalid_proposals"] += 1
                    rejection_hints.append(f"{tax_op.operation} rejected: {reason}")
                    rejected_ops_log.append({"op": tax_op.model_dump(), "reason": reason})
                    rejected = TaxonomyOperation(
                        step=step, operation="NONE", details={},
                        triggered_by=tax_op.triggered_by,
                        justification=f"[rejected: {reason}] {tax_op.justification}",
                    )
                    state.record_operation(rejected, triggered_change=False)

            # Silent retry if any ops were rejected
            if rejection_hints:
                try:
                    retried = call_labeler(
                        config.labeler, sub_text, state.taxonomy, step, config.max_labels,
                        prev_sub_text=prev_sub_text,
                        history_block=history_block,
                        response_log=response_log,
                    )
                    for raw_op in retried.operations:
                        tax_op = operation_to_taxonomy_op(raw_op, sub_text, step, retried.justification)
                        valid, reason = validate_operation(tax_op, state.taxonomy, config.max_labels)
                        if valid:
                            state.taxonomy.apply_operation(tax_op)
                            state.record_operation(tax_op, triggered_change=True)
                            applied_ops.append(tax_op)
                            triggered_change = True
                        else:
                            state.stats["total_invalid_proposals"] += 1
                except Exception as e:
                    logging.warning("Retry after rejection failed: %s", e)

            # If no operations proposed, record a NONE
            if not response.operations:
                none_op = TaxonomyOperation(
                    step=step, operation="NONE", details={},
                    triggered_by={"text": sub_text.text[:300],
                                  "composite_id": sub_text.composite_id,
                                  "sub_text_idx": sub_text.sub_text_idx},
                    justification=response.justification,
                )
                state.record_operation(none_op, triggered_change=False)

            state.taxonomy.increment_usage(response.labels)

            state.mark_sub_text(
                composite_id=sub_text.composite_id,
                sub_text_idx=sub_text.sub_text_idx,
                labels=response.labels,
                triggered_change=triggered_change,
                is_revisit=is_revisit,
            )

            system_prompt = build_system_prompt(config.max_labels, not state.taxonomy.within_limit(config.max_labels))
            user_prompt = build_user_prompt(sub_text, state.taxonomy, step, prev_sub_text, history_block)
            prompt_log = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # Save full per-step context for the visualizer
            state.log_step({
                "step": step,
                "is_revisit": is_revisit,
                "prompt": prompt_log,
                "sub_text": {
                    "text": sub_text.text,
                    "composite_id": sub_text.composite_id,
                    "sub_text_idx": sub_text.sub_text_idx,
                    "model_stem": sub_text.model_stem,
                    "user_prompt": sub_text.user_prompt,
                },
                "prev_sub_text": {"text": prev_sub_text.text} if prev_sub_text else None,
                "taxonomy_at_step": taxonomy_at_step,
                "thinking": response.thinking,
                "internal_reasoning": response.internal_reasoning,
                "labels": response.labels,
                "justification": response.justification,
                "raw_operations": [op if isinstance(op, dict) else {"type": op}
                                   for op in response.operations],
                "applied_ops": [op.model_dump() for op in applied_ops],
                "rejected_ops": rejected_ops_log,
            })

            state.stats["steps_completed"] += 1

            if step % 25 == 0:
                state.snapshot(step)

            state.save(state_path)

            _print_step(step, is_revisit, response.labels, applied_ops)

    except KeyboardInterrupt:
        console.print(f"\n[yellow]Interrupted at step {state.stats['steps_completed']}.[/yellow]")
        console.print(f"[bold yellow]To resume this run, use:[/bold yellow]")
        console.print(f"  [cyan]uv run run.py run --run-id {state.run_id}[/cyan]")
        sys.exit(0)

    console.print(f"\n[bold green]Done.[/bold green] State saved to {state_path}")
    _print_summary(state)

    # Offer to open the visualizer
    if not no_viz:
        try:
            answer = console.input(
                f"\n[bold cyan]Open visualizer for run [yellow]{state.run_id}[/yellow]? [Y/n][/bold cyan] "
            ).strip().lower()
            if answer in ("", "y", "yes"):
                import subprocess
                subprocess.Popen(
                    [sys.executable, str(Path(__file__).parent / "visualizer.py"), "--run-id", state.run_id],
                    start_new_session=True,
                )
                console.print(f"[dim]Visualizer starting at http://localhost:7860?run={state.run_id}[/dim]")
        except (EOFError, KeyboardInterrupt):
            pass  # non-interactive environment, skip prompt


def _print_step(step: int, is_revisit: bool, labels: list[str], applied_ops: list[TaxonomyOperation]) -> None:
    """Print one step line, plus extra lines for newly created labels."""
    revisit_marker = "[dim][R][/dim]" if is_revisit else "   "

    # Build compact op tags
    op_tags = []
    for tax_op in applied_ops:
        t = tax_op.operation
        d = tax_op.details
        if t == "CREATE":
            op_tags.append(f"[bold green]+{d.get('name', '?')}[/bold green]")
        elif t == "MERGE":
            result = d.get("result", "?")
            sources = d.get("sources", [])
            op_tags.append(f"[blue]merge({'+'.join(sources)})→{result}[/blue]")
        elif t == "SPLIT":
            source = d.get("source", "?")
            results = [r.get("name", "?") for r in d.get("results", [])]
            op_tags.append(f"[cyan]split {source}→{','.join(results)}[/cyan]")
        elif t == "RENAME":
            op_tags.append(f"[yellow]rename {d.get('old', '?')}→{d.get('new', '?')}[/yellow]")
        elif t == "DELETE":
            op_tags.append(f"[red]-{d.get('name', '?')}[/red]")

    ops_str = " ".join(op_tags) if op_tags else "[dim]NONE[/dim]"
    labels_str = f"[dim]{labels}[/dim]" if labels else "[dim][][/dim]"

    console.print(f"Step {step:4d} {revisit_marker} | labels: {labels_str} | {ops_str}")

    # Extra line for each new CREATE label
    for tax_op in applied_ops:
        if tax_op.operation == "CREATE":
            name = tax_op.details.get("name", "?")
            desc = tax_op.details.get("description", "")
            console.print(f"       [bold green]NEW[/bold green] [cyan]{name}[/cyan]: {desc}")


# ── visualize ─────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--run-id", default=None, help="Run ID to visualize (default: most recent)")
def visualize(run_id):
    """Show current taxonomy state and run statistics."""
    state, _ = _load_run(run_id)

    console.print(f"\n[bold]Run:[/bold] {state.run_id}  |  Created: {state.created_at}")
    console.print(f"[bold]Labeler:[/bold] {state.config.labeler}  |  Max labels: {state.config.max_labels}")
    console.print(f"[bold]Models:[/bold] {', '.join(state.config.models)}")
    console.print()

    t = Table(title=f"Active Taxonomy ({len(state.taxonomy.active)} labels)", show_lines=True)
    t.add_column("Label", style="cyan", no_wrap=True)
    t.add_column("Description")
    t.add_column("Uses", justify="right")
    for name, entry in sorted(state.taxonomy.active.items(), key=lambda x: -x[1].usage_count):
        t.add_row(name, entry.description, str(entry.usage_count))
    console.print(t)

    s = state.stats
    console.print(f"\n[bold]Steps completed:[/bold] {s['steps_completed']}")
    console.print(f"[bold]Taxonomy changes:[/bold] {s['total_changes']}")
    console.print(f"[bold]Revisits:[/bold] {s['total_revisits']}")
    console.print(f"[bold]Errors:[/bold] {s['errors']}")
    console.print(f"[bold]Invalid proposals:[/bold] {s.get('total_invalid_proposals', 0)}")

    ops = [o for o in state.history["operations"] if o["operation"] != "NONE"]
    if ops:
        console.print()
        ot = Table(title=f"Taxonomy Operations ({len(ops)} changes)", show_lines=True)
        ot.add_column("Step", justify="right")
        ot.add_column("Op", style="yellow")
        ot.add_column("Details")
        for op in ops[-20:]:
            details = op.get("details", {})
            if op["operation"] == "CREATE":
                detail_str = f"{details.get('name', '?')}: {details.get('description', '')[:60]}"
            elif op["operation"] == "MERGE":
                detail_str = f"{details.get('sources', [])} → {details.get('result', '?')}"
            elif op["operation"] == "SPLIT":
                results = [r.get("name", "?") for r in details.get("results", [])]
                detail_str = f"{details.get('source', '?')} → {results}"
            elif op["operation"] == "RENAME":
                detail_str = f"{details.get('old', '?')} → {details.get('new', '?')}"
            elif op["operation"] == "DELETE":
                detail_str = details.get("name", "?")
            else:
                detail_str = str(details)[:80]
            ot.add_row(str(op["step"]), op["operation"], detail_str)
        console.print(ot)

    if state.taxonomy.graveyard:
        console.print(f"\n[dim]Graveyard: {list(state.taxonomy.graveyard.keys())}[/dim]")


# ── list ──────────────────────────────────────────────────────────────────────

@cli.command(name="list")
def list_runs_cmd():
    """List all runs."""
    runs = list_runs()
    if not runs:
        console.print("No runs found.")
        return

    t = Table(title="All Runs", show_lines=True)
    t.add_column("Run ID", style="cyan")
    t.add_column("Created")
    t.add_column("Steps", justify="right")
    t.add_column("Labels", justify="right")
    t.add_column("Changes", justify="right")
    t.add_column("Seed", justify="right")
    t.add_column("Labeler")
    t.add_column("Models")

    for r in runs:
        if "error" in r:
            t.add_row(r["run_id"], "ERROR", "-", "-", "-", "-", "-", "-")
        else:
            t.add_row(
                r["run_id"],
                r["created_at"][:19],
                str(r["steps"]),
                str(r["n_labels"]),
                str(r["n_changes"]),
                str(r.get("sampling_seed", "?")),
                r["labeler"],
                ", ".join(r["models"][:2]) + ("..." if len(r["models"]) > 2 else ""),
            )
    console.print(t)


# ── inspect ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--model", required=True, help="Model name substring to inspect")
@click.option("--run-id", default=None, help="Run ID for per-run stats (optional)")
def inspect(model, run_id):
    """Show detected split strategy and sample sub-texts for a model."""
    available = discover_models()
    matched = match_models([model], available)
    if not matched:
        console.print(f"[red]No models matched '{model}'[/red]")
        console.print(f"Available: {available}")
        sys.exit(1)

    console.print(f"\n[bold]Matched files:[/bold] {matched}")

    for stem in matched:
        profile = get_profile(stem)
        console.print(f"\n[bold cyan]{stem}[/bold cyan]")
        console.print(f"  Split strategy: {profile.description}")
        console.print(f"  Pattern matched: '{profile.pattern}'")

    console.print("\nLoading sub-texts...")
    sub_texts = load_sub_texts(matched)
    total = len(sub_texts)
    console.print(f"Total sub-texts: {total}")

    if not sub_texts:
        return

    lengths = [len(st.text) for st in sub_texts]
    short = sum(1 for l in lengths if l < 50)
    medium = sum(1 for l in lengths if 50 <= l < 200)
    long_ = sum(1 for l in lengths if l >= 200)
    console.print(f"Length stats: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)//len(lengths)}")
    console.print(f"Short (<50): {short}  Medium (50-200): {medium}  Long (>200): {long_}")

    if run_id:
        state, _ = _load_run(run_id)
        model_subs = [st for st in sub_texts if st.model_stem in matched]
        processed = sum(1 for st in model_subs if st.composite_id in state.processed)
        pct = 100 * processed // total if total else 0
        console.print(f"\n[bold]Run {run_id}:[/bold] {processed}/{total} sub-texts processed ({pct}%)")

        label_counts: dict[str, int] = {}
        for cid, cdata in state.processed.items():
            if not any(cid.startswith(m + "::") for m in matched):
                continue
            for entry in cdata.get("sub_texts", []):
                for lbl in entry.get("labels", []):
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1
        if label_counts:
            lt = Table(title="Label Distribution", show_lines=False)
            lt.add_column("Label", style="cyan")
            lt.add_column("Count", justify="right")
            for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
                lt.add_row(lbl, str(cnt))
            console.print(lt)

        model_ops = [
            op for op in state.history["operations"]
            if op["operation"] != "NONE"
            and any(op["triggered_by"]["composite_id"].startswith(m + "::") for m in matched)
        ]
        console.print(f"\nOperations triggered by this model: {len(model_ops)} / {state.stats['total_changes']} total")

    import random as _random
    sample = _random.sample(sub_texts, min(5, len(sub_texts)))
    console.print("\n[bold]Sample sub-texts:[/bold]")
    for st in sample:
        console.print(f"\n  [{st.model_stem} | {st.sample_id} | seg {st.sub_text_idx}]")
        console.print(f"  {st.text[:200]!r}{'...' if len(st.text) > 200 else ''}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _filter_variant(stems: list[str], variant: str) -> list[str]:
    if variant == "both":
        return stems
    if variant == "heretic":
        return [s for s in stems if "heretic" in s.lower()]
    return [s for s in stems if "heretic" not in s.lower()]


def _filter_dataset(stems: list[str], datasets: list[str], include_benign: bool) -> list[str]:
    BENIGN_MARKERS = ("lima",)
    if datasets:
        return [s for s in stems if any(d.lower() in s.lower() for d in datasets)]
    if not include_benign:
        return [s for s in stems if not any(m in s.lower() for m in BENIGN_MARKERS)]
    return stems


def _load_run(run_id: str | None) -> tuple[RunState, Path]:
    if run_id:
        path = RunState.get_run_path(run_id)
        if not path.exists():
            console.print(f"[red]Run '{run_id}' not found[/red]")
            sys.exit(1)
        return RunState.load(path), path

    runs = list_runs()
    if not runs:
        console.print("[red]No runs found[/red]")
        sys.exit(1)
    valid = [r for r in runs if "error" not in r]
    if not valid:
        console.print("[red]No valid runs found[/red]")
        sys.exit(1)
    latest = sorted(valid, key=lambda r: r["created_at"], reverse=True)[0]
    path = Path(latest["path"])
    return RunState.load(path), path


def _print_summary(state: RunState) -> None:
    s = state.stats
    n = len(state.taxonomy.active)
    console.print(
        f"\nSteps: {s['steps_completed']}  |  "
        f"Labels: {n}/{state.config.max_labels}  |  "
        f"Changes: {s['total_changes']}  |  "
        f"Invalid: {s.get('total_invalid_proposals', 0)}  |  "
        f"Errors: {s['errors']}"
    )


if __name__ == "__main__":
    cli()
