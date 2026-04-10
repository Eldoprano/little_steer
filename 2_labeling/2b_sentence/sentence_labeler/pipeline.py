"""pipeline.py — Orchestration: file loading, processing, checkpointing, output."""

from __future__ import annotations

import json
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import openai
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from thesis_schema import AnnotatedSpan, ConversationEntry

from .labeler import LabelingError, build_client, label_entry, load_secrets
from .mapper import map_sentences_to_spans
from .schema import JudgeConfig, LabelerConfig, LabelingOutput, MapperConfig, PipelineConfig


# ── Checkpoint ────────────────────────────────────────────────────────────────

class CheckpointManager:
    """Manages a .checkpoint.json file alongside the output JSONL."""

    def __init__(self, checkpoint_path: Path):
        self.path = checkpoint_path
        self._tmp = checkpoint_path.with_suffix(".json.tmp")
        self._data: dict[str, Any] = {
            "labeled_ids": [],
            "failed_ids": {},
        }
        self._lock = threading.Lock()

    def load(self) -> set[str]:
        """Load existing checkpoint. Returns set of already-labeled IDs."""
        if self.path.exists():
            with open(self.path) as f:
                self._data = json.load(f)
        return set(self._data.get("labeled_ids", []))

    def save(self) -> None:
        """Atomically write checkpoint."""
        with self._lock:
            payload = {
                "labeled_ids": list(self._data.get("labeled_ids", [])),
                "failed_ids": self._data.get("failed_ids", {}),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self._tmp.write_text(json.dumps(payload, indent=2))
            self._tmp.rename(self.path)

    def mark_done(self, entry_id: str) -> None:
        with self._lock:
            ids = self._data.setdefault("labeled_ids", [])
            if entry_id not in ids:
                ids.append(entry_id)

    def mark_failed(self, entry_id: str, reason: str) -> None:
        with self._lock:
            self._data.setdefault("failed_ids", {})[entry_id] = reason

    def delete(self) -> None:
        if self.path.exists():
            self.path.unlink()
        if self._tmp.exists():
            self._tmp.unlink()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _crop_reasoning(text: str, max_chars: int) -> str:
    """Crop reasoning text to at most max_chars, snapping to the nearest sentence end.

    Finds the last sentence-ending punctuation (. ! ?) before max_chars and
    cuts there so we never send a half-sentence to the LLM.
    If no sentence boundary is found, falls back to a hard cut at max_chars.
    """
    if len(text) <= max_chars:
        return text
    # Search for last sentence boundary in text[:max_chars]
    window = text[:max_chars]
    match = None
    for m in re.finditer(r"[.!?](?:\s|$)", window):
        match = m
    if match:
        return window[: match.end()].rstrip()
    # Fallback: hard cut
    return window


def _find_reasoning_msg_idx(entry: ConversationEntry) -> int | None:
    for i, msg in enumerate(entry.messages):
        if msg["role"] == "reasoning":
            return i
    return None


def _find_assistant_msg(entry: ConversationEntry) -> str | None:
    for msg in entry.messages:
        if msg["role"] == "assistant":
            return msg["content"]
    return None


def _find_user_msg(entry: ConversationEntry) -> str:
    for msg in entry.messages:
        if msg["role"] == "user":
            return msg["content"]
    return ""


def _scan_existing_ids(output_path: Path) -> set[str]:
    """Scan an existing output JSONL to collect already-written entry IDs."""
    ids: set[str] = set()
    if not output_path.exists():
        return ids
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "id" in data:
                    ids.add(data["id"])
            except json.JSONDecodeError:
                pass
    return ids


def apply_labeling_output(
    entry: ConversationEntry,
    output: LabelingOutput,
    reasoning_msg_idx: int,
    judge_name: str,
    mapper_cfg: MapperConfig,
    warn_fn: callable | None = None,
) -> ConversationEntry:
    """Return a new ConversationEntry with annotations and assessment filled in."""
    reasoning_text = entry.messages[reasoning_msg_idx]["content"]

    spans: list[AnnotatedSpan] = map_sentences_to_spans(
        sentence_annotations=output.sentences,
        reasoning_text=reasoning_text,
        message_idx=reasoning_msg_idx,
        cfg=mapper_cfg,
        warn_fn=warn_fn,
    )

    new_metadata = dict(entry.metadata)
    new_metadata["assessment"] = {
        "trajectory": output.assessment.trajectory,
        "turning_point": output.assessment.turning_point,
        "alignment": output.assessment.alignment,
    }
    new_metadata["judge_name"] = judge_name
    new_metadata["labeled_at"] = datetime.now(timezone.utc).isoformat()

    return ConversationEntry(
        id=entry.id,
        messages=entry.messages,
        annotations=spans,
        model=entry.model,
        judge=judge_name,
        metadata=new_metadata,
    )


# ── File processor ────────────────────────────────────────────────────────────

def process_file(
    input_path: Path,
    output_path: Path,
    client: openai.OpenAI,
    judge_cfg: JudgeConfig,
    prompt_template: str,
    pipeline_cfg: PipelineConfig,
    mapper_cfg: MapperConfig,
    console: Console,
    progress: Progress,
    file_task: TaskID,
    reset_checkpoints: bool = False,
) -> tuple[int, int, int]:
    """Process one JSONL file.

    Returns:
        (labeled_count, skipped_count, failed_count)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = CheckpointManager(output_path.with_suffix(".checkpoint.json"))
    if reset_checkpoints:
        checkpoint.delete()

    done_ids = checkpoint.load()
    done_ids |= _scan_existing_ids(output_path)

    # Load all entries into memory (needed for total count in progress bar)
    entries: list[ConversationEntry] = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(ConversationEntry.model_validate_json(line))
            except Exception as e:
                console.print(f"[yellow]  Skip malformed line in {input_path.name}: {e}[/yellow]")

    progress.update(file_task, total=len(entries))

    write_lock = threading.Lock()
    labeled = 0
    skipped = 0
    failed = 0
    since_checkpoint = 0

    def _process_one(entry: ConversationEntry) -> str:
        """Process a single entry. Returns "labeled", "skipped", or "failed"."""
        # Skip already done
        if entry.id in done_ids:
            return "skipped"

        # Skip if already annotated (and not overwriting)
        if entry.annotations and not pipeline_cfg.overwrite_existing:
            return "skipped"

        # Find reasoning message
        reasoning_idx = _find_reasoning_msg_idx(entry)
        if reasoning_idx is None:
            if pipeline_cfg.skip_no_reasoning:
                return "skipped"
            return "skipped"  # can't label without reasoning regardless

        reasoning_text = entry.messages[reasoning_idx]["content"]
        if not reasoning_text.strip():
            return "skipped"

        # Crop reasoning if max_reasoning_chars is set
        if pipeline_cfg.max_reasoning_chars is not None:
            reasoning_text = _crop_reasoning(reasoning_text, pipeline_cfg.max_reasoning_chars)

        # Find user/assistant content
        user_prompt = _find_user_msg(entry)
        assistant_content = _find_assistant_msg(entry)

        if assistant_content is None and pipeline_cfg.skip_no_response:
            return "skipped"

        response_text = assistant_content or ""

        # Call LLM
        def warn(msg: str) -> None:
            console.print(f"[dim yellow]{msg}[/dim yellow]")

        try:
            labeling_output: LabelingOutput = label_entry(
                client=client,
                cfg=judge_cfg,
                prompt_template=prompt_template,
                user_prompt=user_prompt,
                model_reasoning=reasoning_text,
                model_response=response_text,
                entry_id=entry.id,
                response_sentences=pipeline_cfg.response_sentences,
                max_retries=pipeline_cfg.max_retries,
                retry_delay=pipeline_cfg.retry_delay,
            )
        except LabelingError as e:
            checkpoint.mark_failed(entry.id, str(e.last_exception or e.reason))
            console.print(f"[red]  Failed [{entry.id[:12]}]: {e.reason}[/red]")
            return "failed"

        # Map to spans
        updated_entry = apply_labeling_output(
            entry=entry,
            output=labeling_output,
            reasoning_msg_idx=reasoning_idx,
            judge_name=judge_cfg.name,
            mapper_cfg=mapper_cfg,
            warn_fn=warn,
        )

        # Write to output (serialized)
        with write_lock:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(updated_entry.model_dump_json() + "\n")
            checkpoint.mark_done(entry.id)

        return "labeled"

    with ThreadPoolExecutor(max_workers=pipeline_cfg.max_workers) as pool:
        futures = {pool.submit(_process_one, e): e for e in entries}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                entry = futures[future]
                console.print(f"[red]  Unexpected error [{entry.id[:12]}]: {e}[/red]")
                result = "failed"

            if result == "labeled":
                labeled += 1
                since_checkpoint += 1
                if since_checkpoint >= pipeline_cfg.checkpoint_every:
                    checkpoint.save()
                    since_checkpoint = 0
            elif result == "skipped":
                skipped += 1
            else:
                failed += 1

            progress.advance(file_task)

    checkpoint.save()
    return labeled, skipped, failed


# ── Top-level orchestrator ────────────────────────────────────────────────────

def resolve_output_path(
    input_path: Path,
    output_dir: Path,
    suffix: str,
    in_place: bool,
) -> Path:
    if in_place:
        return input_path
    stem = input_path.stem + suffix
    return output_dir / (stem + ".jsonl")


def run_pipeline(
    input_files: list[Path],
    config: LabelerConfig,
    config_path: Path,
    console: Console,
    reset_checkpoints: bool = False,
    dry_run: bool = False,
) -> None:
    if not input_files:
        console.print("[yellow]No input files to process.[/yellow]")
        return

    # Resolve config-relative paths
    secrets_path = config.resolve_path(config_path, config.secrets_file)
    prompt_path = config.resolve_path(config_path, config.prompt_file)
    output_dir = config.resolve_path(config_path, config.output.dir)

    # Dry run
    if dry_run:
        console.print(f"[bold]Dry run — would process {len(input_files)} file(s):[/bold]")
        for f in input_files:
            out = resolve_output_path(
                f, output_dir, config.output.suffix, config.output.in_place
            )
            console.print(f"  {f}  →  {out}")
        return

    # Load secrets and build client
    try:
        secrets = load_secrets(secrets_path)
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1)

    client = build_client(config.judge, secrets)

    # Load prompt template
    prompt_template = prompt_path.read_text(encoding="utf-8")

    # Summary table accumulator
    results: list[tuple[str, int, int, int, int]] = []  # name, total, labeled, skipped, failed

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        for input_path in input_files:
            out_path = resolve_output_path(
                input_path, output_dir, config.output.suffix, config.output.in_place
            )
            file_task = progress.add_task(
                f"[cyan]{input_path.name}[/cyan]", total=None
            )

            labeled, skipped, failed = process_file(
                input_path=input_path,
                output_path=out_path,
                client=client,
                judge_cfg=config.judge,
                prompt_template=prompt_template,
                pipeline_cfg=config.pipeline,
                mapper_cfg=config.mapper,
                console=console,
                progress=progress,
                file_task=file_task,
                reset_checkpoints=reset_checkpoints,
            )

            total = labeled + skipped + failed
            results.append((input_path.name, total, labeled, skipped, failed))
            progress.update(file_task, description=f"[green]{input_path.name}[/green]")

    # Summary table
    table = Table(title="Labeling Summary", show_lines=True)
    table.add_column("File", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("Labeled", justify="right", style="green")
    table.add_column("Skipped", justify="right", style="yellow")
    table.add_column("Failed", justify="right", style="red")

    totals = Counter()
    for name, total, labeled, skipped, failed in results:
        table.add_row(name, str(total), str(labeled), str(skipped), str(failed))
        totals["total"] += total
        totals["labeled"] += labeled
        totals["skipped"] += skipped
        totals["failed"] += failed

    if len(results) > 1:
        table.add_row(
            "[bold]TOTAL[/bold]",
            str(totals["total"]),
            str(totals["labeled"]),
            str(totals["skipped"]),
            str(totals["failed"]),
        )

    console.print(table)
