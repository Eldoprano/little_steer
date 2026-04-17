"""pipeline.py — Orchestration: file loading, processing, checkpointing, output."""

from __future__ import annotations

import json
import random
import re
import threading
import time
from collections import Counter, deque
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

from .labeler import BillingQuotaError, LabelingError, build_client, format_prompt, label_entry, load_secrets
from .mapper import map_sentences_to_spans
from .schema import JudgeConfig, LabelerConfig, LabelingOutput, MapperConfig, PipelineConfig
from .taxonomy_loader import get_taxonomy_version, inject_taxonomy


# ── Token budget ──────────────────────────────────────────────────────────────

class TokenBudget:
    """Thread-safe daily token budget tracker with file persistence."""

    def __init__(self, daily_limit: int, state_file: Path):
        self.daily_limit = daily_limit
        self.safety_limit = int(daily_limit * 0.95)  # Stop at 95% to avoid overages with concurrent workers
        self.state_file = state_file
        self._lock = threading.Lock()
        self._today = datetime.now(timezone.utc).date().isoformat()
        self._data: dict[str, int] = {}
        self._load()
        self.stop_event = threading.Event()
        if self.tokens_used_today() >= self.safety_limit:
            self.stop_event.set()

    def _load(self) -> None:
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def _save(self) -> None:
        tmp = self.state_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._data, indent=2))
        tmp.rename(self.state_file)

    def tokens_used_today(self) -> int:
        return self._data.get(self._today, 0)

    def remaining(self) -> int:
        return max(0, self.safety_limit - self.tokens_used_today())

    def can_proceed(self) -> bool:
        return not self.stop_event.is_set()

    def can_afford(self, estimated_tokens: int) -> bool:
        """Return False if spending estimated_tokens would push us over the safety limit."""
        with self._lock:
            return self._data.get(self._today, 0) + estimated_tokens <= self.safety_limit

    def consume(self, tokens: int) -> None:
        with self._lock:
            self._data[self._today] = self._data.get(self._today, 0) + tokens
            self._save()
            if self._data[self._today] >= self.safety_limit:
                self.stop_event.set()


# ── Request rate limiter (RPM + RPD) ──────────────────────────────────────────

class RequestRateLimiter:
    """Thread-safe RPM + RPD rate limiter with persistent daily request tracking.

    RPM is enforced via a sliding-window queue of recent request timestamps.
    Workers calling ``acquire()`` block until a slot is available.
    RPD is tracked in a JSON state file; once the daily limit is reached the
    ``stop_event`` is set so workers return early without acquiring new slots.
    """

    def __init__(self, rpm: int, rpd: int, state_file: Path):
        self.rpm = rpm
        self.rpd = rpd
        self.state_file = state_file
        self._lock = threading.Lock()
        self._today = datetime.now(timezone.utc).date().isoformat()
        self._rpm_window: deque[float] = deque()
        self._daily_counts: dict[str, int] = {}
        self._load()
        self.stop_event = threading.Event()
        if self.requests_today() >= self.rpd:
            self.stop_event.set()

    def _load(self) -> None:
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    self._daily_counts = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._daily_counts = {}

    def _save(self) -> None:
        tmp = self.state_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._daily_counts, indent=2))
        tmp.rename(self.state_file)

    def requests_today(self) -> int:
        return self._daily_counts.get(self._today, 0)

    def remaining_today(self) -> int:
        return max(0, self.rpd - self.requests_today())

    def can_proceed(self) -> bool:
        return not self.stop_event.is_set()

    def acquire(self) -> bool:
        """Block until an RPM slot is available. Returns False if RPD limit hit."""
        while True:
            if not self.can_proceed():
                return False
            with self._lock:
                now = time.monotonic()
                cutoff = now - 60.0
                while self._rpm_window and self._rpm_window[0] <= cutoff:
                    self._rpm_window.popleft()
                if len(self._rpm_window) < self.rpm:
                    self._rpm_window.append(now)
                    return True
                wait_until = self._rpm_window[0] + 60.1
            sleep_secs = wait_until - time.monotonic()
            if sleep_secs > 0:
                time.sleep(min(sleep_secs, 5.0))

    def record(self) -> None:
        """Increment the daily request counter after a completed API call."""
        with self._lock:
            self._daily_counts[self._today] = self._daily_counts.get(self._today, 0) + 1
            self._save()
            if self._daily_counts[self._today] >= self.rpd:
                self.stop_event.set()

    def exhaust_today(self) -> None:
        """Record that the billing API quota was exhausted today.

        Uses a separate flag rather than inflating the request counter, so the
        dashboard still shows the real number of successful calls made today.
        On the next run, the flag is detected and the judge skips immediately.
        """
        with self._lock:
            self._daily_counts[f"{self._today}:billing_exhausted"] = True
            self._save()
            self.stop_event.set()

    def is_billing_exhausted_today(self) -> bool:
        return bool(self._daily_counts.get(f"{self._today}:billing_exhausted", False))


# ── Entry limit ───────────────────────────────────────────────────────────────

class EntryLimit:
    """Semaphore that ensures exactly N entries are labeled across all files."""

    def __init__(self, max_entries: int):
        self.max_entries = max_entries
        self._sem = threading.Semaphore(max_entries)
        self.stop_event = threading.Event()

    def can_proceed(self) -> bool:
        return not self.stop_event.is_set()

    def try_acquire(self) -> bool:
        """Atomically claim one entry slot. Returns False if limit already reached."""
        if self.stop_event.is_set():
            return False
        acquired = self._sem.acquire(blocking=False)
        if not acquired:
            self.stop_event.set()
        return acquired


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

    @staticmethod
    def _is_transient_error(reason: str) -> bool:
        """Return True if the failure reason is transient (quota/rate-limit), not permanent."""
        r = reason.lower()
        return any(kw in r for kw in (
            "quota", "billing", "check your plan", "429", "rate limit", "resource_exhausted",
        ))

    def load(self) -> set[str]:
        """Load existing checkpoint. Returns set of already-labeled IDs.

        Transient failures (quota, billing, 429) are purged from failed_ids on load
        so they are retried on the next run rather than staying stuck permanently.
        """
        if self.path.exists():
            with open(self.path) as f:
                self._data = json.load(f)

        # Purge transient failures — they were caused by quota/rate-limit, not bad data.
        # They'll be retried fresh next run; if quota is available they'll succeed.
        failed = self._data.get("failed_ids", {})
        if isinstance(failed, dict):
            permanent = {k: v for k, v in failed.items() if not self._is_transient_error(v)}
            if len(permanent) < len(failed):
                self._data["failed_ids"] = permanent

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
    taxonomy_version: str = "",
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
    if taxonomy_version:
        new_metadata["taxonomy_version"] = taxonomy_version

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
    system_prompt_text: str | None,
    pipeline_cfg: PipelineConfig,
    mapper_cfg: MapperConfig,
    console: Console,
    progress: Progress,
    file_task: TaskID,
    reset_checkpoints: bool = False,
    token_budget: TokenBudget | None = None,
    rate_limiter: RequestRateLimiter | None = None,
    entry_limit: EntryLimit | None = None,
    per_file_max: int | None = None,
    allowed_ids: list[str] | None = None,
    taxonomy_version: str = "",
) -> tuple[int, int, int, int]:
    """Process one JSONL file.

    Returns:
        (labeled_count, skipped_count, failed_count, budget_stopped_count)
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

    # When a work order is active, restrict to the specified IDs and honour their order.
    # This ensures all judges target the same entries regardless of per-judge skips.
    if allowed_ids is not None:
        id_to_entry = {e.id: e for e in entries}
        entries = [id_to_entry[eid] for eid in allowed_ids if eid in id_to_entry]

    progress.update(file_task, total=len(entries))

    write_lock = threading.Lock()
    labeled = 0
    skipped = 0
    failed = 0
    budget_stopped = 0
    since_checkpoint = 0

    # Per-file entry cap: tracks successful labels (not attempts) so that
    # content-filter failures don't eat the per-file budget.
    per_file_labeled = [0]
    per_file_stop = threading.Event()

    def _process_one(entry: ConversationEntry) -> str:
        """Process a single entry. Returns "labeled", "skipped", "budget", or "failed"."""
        # Skip already done
        if entry.id in done_ids:
            return "skipped"

        # Quick limit checks before any heavy work
        if token_budget is not None and not token_budget.can_proceed():
            return "budget"
        if rate_limiter is not None and not rate_limiter.can_proceed():
            return "budget"
        if entry_limit is not None and not entry_limit.can_proceed():
            return "budget"
        if per_file_stop.is_set():
            return "budget"

        # Skip entries flagged as low quality (max_length, repetition, etc.)
        # approved=False is set by the generator; missing key means legacy entry (treat as ok).
        if entry.metadata.get("approved") is False:
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

        # Pre-flight token estimate: skip if this request would push us over the daily budget.
        # Estimate = prompt chars / 4 (rough token heuristic) + max output tokens.
        if token_budget is not None:
            estimated_prompt_tokens = (
                len(system_prompt_text or "") +
                len(format_prompt(
                    prompt_template, user_prompt, reasoning_text, response_text,
                    pipeline_cfg.response_sentences,
                ))
            ) // 4
            max_comp = judge_cfg.max_completion_tokens or judge_cfg.max_tokens or 8192
            if not token_budget.can_afford(estimated_prompt_tokens + max_comp):
                token_budget.stop_event.set()
                return "budget"

        # Atomically claim one entry slot (prevents over-labeling with concurrent workers)
        if entry_limit is not None:
            if not entry_limit.try_acquire():
                return "budget"

        # Acquire RPM slot — blocks until a slot is free, returns False if RPD hit
        if rate_limiter is not None:
            if not rate_limiter.acquire():
                return "budget"

        # Call LLM
        def warn(msg: str) -> None:
            console.print(f"[dim yellow]{msg}[/dim yellow]")

        try:
            is_unsafe = entry.metadata.get("prompt_safety") == "unsafe"
            
            labeling_output: LabelingOutput = label_entry(
                client=client,
                cfg=judge_cfg,
                prompt_template=prompt_template,
                user_prompt=user_prompt,
                model_reasoning=reasoning_text,
                model_response=response_text,
                entry_id=entry.id,
                system_prompt=system_prompt_text,
                response_sentences=pipeline_cfg.response_sentences,
                max_retries=pipeline_cfg.max_retries,
                retry_delay=pipeline_cfg.retry_delay,
                is_unsafe=is_unsafe,
            )
        except BillingQuotaError as e:
            console.print(f"[bold red]  Billing quota exceeded — stopping judge.[/bold red]")
            if rate_limiter is not None:
                rate_limiter.stop_event.set()
            if token_budget is not None:
                token_budget.stop_event.set()
            return "budget"
        except LabelingError as e:
            detail = str(e.last_exception) if e.last_exception else e.reason
            checkpoint.mark_failed(entry.id, detail)
            console.print(f"[red]  Failed [{entry.id[:12]}]: {detail[:120]}[/red]")
            return "failed"

        # Update limiters after a successful call
        if rate_limiter is not None:
            rate_limiter.record()
        if token_budget is not None:
            tokens_used = labeling_output.metadata.get("usage", {}).get("total_tokens", 0)
            if tokens_used:
                token_budget.consume(tokens_used)
        # entry_limit: slot was already claimed atomically via try_acquire()

        # Map to spans
        updated_entry = apply_labeling_output(
            entry=entry,
            output=labeling_output,
            reasoning_msg_idx=reasoning_idx,
            judge_name=judge_cfg.name,
            mapper_cfg=mapper_cfg,
            warn_fn=warn,
            taxonomy_version=taxonomy_version,
        )

        # Write to output (serialized)
        with write_lock:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(updated_entry.model_dump_json() + "\n")
            checkpoint.mark_done(entry.id)
            per_file_labeled[0] += 1
            if per_file_max is not None and per_file_labeled[0] >= per_file_max:
                per_file_stop.set()

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
            elif result == "budget":
                budget_stopped += 1
            else:
                failed += 1

            progress.advance(file_task)

    checkpoint.save()
    return labeled, skipped, failed, budget_stopped


# ── Breadth-first processor ───────────────────────────────────────────────────

def process_breadth_first(
    flat_order: list[dict[str, str]],
    input_dir: Path,
    output_dir: Path,
    output_suffix: str,
    client: openai.OpenAI,
    judge_cfg: JudgeConfig,
    prompt_template: str,
    system_prompt_text: str | None,
    pipeline_cfg: PipelineConfig,
    mapper_cfg: MapperConfig,
    console: Console,
    progress: Progress,
    task_id: TaskID,
    reset_checkpoints: bool = False,
    token_budget: TokenBudget | None = None,
    rate_limiter: RequestRateLimiter | None = None,
    entry_limit: EntryLimit | None = None,
    taxonomy_version: str = "",
) -> tuple[int, int, int, int]:
    """Process entries from a breadth-first flat_order across multiple source files.

    Entries are processed in the interleaved order defined by flat_order, which
    alternates across all source files (round-robin).  A budget-limited run will
    therefore label a few entries from every file before going deeper into any one.

    Returns (labeled, skipped, failed, budget_stopped).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect unique source files (preserving appearance order)
    file_names: list[str] = list(dict.fromkeys(item["file"] for item in flat_order))

    # Per-file output paths, checkpoints, write locks
    output_paths: dict[str, Path] = {}
    checkpoints: dict[str, CheckpointManager] = {}
    done_ids_per_file: dict[str, set[str]] = {}
    write_locks: dict[str, threading.Lock] = {}

    for fname in file_names:
        out = output_dir / (Path(fname).stem + output_suffix + ".jsonl")
        output_paths[fname] = out
        cp = CheckpointManager(out.with_suffix(".checkpoint.json"))
        if reset_checkpoints:
            cp.delete()
        checkpoints[fname] = cp
        done_ids_per_file[fname] = cp.load() | _scan_existing_ids(out)
        write_locks[fname] = threading.Lock()

    # Load all source entries into memory
    all_entries: dict[str, dict[str, ConversationEntry]] = {}
    for fname in file_names:
        fpath = input_dir / fname
        entries: dict[str, ConversationEntry] = {}
        if fpath.exists():
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = ConversationEntry.model_validate_json(line)
                        entries[e.id] = e
                    except Exception as exc:
                        console.print(f"[yellow]  Skip malformed line in {fname}: {exc}[/yellow]")
        all_entries[fname] = entries

    # Filter to entries that still need labeling
    todo = [
        item for item in flat_order
        if item["file"] in all_entries
        and item["id"] in all_entries[item["file"]]
        and item["id"] not in done_ids_per_file.get(item["file"], set())
    ]
    progress.update(task_id, total=len(todo))

    labeled = skipped = failed = budget_stopped = 0
    since_checkpoint = 0

    def _process_item(item: dict[str, str]) -> str:
        fname = item["file"]
        eid = item["id"]
        entry = all_entries[fname].get(eid)

        if entry is None or eid in done_ids_per_file.get(fname, set()):
            return "skipped"
        if token_budget is not None and not token_budget.can_proceed():
            return "budget"
        if rate_limiter is not None and not rate_limiter.can_proceed():
            return "budget"
        if entry_limit is not None and not entry_limit.can_proceed():
            return "budget"
        # Skip entries flagged as low quality.
        if entry.metadata.get("approved") is False:
            return "skipped"
        if entry.annotations and not pipeline_cfg.overwrite_existing:
            return "skipped"

        reasoning_idx = _find_reasoning_msg_idx(entry)
        if reasoning_idx is None:
            return "skipped"
        reasoning_text = entry.messages[reasoning_idx]["content"]
        if not reasoning_text.strip():
            return "skipped"
        if pipeline_cfg.max_reasoning_chars is not None:
            reasoning_text = _crop_reasoning(reasoning_text, pipeline_cfg.max_reasoning_chars)

        user_prompt = _find_user_msg(entry)
        assistant_content = _find_assistant_msg(entry)
        if assistant_content is None and pipeline_cfg.skip_no_response:
            return "skipped"
        response_text = assistant_content or ""

        if token_budget is not None:
            estimated = (
                len(system_prompt_text or "") +
                len(format_prompt(
                    prompt_template, user_prompt, reasoning_text, response_text,
                    pipeline_cfg.response_sentences,
                ))
            ) // 4
            max_comp = judge_cfg.max_completion_tokens or judge_cfg.max_tokens or 8192
            if not token_budget.can_afford(estimated + max_comp):
                token_budget.stop_event.set()
                return "budget"

        if entry_limit is not None and not entry_limit.try_acquire():
            return "budget"
        if rate_limiter is not None and not rate_limiter.acquire():
            return "budget"

        def warn(msg: str) -> None:
            console.print(f"[dim yellow]{msg}[/dim yellow]")

        try:
            is_unsafe = entry.metadata.get("prompt_safety") == "unsafe"
            
            labeling_output = label_entry(
                client=client,
                cfg=judge_cfg,
                prompt_template=prompt_template,
                user_prompt=user_prompt,
                model_reasoning=reasoning_text,
                model_response=response_text,
                entry_id=eid,
                system_prompt=system_prompt_text,
                response_sentences=pipeline_cfg.response_sentences,
                max_retries=pipeline_cfg.max_retries,
                retry_delay=pipeline_cfg.retry_delay,
                is_unsafe=is_unsafe,
            )
        except BillingQuotaError:
            console.print("[bold red]  Billing quota exceeded — stopping judge.[/bold red]")
            if rate_limiter is not None:
                rate_limiter.stop_event.set()
            if token_budget is not None:
                token_budget.stop_event.set()
            return "budget"
        except LabelingError as e:
            detail = str(e.last_exception) if e.last_exception else e.reason
            checkpoints[fname].mark_failed(eid, detail)
            console.print(f"[red]  Failed [{eid[:12]}]: {detail[:120]}[/red]")
            return "failed"

        if rate_limiter is not None:
            rate_limiter.record()
        if token_budget is not None:
            tokens_used = labeling_output.metadata.get("usage", {}).get("total_tokens", 0)
            if tokens_used:
                token_budget.consume(tokens_used)

        updated_entry = apply_labeling_output(
            entry=entry,
            output=labeling_output,
            reasoning_msg_idx=reasoning_idx,
            judge_name=judge_cfg.name,
            mapper_cfg=mapper_cfg,
            warn_fn=warn,
            taxonomy_version=taxonomy_version,
        )

        with write_locks[fname]:
            with open(output_paths[fname], "a", encoding="utf-8") as f:
                f.write(updated_entry.model_dump_json() + "\n")
            checkpoints[fname].mark_done(eid)

        return "labeled"

    with ThreadPoolExecutor(max_workers=pipeline_cfg.max_workers) as pool:
        futures = {pool.submit(_process_item, item): item for item in todo}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                item = futures[future]
                console.print(f"[red]  Unexpected error [{item['id'][:12]}]: {exc}[/red]")
                result = "failed"

            if result == "labeled":
                labeled += 1
                since_checkpoint += 1
                if since_checkpoint >= pipeline_cfg.checkpoint_every:
                    for cp in checkpoints.values():
                        cp.save()
                    since_checkpoint = 0
            elif result == "skipped":
                skipped += 1
            elif result == "budget":
                budget_stopped += 1
            else:
                failed += 1

            progress.advance(task_id)

    for cp in checkpoints.values():
        cp.save()

    return labeled, skipped, failed, budget_stopped


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
    judge_cfg: JudgeConfig | None = None,
    output_dir_override: Path | None = None,
) -> None:
    """Run the labeling pipeline for a single judge.

    Args:
        judge_cfg: If provided, use this judge instead of the first entry in
            config.judges. Allows the multi-judge runner to call this function
            once per judge without duplicating config loading.
        output_dir_override: If provided, use this as the output directory
            instead of the one derived from config. Used by the multi-judge
            runner to give each judge its own sub-directory.
    """
    if not input_files:
        console.print("[yellow]No input files to process.[/yellow]")
        return

    # Resolve config-relative paths
    secrets_path = config.resolve_path(config_path, config.secrets_file)
    prompt_path = config.resolve_path(config_path, config.prompt_file)
    output_dir = output_dir_override or config.resolve_path(config_path, config.output.dir)

    # Auto-load taxonomy version from taxonomy.json if not set in config
    taxonomy_version = config.output.taxonomy_version or get_taxonomy_version()
    if not output_dir_override and taxonomy_version:
        output_dir = output_dir / taxonomy_version

    effective_judge = judge_cfg or config.judges[0]

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

    client = build_client(effective_judge, secrets)

    # Load prompt template and optional system prompt
    prompt_template = inject_taxonomy(prompt_path.read_text(encoding="utf-8"))
    system_prompt_text: str | None = None
    if config.system_prompt_file:
        sp_path = config.resolve_path(config_path, config.system_prompt_file)
        if sp_path.exists():
            system_prompt_text = inject_taxonomy(sp_path.read_text(encoding="utf-8"))

    # Set up token budget (daily token limit for OpenAI-style billing)
    token_budget: TokenBudget | None = None
    if config.pipeline.token_budget is not None:
        budget_state_path = config.resolve_path(config_path, config.pipeline.budget_state_file)
        token_budget = TokenBudget(
            daily_limit=config.pipeline.token_budget,
            state_file=budget_state_path,
        )
        used = token_budget.tokens_used_today()
        console.print(
            f"[bold]Token budget:[/bold] {config.pipeline.token_budget:,} / day — "
            f"used today: [yellow]{used:,}[/yellow] — "
            f"remaining: [green]{token_budget.remaining():,}[/green]"
        )
        if not token_budget.can_proceed():
            console.print("[bold red]Daily token budget already exhausted. Nothing to do.[/bold red]")
            return

    # Set up request rate limiter (RPM + RPD from judge config)
    rate_limiter: RequestRateLimiter | None = None
    if effective_judge.rpd is not None or effective_judge.rpm is not None:
        rpm = effective_judge.rpm or 9999
        rpd = effective_judge.rpd or 9999999
        req_state_path = config.resolve_path(config_path, config.pipeline.request_state_file)
        rate_limiter = RequestRateLimiter(rpm=rpm, rpd=rpd, state_file=req_state_path)
        reqs = rate_limiter.requests_today()
        console.print(
            f"[bold]Request budget:[/bold] {rpd:,} req/day, {rpm} req/min — "
            f"used today: [yellow]{reqs:,}[/yellow] — "
            f"remaining: [green]{rate_limiter.remaining_today():,}[/green]"
        )
        if not rate_limiter.can_proceed():
            console.print("[bold red]Daily request budget already exhausted. Nothing to do.[/bold red]")
            return

    # Set up entry limit (--max-entries)
    entry_limit: EntryLimit | None = None
    if config.pipeline.max_entries is not None:
        entry_limit = EntryLimit(max_entries=config.pipeline.max_entries)
        console.print(
            f"[bold]Entry limit:[/bold] stopping after [cyan]{config.pipeline.max_entries}[/cyan] labeled entries"
        )

    def _any_limit_exhausted() -> bool:
        if token_budget is not None and not token_budget.can_proceed():
            return True
        if rate_limiter is not None and not rate_limiter.can_proceed():
            return True
        if entry_limit is not None and not entry_limit.can_proceed():
            return True
        return False

    # Work order: shared pre-generated list of entry IDs all judges target.
    # When set, it overrides shuffle_files and max_entries_per_file.
    work_order: dict[str, list[str]] | None = None  # filename → [id, ...]
    work_order_file_order: list[Path] | None = None
    flat_order: list[dict[str, str]] | None = None   # breadth-first interleaved order

    if config.pipeline.work_order_file:
        wo_path = config.resolve_path(config_path, config.pipeline.work_order_file)
        if wo_path.exists():
            with open(wo_path) as _f:
                _wo = json.load(_f)
            work_order = _wo.get("per_file", {})
            flat_order = _wo.get("flat_order")  # present in breadth-first work orders
            # Reconstruct absolute paths in the order the work order specifies
            input_by_name = {p.name: p for p in input_files}
            work_order_file_order = [
                input_by_name[name]
                for name in _wo.get("file_order", [])
                if name in input_by_name
            ]
            if flat_order:
                console.print(
                    f"[bold]Work order (breadth-first):[/bold] {wo_path.name} — "
                    f"{len(flat_order)} entries across "
                    f"{len(set(item['file'] for item in flat_order))} files"
                )
            else:
                console.print(
                    f"[bold]Work order:[/bold] {wo_path.name} — "
                    f"{sum(len(v) for v in work_order.values())} target entries across "
                    f"{len(work_order)} files"
                )
        else:
            console.print(f"[yellow]Work order file not found: {wo_path} — falling back to normal mode[/yellow]")

    if work_order_file_order is not None:
        input_files = work_order_file_order
    elif config.pipeline.shuffle_files and len(input_files) > 1:
        # Shuffle file order for breadth-first coverage across datasets.
        # Seed with the UTC date so every judge running on the same day visits
        # files in the same order — ensuring all judges label the same entries
        # and their outputs are directly comparable.
        input_files = list(input_files)
        day_seed = datetime.now(timezone.utc).date().isoformat()
        random.Random(day_seed).shuffle(input_files)

    # ── Breadth-first mode (flat_order present in work order) ────────────────
    if flat_order is not None:
        input_dir = input_files[0].parent if input_files else output_dir
        n_files = len(set(item["file"] for item in flat_order))
        console.print(
            f"[bold]Breadth-first labeling:[/bold] "
            f"{len(flat_order)} entries across {n_files} files"
        )
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("[cyan]Breadth-first labeling…[/cyan]", total=len(flat_order))
            labeled, skipped, failed, budget_stopped = process_breadth_first(
                flat_order=flat_order,
                input_dir=input_dir,
                output_dir=output_dir,
                output_suffix=config.output.suffix,
                client=client,
                judge_cfg=effective_judge,
                prompt_template=prompt_template,
                system_prompt_text=system_prompt_text,
                pipeline_cfg=config.pipeline,
                mapper_cfg=config.mapper,
                console=console,
                progress=progress,
                task_id=task_id,
                reset_checkpoints=reset_checkpoints,
                token_budget=token_budget,
                rate_limiter=rate_limiter,
                entry_limit=entry_limit,
                taxonomy_version=taxonomy_version,
            )

        total = labeled + skipped + failed + budget_stopped
        table = Table(title="Breadth-First Labeling Summary", show_lines=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right")
        table.add_row("Total processed", str(total))
        table.add_row("Labeled", f"[green]{labeled}[/green]")
        table.add_row("Skipped (done)", f"[yellow]{skipped}[/yellow]")
        table.add_row("Failed", f"[red]{failed}[/red]")
        table.add_row("Budget stopped", f"[magenta]{budget_stopped}[/magenta]")
        console.print(table)

        if token_budget is not None:
            console.print(
                f"[bold]Token budget after run:[/bold] "
                f"used today: [yellow]{token_budget.tokens_used_today():,}[/yellow] / {token_budget.daily_limit:,} — "
                f"remaining: [green]{token_budget.remaining():,}[/green]"
            )
        if rate_limiter is not None:
            console.print(
                f"[bold]Request budget after run:[/bold] "
                f"used today: [yellow]{rate_limiter.requests_today():,}[/yellow] / {rate_limiter.rpd:,} — "
                f"remaining: [green]{rate_limiter.remaining_today():,}[/green]"
            )
        if _any_limit_exhausted():
            console.print(
                "[bold magenta]Daily limit reached. "
                "Run again tomorrow to continue from where you left off.[/bold magenta]"
            )
        return

    # ── Per-file mode (legacy / no flat_order) ────────────────────────────────

    # Summary table accumulator
    results: list[tuple[str, int, int, int, int, int]] = []  # name, total, labeled, skipped, failed, budget_stopped

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

            # Skip file if any limit was exhausted in a prior file
            if _any_limit_exhausted():
                console.print(
                    f"[yellow]Limit reached — skipping {input_path.name} and remaining files[/yellow]"
                )
                break

            labeled, skipped, failed, budget_stopped = process_file(
                input_path=input_path,
                output_path=out_path,
                client=client,
                judge_cfg=effective_judge,
                prompt_template=prompt_template,
                system_prompt_text=system_prompt_text,
                pipeline_cfg=config.pipeline,
                mapper_cfg=config.mapper,
                console=console,
                progress=progress,
                file_task=file_task,
                reset_checkpoints=reset_checkpoints,
                token_budget=token_budget,
                rate_limiter=rate_limiter,
                entry_limit=entry_limit,
                per_file_max=None if work_order else config.pipeline.max_entries_per_file,
                allowed_ids=work_order.get(input_path.name) if work_order else None,
                taxonomy_version=taxonomy_version,
            )

            total = labeled + skipped + failed + budget_stopped
            results.append((input_path.name, total, labeled, skipped, failed, budget_stopped))
            progress.update(file_task, description=f"[green]{input_path.name}[/green]")

    # Summary table
    table = Table(title="Labeling Summary", show_lines=True)
    table.add_column("File", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("Labeled", justify="right", style="green")
    table.add_column("Skipped", justify="right", style="yellow")
    table.add_column("Failed", justify="right", style="red")
    table.add_column("Budget stop", justify="right", style="magenta")

    totals = Counter()
    for name, total, labeled, skipped, failed, budget_stopped in results:
        table.add_row(name, str(total), str(labeled), str(skipped), str(failed), str(budget_stopped))
        totals["total"] += total
        totals["labeled"] += labeled
        totals["skipped"] += skipped
        totals["failed"] += failed
        totals["budget_stopped"] += budget_stopped

    if len(results) > 1:
        table.add_row(
            "[bold]TOTAL[/bold]",
            str(totals["total"]),
            str(totals["labeled"]),
            str(totals["skipped"]),
            str(totals["failed"]),
            str(totals["budget_stopped"]),
        )

    console.print(table)

    # Final status
    if token_budget is not None:
        console.print(
            f"[bold]Token budget after run:[/bold] "
            f"used today: [yellow]{token_budget.tokens_used_today():,}[/yellow] / {token_budget.daily_limit:,} — "
            f"remaining: [green]{token_budget.remaining():,}[/green]"
        )
    if rate_limiter is not None:
        console.print(
            f"[bold]Request budget after run:[/bold] "
            f"used today: [yellow]{rate_limiter.requests_today():,}[/yellow] / {rate_limiter.rpd:,} — "
            f"remaining: [green]{rate_limiter.remaining_today():,}[/green]"
        )
    if _any_limit_exhausted():
        console.print(
            "[bold magenta]Daily limit reached. "
            "Run again tomorrow to continue from where you left off.[/bold magenta]"
        )
