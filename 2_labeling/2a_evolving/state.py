"""
Run state: full persistence with atomic writes and resume logic.
"""
from __future__ import annotations

import copy
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from taxonomy import LabelEntry, TaxonomyOperation, TaxonomyState

RUNS_DIR = Path(__file__).parent / "runs"
SNAPSHOT_INTERVAL = 25  # take a taxonomy snapshot every N steps


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProcessedSubText(BaseModel):
    idx: int
    times_processed: int = 0
    triggered_change: bool = False
    labels: list[str] = Field(default_factory=list)


class RunConfig(BaseModel):
    models: list[str]
    labeler: str
    max_labels: int
    seed_file: str | None = None
    sampling_seed: int = 42


class RunState(BaseModel):
    run_id: str = Field(default_factory=lambda: (
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M") + "_" + uuid.uuid4().hex[:4]
    ))
    created_at: str = Field(default_factory=_now)
    config: RunConfig
    taxonomy: TaxonomyState = Field(default_factory=TaxonomyState)
    history: dict = Field(default_factory=lambda: {
        "labels_ever": {},      # label_id -> LabelEntry dict
        "operations": [],       # list of TaxonomyOperation dicts
        "snapshots": [],        # list of {step, taxonomy} dicts
        "steps_log": [],        # list of per-step full context dicts
    })
    processed: dict = Field(default_factory=dict)  # composite_id -> {"sub_texts": [...]}
    stats: dict = Field(default_factory=lambda: {
        "steps_completed": 0,
        "total_revisits": 0,
        "total_changes": 0,
        "errors": 0,
        "total_invalid_proposals": 0,
    })

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Atomic write: write to .tmp sibling then os.replace()."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        data = self.model_dump(mode="json")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "RunState":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.model_validate(data)

    @classmethod
    def get_run_path(cls, run_id: str) -> Path:
        return RUNS_DIR / run_id / "state.json"

    @classmethod
    def load_or_create(cls, run_id: str | None, config: RunConfig) -> tuple["RunState", Path]:
        """
        If run_id given and state file exists, resume.
        Otherwise create a new run.
        """
        if run_id:
            path = cls.get_run_path(run_id)
            if path.exists():
                state = cls.load(path)
                return state, path
            # run_id given but not found — create fresh with that id
        state = cls(config=config)
        if run_id:
            state.run_id = run_id
        path = cls.get_run_path(state.run_id)
        return state, path

    # ── Mutations ─────────────────────────────────────────────────────────────

    def record_operation(self, op: TaxonomyOperation, triggered_change: bool) -> None:
        """Append operation to history and update stats."""
        self.history["operations"].append(op.model_dump())
        if triggered_change:
            self.stats["total_changes"] += 1
        # Track all labels ever seen
        for entry in self.taxonomy.active.values():
            self.history["labels_ever"][entry.label_id] = entry.model_dump()
        for entry in self.taxonomy.graveyard.values():
            self.history["labels_ever"][entry.label_id] = entry.model_dump()

    def mark_sub_text(
        self,
        composite_id: str,
        sub_text_idx: int,
        labels: list[str],
        triggered_change: bool,
        is_revisit: bool,
    ) -> None:
        if composite_id not in self.processed:
            self.processed[composite_id] = {"sub_texts": []}
        sub_texts = self.processed[composite_id]["sub_texts"]
        # Find existing entry or create
        for entry in sub_texts:
            if entry["idx"] == sub_text_idx:
                entry["times_processed"] += 1
                entry["triggered_change"] = entry["triggered_change"] or triggered_change
                entry["labels"] = labels
                break
        else:
            sub_texts.append({
                "idx": sub_text_idx,
                "times_processed": 1,
                "triggered_change": triggered_change,
                "labels": labels,
            })
        if is_revisit:
            self.stats["total_revisits"] += 1

    def log_step(self, entry: dict) -> None:
        """Append a full per-step context record to history['steps_log']."""
        if "steps_log" not in self.history:
            self.history["steps_log"] = []
        self.history["steps_log"].append(entry)

    def snapshot(self, step: int) -> None:
        snap = {
            "step": step,
            "taxonomy": self.taxonomy.model_dump(mode="json"),
        }
        self.history["snapshots"].append(snap)

    def load_seed(self, seed_file: str) -> None:
        """Load labels from a seed JSON file into the initial taxonomy."""
        import json as _json
        data = _json.loads(Path(seed_file).read_text(encoding="utf-8"))
        # Support two formats: {name: {description: ...}} or list of {name, description}
        if isinstance(data, dict):
            # Could be label_registry format: {label_id: {name, description, ...}}
            for key, val in data.items():
                if isinstance(val, dict) and "name" in val and "description" in val:
                    entry = LabelEntry(
                        label_id=val.get("label_id", key),
                        name=val["name"],
                        description=val["description"],
                        usage_count=val.get("usage_count", 0),
                    )
                    self.taxonomy.active[entry.name] = entry
        elif isinstance(data, list):
            for item in data:
                if "name" in item and "description" in item:
                    entry = LabelEntry(name=item["name"], description=item["description"])
                    self.taxonomy.active[entry.name] = entry


def list_runs() -> list[dict]:
    """Return summary dicts for all existing runs."""
    runs = []
    if not RUNS_DIR.exists():
        return runs
    for run_dir in sorted(RUNS_DIR.iterdir()):
        state_file = run_dir / "state.json"
        if not state_file.exists():
            continue
        try:
            state = RunState.load(state_file)
            runs.append({
                "run_id": state.run_id,
                "created_at": state.created_at,
                "steps": state.stats["steps_completed"],
                "models": state.config.models,
                "labeler": state.config.labeler,
                "n_labels": len(state.taxonomy.active),
                "n_changes": state.stats["total_changes"],
                "sampling_seed": state.config.sampling_seed,
                "path": str(state_file),
            })
        except Exception:
            runs.append({"run_id": run_dir.name, "error": "could not load"})
    return runs
