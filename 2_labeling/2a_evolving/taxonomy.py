"""
Taxonomy state: active labels, graveyard, operation application.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_id() -> str:
    return uuid.uuid4().hex[:8]


class LabelEntry(BaseModel):
    label_id: str = Field(default_factory=_short_id)
    name: str
    description: str
    created_at: str = Field(default_factory=_now)
    created_at_step: int = 0          # step number when this label was born
    usage_count: int = 0
    parent_ids: list[str] = Field(default_factory=list)  # IDs of source labels (MERGE/SPLIT)


class GraveyardEntry(BaseModel):
    """A label that was retired — stores full label + retirement metadata for animation."""
    label: LabelEntry
    retired_at_step: int
    retired_reason: str   # e.g. "merged into Compliance refusal", "split into X, Y", "deleted", "renamed to X"
    child_ids: list[str] = Field(default_factory=list)   # IDs of labels that replaced this one


class TaxonomyOperation(BaseModel):
    step: int
    operation: str          # NONE | CREATE | MERGE | SPLIT | RENAME | DELETE
    details: dict
    triggered_by: dict      # {text, composite_id, sub_text_idx}
    justification: str


class TaxonomyState(BaseModel):
    active: dict[str, LabelEntry] = Field(default_factory=dict)         # keyed by label name
    graveyard: dict[str, GraveyardEntry] = Field(default_factory=dict)  # keyed by label name

    def label_names(self) -> list[str]:
        return list(self.active.keys())

    def within_limit(self, max_labels: int) -> bool:
        return len(self.active) < max_labels

    def to_prompt_block(self) -> str:
        if not self.active:
            return "(empty — no labels exist yet. You SHOULD propose CREATE for meaningful behaviors you observe.)"
        lines = []
        for i, (name, entry) in enumerate(self.active.items(), 1):
            lines.append(f"{i}. {name} — {entry.description}")
        return "\n".join(lines)

    def increment_usage(self, names: list[str]) -> None:
        for name in names:
            if name in self.active:
                self.active[name].usage_count += 1

    def _retire(self, name: str, step: int, reason: str, child_ids: list[str] | None = None) -> LabelEntry:
        """Move a label from active to graveyard. Returns the retired entry."""
        entry = self.active.pop(name)
        self.graveyard[name] = GraveyardEntry(
            label=entry,
            retired_at_step=step,
            retired_reason=reason,
            child_ids=child_ids or [],
        )
        return entry

    def apply_operation(self, op: TaxonomyOperation) -> None:
        t = op.operation
        d = op.details
        step = op.step

        if t == "NONE":
            return

        elif t == "CREATE":
            name = d.get("name", "").strip()
            if not name or name in self.active:
                return
            self.active[name] = LabelEntry(
                name=name,
                description=d.get("description", ""),
                created_at_step=step,
            )

        elif t == "MERGE":
            sources = d.get("sources", [])
            result_name = d.get("result", "").strip()
            desc = d.get("description", "")
            if not result_name or len(sources) < 2:
                return
            # Create merged label first to get its ID
            merged = LabelEntry(
                name=result_name,
                description=desc,
                created_at_step=step,
                parent_ids=[self.active[s].label_id for s in sources if s in self.active],
            )
            # Retire source labels, pointing to the merged label's ID
            for src in sources:
                if src in self.active:
                    self._retire(src, step,
                                 reason=f"merged into '{result_name}'",
                                 child_ids=[merged.label_id])
            if result_name not in self.active:
                self.active[result_name] = merged

        elif t == "SPLIT":
            source = d.get("source", "").strip()
            results = d.get("results", [])
            if not source or source not in self.active or len(results) < 2:
                return
            # Create child labels first to get their IDs
            children = []
            source_id = self.active[source].label_id
            for r in results:
                name = r.get("name", "").strip()
                if name and name not in self.active:
                    child = LabelEntry(
                        name=name,
                        description=r.get("description", ""),
                        created_at_step=step,
                        parent_ids=[source_id],
                    )
                    children.append(child)
            # Retire source
            self._retire(source, step,
                         reason=f"split into {', '.join(repr(c.name) for c in children)}",
                         child_ids=[c.label_id for c in children])
            for child in children:
                self.active[child.name] = child

        elif t == "RENAME":
            old = d.get("old", "").strip()
            new = d.get("new", "").strip()
            desc = d.get("description", "")
            if not old or not new or old not in self.active:
                return
            old_entry = self.active[old]
            new_entry = LabelEntry(
                name=new,
                description=desc or old_entry.description,
                created_at_step=step,
                usage_count=old_entry.usage_count,
                parent_ids=[old_entry.label_id],
            )
            self._retire(old, step,
                         reason=f"renamed to '{new}'",
                         child_ids=[new_entry.label_id])
            self.active[new] = new_entry

        elif t == "DELETE":
            name = d.get("name", "").strip()
            if name and name in self.active:
                self._retire(name, step, reason="deleted")
