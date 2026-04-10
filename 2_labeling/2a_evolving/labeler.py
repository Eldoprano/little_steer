"""
LLM labeler: prompt construction, LiteLLM calls, response parsing.
"""
from __future__ import annotations

import json
import logging
import re
import os
from typing import Annotated, Any, Union

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from data_loader import SubText
from taxonomy import TaxonomyOperation, TaxonomyState

logger = logging.getLogger(__name__)

# Cache of negotiated kwargs per model — avoids re-probing on every step
_model_kwargs_cache: dict[str, dict] = {}


# ── JSON schema for structured output (LM-Studio / OpenAI-compatible) ─────────

def get_response_format() -> dict:
    """Generate the JSON schema for OpenAI structured output from the Pydantic model."""
    schema = RawLabelResponse.model_json_schema()
    # Remove things OpenAI doesn't like in the schema if necessary
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "label_response",
            "strict": True,
            "schema": schema,
        },
    }


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
You are a behavioral taxonomy labeler for AI safety research. Your task is to identify safety-relevant reasoning behaviors in excerpts from AI reasoning traces.

## WHAT YOU ARE LABELING

Each excerpt is a paragraph from an AI's internal reasoning chain. Label the BEHAVIORS present — not the topics discussed.

**What makes a good label:**
- Describes a reasoning PATTERN (what the model is cognitively doing, the decision it's making)
- Reusable across many topics and domains. Name should not contain domain-specific nouns or proper names
- Distinct enough that it doesn't overlap heavily with other labels

Safety-related behaviors are the highest priority, their labels should be precise.
General reasoning behaviors are also worth labeling.

These labels will extract steering vectors from model activations. Descriptions must be specific enough that a future labeler can identify exactly which sentences exhibit the behavior.

## USING EXISTING LABELS

Read each label's DESCRIPTION carefully before applying it. Apply a label ONLY when the behavior in the text matches the description, not just the name.

If you are stretching a label beyond its description to cover something it was not designed for, CREATE a more specific label instead of overloading the existing one.

## WHEN TO PROPOSE TAXONOMY OPERATIONS

You may propose up to 3 operations per response (one per genuinely distinct missing behavior). The default is an empty list. Propose only when:
1. A sentence shows a behavior that no existing label fits, even loosely.
2. That behavior would likely recur across many other reasoning traces (not a one-off phrasing).
3. The label is specific enough — not too generic and not too narrow.
4. If the taxonomy is empty, you are expected to propose CREATE for any meaningful behavior you observe.

If unsure: use the closest existing label and leave operations empty.
{limit_notice}
## OUTPUT FORMAT

Respond with a single JSON object — no markdown, no code fences, nothing else.
{{"thinking": "<reason through which behaviors are contained in the excerpt, and whether any behaviors are missing from the taxonomy>", "labels": ["ExactLabelName", ...], "operations": [], "justification": "<one sentence>"}}

"thinking": Use this field to reason about the excerpt, the current taxonomy, and any operations you are proposing.

"labels": Use this field to list the labels that apply. Only EXACT names from the CURRENT TAXONOMY. Labels you are proposing via CREATE do not exist yet — they go in "operations", never in "labels".

"operations" is a list of 0–3 operation objects. Leave it empty ([]) if no changes are needed. Each operation is one of:

  CREATE — Add a truly new label (no existing label covers this behavior even loosely):
    {{"type": "CREATE", "name": "NewLabelName", "description": "Describe the new label precisely here."}}

  MERGE — Two labels have become redundant; combine into one (source labels are automatically removed):
    {{"type": "MERGE", "sources": ["Label A", "Label B"], "result": "New Combined Name", "description": "Describe the new label precisely here."}}

  SPLIT — A label covers two distinct behaviors; break it into specifics (source label is automatically removed):
    {{"type": "SPLIT", "source": "oldLabelName", "results": [{{"name": "newLabelName1", "description": "Describe the new label precisely here."}}, {{"name": "newLabelName2", "description": "Describe the new label precisely here."}}]}}

  RENAME — A label's name or description needs clarification:
    {{"type": "RENAME", "old": "oldLabelName", "new": "newLabelName", "description": "Either keep the same old description or define a new one."}}

  DELETE — Use only when you need to CREATE a new label, there is no space, and there are no good merge candidates:
    {{"type": "DELETE", "name": "oldLabelName", "reason": "Reason for deleting the label."}}

## RULES
- "labels" must only contain names that EXACTLY match names in the CURRENT TAXONOMY below. If a name is not in the taxonomy, it does NOT go in "labels" — it goes in "operations" as a CREATE.
- If you propose CREATE, the new label name must NOT appear in "labels" (it doesn't exist yet).
- When creating a new label:
    - name: Max 3 words, camelCase, describes a BEHAVIOR not a topic (no proper nouns, no domain-specific terms).
    - description: What the model is cognitively doing or deciding. 1-2 sentences. Specific enough to distinguish from similar behaviors.
- If you propose MERGE, the new combined name goes in "result" — do not put source names in "labels".
- For SPLIT/RENAME/DELETE, the target label must exist in the taxonomy.
- Do not propose cosmetic renames.\
"""

USER_PROMPT_TEMPLATE = """\
## CURRENT TAXONOMY (Step {step}, {n_active} active labels)

{taxonomy_block}
{history_block}
---
{context_block}
## PARAGRAPH TO LABEL

{sub_text}

---

Step through each sentence: what is the model cognitively doing? Assign matching taxonomy labels. If any behavior is genuinely missing, propose a CREATE.\
"""

CONTEXT_BLOCK_TEMPLATE = """\
## PRECEDING PARAGRAPH (for context only)

{prev_text}

---
"""

HISTORY_BLOCK_TEMPLATE = """\

## RECENT TAXONOMY CHANGES
{recent_ops}
{graveyard_section}"""

GRAVEYARD_SECTION_TEMPLATE = """\
## RETIRED LABELS
{graveyard_lines}
"""

LIMIT_NOTICE = """
The taxonomy has reached its maximum of {max_labels} labels. Do NOT propose CREATE. MERGE, SPLIT, RENAME, and DELETE are still allowed if genuinely needed.
"""


# ── Per-operation Pydantic models ─────────────────────────────────────────────
# extra="ignore": don't reject harmless extra keys that some models add.
# _NonEmpty: required + min_length=1 so Pydantic auto-catches:
#   - typos like 'descr1ption'  → 'description' absent  → "field required"
#   - LLM left the field blank  → min_length violated    → "at least 1 char"

_NonEmpty = Annotated[str, Field(min_length=1)]


class _OpBase(BaseModel):
    model_config = ConfigDict(extra="ignore")


class CreateOp(_OpBase):
    type: str
    name: _NonEmpty
    description: _NonEmpty


class MergeOp(_OpBase):
    type: str
    sources: list[str]
    result: _NonEmpty
    description: _NonEmpty


class SplitResult(_OpBase):
    name: _NonEmpty
    description: _NonEmpty


class SplitOp(_OpBase):
    type: str
    source: _NonEmpty
    results: list[SplitResult]  # Pydantic recurses into each result dict too


class RenameOp(_OpBase):
    type: str
    old: _NonEmpty
    new: _NonEmpty
    description: str = ""


class DeleteOp(_OpBase):
    type: str
    name: _NonEmpty
    reason: str = ""


_OP_MODELS: dict[str, type[_OpBase]] = {
    "CREATE": CreateOp,
    "MERGE":  MergeOp,
    "SPLIT":  SplitOp,
    "RENAME": RenameOp,
    "DELETE": DeleteOp,
}


# ── Top-level response model ───────────────────────────────────────────────────

class RawLabelResponse(BaseModel):
    """Parses + validates the raw JSON dict from the LLM.

    All coercion and normalization lives here so parse_response() stays thin:
    - thinking / labels / justification are plain typed fields
    - operations: legacy singular 'operation' key is folded in; 'NONE' strings
      are stripped; each op dict is validated against its typed model
    - justification must be present and non-empty
    """
    model_config = ConfigDict(extra="ignore")

    thinking: str = ""
    labels: list[str] = []
    operations: list[dict] = []   # always a clean list of dicts after validation
    justification: _NonEmpty

    @model_validator(mode="before")
    @classmethod
    def coerce_input(cls, data: Any) -> Any:
        """Normalize the raw dict before field-level validation runs."""
        if not isinstance(data, dict):
            return data

        # 1. Fold legacy singular 'operation' field into 'operations' list
        if "operations" not in data or data["operations"] is None:
            legacy = data.get("operation")
            if legacy and legacy != "NONE":
                data = {**data, "operations": [legacy]}
            else:
                data = {**data, "operations": []}

        # 2. Ensure it's a list
        ops = data["operations"]
        if not isinstance(ops, list):
            ops = [ops] if ops else []

        # 3. Drop bare "NONE" strings; uppercase every op's 'type' field
        cleaned = []
        for op in ops:
            if op == "NONE" or op is None:
                continue
            if isinstance(op, dict) and "type" in op:
                op = {**op, "type": str(op["type"]).upper()}
            cleaned.append(op)

        return {**data, "operations": cleaned}

    @field_validator("operations")
    @classmethod
    def validate_each_operation(cls, ops: list[Any]) -> list[dict]:
        """Validate each op dict against its specific typed model.
        Pydantic will raise ValidationError automatically for:
        - missing required fields (e.g. 'descr1ption' typo → 'description' absent)
        - empty required strings (min_length=1)
        - wrong types
        """
        for i, op in enumerate(ops):
            if not isinstance(op, dict):
                raise ValueError(f"Operation {i} must be a dict, got {type(op).__name__}")
            op_type = op.get("type", "")
            model_cls = _OP_MODELS.get(op_type)
            if model_cls is None:
                continue  # unknown type — downstream validate_operation() handles it
            model_cls.model_validate(op)  # raises ValidationError if anything is wrong
        return ops

    @field_validator("labels")
    @classmethod
    def warn_invented_labels(cls, labels: list[str], info: Any) -> list[str]:
        """When active taxonomy is passed via context, log a warning for any
        label the model invented (not in the taxonomy). Does NOT raise — filtering
        happens in parse_response which has the full domain context.
        """
        active_names: set[str] | None = (info.context or {}).get("active_names")
        if active_names is not None:
            invented = [l for l in labels if l not in active_names]
            if invented:
                logger.warning("Model used labels not in taxonomy (will be filtered): %s", invented)
        return labels




class LabelResponse(BaseModel):
    """The clean, processed result returned by parse_response()."""
    thinking: str = ""
    internal_reasoning: str = ""
    labels: list[str]
    operations: list[dict]
    justification: str


def build_system_prompt(max_labels: int, at_limit: bool) -> str:
    notice = LIMIT_NOTICE.format(max_labels=max_labels) if at_limit else "\n"
    return SYSTEM_PROMPT_TEMPLATE.format(limit_notice=notice)


def build_history_block(
    operations: list[dict],
    graveyard: dict,
    max_recent_ops: int = 3,
    max_graveyard: int = 5,
) -> str:
    """Build the optional history section shown in the user prompt."""
    non_none = [o for o in operations if o.get("operation") != "NONE"]
    recent = non_none[-max_recent_ops:] if non_none else []

    if not recent and not graveyard:
        return ""

    recent_lines = []
    for op in recent:
        op_type = op.get("operation", "")
        details = op.get("details", {})
        step_n = op.get("step", "?")

        if op_type == "CREATE":
            name = details.get("name", "?")
            recent_lines.append(f"- Step {step_n} | CREATE \"{name}\"")
        elif op_type == "MERGE":
            sources = details.get("sources", [])
            result = details.get("result", "?")
            recent_lines.append(f"- Step {step_n} | MERGE {sources} → \"{result}\"")
        elif op_type == "SPLIT":
            source = details.get("source", "?")
            results = [r.get("name", "?") for r in details.get("results", [])]
            recent_lines.append(f"- Step {step_n} | SPLIT \"{source}\" → {results}")
        elif op_type == "RENAME":
            recent_lines.append(f"- Step {step_n} | RENAME \"{details.get('old')}\" → \"{details.get('new')}\"")
        elif op_type == "DELETE":
            recent_lines.append(f"- Step {step_n} | DELETE \"{details.get('name')}\"")

    graveyard_section = ""
    if graveyard:
        entries = sorted(
            graveyard.items(),
            key=lambda kv: kv[1].get("retired_at_step", 0) if isinstance(kv[1], dict) else kv[1].retired_at_step,
            reverse=True,
        )[:max_graveyard]
        lines = []
        for name, entry in entries:
            if isinstance(entry, dict):
                reason = entry.get("retired_reason", "")
                step_n = entry.get("retired_at_step", "?")
                label_id = entry.get("label", {}).get("label_id", "?")
            else:
                reason = entry.retired_reason
                step_n = entry.retired_at_step
                label_id = entry.label.label_id
            lines.append(f"- [{label_id}] \"{name}\" — {reason} (step {step_n})")
        graveyard_section = GRAVEYARD_SECTION_TEMPLATE.format(graveyard_lines="\n".join(lines))

    return HISTORY_BLOCK_TEMPLATE.format(
        recent_ops="\n".join(recent_lines) if recent_lines else "(none yet)",
        graveyard_section=graveyard_section,
    )


def build_user_prompt(
    sub_text: SubText,
    taxonomy: TaxonomyState,
    step: int,
    prev_sub_text: SubText | None = None,
    history_block: str = "",
) -> str:
    context_block = ""
    if prev_sub_text is not None:
        context_block = CONTEXT_BLOCK_TEMPLATE.format(
            prev_text=prev_sub_text.text[:800]
        )
    return USER_PROMPT_TEMPLATE.format(
        step=step,
        n_active=len(taxonomy.active),
        taxonomy_block=taxonomy.to_prompt_block(),
        history_block=history_block,
        context_block=context_block,
        sub_text=sub_text.text[:1500],
    )


def _extract_json(raw: str) -> dict:
    """Try direct parse, then find first { to last }."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from response: {raw[:200]!r}")


def parse_response(raw_json: str, taxonomy: TaxonomyState) -> LabelResponse:
    # Extract internal reasoning via <think> tags if present (before JSON parsing)
    internal_reasoning = ""
    think_match = re.search(r"<think>(.*?)</think>", raw_json, re.DOTALL | re.IGNORECASE)
    if think_match:
        internal_reasoning = think_match.group(1).strip()

    # Parse JSON, then validate + normalize with Pydantic.
    data = _extract_json(raw_json)
    active_names = set(taxonomy.label_names())
    parsed = RawLabelResponse.model_validate(data, context={"active_names": active_names})

    # Only return labels that actually exist in the taxonomy
    valid_labels = [l for l in parsed.labels if l in active_names]

    return LabelResponse(
        thinking=parsed.thinking,
        internal_reasoning=internal_reasoning,
        labels=valid_labels,
        operations=parsed.operations,
        justification=parsed.justification,
    )



def call_labeler(
    model: str,
    sub_text: SubText,
    taxonomy: TaxonomyState,
    step: int,
    max_labels: int,
    prev_sub_text: SubText | None = None,
    history_block: str = "",
    response_log: "Path | None" = None,
) -> LabelResponse:
    """Call OpenAI and parse the response. Retries on parse errors (up to 3x)
    and on rate limits (up to 8x with exponential backoff).

    Args:
        response_log: If given, each raw API response is appended as a JSON line
                      to this file immediately after the call, before any parsing.
                      Provides a crash-safe backup of all API responses.
    """
    import datetime
    import time
    from openai import RateLimitError

    at_limit = not taxonomy.within_limit(max_labels)
    system_prompt = build_system_prompt(max_labels, at_limit)
    user_prompt = build_user_prompt(sub_text, taxonomy, step, prev_sub_text, history_block)

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    base_url = os.environ.get("OPENAI_BASE_URL", os.environ.get("OPENAI_API_BASE"))
    client = OpenAI(base_url=base_url)

    if model in _model_kwargs_cache:
        kwargs: dict = {**_model_kwargs_cache[model], "messages": messages}
    else:
        kwargs: dict = dict(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=2500,
            reasoning_effort="low",
        )

    def _raw_api_call() -> tuple:
        """Make one API call with rate-limit backoff. Returns (message, raw, response)."""
        for rl_try in range(8):
            try:
                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message, resp.choices[0].message.content or "", resp
            except RateLimitError as e:
                wait = min(30 * (2 ** rl_try), 600)  # 30s → 60 → 120 → … → 600s cap
                logger.warning(
                    "Rate limited (retry %d/8), waiting %ds: %s", rl_try + 1, wait, e
                )
                time.sleep(wait)
        raise RuntimeError("Rate limit retries exhausted")

    _fixed: set[str] = set()  # tracks which param fixes have been applied

    for attempt in range(6):  # up to 4 param fixes + 2 real parse retries
        try:
            message, raw, response = _raw_api_call()

            if response_log is not None:
                try:
                    log_entry = {
                        "step": step,
                        "attempt": attempt,
                        "composite_id": sub_text.composite_id,
                        "sub_text_idx": sub_text.sub_text_idx,
                        "model": model,
                        "timestamp": datetime.datetime.utcnow().isoformat(),
                        "usage": response.usage.model_dump() if response.usage else None,
                        "raw": raw,
                    }
                    with open(response_log, "a", encoding="utf-8") as _f:
                        _f.write(json.dumps(log_entry) + "\n")
                except Exception as _log_err:
                    logger.warning("Failed to write response log: %s", _log_err)

            # Cache the working kwargs for this model (minus messages, which change every call)
            if model not in _model_kwargs_cache:
                _model_kwargs_cache[model] = {k: v for k, v in kwargs.items() if k != "messages"}

            parsed = parse_response(raw, taxonomy)

            extracted_reasoning = message.model_dump().get("reasoning", "")
            if not extracted_reasoning:
                extracted_reasoning = getattr(message, "reasoning_content", "") or ""
            parsed.internal_reasoning = str(extracted_reasoning).strip()

            return parsed

        except (ValueError, KeyError, ValidationError) as e:
            parse_attempts = attempt - len(_fixed)  # don't count param-fix attempts
            if parse_attempts < 2:
                err_msg = str(e)
                if isinstance(e, ValidationError):
                    clean_errors = []
                    for err in e.errors():
                        loc = ".".join(str(p) for p in err["loc"])
                        clean_errors.append(f"[{loc}] {err['msg']}")
                    err_msg = "; ".join(clean_errors)
                logger.warning("Attempt %d failed validation, retrying: %s", attempt + 1, err_msg)
                continue
            raise

        except Exception as e:
            err_msg = str(e).lower()
            # Parameter compatibility fixes — each applied at most once (tracked by _fixed)
            fixed = False
            if "response_format" in err_msg and "response_format" not in _fixed:
                logger.warning("Model does not support response_format, retrying without: %s", e)
                kwargs.pop("response_format", None)
                _fixed.add("response_format")
                fixed = True
            if ("reasoning_effort" in err_msg or "extra_body" in err_msg or "unrecognized parameter" in err_msg) \
                    and "reasoning_effort" not in _fixed:
                logger.warning("Server does not support reasoning_effort, retrying without: %s", e)
                kwargs.pop("reasoning_effort", None)
                _fixed.add("reasoning_effort")
                fixed = True
            if "max_tokens" in err_msg and "max_completion_tokens" in err_msg \
                    and "max_tokens" not in _fixed:
                logger.warning("Model requires max_completion_tokens, retrying: %s", e)
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens", 2500)
                _fixed.add("max_tokens")
                fixed = True
            if "temperature" in err_msg and "temperature" not in _fixed:
                logger.warning("Model does not support custom temperature, retrying without: %s", e)
                kwargs.pop("temperature", None)
                _fixed.add("temperature")
                fixed = True
            if fixed:
                continue
            logger.error("OpenAI call failed: %s", e)
            raise


def operation_to_taxonomy_op(
    op: dict,
    sub_text: SubText,
    step: int,
    justification: str = "",
) -> TaxonomyOperation:
    """Convert a validated operation dict into a TaxonomyOperation."""
    op_type = op.get("type", "NONE").upper()
    details = {k: v for k, v in op.items() if k != "type"}

    return TaxonomyOperation(
        step=step,
        operation=op_type,
        details=details,
        triggered_by={
            "text": sub_text.text[:300],
            "composite_id": sub_text.composite_id,
            "sub_text_idx": sub_text.sub_text_idx,
        },
        justification=justification,
    )


def validate_operation(op: TaxonomyOperation, taxonomy: TaxonomyState, max_labels: int) -> tuple[bool, str]:
    """Returns (valid, rejection_reason). rejection_reason is empty string if valid."""
    t = op.operation
    d = op.details

    if t == "NONE":
        return True, ""

    if t == "CREATE":
        if not taxonomy.within_limit(max_labels):
            return False, f"taxonomy is at its limit of {max_labels} labels — use MERGE or DELETE first"
        name = d.get("name", "").strip()
        if not name:
            return False, "CREATE requires a non-empty 'name' field"
        if name in taxonomy.active:
            return False, f"label '{name}' already exists — use RENAME or MERGE instead"
        return True, ""

    if t == "MERGE":
        sources = d.get("sources", [])
        result = d.get("result", "").strip()
        if len(sources) < 2 or not result:
            return False, "MERGE requires 'sources' (2+ names) and 'result'"
        missing = [s for s in sources if s not in taxonomy.active]
        if missing:
            return False, f"MERGE sources not found in taxonomy: {missing}. Active labels: {taxonomy.label_names()}"
        return True, ""

    if t == "SPLIT":
        source = d.get("source", "").strip()
        results = d.get("results", [])
        if not source or len(results) < 2:
            return False, "SPLIT requires 'source' and at least 2 'results'"
        if source not in taxonomy.active:
            return False, f"SPLIT source '{source}' not found. Active labels: {taxonomy.label_names()}"
        return True, ""

    if t == "RENAME":
        old = d.get("old", "").strip()
        new = d.get("new", "").strip()
        if not old or not new:
            return False, "RENAME requires 'old' and 'new' fields"
        if old not in taxonomy.active:
            return False, f"RENAME target '{old}' not found. Active labels: {taxonomy.label_names()}"
        return True, ""

    if t == "DELETE":
        name = d.get("name", "").strip()
        if not name or name not in taxonomy.active:
            return False, f"DELETE target '{name}' not found. Active labels: {taxonomy.label_names()}"
        return True, ""

    return False, f"Unknown operation type '{t}'"
