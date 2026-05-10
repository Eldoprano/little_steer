"""Internal helpers shared across the public API.

This package is private — its contents may change between minor versions.
External users should depend on the public modules (extraction, vectors,
probing, steering) instead.
"""

from .forward import ForwardPassResult, run_forward_pass
from .span_eval import (
    EntryContext,
    SpanActivation,
    prepare_entry,
    select_span_activation,
    stack_vectors,
    cosine_against_stack,
)

__all__ = [
    "ForwardPassResult",
    "run_forward_pass",
    "EntryContext",
    "SpanActivation",
    "prepare_entry",
    "select_span_activation",
    "stack_vectors",
    "cosine_against_stack",
]
