"""
Smoke test: runs 5 steps with a mocked LLM call to verify the full pipeline.
"""
import sys
import json
import random
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import discover_models, load_sub_texts, match_models
from state import RunConfig, RunState, list_runs
from sampling import sample_next
from labeler import call_labeler, operation_to_taxonomy_op, validate_operation, LabelResponse
from taxonomy import TaxonomyOperation

# ── Mock LLM responses ────────────────────────────────────────────────────────

STEP = 0
def mock_call_labeler(model, sub_text, taxonomy, step, max_labels):
    global STEP
    STEP += 1
    labels = list(taxonomy.label_names())[:2] if taxonomy.label_names() else []
    # On step 2 propose a CREATE
    if STEP == 2:
        return LabelResponse(
            labels=labels,
            operation={"type": "CREATE", "name": "Harm Assessment", "description": "Evaluating whether a request could cause harm."},
            justification="No existing label covers harm evaluation.",
        )
    # On step 4 propose an invalid CREATE (taxonomy may or may not be at limit)
    if STEP == 4:
        return LabelResponse(
            labels=labels,
            operation={"type": "CREATE", "name": "Harm Assessment", "description": "Duplicate."},
            justification="Duplicate — should be rejected.",
        )
    return LabelResponse(labels=labels, operation="NONE", justification="Existing labels sufficient.")


# ── Run smoke test ────────────────────────────────────────────────────────────

def main():
    print("=== Smoke Test: 5-step run with mocked LLM ===\n")

    # 1. Discover models
    available = discover_models()
    print(f"Available models: {len(available)} JSONL files")

    matched = match_models(["deepseek"], available)
    print(f"Matched for 'deepseek': {matched[:3]}... ({len(matched)} files)")

    # 2. Load sub-texts (just one file for speed)
    sub_texts = load_sub_texts([matched[0]])
    print(f"Sub-texts from first file: {len(sub_texts)}")
    assert len(sub_texts) > 0, "No sub-texts loaded"

    # 3. Create run state
    config = RunConfig(models=matched[:1], labeler="test/mock", max_labels=20)
    state, state_path = RunState.load_or_create(None, config)
    state.snapshot(0)
    state.save(state_path)
    print(f"\nRun ID: {state.run_id}")
    print(f"State path: {state_path}")

    rng = random.Random(42)

    # 4. Run 5 steps with mocked LLM
    print("\n--- Steps ---")
    for i in range(5):
        step = i + 1
        sub_text, is_revisit = sample_next(sub_texts, state, rng=rng)

        response = mock_call_labeler("test/mock", sub_text, state.taxonomy, step, 20)
        tax_op = operation_to_taxonomy_op(response, sub_text, step)

        triggered_change = False
        if tax_op.operation != "NONE":
            if validate_operation(tax_op, state.taxonomy, 20):
                state.taxonomy.apply_operation(tax_op)
                triggered_change = True
            else:
                state.stats["total_invalid_proposals"] += 1
                tax_op = TaxonomyOperation(
                    step=step, operation="NONE", details={},
                    triggered_by=tax_op.triggered_by,
                    justification=f"[rejected] {tax_op.justification}",
                )

        state.taxonomy.increment_usage(response.labels)
        state.record_operation(tax_op, triggered_change)
        state.mark_sub_text(
            composite_id=sub_text.composite_id,
            sub_text_idx=sub_text.sub_text_idx,
            labels=response.labels,
            triggered_change=triggered_change,
            is_revisit=is_revisit,
        )
        state.stats["steps_completed"] += 1
        state.save(state_path)

        print(f"  Step {step}: op={tax_op.operation} change={triggered_change} labels={response.labels}")

    # 5. Verify state
    print("\n--- Verification ---")
    loaded = RunState.load(state_path)
    assert loaded.stats["steps_completed"] == 5, f"Expected 5 steps, got {loaded.stats['steps_completed']}"
    assert len(loaded.history["operations"]) == 5
    assert "Harm Assessment" in loaded.taxonomy.active, "CREATE not applied"
    assert loaded.stats["total_invalid_proposals"] == 1, "Duplicate CREATE should have been rejected"
    assert loaded.stats["total_changes"] == 1
    assert len(loaded.history["snapshots"]) >= 1  # step-0 snapshot

    print(f"steps_completed: {loaded.stats['steps_completed']} ✓")
    print(f"operations recorded: {len(loaded.history['operations'])} ✓")
    print(f"active labels: {list(loaded.taxonomy.active.keys())} ✓")
    print(f"invalid proposals rejected: {loaded.stats['total_invalid_proposals']} ✓")
    print(f"snapshots: {len(loaded.history['snapshots'])} ✓")

    # 6. Test list_runs
    runs = list_runs()
    assert any(r["run_id"] == loaded.run_id for r in runs)
    print(f"list_runs: found {len(runs)} run(s) ✓")

    print("\n=== All assertions passed ✓ ===")


if __name__ == "__main__":
    main()
