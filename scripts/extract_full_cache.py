#!/usr/bin/env python3
"""
Extract activations for ALL 994 conversations with ALL specs in one pass.
Saves to full_cache.pt — everything needed for the sweep.

Specs extracted (all in a single forward pass per conversation):
  - whole_sentence  : all tokens, mean
  - last_1          : last token only
  - first_1         : first token only
  - last_3          : last 3 tokens, mean
  - first_3         : first 3 tokens, mean
  - bleed_3         : all + 3 context tokens before/after, mean
  - top_confident   : tokens where model is most confident (logit-based)
  - entropy_w       : entropy-weighted mean (logit-based)

Layers: 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30  (dense coverage)

Run from: /home/eldoprano/Studium/GeneralCode/Reasoning_behaviours/
  python little_steer/scripts/extract_full_cache.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from little_steer import (
    LittleSteerModel, ActivationExtractor,
    ExtractionPlan, ExtractionSpec, TokenSelection, ExtractionResult,
    load_dataset,
)
import numpy as np

DATA_PATH  = REPO_ROOT / "data" / "little_steer_dataset.jsonl"
OUT_PATH   = REPO_ROOT / "little_steer" / "full_cache.pt"
MODEL_ID   = "Qwen/Qwen3.5-4B"

LAYERS = list(range(12, 28, 2))  # [12,14,...,26]  → 8 layers (consistent with training)

def main():
    import torch
    print(f"Device: {'CUDA ('+torch.cuda.get_device_name(0)+')' if torch.cuda.is_available() else 'CPU'}")

    if OUT_PATH.exists():
        print(f"Cache already exists: {OUT_PATH}")
        r = ExtractionResult.load(str(OUT_PATH))
        print(r.summary())
        return

    print("Loading dataset...")
    all_entries = load_dataset(str(DATA_PATH))
    print(f"  {len(all_entries)} conversations")

    print("\nLoading model...")
    model = LittleSteerModel(
        model_id=MODEL_ID,
        use_pretrained_loading=True,
        allow_multimodal=True,
        check_renaming=False,
    )
    extractor = ActivationExtractor(model)

    # Build plan — all specs in one forward pass
    plan = ExtractionPlan(name="full_cache")

    plan.add_spec("whole_sentence",
        ExtractionSpec(TokenSelection("all", aggregation="mean"), layers=LAYERS))

    plan.add_spec("last_1",
        ExtractionSpec(TokenSelection("last", aggregation="mean"), layers=LAYERS))

    plan.add_spec("first_1",
        ExtractionSpec(TokenSelection("first", aggregation="mean"), layers=LAYERS))

    plan.add_spec("last_3",
        ExtractionSpec(TokenSelection("last_n", n=3, aggregation="mean"), layers=LAYERS))

    plan.add_spec("first_3",
        ExtractionSpec(TokenSelection("first_n", n=3, aggregation="mean"), layers=LAYERS))

    plan.add_spec("bleed_3",
        ExtractionSpec(TokenSelection("all", bleed_before=3, bleed_after=3, aggregation="mean"), layers=LAYERS))

    # Logit-based aggregations require full LM-head pass → very slow, skip for now
    # plan.add_spec("top_confident", ...)
    # plan.add_spec("entropy_weighted", ...)

    print(f"\nExtracting {len(all_entries)} conversations, {len(LAYERS)} layers, {len(plan.specs)} specs...")
    print(f"  Specs: {list(plan.specs.keys())}")
    print(f"  Layers: {LAYERS}\n")

    t0 = time.time()
    result = extractor.extract(all_entries, plan)
    elapsed = time.time() - t0

    print(f"\nExtraction done in {elapsed:.0f}s ({elapsed/len(all_entries):.1f}s/conv)")
    result.save(str(OUT_PATH))
    print(f"Saved → {OUT_PATH}")
    print(result.summary())

if __name__ == "__main__":
    main()
