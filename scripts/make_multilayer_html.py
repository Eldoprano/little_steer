#!/usr/bin/env python3
"""
Generate multi-layer token similarity grids for the top-performing labels.

Loads the best steering vectors from sweep_results.json, runs one forward pass
per label on a representative test entry, and builds a standalone HTML page.

Run from: /home/eldoprano/Studium/GeneralCode/Reasoning_behaviours/
  python little_steer/scripts/make_multilayer_html.py

Produces: little_steer/multilayer_grids.html
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT    = Path(__file__).parent.parent.parent
CACHE_PATH   = REPO_ROOT / "little_steer" / "full_cache.pt"
RESULTS_PATH = REPO_ROOT / "little_steer" / "sweep_results.json"
OUT_PATH     = REPO_ROOT / "little_steer" / "multilayer_grids.html"
DATA_PATH    = REPO_ROOT / "data" / "little_steer_dataset.jsonl"
MODEL_ID     = "Qwen/Qwen3.5-4B"

sys.path.insert(0, str(REPO_ROOT))
from little_steer import (
    ExtractionResult, LittleSteerModel, load_dataset,
)
from little_steer.vectors.steering_vector import SteeringVector
from little_steer.visualization.token_view import render_multilayer_html
from little_steer.probing import get_token_similarities

LAYERS = list(range(12, 28, 2))   # [12,14,...,26] — same as extraction
TOP_N  = 8                         # how many labels to show


def build_vec_from_cache(np_cache, spec, label, layer, method, baseline, all_labels, train_frac=0.70, seed=42):
    """Reconstruct the steering vector that produced the sweep result."""
    MIN_POS = 5

    def split_arr(arr, rng_seed):
        rng = np.random.default_rng(rng_seed)
        idx = np.arange(len(arr))
        rng.shuffle(idx)
        n_train = max(1, int(len(arr) * train_frac))
        return arr[idx[:n_train]]

    pos_arr = np_cache.get((spec, label, layer))
    if pos_arr is None:
        return None
    pos_tr = split_arr(pos_arr, seed + hash(label) % (2**20))

    if method == "mean_difference":
        if baseline == "all_others":
            neg_parts = []
            for lbl in all_labels:
                if lbl == label:
                    continue
                arr = np_cache.get((spec, lbl, layer))
                if arr is not None:
                    neg_parts.append(split_arr(arr, seed + hash(lbl) % (2**20)))
            if not neg_parts:
                return None
            neg_tr = np.vstack(neg_parts)
        else:
            arr = np_cache.get((spec, baseline, layer))
            if arr is None:
                return None
            neg_tr = split_arr(arr, seed + hash(baseline) % (2**20))
        vec_np = pos_tr.mean(0) - neg_tr.mean(0)

    elif method == "mean_centering":
        other_means = []
        for lbl in all_labels:
            if lbl == label:
                continue
            arr = np_cache.get((spec, lbl, layer))
            if arr is not None:
                other_means.append(split_arr(arr, seed + hash(lbl) % (2**20)).mean(0))
        if not other_means:
            return None
        vec_np = pos_tr.mean(0) - np.stack(other_means).mean(0)

    elif method == "pca_direction":
        from sklearn.decomposition import PCA
        if len(pos_tr) < 3:
            return None
        Xc = pos_tr - pos_tr.mean(0)
        pca = PCA(n_components=1, svd_solver="randomized", random_state=42)
        pca.fit(Xc)
        vec_np = pca.components_[0]

    else:
        return None

    return torch.from_numpy(vec_np.copy()).float()


def main():
    # ── Load sweep results, pick top N labels ─────────────────────────────────
    print(f"Loading sweep results from {RESULTS_PATH}...")
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    by_label = {}
    for r in results:
        t = r["target"]
        if t not in by_label or r["auroc"] > by_label[t]["auroc"]:
            by_label[t] = r

    top_labels = sorted(by_label.items(), key=lambda kv: -kv[1]["auroc"])[:TOP_N]
    print(f"  Top {TOP_N} labels:")
    for lbl, r in top_labels:
        short = lbl.replace("I_","").replace("II_","").replace("III_","").replace("IV_","")
        print(f"    {short:<35} AUROC={r['auroc']:.4f}  {r['spec']} L{r['layer']}  {r['method']}  {r['baseline']}")

    # ── Load activation cache ──────────────────────────────────────────────────
    print(f"\nLoading activation cache {CACHE_PATH}...")
    t0 = time.time()
    res = ExtractionResult.load(str(CACHE_PATH))
    all_labels = sorted(res.labels())
    print(f"  {res.metadata.n_conversations} convs, {len(all_labels)} labels  [{time.time()-t0:.1f}s]")

    # Pre-convert to numpy (fast vector reconstruction)
    print("  Pre-converting to numpy... ", end="", flush=True)
    t1 = time.time()
    np_cache: dict[tuple, np.ndarray] = {}
    needed_specs = {r["spec"] for _, r in top_labels}
    for spec_s in needed_specs:
        for lbl in all_labels:
            for lyr in LAYERS:
                tensors = res.get(spec_s, lbl, lyr)
                if tensors:
                    if tensors[0].dim() == 1:
                        arr = torch.stack([t.float() for t in tensors]).numpy().astype(np.float32)
                    else:
                        arr = torch.stack([t.float().mean(0) for t in tensors]).numpy().astype(np.float32)
                    np_cache[(spec_s, lbl, lyr)] = arr
    del res; gc.collect()
    print(f"done in {time.time()-t1:.1f}s")

    # ── Load dataset ───────────────────────────────────────────────────────────
    print(f"\nLoading dataset {DATA_PATH}...")
    entries = load_dataset(str(DATA_PATH))
    print(f"  {len(entries)} conversations")

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"\nLoading model {MODEL_ID}...")
    t2 = time.time()
    model = LittleSteerModel(
        model_id=MODEL_ID,
        use_pretrained_loading=True,
        allow_multimodal=True,
        check_renaming=False,
    )
    print(f"  Model loaded in {time.time()-t2:.1f}s")

    # ── Generate grids ─────────────────────────────────────────────────────────
    grid_sections = []

    for target, best_r in top_labels:
        spec   = best_r["spec"]
        layer  = best_r["layer"]
        method = best_r["method"]
        baseline = best_r["baseline"]
        short  = target.replace("I_","").replace("II_","").replace("III_","").replace("IV_","")

        print(f"\n[{short}] spec={spec} L{layer} {method} {baseline}")

        # Build vector
        vec_tensor = build_vec_from_cache(np_cache, spec, target, layer, method, baseline, all_labels)
        if vec_tensor is None:
            print(f"  SKIP: could not build vector")
            continue

        sv = SteeringVector(
            vector=vec_tensor,
            layer=layer,
            label=target,
            method=method,
            extraction_spec=spec,
        )

        # Find a test entry that has this label annotated
        test_entry = None
        for e in entries:
            if any(target in ann.labels for ann in e.annotations):
                test_entry = e
                break
        if test_entry is None:
            print(f"  SKIP: no entry found with label")
            continue

        # Count labeled spans in this entry
        n_spans = sum(1 for ann in test_entry.annotations if target in ann.labels)
        print(f"  Entry found: {n_spans} labeled spans")

        # Run forward pass & build token similarities at all extraction layers
        t3 = time.time()
        try:
            token_sims = get_token_similarities(model, test_entry, sv, layers=LAYERS)
        except Exception as ex:
            print(f"  ERROR: {ex}")
            continue
        print(f"  Forward pass done in {time.time()-t3:.1f}s")

        # Render multi-layer HTML
        grid_html = render_multilayer_html(token_sims, layers=LAYERS)

        auroc_str = f"{best_r['auroc']:.4f}"
        f1_str    = f"{best_r['f1']:.4f}"
        section = f"""
<section class="label-section">
  <h2>{short.replace("_"," ")}</h2>
  <div class="meta">
    AUROC <b>{auroc_str}</b> &nbsp;·&nbsp; F1 <b>{f1_str}</b> &nbsp;·&nbsp;
    spec <code>{spec}</code> &nbsp;·&nbsp; Layer <b>{layer}</b> &nbsp;·&nbsp;
    method <code>{method}</code> &nbsp;·&nbsp; baseline <code>{baseline}</code>
  </div>
  <p class="hint">Red = high similarity (behaviour present). Blue = absent.
  Underlined tokens fall inside an annotated span for this label.
  Each row is a transformer layer — watch how signal builds with depth.</p>
  {grid_html}
</section>
"""
        grid_sections.append(section)
        gc.collect()

    # ── Assemble HTML ──────────────────────────────────────────────────────────
    CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #f8f8f8; color: #222; }
header { background: #1c3557; color: white; padding: 2rem 3rem; }
header h1 { font-size: 1.6rem; margin-bottom: 0.4rem; }
header p { opacity: 0.8; font-size: 0.95rem; }
main { max-width: 1600px; margin: 2rem auto; padding: 0 2rem; }
.label-section { margin-bottom: 4rem; padding: 1.5rem; background: white;
                 border-radius: 8px; border: 1px solid #ddd; }
h2 { font-size: 1.2rem; color: #1c3557; margin-bottom: 0.5rem; }
.meta { font-size: 0.85rem; color: #555; margin-bottom: 0.8rem; }
.hint { font-size: 0.82rem; color: #888; margin-bottom: 1rem; }
code { background: #eef; padding: 1px 4px; border-radius: 3px; font-size: 0.82rem; }
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Multi-layer Token Similarity Grids</title>
  <style>{CSS}</style>
</head>
<body>
<header>
  <h1>Multi-layer Token Similarity Grids</h1>
  <p>Top {TOP_N} labels by honest AUROC · Qwen3.5-4B · layers 12–26</p>
</header>
<main>
{"".join(grid_sections)}
</main>
</body>
</html>"""

    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
