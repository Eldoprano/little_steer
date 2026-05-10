import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium", app_title="Behavior Detection Explorer")


@app.cell
def _():
    import marimo as mo
    import sys, os

    # Force LD_LIBRARY_PATH to include nvidia libraries in the venv
    _venv_site = os.path.join(os.path.dirname(__file__), ".venv", "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
    if os.path.exists(_venv_site):
        _nv_paths = []
        for _pkg in ["cudnn", "cublas", "cuda_runtime", "nccl", "cuda_nvrtc"]:
            _p = os.path.join(_venv_site, "nvidia", _pkg, "lib")
            if os.path.exists(_p):
                _nv_paths.append(_p)

        if _nv_paths:
            _current = os.environ.get("LD_LIBRARY_PATH", "")
            _new = ":".join(_nv_paths)
            os.environ["LD_LIBRARY_PATH"] = f"{_new}:{_current}" if _current else _new

    import torch
    import numpy as np
    import polars as pl
    import altair as alt

    # Ensure we can import local modules
    if os.path.dirname(__file__) not in sys.path:
        sys.path.insert(0, os.path.dirname(__file__))

    import little_steer as ls
    return alt, ls, mo, np, os, pl, sys, torch


@app.cell
def _(mo):
    _css = """
    :root {
      --bg: #2d353b;
      --bg-dim: #232a2e;
      --bg-1: #343f44;
      --bg-2: #3d484d;
      --bg-3: #475258;
      --fg: #d3c6aa;
      --fg-dim: #9da9a0;
      --red: #e67e80;
      --orange: #e69875;
      --yellow: #dbbc7f;
      --green: #a7c080;
      --aqua: #83c092;
      --blue: #7fbbb3;
      --purple: #d699b6;
      --grey: #7a8478;
    }

    [data-theme="light"] {
      --bg: #fdf6e3;
      --bg-dim: #f4f0d9;
      --bg-1: #e9e4ca;
      --bg-2: #ddd8be;
      --bg-3: #cac9ad;
      --fg: #5c6a72;
      --fg-dim: #829181;
      --red: #f85552;
      --orange: #f57d26;
      --yellow: #dfa000;
      --green: #8da101;
      --aqua: #35a77c;
      --blue: #3a94c5;
      --purple: #df69ba;
      --grey: #a6b0a0;
    }

    body {
      font-family: 'Inter', system-ui, sans-serif;
      background-color: var(--bg-dim) !important;
      color: var(--fg) !important;
    }

    .marimo-content {
        max-width: 1200px;
        margin: 0 auto;
    }

    .card {
      background: var(--bg-1);
      border: 1px solid var(--bg-3);
      border-radius: 8px;
      padding: 24px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
      margin-bottom: 24px;
    }

    #theme-toggle {
      position: fixed;
      top: 20px;
      right: 20px;
      background: var(--bg-1);
      border: 1px solid var(--bg-3);
      border-radius: 50%;
      width: 44px;
      height: 44px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 20px;
      z-index: 9999;
      box-shadow: 0 2px 8px rgba(0,0,0,0.4);
      transition: all 0.2s ease;
    }

    #theme-toggle:hover {
      transform: scale(1.1);
      background: var(--bg-2);
    }

    h1, h2, h3 { color: var(--green); }
    a { color: var(--blue); }

    /* Style marimo elements to match theme */
    .mo-md { color: var(--fg); }
    """

    _script = """
    (function () {
        const root = document.documentElement;
        const btn = document.getElementById("theme-toggle");
        const saved = localStorage.getItem("theme") || "dark";
        root.dataset.theme = saved;
        if (btn) btn.textContent = saved === "dark" ? "☀️" : "🌙";

        if (btn) {
            btn.onclick = () => {
                const next = root.dataset.theme === "dark" ? "light" : "dark";
                root.dataset.theme = next;
                localStorage.setItem("theme", next);
                btn.textContent = next === "dark" ? "☀️" : "🌙";
            };
        }
    })();
    """
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md("""
    # Behavior Detection Explorer

    Loads entries from the dataset, extracts activations across all layers,
    trains an MLP probe, and lets you interactively visualize which behaviors are
    detected where in the reasoning traces.

    **Steps:** Configure → Load → Extract → (Layer comparison) → Train probe → Evaluate → Browse entries
    """)
    return


@app.cell
def _():
    import json as _json
    import os as _os
    _tax_path = _os.path.normpath(
        _os.path.join(_os.path.dirname(__file__), "..", "2_labeling", "taxonomy.json")
    )
    try:
        with open(_tax_path) as _f:
            _taxonomy = _json.load(_f)
        ALL_LABELS = [
            lbl["id"]
            for g in _taxonomy["groups"]
            for lbl in g["labels"]
            if lbl["id"] != "none"
        ]
    except Exception:
        ALL_LABELS = [
            "rephrasePrompt", "speculateUserMotive", "flagEvaluationAwareness",
            "reframeTowardSafety", "flagAsHarmful", "enumerateHarms",
            "stateSafetyConcern", "stateLegalConcern", "stateEthicalConcern",
            "referenceOwnPolicy", "cautiousFraming", "stateFactOrKnowledge",
            "stateFalseClaim", "detailHarmfulMethod", "noteRiskWhileDetailingHarm",
            "intendRefusal", "intendHarmfulCompliance", "planResponseStructure",
            "suggestSafeAlternative", "produceResponseDraft", "expressUncertainty",
            "selfCorrect", "planReasoningStep", "summarizeReasoning", "neutralFiller",
        ]
    return (ALL_LABELS,)


@app.cell
def _(mo):
    mo.md("""
    ## § 1 — Configuration
    """)
    return


@app.cell
def _(mo):
    cfg_model_id = mo.ui.text(
        value="Qwen/Qwen3.5-4B",
        label="Model ID",
        full_width=True,
    )
    cfg_data_path = mo.ui.text(
        value="../data/dataset.jsonl",
        label="Dataset path",
        full_width=True,
    )
    cfg_n_entries = mo.ui.slider(
        start=10, stop=500, step=10, value=80,
        label="Entries to sample",
        show_value=True,
    )
    cfg_layer_stride = mo.ui.slider(
        start=1, stop=8, step=1, value=2,
        label="Layer stride (1=every layer, 2=every other, …)",
        show_value=True,
    )
    mo.vstack([
        mo.hstack([cfg_model_id, cfg_data_path], justify="start", gap=2),
        mo.hstack([cfg_n_entries, cfg_layer_stride], justify="start", gap=2),
    ])
    return cfg_data_path, cfg_layer_stride, cfg_model_id, cfg_n_entries


@app.cell
def _(ALL_LABELS, mo):
    cfg_labels = mo.ui.multiselect(
        options=ALL_LABELS,
        value=[
            "stateSafetyConcern", "intendRefusal", "intendHarmfulCompliance",
            "rephrasePrompt", "neutralFiller",
        ],
        label="Behavior labels to probe",
    )
    cfg_labels
    return (cfg_labels,)


@app.cell
def _(mo):
    mo.md("""
    ---
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## § 2 — Load model and dataset
    """)
    return


@app.cell
def _(mo):
    load_btn = mo.ui.run_button(label="Load model + dataset")
    load_btn
    return (load_btn,)


@app.cell
def _(cfg_data_path, cfg_model_id, load_btn, ls, mo, torch):
    import random
    mo.stop(not load_btn.value)

    with mo.status.spinner("Loading model…"):
        _model_id = cfg_model_id.value
        _kwargs = {}
        if "Qwen" in _model_id:
            _kwargs["attn_rename"] = "linear_attn"
        model = ls.LittleSteerModel(
            _model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            use_pretrained_loading=True,
            check_renaming=False,
            **_kwargs,
        )

    with mo.status.spinner("Loading dataset…"):
        _all = list(ls.iter_dataset(cfg_data_path.value))
        all_annotated = [e for e in _all if e.annotations]

    # Collect all judge names and how many entries each has
    _judge_counts: dict[str, int] = {}
    for _e in all_annotated:
        for _lr in _e.label_runs:
            _judge_counts[_lr.judge_name] = _judge_counts.get(_lr.judge_name, 0) + 1
    available_judges = sorted(_judge_counts, key=lambda j: -_judge_counts[j])

    mo.md(
        f"✅ **{model}**  \n"
        f"Dataset: {len(all_annotated)} annotated entries loaded  \n"
        f"Layers: {model.num_layers} &nbsp;·&nbsp; Hidden size: {model.hidden_size}  \n"
        f"Judges: {', '.join(f'`{j}` ({_judge_counts[j]})' for j in available_judges)}"
    )
    return all_annotated, available_judges, model


@app.cell
def _(available_judges, mo):
    cfg_judge = mo.ui.dropdown(
        options=available_judges,
        value=available_judges[0] if available_judges else None,
        label="Judge (labeler)",
    )
    cfg_judge
    return (cfg_judge,)


@app.cell
def _(all_annotated, cfg_judge, cfg_n_entries, mo):
    """Filter entries by selected judge, apply activation, sample N."""
    import random as _random
    
    try:
        _judge_name = cfg_judge.value
    except (NameError, AttributeError):
        mo.stop(True, mo.md("⚠️ `cfg_judge` is not defined. Try restarting the marimo server."))
        
    mo.stop(_judge_name is None, mo.md("⚠️ No judge selected."))

    # Filter to entries that have a label_run from this judge
    _filtered = [
        e for e in all_annotated
        if any(lr.judge_name == _judge_name for lr in e.label_runs)
    ]
    if not _filtered:
        mo.stop(True, mo.md(f"⚠️ No entries found with judge `{_judge_name}`."))

    # Sample and activate
    _random.seed(42)
    _sampled = _random.sample(_filtered, min(cfg_n_entries.value, len(_filtered)))
    for _e in _sampled:
        _lr = next(lr for lr in _e.label_runs if lr.judge_name == _judge_name)
        _e.set_active_label_run(_lr)

    entries = _sampled
    mo.callout(
        mo.md(
            f"Judge **{_judge_name}** · {len(entries)} entries sampled "
            f"(from {len(_filtered)} available)"
        ),
        kind="success",
    )
    return (entries,)


@app.cell
def _():
    _cache = {"result": None, "entry_ids": set()}
    extraction_cache = _cache
    return (extraction_cache,)


@app.cell
def _(mo):
    probe_state, set_probe_state = mo.state(None)
    vectors_state, set_vectors_state = mo.state(None)
    return probe_state, set_probe_state, set_vectors_state, vectors_state


@app.cell
def _(mo):
    mo.md("""
    ## § 3 — Activation extraction
    """)
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.callout(
            mo.md(
                "Extracts activations at **all layers** with the configured stride in a single forward pass per entry.  \n"
                "Re-clicking **Extract** with more entries accumulates — previously extracted entries are skipped."
            ),
            kind="info",
        ),
    ])
    return


@app.cell
def _(mo):
    extract_btn = mo.ui.run_button(label="Extract activations")
    extract_btn
    return (extract_btn,)


@app.cell
def _(cfg_layer_stride, entries, extract_btn, extraction_cache, ls, mo, model):
    mo.stop(not extract_btn.value)

    # 80/20 train/test split — extraction only on train_entries
    # test_entries are held out for unbiased evaluation in § 5
    import random as _rnd
    _rnd.seed(0)
    _shuffled = list(entries)
    _rnd.shuffle(_shuffled)
    _n_train = int(0.8 * len(_shuffled))
    train_entries = _shuffled[:_n_train]
    test_entries = _shuffled[_n_train:]

    _stride = cfg_layer_stride.value
    _layers_to_extract = list(range(0, model.num_layers, _stride))
    _plan = ls.ExtractionPlan("explore", specs={
        "whole_sentence": ls.ExtractionSpec(
            ls.TokenSelection("all"), layers=_layers_to_extract
        ),
        "last_token": ls.ExtractionSpec(
            ls.TokenSelection("last"), layers=_layers_to_extract
        ),
    })

    _prev_ids = extraction_cache["entry_ids"]
    _new_entries = [e for e in train_entries if e.id not in _prev_ids]

    if _new_entries:
        _extractor = ls.ActivationExtractor(model, max_seq_len=3072)
        _new_result = _extractor.extract(
            _new_entries,
            _plan,
            show_progress=False,
            progress_fn=lambda ds: mo.status.progress_bar(
                ds,
                title=f"Extracting {len(_new_entries)} train entries ({len(_layers_to_extract)} layers each)…",
                completion_title="Extraction done!",
            ),
        )
        _prev = extraction_cache["result"]
        if _prev is not None:
            _new_result.merge_from(_prev)
        extraction_cache["result"] = _new_result
        extraction_cache["entry_ids"] = _prev_ids | {e.id for e in _new_entries}

    extraction_result = extraction_cache["result"]
    if extraction_result is None:
        mo.stop(True, mo.md("⚠️ No extraction data yet — click **Extract activations** above."))
    _avail_layers = extraction_result.layers()
    _label_counts = {
        lbl: len(extraction_result.get("whole_sentence", lbl, _avail_layers[0]))
        for lbl in sorted(extraction_result.labels())
    }

    mo.vstack([
        mo.md(
            f"✅ **{len(extraction_cache['entry_ids'])}** train entries extracted &nbsp;·&nbsp; "
            f"**{len(test_entries)}** held out for evaluation &nbsp;·&nbsp; "
            f"**{len(_avail_layers)}** layers ({_avail_layers[0]}–{_avail_layers[-1]})"
        ),
        mo.ui.table(
            [{"label": k, "samples": v} for k, v in sorted(_label_counts.items(), key=lambda x: -x[1])],
            selection=None,
        ),
    ])
    return extraction_result, test_entries


@app.cell
def _(mo):
    mo.md("""
    ## § 3.5 — Layer comparison
    """)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            "Trains a small probe on each extracted layer (80 / 20 train-val split, 30 epochs).  \n"
            "The chart shows which layers best detect the selected behaviors."
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    compare_btn = mo.ui.run_button(label="Run layer comparison")
    compare_btn
    return (compare_btn,)


@app.cell
def _(cfg_labels, compare_btn, extraction_result, ls, mo, np, torch):
    import matplotlib.pyplot as _plt
    mo.stop(not compare_btn.value)
    if extraction_result is None:
        mo.stop(True, mo.md("⚠️ Run extraction first (§ 3)."))

    _labels = cfg_labels.value
    if not _labels:
        mo.stop(True, mo.md("⚠️ Select at least one label in § 1."))

    # Filter to labels actually present in the extraction result
    _available_labels = set(extraction_result.labels())
    _labels_present = [l for l in _labels if l in _available_labels]
    if not _labels_present:
        mo.stop(True, mo.md(
            f"⚠️ None of the selected labels `{_labels}` are in the extraction result. "
            f"Available: `{sorted(_available_labels)[:10]}`"
        ))

    _trainer = ls.MLPProbeTrainer()
    _layers = extraction_result.layers()
    _layer_results = []
    _errors = []

    def _auroc_numpy(y_true, y_scores):
        order = np.argsort(-y_scores)
        ys = y_true[order].astype(float)
        n_pos, n_neg = ys.sum(), len(ys) - ys.sum()
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        tpr = np.concatenate([[0.0], np.cumsum(ys) / n_pos])
        fpr = np.concatenate([[0.0], np.cumsum(1 - ys) / n_neg])
        _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        return float(_trapz(tpr, fpr))

    for _layer in mo.status.progress_bar(
        _layers, title="Comparing layers…", completion_title="Done!"
    ):
        try:
            _X, _Y = _trainer._build_training_data(
                extraction_result, "whole_sentence", _layer, _labels_present, None
            )
            if len(_X) < 10:
                _errors.append(f"Layer {_layer}: only {len(_X)} samples")
                continue
            _n = len(_X)
            _rng = np.random.RandomState(42)
            _perm = torch.from_numpy(_rng.permutation(_n))
            _tr = _perm[:int(0.8 * _n)]
            _vl = _perm[int(0.8 * _n):]

            _probe = ls.MLPProbe(_X.shape[1], len(_labels_present), hidden_dim=128, labels=_labels_present)
            _trainer._train_mlp(
                _probe, _X[_tr], _Y[_tr],
                epochs=30, batch_size=32, lr=1e-3,
                device="cuda" if torch.cuda.is_available() else "cpu",
                show_progress=False,
            )
            with torch.no_grad():
                _vp = torch.sigmoid(_probe(_X[_vl].float())).numpy()
            _vy = _Y[_vl].numpy()
            _aurocs = [_auroc_numpy(_vy[:, j], _vp[:, j]) for j in range(len(_labels_present))]
            _valid = [a for a in _aurocs if not np.isnan(a)]
            _mean = float(np.mean(_valid)) if _valid else float("nan")
            _layer_results.append({"layer": _layer, "auroc": _mean})
        except Exception as _exc:
            _errors.append(f"Layer {_layer}: {_exc}")
            _layer_results.append({"layer": _layer, "auroc": float("nan")})

    _valid_results = [r for r in _layer_results if not np.isnan(r["auroc"])]
    _best = max(_valid_results, key=lambda r: r["auroc"]) if _valid_results else None

    if not _valid_results:
        _err_summary = "\n".join(_errors[:5])
        mo.stop(True, mo.md(
            f"⚠️ No valid layer results.  \n"
            f"Labels checked: `{_labels_present}`  \n"
            f"Errors (first 5):\n```\n{_err_summary}\n```"
        ))

    _fig, _ax = _plt.subplots(figsize=(9, 3.5))
    _xs = [r["layer"] for r in _valid_results]
    _ys = [r["auroc"] for r in _valid_results]
    _ax.plot(_xs, _ys, "-o", color="#2166ac", linewidth=1.8, markersize=5)
    _ax.axhline(0.5, color="#aaa", linestyle="--", linewidth=1)
    if _best:
        _ax.axvline(_best["layer"], color="#d6604d", linestyle=":", linewidth=1.5, alpha=0.7)
    _ax.set_xlabel("Layer")
    _ax.set_ylabel("Mean val AUROC")
    _ax.set_title(f"Layer comparison — {len(_labels_present)} labels")
    _ax.set_ylim(0.4, 1.0)
    _ax.grid(True, alpha=0.25)
    _fig.tight_layout()

    mo.vstack([
        mo.md(f"Best layer: **{_best['layer']}** (AUROC = {_best['auroc']:.3f})" if _best else ""),
        mo.mpl.interactive(_fig),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## § 4 — Train MLP probe
    """)
    return


@app.cell
def _(extraction_result, mo):
    _available = extraction_result.layers() if extraction_result else []
    _mid = _available[len(_available) // 2] if _available else None
    cfg_layer = mo.ui.dropdown(
        options={str(l): l for l in _available},
        value=str(_mid) if _mid is not None else None,
        label="Probe layer",
    )
    cfg_epochs = mo.ui.slider(
        start=10, stop=300, step=10, value=60,
        label="Epochs", show_value=True,
    )
    cfg_lr = mo.ui.dropdown(
        options={"1e-2": 1e-2, "1e-3": 1e-3, "5e-4": 5e-4, "1e-4": 1e-4},
        value="1e-3",
        label="Learning rate",
    )
    cfg_hidden = mo.ui.slider(
        start=64, stop=512, step=64, value=256,
        label="Hidden dim", show_value=True,
    )
    cfg_batch = mo.ui.slider(
        start=8, stop=128, step=8, value=32,
        label="Batch size", show_value=True,
    )
    mo.vstack([
        cfg_layer,
        mo.hstack([cfg_epochs, cfg_lr, cfg_hidden, cfg_batch], justify="start", gap=2),
    ])
    return cfg_batch, cfg_epochs, cfg_hidden, cfg_layer, cfg_lr


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            "Training runs **on CPU** using the already-extracted activations — "
            "the model stays in VRAM and is not touched during training."
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    train_btn = mo.ui.run_button(label="Train probe")
    train_btn
    return (train_btn,)


@app.cell
def _(
    cfg_batch,
    cfg_epochs,
    cfg_hidden,
    cfg_labels,
    cfg_layer,
    cfg_lr,
    extraction_result,
    ls,
    mo,
    set_probe_state,
    train_btn,
):
    import matplotlib.pyplot as _plt
    mo.stop(not train_btn.value)
    _labels = cfg_labels.value
    if not _labels:
        mo.stop(True, mo.md("⚠️ Select at least one label in § 1."))
    if cfg_layer.value is None:
        mo.stop(True, mo.md("⚠️ No layers available — run extraction first (§ 3)."))

    _trainer = ls.MLPProbeTrainer()
    _probe, _history = _trainer.train(
        extraction_result,
        spec="whole_sentence",
        layer=cfg_layer.value,
        labels=_labels,
        method="mlp",
        epochs=cfg_epochs.value,
        batch_size=cfg_batch.value,
        lr=cfg_lr.value,
        hidden_dim=cfg_hidden.value,
        show_progress=False,
        progress_fn=lambda itr: mo.status.progress_bar(
            itr, title="Training MLP probe…", completion_title="Training done!"
        ),
        return_history=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    set_probe_state(_probe)

    _fig, (_ax1, _ax2) = _plt.subplots(1, 2, figsize=(10, 3))
    _epochs_x = [h["epoch"] for h in _history]
    _ax1.plot(_epochs_x, [h["loss"] for h in _history], color="#2166ac", linewidth=1.8)
    _ax1.set_xlabel("Epoch"); _ax1.set_ylabel("BCE Loss"); _ax1.set_title("Training Loss")
    _ax1.grid(True, alpha=0.3)
    _ax2.plot(_epochs_x, [h["acc"] for h in _history], color="#d6604d", linewidth=1.8)
    _ax2.set_xlabel("Epoch"); _ax2.set_ylabel("Exact-match Accuracy"); _ax2.set_title("Training Accuracy")
    _ax2.set_ylim(0, 1); _ax2.grid(True, alpha=0.3)
    _fig.tight_layout()

    mo.vstack([
        mo.md(
            f"✅ Probe trained · layer **{cfg_layer.value}** · {len(_labels)} labels  \n"
            f"Final loss: **{_history[-1]['loss']:.4f}** · accuracy: **{_history[-1]['acc']:.3f}**"
        ),
        mo.mpl.interactive(_fig),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## § 5 — Evaluation (steering vectors)
    """)
    return


@app.cell
def _(mo):
    eval_btn = mo.ui.run_button(label="Run evaluation")
    eval_btn
    return (eval_btn,)


@app.cell
def _(
    cfg_labels,
    cfg_layer,
    eval_btn,
    extraction_result,
    ls,
    mo,
    model,
    np,
    set_vectors_state,
    test_entries,
):
    import matplotlib.pyplot as _plt2
    mo.stop(not eval_btn.value)
    if cfg_layer.value is None:
        mo.stop(True, mo.md("⚠️ No layers available — run extraction first (§ 3)."))

    _layer = cfg_layer.value
    _labels = cfg_labels.value

    with mo.status.spinner("Building steering vectors…"):
        _builder = ls.SteeringVectorBuilder()
        _v_dict = _builder.build_all_labels(extraction_result, methods=["mean_centering"])
        _all_vecs = ls.SteeringVectorSet([v for vset in _v_dict.values() for v in vset])
        _vecs = _all_vecs.filter(layer=_layer)

    _eval_results = ls.evaluate_dataset(
        model, test_entries, _vecs, label_filter=_labels,
        show_progress=False,
        progress_fn=lambda ds: mo.status.progress_bar(
            ds, title=f"Evaluating on {len(test_entries)} held-out entries…",
            completion_title="Evaluation done!"
        ),
    )

    if not _eval_results:
        mo.stop(True, mo.md("⚠️ No evaluation results — check that labels have enough samples."))

    set_vectors_state(_vecs)

    _rows = sorted(
        [
            {
                "label": r.label,
                "method": r.method,
                "auroc": r.auroc if r.auroc == r.auroc else 0.5,
                "f1": r.f1,
                "n_present": r.n_present,
                "n_absent": r.n_absent,
            }
            for r in _eval_results
        ],
        key=lambda r: -r["auroc"],
    )

    _n_labels = len(set(r["label"] for r in _rows))
    _mean_auroc = float(np.mean([r["auroc"] for r in _rows]))

    # ── Chart ──────────────────────────────────────────────────────────────
    _labels_sorted = [r["label"] for r in _rows]
    _aurocs = [r["auroc"] for r in _rows]
    _colors = ["#4dac26" if a >= 0.8 else "#f4a582" if a >= 0.65 else "#d6604d" for a in _aurocs]

    _fig2, _ax = _plt2.subplots(figsize=(8, max(3, len(_rows) * 0.32 + 1)))
    _ax.barh(_labels_sorted[::-1], _aurocs[::-1], color=_colors[::-1], height=0.65)
    _ax.axvline(0.5, color="#aaa", linestyle="--", linewidth=1, label="random")
    _ax.axvline(0.7, color="#f4a582", linestyle=":", linewidth=1, label="useful")
    _ax.axvline(0.8, color="#4dac26", linestyle=":", linewidth=1, label="strong")
    _ax.set_xlim(0.3, 1.0)
    _ax.set_xlabel("AUROC")
    _ax.set_title(f"Layer {_layer} — Steering vector AUROC per label")
    _ax.legend(loc="lower right", fontsize=8)
    _fig2.tight_layout()

    # ── Explanation ─────────────────────────────────────────────────────
    _explanation = mo.callout(
        mo.md(f"""
    **How to read this:**
    - **AUROC** measures how well the steering vector direction separates *spans with* a label from *spans without* it — threshold-free.
    - **0.5** = random &nbsp;·&nbsp; **≥ 0.65** = useful &nbsp;·&nbsp; **≥ 0.80** = strong detection
    - Vectors are built on the **train set** (80%), evaluated on the **held-out test set** (20%) — scores are honest.
    - With **{_n_labels} label{"s" if _n_labels > 1 else ""}**, rare behaviors naturally have fewer training samples and tend to score lower.
    - Mean AUROC across all labels: **{_mean_auroc:.3f}**
        """),
        kind="neutral",
    )

    _table_rows = [
        {
            "label": r["label"], "method": r["method"],
            "AUROC": f"{r['auroc']:.3f}", "F1": f"{r['f1']:.3f}",
            "n_present": r["n_present"], "n_absent": r["n_absent"],
        }
        for r in _rows
    ]

    mo.vstack([
        _explanation,
        mo.mpl.interactive(_fig2),
        mo.accordion({"Detailed table": mo.ui.table(_table_rows, selection=None)}),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## § 6 — Entry browser
    """)
    return


@app.cell
def _(entries, mo, test_entries):
    def _user_prompt(e):
        for m in e.messages:
            if m["role"] == "user":
                return m["content"][:70].replace("\n", " ")
        return "(no user message)"

    # Build options with dedup suffix for duplicate prompt+model combos
    _seen: dict[str, int] = {}
    _options: dict[str, int] = {}
    _all_entries = entries  # browse all entries
    for _i, _e in enumerate(_all_entries):
        _split = "test" if _e in test_entries else "train"
        _base = f"[{_split}] {_user_prompt(_e)}  ·  {_e.model.split('/')[-1][:25]}"
        _count = _seen.get(_base, 0)
        _seen[_base] = _count + 1
        _key = _base if _count == 0 else f"{_base} #{_count + 1}"
        _options[_key] = _i

    entry_selector = mo.ui.dropdown(options=_options, label="Select entry")
    entry_selector
    return (entry_selector,)


@app.cell
def _(mo):
    cfg_threshold = mo.ui.slider(
        start=0.05, stop=0.95, step=0.05, value=0.40,
        label="Detection threshold", show_value=True,
    )
    cfg_mode = mo.ui.radio(
        options={
            "Token level": "token",
            "Span level (avg probabilities per annotated span)": "sentence",
        },
        value="Span level (avg probabilities per annotated span)",
        label="Display mode",
    )
    cfg_show_gt = mo.ui.checkbox(label="Ground truth", value=True)
    cfg_show_probe = mo.ui.checkbox(label="MLP probe", value=True)
    cfg_show_vectors = mo.ui.checkbox(label="Mean vector", value=False)
    cfg_skip_prefix = mo.ui.checkbox(label="Hide system/user", value=True)
    mo.vstack([
        mo.hstack([cfg_threshold, cfg_mode], justify="start", gap=2),
        mo.hstack([
            mo.md("**Show panels:**"),
            cfg_show_gt, cfg_show_probe, cfg_show_vectors,
            mo.md("  "),
            cfg_skip_prefix,
        ], justify="start", gap=1),
    ])
    return (
        cfg_mode,
        cfg_show_gt,
        cfg_show_probe,
        cfg_show_vectors,
        cfg_skip_prefix,
        cfg_threshold,
    )


@app.cell
def _(
    cfg_layer,
    cfg_mode,
    cfg_show_gt,
    cfg_show_probe,
    cfg_show_vectors,
    cfg_skip_prefix,
    cfg_threshold,
    entries,
    entry_selector,
    ls,
    mo,
    model,
    np,
    probe_state,
    vectors_state,
):
    mo.stop(entry_selector.value is None)
    if cfg_layer.value is None:
        mo.stop(True, mo.md("Run extraction first."))

    import importlib
    import little_steer.visualization.probe_view as _pv
    importlib.reload(_pv)

    _probe = probe_state()
    _vectors = vectors_state()
    _want_probe = cfg_show_probe.value
    _want_vectors = cfg_show_vectors.value
    _want_gt = cfg_show_gt.value

    if not (_want_gt or _want_probe or _want_vectors):
        mo.stop(True, mo.md("Select at least one panel to display."))
    if _want_probe and _probe is None:
        mo.stop(True, mo.md("Train the MLP probe first to show the probe panel."))
    if _want_vectors and _vectors is None:
        mo.stop(True, mo.md("Run evaluation to show the mean vector panel."))

    _entry = entries[entry_selector.value]
    _layer = cfg_layer.value
    _user_msg = next((m["content"] for m in _entry.messages if m["role"] == "user"), "")
    _model_name = _entry.model.split("/")[-1]

    _det_probe = None
    _det_vectors = None
    if _want_probe:
        _det_probe = ls.get_probe_predictions(model, _entry, _probe, layer=_layer)
    if _want_vectors:
        _det_vectors = ls.get_multilabel_token_scores(model, _entry, _vectors, layer=_layer)

    _det = _det_probe or _det_vectors
    if _det is None:
        if _probe is not None:
            _det = ls.get_probe_predictions(model, _entry, _probe, layer=_layer)
        elif _vectors is not None:
            _det = ls.get_multilabel_token_scores(model, _entry, _vectors, layer=_layer)
        else:
            mo.stop(True, mo.md("Need either a trained probe or vectors to tokenize the entry."))

    _, _msg_offsets = model.format_messages_with_offsets(_entry.messages)
    _role_names = {
        "system": "System prompt", "user": "User",
        "reasoning": "Reasoning", "assistant": "Response",
    }
    _section_markers: dict[int, str] = {}
    for _msg_idx, _msg in enumerate(_entry.messages):
        _char_offset = _msg_offsets.get(_msg_idx)
        if _char_offset is None:
            continue
        for _tok_i, (_cs, _ce) in enumerate(_det.token_char_spans):
            if _ce == 0:
                continue
            if _cs >= _char_offset:
                _section_markers[_tok_i] = _role_names.get(_msg["role"], _msg["role"].capitalize())
                break

    # ── Compute start token (skip system/user if requested) ────────────
    _start_tok = 0
    if cfg_skip_prefix.value:
        # Find the first token of the reasoning or assistant section
        _reasoning_keys = [k for k, v in _section_markers.items() if v in ("Reasoning", "Response")]
        if _reasoning_keys:
            _start_tok = min(_reasoning_keys)

    def _get_sentence_spans(tokens):
        spans = []
        start = 0
        for i, tok in enumerate(tokens):
            # Aggressive split: any newline or punctuation
            if "\n" in tok or (tok.strip() and tok.strip()[-1] in (".", "!", "?")):
                spans.append(ls.TokenSpan(token_start=start, token_end=i+1, labels=[]))
                start = i + 1
        if start < len(tokens):
            spans.append(ls.TokenSpan(token_start=start, token_end=len(tokens), labels=[]))
        return spans

    def _slice_det(det_obj, scores_arr):
        """Slice tokens/scores/spans to start from _start_tok."""
        _st = _start_tok
        _tokens = det_obj.tokens[_st:]
        _char_spans = det_obj.token_char_spans[_st:]
        _scores = scores_arr[_st:]

        if cfg_mode.value == "sentence":
            # Partition the visible text into sentences for averaging
            _spans = _get_sentence_spans(_tokens)
        else:
            # Use original annotation spans (offset to match slice)
            _spans = [
                ls.TokenSpan(token_start=max(0, ts.token_start - _st),
                             token_end=ts.token_end - _st, labels=ts.labels)
                for ts in det_obj.token_spans
                if ts.token_end > _st and ts.token_start - _st < len(_tokens)
            ] if det_obj.token_spans else []

        _markers = {k - _st: v for k, v in _section_markers.items() if k >= _st}
        return _tokens, _char_spans, _scores, _spans, _markers, det_obj.formatted_text

    def _dedup(det_obj):
        seen = set()
        labels = []
        col_indices = []
        for i, lbl in enumerate(det_obj.labels):
            if lbl not in seen:
                seen.add(lbl)
                labels.append(lbl)
                col_indices.append(i)
        scores = det_obj.scores[:, col_indices]
        return labels, scores

    _panels = []
    _legend_labels = None

    if _want_gt:
        _u_labels, _ = _dedup(_det)
        _legend_labels = _u_labels
        _lbl_to_idx = {l: i for i, l in enumerate(_u_labels)}
        _gt_scores = np.zeros((len(_det.tokens), len(_u_labels)), dtype=np.float32)
        for _ts in _det.token_spans:
            for _lbl in _ts.labels:
                if _lbl in _lbl_to_idx:
                    _gt_scores[_ts.token_start:min(_ts.token_end, len(_det.tokens)), _lbl_to_idx[_lbl]] = 1.0
        _toks, _cspans, _gts, _spans, _markers, _ftxt = _slice_det(_det, _gt_scores)
        _html_gt = _pv.render_probe_detection_html(
            tokens=_toks, token_char_spans=_cspans,
            scores=_gts, labels=_u_labels,
            formatted_text=_ftxt, token_spans=_spans,
            threshold=0.5, mode=cfg_mode.value,
            show_ground_truth=False, normalize_scores=False,
            section_markers=_markers, show_legend=False, show_header=False,
        )
        _panels.append(mo.vstack([mo.md("#### Ground truth"), mo.Html(_html_gt)]))

    if _want_probe and _det_probe is not None:
        _u_labels_p, _scores_p = _dedup(_det_probe)
        if _legend_labels is None:
            _legend_labels = _u_labels_p
        _toks, _cspans, _sp, _spans, _markers, _ftxt = _slice_det(_det_probe, _scores_p)
        _html_probe = _pv.render_probe_detection_html(
            tokens=_toks, token_char_spans=_cspans,
            scores=_sp, labels=_u_labels_p,
            formatted_text=_ftxt, token_spans=_spans,
            threshold=cfg_threshold.value, mode=cfg_mode.value,
            show_ground_truth=False, normalize_scores=False,
            section_markers=_markers, show_legend=False, show_header=False,
        )
        _panels.append(mo.vstack([mo.md("#### MLP probe"), mo.Html(_html_probe)]))

    if _want_vectors and _det_vectors is not None:
        _u_labels_v, _scores_v = _dedup(_det_vectors)
        if _legend_labels is None:
            _legend_labels = _u_labels_v
        _toks, _cspans, _sv, _spans, _markers, _ftxt = _slice_det(_det_vectors, _scores_v)
        _html_vec = _pv.render_probe_detection_html(
            tokens=_toks, token_char_spans=_cspans,
            scores=_sv, labels=_u_labels_v,
            formatted_text=_ftxt, token_spans=_spans,
            threshold=cfg_threshold.value, mode=cfg_mode.value,
            show_ground_truth=False, normalize_scores=True,
            section_markers=_markers, show_legend=False, show_header=False,
        )
        _panels.append(mo.vstack([mo.md("#### Mean vector"), mo.Html(_html_vec)]))

    _shared_legend = mo.Html(_pv.legend_html(_legend_labels, cfg_threshold.value))
    _ellipsis = "\u2026"
    _dot = "\u00b7"
    _title = _user_msg[:120].replace(chr(10), " ")
    if len(_user_msg) > 120:
        _title += _ellipsis

    mo.vstack([
        mo.md(f"### {_title}"),
        mo.md(
            f"**Model:** `{_model_name}` &nbsp;{_dot}&nbsp; "
            f"**Layer:** `{_layer}`"
        ),
        _shared_legend,
        mo.hstack(_panels, justify="start", gap=4),
    ])
    return


if __name__ == "__main__":
    app.run()
