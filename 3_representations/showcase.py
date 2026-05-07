"""little_steer showcase — interactive comparison harness.

Sections:
  §1  Configure          — model, dataset, labels, judge
  §2  Load               — model + dataset loading
  §3  Extract            — activations (both whole-sentence and last-token)
  §4  Build vectors      — mean_centering + PCA for all selected labels
  §5  Vector Comparison  — cosine similarity heatmap + PCA across behaviours
  §6  Scoring            — AUROC separation on test split + hard-to-game diagnostics
  §7  Entry browser      — per-token detection visualization
  §8  Steering playground — baseline vs steered; alpha sweep

Note on activation context (§3):
  TokenSelection("all") captures activations from a SINGLE forward pass on the
  FULL formatted conversation (system+user+reasoning+assistant joined by the
  chat template). Activations at each annotated sentence position therefore
  reflect the full prior context through the model's residual stream — the
  model has "read" everything before that sentence before we capture its
  hidden state. No sentence is tokenized in isolation.
"""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium", app_title="little_steer showcase")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os as _os
    import sys as _sys
    if _os.path.dirname(__file__) not in _sys.path:
        _sys.path.insert(0, _os.path.dirname(__file__))
    return


@app.cell
def _():
    import numpy as np
    import polars as pl
    import altair as alt

    return alt, np, pl


@app.cell
def _(mo):
    _css = """
    :root {
      --bg: #2d353b; --bg-dim: #232a2e; --bg-1: #343f44;
      --bg-2: #3d484d; --bg-3: #475258;
      --fg: #d3c6aa; --fg-dim: #9da9a0;
      --red: #e67e80; --orange: #e69875; --yellow: #dbbc7f;
      --green: #a7c080; --aqua: #83c092; --blue: #7fbbb3;
      --purple: #d699b6; --grey: #7a8478;
    }
    [data-theme="light"] {
      --bg: #fdf6e3; --bg-dim: #f4f0d9; --bg-1: #e9e4ca;
      --bg-2: #ddd8be; --bg-3: #cac9ad;
      --fg: #5c6a72; --fg-dim: #829181;
      --red: #f85552; --orange: #f57d26; --yellow: #dfa000;
      --green: #8da101; --aqua: #35a77c; --blue: #3a94c5;
      --purple: #df69ba; --grey: #a6b0a0;
    }
    body { font-family: 'Inter', system-ui, sans-serif;
           background-color: var(--bg-dim) !important; color: var(--fg) !important; }
    h1, h2, h3 { color: var(--green); }
    a { color: var(--blue); }
    #theme-toggle {
      position: fixed; top: 20px; right: 20px;
      background: var(--bg-1); border: 1px solid var(--bg-3);
      border-radius: 50%; width: 44px; height: 44px;
      cursor: pointer; display: flex; align-items: center; justify-content: center;
      font-size: 20px; z-index: 9999;
    }
    """
    _script = """
    (function(){
      const root=document.documentElement;
      const btn=document.getElementById("theme-toggle");
      const saved=localStorage.getItem("theme")||"dark";
      root.dataset.theme=saved;
    })();
    """
    mo.Html(f"""
    <style>{_css}</style>
    <script>{_script}</script>
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    # little_steer showcase

    **What this notebook does:**
    - Extract activations for annotated reasoning traces at every layer (using the full conversation context).
    - Build steering vectors with multiple methods (mean-centering, PCA) and token selections (whole-sentence, last-token).
    - Score and compare all (method × spec × layer × label) combinations in one Session-cached pass.
    - Run the cheap hard-to-game diagnostics: logit lens, keyword overlap, token-ablation, neutral-PCA.
    - Interactively steer generation with a chosen vector and visualize the effect.

    > For probe training and the full per-entry detection browser, see `explore.py`.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## § 1 — Configure
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
        start=10, stop=2000, step=10, value=60,
        label="Entries to sample (total pool before 80/20 split)",
        show_value=True,
    )
    cfg_layer_stride = mo.ui.slider(
        start=1, stop=6, step=1, value=2,
        label="Layer stride (1 = every layer, 2 = every other, …)",
        show_value=True,
    )
    mo.vstack([
        mo.hstack([cfg_model_id, cfg_data_path], justify="start", gap=2),
        mo.hstack([cfg_n_entries, cfg_layer_stride], justify="start", gap=2),
    ])
    return cfg_data_path, cfg_layer_stride, cfg_model_id, cfg_n_entries


@app.cell
def _():
    import json as _json
    import os as _os2
    _tax_path = _os2.path.normpath(
        _os2.path.join(_os2.path.dirname(__file__), "..", "2_labeling", "taxonomy.json")
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
def _(ALL_LABELS, mo):
    cfg_labels = mo.ui.multiselect(
        options=ALL_LABELS,
        value=ALL_LABELS,
        label="Behavior labels to extract and compare",
    )
    cfg_labels
    return (cfg_labels,)


@app.cell
def _(mo):
    SAFE_PROMPT_LABEL = "__safe_prompt_response__"
    mo.callout(
        mo.md(
            "Readout choices now live in **§ 3 Extract activations**. "
            "Vector methods, contrastive baselines, safe-response cleanup, "
            "and the small MLP method now live in **§ 4 Build steering vectors**."
        ),
        kind="neutral",
    )
    return (SAFE_PROMPT_LABEL,)


@app.cell
def _(mo):
    mo.md("""
    ## § 2 — Load model + dataset
    """)
    return


@app.cell
def _(mo):
    load_btn = mo.ui.run_button(label="Load model + dataset")
    load_btn
    return (load_btn,)


@app.cell
def _():
    import torch
    import little_steer as ls

    return ls, torch


@app.cell
def _(cfg_data_path, cfg_model_id, load_btn, ls, mo, torch):
    mo.stop(not load_btn.value, mo.md("_Click **Load model + dataset** to start._"))

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
        _all_entries = list(ls.iter_dataset(cfg_data_path.value))
        all_annotated = [e for e in _all_entries if e.annotations]

    def _is_safe_prompt_entry(entry):
        _meta = entry.metadata or {}
        _meta_safety = str(
            _meta.get("prompt_safety")
            or _meta.get("prompt_safety_label")
            or _meta.get("safety")
            or ""
        ).lower()
        if _meta_safety == "safe":
            return True
        if str(_meta.get("prompt_harmfulness") or "").lower() == "unharmful":
            return True
        _source_blob = " ".join(
            str(_meta.get(k) or "")
            for k in ("dataset", "dataset_source", "source", "prompt_source", "split")
        ).lower()
        if "lima" in _source_blob:
            return True
        for _run in entry.safety_runs:
            _result = _run.result or {}
            if str(_result.get("prompt_safety") or "").lower() == "safe":
                return True
            if str(_result.get("prompt_harmfulness") or "").lower() == "unharmful":
                return True
        return False

    safe_prompt_entries = [e for e in _all_entries if _is_safe_prompt_entry(e)]
    del _all_entries

    _judge_counts: dict[str, int] = {}
    for _e in all_annotated:
        for _lr in _e.label_runs:
            _judge_counts[_lr.judge_name] = _judge_counts.get(_lr.judge_name, 0) + 1
    available_judges = sorted(_judge_counts, key=lambda j: -_judge_counts[j])

    mo.md(
        f"✅ **{model}**  \n"
        f"Dataset: **{len(all_annotated)}** annotated entries  \n"
        f"Safe-prompt baseline candidates: **{len(safe_prompt_entries)}**  \n"
        f"Layers: **{model.num_layers}** · Hidden size: **{model.hidden_size}**  \n"
        f"Judges available: {', '.join(f'`{j}` ({_judge_counts[j]})' for j in available_judges)}"
    )
    return all_annotated, available_judges, model, safe_prompt_entries


@app.cell
def _(available_judges, mo):
    cfg_judge = mo.ui.dropdown(
        options=available_judges,
        value=available_judges[0] if available_judges else None,
        label="Judge (which labeler's annotations to use)",
    )
    cfg_judge
    return (cfg_judge,)


@app.cell
def _(all_annotated, mo):
    import json as _json2
    _avail_models = sorted({e.model for e in all_annotated if e.model})
    cfg_model_filter = mo.ui.multiselect(
        options=_avail_models,
        value=_avail_models,
        label="Models to include in sample pool",
    )
    cfg_label_mode = mo.ui.radio(
        options={"All labels (current default)": "all", "First label only (highest priority)": "first"},
        value="All labels (current default)",
        label="Label mode — which labels per span to use",
    )
    cfg_logprob_filter = mo.ui.checkbox(
        label="Filter by annotation confidence (logprob threshold)",
        value=False,
    )
    cfg_logprob_threshold = mo.ui.slider(
        start=0.05, stop=0.99, step=0.05, value=0.50,
        label="Min label confidence",
        show_value=True,
    )
    mo.vstack([
        mo.md("### § 2b — Sample filters"),
        mo.callout(
            mo.md(
                "**Model filter**: restrict which generation models contribute entries to the sample pool.\n\n"
                "**Label mode**: each span can have up to 3 labels sorted by importance. "
                "'First label only' keeps only the highest-priority label per span — this is "
                "stricter and produces cleaner vectors but fewer samples per label.\n\n"
                "**Logprob confidence**: only available for judges that report per-label log-probabilities "
                "(e.g. GPT-4o-mini). Confidence = exp(sum of token log-probabilities for that label name). "
                "Near 1.0 = model was very certain; near 0.1 = it was unsure. "
                "Recommended threshold: 0.5 (moderate) · 0.8 (strict). "
                "Labels with no logprob data are always kept. "
                "**Note**: changing any of these requires re-extracting activations (§ 3)."
            ),
            kind="info",
        ),
        mo.hstack([cfg_model_filter], justify="start"),
        mo.hstack([cfg_label_mode, cfg_logprob_filter, cfg_logprob_threshold], justify="start", gap=2),
    ])
    return (
        cfg_label_mode,
        cfg_logprob_filter,
        cfg_logprob_threshold,
        cfg_model_filter,
    )


@app.cell
def _(
    all_annotated,
    alt,
    cfg_judge,
    cfg_label_mode,
    cfg_labels,
    cfg_logprob_filter,
    cfg_logprob_threshold,
    cfg_model_filter,
    mo,
    pl,
):
    import math as _math2

    _judge_name_prev = cfg_judge.value
    _model_set_prev = set(cfg_model_filter.value) if cfg_model_filter.value else set()
    _use_first_prev = cfg_label_mode.value == "first"
    _use_logprob_prev = cfg_logprob_filter.value
    _lp_thresh_prev = cfg_logprob_threshold.value
    _selected_labels_prev = set(cfg_labels.value) if cfg_labels.value else None

    _label_counts: dict[str, int] = {}
    _matching_entries_count = 0

    for _e_prev in all_annotated:
        if _model_set_prev and _e_prev.model not in _model_set_prev:
            continue
        _lr_prev = next(
            (lr for lr in _e_prev.label_runs if lr.judge_name == _judge_name_prev),
            None,
        )
        if _lr_prev is None:
            continue
        
        _entry_had_match = False
        for _sa_prev, _sp_prev in zip(_lr_prev.sentence_annotations, _lr_prev.spans):
            _labels_here = list(_sp_prev.labels)
            if _use_first_prev and _labels_here:
                _labels_here = _labels_here[:1]
            if _use_logprob_prev:
                _lp_map_prev = _sa_prev.get("label_logprobs") or {}
                def _conf_prev(_lbl, _lpm=_lp_map_prev):
                    _lps = _lpm.get(_lbl)
                    return _math2.exp(sum(_lps)) if _lps else None
                _labels_here = [lbl for lbl in _labels_here if (_c := _conf_prev(lbl)) is None or _c >= _lp_thresh_prev]
        
            for _lbl_prev in _labels_here:
                if _selected_labels_prev is None or _lbl_prev in _selected_labels_prev:
                    _label_counts[_lbl_prev] = _label_counts.get(_lbl_prev, 0) + 1
                    _entry_had_match = True
    
        if _entry_had_match:
            _matching_entries_count += 1

    def _get_output():
        if not _label_counts:
            return mo.callout(mo.md("No spans match the current filters."), kind="danger")
    
        _counts_df = pl.DataFrame([{"label": k, "spans": v} for k, v in sorted(_label_counts.items(), key=lambda x: -x[1])])
        _preview_chart = (
            alt.Chart(_counts_df.to_pandas())
            .mark_bar()
            .encode(
                x=alt.X("spans:Q", title="Spans surviving filter"),
                y=alt.Y("label:N", title=None, sort="-x"),
                color=alt.Color("spans:Q", scale=alt.Scale(scheme="greens"), legend=None),
                tooltip=["label", "spans"],
            )
            .properties(
                width=400,
                height=max(150, 18 * len(_label_counts)),
                title=f"Spans per label ({_matching_entries_count} unique entries)",
            )
        )
        return mo.vstack([
            mo.md(
                f"### § 2b — Live Filter Preview\n"
                f"**{_matching_entries_count}** unique entries match filters · "
                f"**{sum(_label_counts.values())}** total label-spans across **{len(_label_counts)}** labels  \n"
                f"Judge: **{_judge_name_prev}**"
            ),
            _preview_chart,
        ])

    _get_output()
    return


@app.cell
def _(
    all_annotated,
    cfg_judge,
    cfg_label_mode,
    cfg_logprob_filter,
    cfg_logprob_threshold,
    cfg_model_filter,
    cfg_n_entries,
    mo,
):
    import random as _random
    import math as _math3

    _judge_name = cfg_judge.value
    mo.stop(_judge_name is None, mo.md("⚠️ No judge selected."))

    _model_set = set(cfg_model_filter.value) if cfg_model_filter.value else set()
    _filtered = [
        e for e in all_annotated
        if any(lr.judge_name == _judge_name for lr in e.label_runs)
        and (not _model_set or e.model in _model_set)
    ]
    mo.stop(not _filtered, mo.md(f"⚠️ No entries found with judge `{_judge_name}` for the selected models."))

    _random.seed(42)
    _sampled_raw = _random.sample(_filtered, min(cfg_n_entries.value, len(_filtered)))

    _use_first = cfg_label_mode.value == "first"
    _use_logprob = cfg_logprob_filter.value
    _lp_thresh = cfg_logprob_threshold.value

    entries = []
    for _e_raw in _sampled_raw:
        _e = _e_raw.model_copy(deep=True)
        _active_lr = next(lr for lr in _e.label_runs if lr.judge_name == _judge_name)
        _e.set_active_label_run(_active_lr)

        for _sa, _sp in zip(_active_lr.sentence_annotations, _active_lr.spans):
            if _use_first and _sp.labels:
                _sp.labels = _sp.labels[:1]
            if _use_logprob:
                _lp_map = _sa.get("label_logprobs") or {}
                def _conf(_lbl, _lpm=_lp_map):
                    _lps = _lpm.get(_lbl)
                    return _math3.exp(sum(_lps)) if _lps else None
                _sp.labels = [lbl for lbl in _sp.labels if (_c := _conf(lbl)) is None or _c >= _lp_thresh]

        _e.annotations = [sp for sp in _active_lr.spans if sp.labels]
        if _e.annotations:
            entries.append(_e)

    if not entries:
        mo.callout(mo.md("⚠️ No entries remain after applying filters. Loosen the logprob threshold or label mode."), kind="warn")
    else:
        _n_models = len({e.model for e in entries})
        mo.callout(
            mo.md(
                f"Judge **{_judge_name}** · **{len(entries)}** entries sampled "
                f"(from {len(_filtered)} available) · **{_n_models}** models · "
                f"label mode: **{cfg_label_mode.value}**"
                + (f" · logprob ≥ {_lp_thresh:.2f}" if _use_logprob else "")
            ),
            kind="success",
        )
    return (entries,)


@app.cell
def _(mo):
    mo.md("""
    ## § 3 — Extract activations

    Extracts selected behavior labels at every layer with the configured stride.
    The readout controls decide *where inside each labeled sentence/span* to pool
    activations. All selected readouts share one forward pass per entry.

    Activations are computed with the **full conversation context** — the model sees the
    system prompt, user message, and all prior reasoning before each annotated position.
    Cached across reruns: increasing entries only extracts the new ones.
    """)
    return


@app.cell
def _():
    _cache: dict = {"result": None, "entry_ids": set()}
    extraction_cache = _cache
    return (extraction_cache,)


@app.cell
def _(SAFE_PROMPT_LABEL, mo, safe_prompt_entries):
    _spec_options = [
        "whole_sentence",
        "last_token",
        "first_token",
        "first_3_tokens",
        "last_3_tokens",
        "bleed_5_all",
        "skip_first_2_all",
        "pre_context_last",
    ]
    cfg_extraction_specs = mo.ui.multiselect(
        options=_spec_options,
        value=["first_3_tokens", "last_3_tokens", "skip_first_2_all"],
        label="Readout positions inside each labeled span",
    )
    cfg_include_safe_baseline = mo.ui.checkbox(
        label=f"Also extract safe-prompt responses as `{SAFE_PROMPT_LABEL}`",
        value=False,
    )
    cfg_safe_baseline_n = mo.ui.slider(
        start=10,
        stop=max(10, min(500, len(safe_prompt_entries))),
        step=10,
        value=min(100, max(10, len(safe_prompt_entries))) if safe_prompt_entries else 10,
        label="Safe-prompt baseline examples",
        show_value=True,
    )
    _safe_ds_names = sorted({
        str((e.metadata or {}).get("dataset_name") or (e.metadata or {}).get("dataset_source") or "unknown")
        for e in safe_prompt_entries
    })
    cfg_safe_dataset_filter = mo.ui.multiselect(
        options=_safe_ds_names,
        value=_safe_ds_names,
        label="Safe-prompt datasets to use as baseline",
    )
    mo.vstack([
        cfg_extraction_specs,
        mo.hstack([cfg_include_safe_baseline, cfg_safe_baseline_n], justify="start", gap=2),
        mo.callout(
            mo.md(
                "**What are safe-prompt entries?** Entries whose **prompt** was flagged as non-harmful, "
                "identified via metadata fields (e.g. `prompt_safety=safe`, dataset source = LIMA) "
                "or via `safety_runs` — a separate safety-classifier result stored per entry that checks "
                "prompt harmfulness. This is entirely separate from the behavior judge (`label_runs`): "
                "no behavior labels are used here. These entries are used as a 'neutral activation' baseline — "
                "activations from benign contexts with no specific safety-relevant behavior. "
                "Use the dataset filter above to restrict which sources contribute."
            ),
            kind="neutral",
        ),
        cfg_safe_dataset_filter,
    ])
    return (
        cfg_extraction_specs,
        cfg_include_safe_baseline,
        cfg_safe_baseline_n,
        cfg_safe_dataset_filter,
    )


@app.cell
def _(mo):
    extract_btn = mo.ui.run_button(label="Extract activations (§ 3)")
    extract_btn
    return (extract_btn,)


@app.cell
def _(
    SAFE_PROMPT_LABEL,
    cfg_extraction_specs,
    cfg_include_safe_baseline,
    cfg_labels,
    cfg_layer_stride,
    cfg_safe_baseline_n,
    cfg_safe_dataset_filter,
    entries,
    extract_btn,
    extraction_cache,
    ls,
    mo,
    model,
    safe_prompt_entries,
):
    import random as _rnd
    mo.stop(not extract_btn.value)
    mo.stop(not entries, mo.md("⚠️ No entries available — check filter settings in § 2b."))

    _rnd.seed(0)
    _shuffled = list(entries)
    _rnd.shuffle(_shuffled)
    _n_train = int(0.8 * len(_shuffled))
    train_entries = _shuffled[:_n_train]
    test_entries = _shuffled[_n_train:]

    _stride = cfg_layer_stride.value
    _layers = list(range(0, model.num_layers, _stride))
    _label_filter = set(cfg_labels.value) if cfg_labels.value else set()
    if cfg_include_safe_baseline.value:
        _label_filter.add(SAFE_PROMPT_LABEL)
    _spec_factories = {
        "whole_sentence": lambda: ls.TokenSelection("all"),
        "last_token": lambda: ls.TokenSelection("last"),
        "first_token": lambda: ls.TokenSelection("first"),
        "first_3_tokens": lambda: ls.TokenSelection("first_n", n=3),
        "last_3_tokens": lambda: ls.TokenSelection("last_n", n=3),
        "bleed_5_all": lambda: ls.TokenSelection("all", bleed_before=5, bleed_after=5),
        "skip_first_2_all": lambda: ls.TokenSelection("all", bleed_before=-2),
        "pre_context_last": lambda: ls.TokenSelection("last", bleed_before=5),
    }
    _selected_specs = cfg_extraction_specs.value or ["first_3_tokens", "last_3_tokens", "skip_first_2_all"]
    _specs = {
        name: ls.ExtractionSpec(_spec_factories[name](), layers=_layers)
        for name in _selected_specs
        if name in _spec_factories
    }
    mo.stop(not _specs, mo.md("⚠️ Select at least one readout in § 3."))

    _safe_train_entries = []
    if cfg_include_safe_baseline.value and safe_prompt_entries:
        _safe_ds_set = set(cfg_safe_dataset_filter.value) if cfg_safe_dataset_filter.value else None
        _safe_candidates = [
            e for e in safe_prompt_entries
            if _safe_ds_set is None or str((e.metadata or {}).get("dataset_name") or (e.metadata or {}).get("dataset_source") or "unknown") in _safe_ds_set
        ]
        _rnd.shuffle(_safe_candidates)
        for _entry in _safe_candidates[: cfg_safe_baseline_n.value]:
            _msg_idx = next(
                (
                    _i
                    for _i, _msg in reversed(list(enumerate(_entry.messages)))
                    if _msg.get("role") == "assistant" and _msg.get("content")
                ),
                None,
            )
            if _msg_idx is None:
                _msg_idx = next(
                    (
                        _i
                        for _i, _msg in reversed(list(enumerate(_entry.messages)))
                        if _msg.get("role") == "reasoning" and _msg.get("content")
                    ),
                    None,
                )
            if _msg_idx is None:
                continue
            _content = _entry.messages[_msg_idx]["content"]
            _copy = _entry.model_copy(deep=True)
            _copy.id = f"{_entry.id}::safe_prompt_baseline"
            _copy.annotations = [
                ls.AnnotatedSpan(
                    text=_content,
                    message_idx=_msg_idx,
                    char_start=0,
                    char_end=len(_content),
                    labels=[SAFE_PROMPT_LABEL],
                    score=0.0,
                    meta={"synthetic_baseline": "safe_prompt_response"},
                )
            ]
            _safe_train_entries.append(_copy)

    _plan = ls.ExtractionPlan(
        "showcase",
        specs=_specs,
        label_filter=_label_filter,
    )

    _plan_key = (
        tuple(sorted(_specs)),
        tuple(_layers),
        tuple(sorted(_label_filter)) if _label_filter else None,
        cfg_safe_baseline_n.value if cfg_include_safe_baseline.value else 0,
    )
    if extraction_cache.get("plan_key") != _plan_key:
        extraction_cache["result"] = None
        extraction_cache["entry_ids"] = set()
        extraction_cache["plan_key"] = _plan_key

    _prev_ids = extraction_cache["entry_ids"]
    _train_plus_safe = train_entries + _safe_train_entries
    _new_entries = [e for e in _train_plus_safe if e.id not in _prev_ids]

    if _new_entries:
        _extractor = ls.ActivationExtractor(model, max_seq_len=3072)
        _new_result = _extractor.extract(
            _new_entries, _plan,
            show_progress=False,
            progress_fn=lambda ds: mo.status.progress_bar(
                ds,
                title=f"Extracting {len(_new_entries)} train entries ({len(_layers)} layers)…",
                completion_title="Extraction done!",
            ),
        )
        _prev = extraction_cache["result"]
        if _prev is not None:
            _new_result.merge_from(_prev)
        extraction_cache["result"] = _new_result
        extraction_cache["entry_ids"] = _prev_ids | {e.id for e in _new_entries}

    extraction_result = extraction_cache["result"]
    mo.stop(
        extraction_result is None,
        mo.md("⚠️ No extraction data yet."),
    )

    _avail_layers = extraction_result.layers()
    _label_counts = {}
    _count_spec = extraction_result.specs()[0]
    for _lbl in sorted(extraction_result.labels()):
        _acts = extraction_result.get(_count_spec, _lbl, _avail_layers[0])
        _label_counts[_lbl] = len(_acts)

    import altair as _alt
    import pandas as _pd
    _counts_df = _pd.DataFrame([
        {"label": k, "train_samples": v}
        for k, v in _label_counts.items()
    ])
    _chart = (
        _alt.Chart(_counts_df)
        .mark_bar()
        .encode(
            x=_alt.X("train_samples:Q", title="Samples"),
            y=_alt.Y("label:N", title=None, sort="-x"),
            tooltip=["label", "train_samples"],
            color=_alt.Color("train_samples:Q", scale=_alt.Scale(scheme="greens"), legend=None)
        )
        .properties(width=400, height=max(200, 20 * len(_label_counts)))
    )

    mo.vstack([
        mo.md(
            f"✅ **{len(extraction_cache['entry_ids'])}** train entries extracted · "
            f"**{len(test_entries)}** held out for evaluation · "
            f"**{len(_avail_layers)}** layers ({_avail_layers[0]}–{_avail_layers[-1]}) · "
            f"Specs: {', '.join(f'`{s}`' for s in extraction_result.specs())}"
        ),
        _chart,
    ])
    return extraction_result, test_entries


@app.cell
def _(mo):
    mo.md("""
    ## § 4 — Build steering vectors

    Builds every selected behavior × readout × method × layer combination from
    the already-extracted activations. `mean_difference` and `linear_probe` are
    contrastive methods, so they use the selected baseline. The small MLP option
    trains one multilabel MLP per selected readout/layer and converts each
    behavior output into a gradient direction that can be compared and steered
    like the other vectors.
    """)
    return


@app.cell
def _():
    _cache: dict = {"vectors": [], "recipes": []}
    recipes_cache = _cache
    return (recipes_cache,)


@app.cell
def _(SAFE_PROMPT_LABEL, extraction_result, mo):
    _labels = [l for l in extraction_result.labels() if l != SAFE_PROMPT_LABEL]
    vector_method_sel = mo.ui.multiselect(
        options=[
            "mean_centering",
            "pca",
            "mean_difference",
            "linear_probe",
            "small_mlp",
        ],
        value=["mean_centering", "small_mlp"],
        label="Vector methods",
    )
    _baseline_options = {
        "Behavior label": "label",
        "Safe-prompt responses": "safe_prompt",
    }
    vector_baseline_source = mo.ui.radio(
        options=_baseline_options,
        value="Behavior label",
        label="Contrastive baseline",
    )
    _baseline_label_options = _labels or extraction_result.labels()
    vector_baseline_label = mo.ui.dropdown(
        options=_baseline_label_options,
        value="neutralFiller" if "neutralFiller" in _baseline_label_options else (
            _baseline_label_options[0] if _baseline_label_options else None
        ),
        label="Baseline behavior",
    )
    vector_safe_cleanup = mo.ui.checkbox(
        label="Add safe-response mean-cleaned copies (skips PCA)",
        value=False,
    )
    vector_mlp_epochs = mo.ui.slider(
        start=5, stop=80, step=5, value=25,
        label="Small MLP epochs", show_value=True,
    )
    vector_pca_contrastive = mo.ui.checkbox(
        label="Contrastive PCA: subtract centroid of all other labels before fitting PCA (makes PC1 more target-specific)",
        value=False,
    )
    vector_mlp_label_mode = mo.ui.radio(
        options={"All labels (current default)": "all", "First label only (highest priority)": "first"},
        value="All labels (current default)",
        label="MLP: which labels per span to train on",
    )
    return (
        vector_baseline_label,
        vector_baseline_source,
        vector_method_sel,
        vector_mlp_epochs,
        vector_mlp_label_mode,
        vector_pca_contrastive,
        vector_safe_cleanup,
    )


@app.cell
def _(
    SAFE_PROMPT_LABEL,
    mo,
    vector_baseline_label,
    vector_baseline_source,
    vector_method_sel,
    vector_mlp_epochs,
    vector_mlp_label_mode,
    vector_pca_contrastive,
    vector_safe_cleanup,
):
    def _suggest_recipe_name():
        _methods = vector_method_sel.value or []
        _abbr = {
            "mean_centering": "mc",
            "pca": "pca",
            "mean_difference": "md",
            "linear_probe": "lp",
            "small_mlp": "mlp",
        }
        _m_part = "+".join(_abbr.get(m, m) for m in _methods) or "none"
        _bsrc = vector_baseline_source.value
        _baseline = (
            SAFE_PROMPT_LABEL
            if _bsrc in ("safe_prompt", "Safe-prompt responses")
            else vector_baseline_label.value
        )
        _needs_b = bool({"mean_difference", "linear_probe"} & set(_methods))
        _b_part = ""
        if _needs_b and _baseline:
            _short = "safe" if _baseline == SAFE_PROMPT_LABEL else _baseline[:12]
            _b_part = f"_vs_{_short}"
        _c_part = "_clean" if vector_safe_cleanup.value else ""
        _e_part = f"_e{int(vector_mlp_epochs.value)}" if "small_mlp" in _methods else ""
        return f"{_m_part}{_b_part}{_c_part}{_e_part}"

    recipe_suggested_name = _suggest_recipe_name()
    # `value` is reactive: marimo re-instantiates the text input whenever an
    # upstream setting changes, so the field auto-fills with the new
    # suggestion — but the user can still edit it freely before clicking Add.
    recipe_name = mo.ui.text(
        value=recipe_suggested_name,
        label="Recipe name (auto-suggested · editable)",
        full_width=True,
    )
    add_recipe_btn = mo.ui.run_button(label="➕ Add recipe")
    clear_recipes_btn = mo.ui.run_button(label="🗑 Clear all")
    build_btn = mo.ui.run_button(label="🔨 Build")

    _intro = mo.md(
        "**A *recipe* = one full configuration of build settings.** "
        "Set the knobs below, give it a name (auto-suggested from the "
        "settings — feel free to override), then **Add recipe** to stash it. "
        "Repeat with different settings to stack up multiple recipes; **Build** "
        "constructs vectors for every stashed recipe (or just the current "
        "settings if none stashed yet). Activations from §3 are shared across "
        "recipes — only the cheap algebraic / probe step re-runs per recipe."
    )

    _settings_card = mo.vstack([
        mo.md("**1 · Build settings**"),
        mo.hstack([vector_method_sel, vector_mlp_epochs], justify="start", gap=2),
        mo.hstack([vector_baseline_source, vector_baseline_label], justify="start", gap=2),
        vector_safe_cleanup,
        mo.hstack([vector_pca_contrastive, vector_mlp_label_mode], justify="start", gap=2),
    ], gap=0.5)

    _name_card = mo.vstack([
        mo.md("**2 · Name & save this recipe**"),
        recipe_name,
        mo.md(f"_Suggested: `{recipe_suggested_name}`_"),
        mo.hstack([add_recipe_btn, clear_recipes_btn], justify="start", gap=1),
    ], gap=0.5)

    _build_card = mo.vstack([
        mo.md("**3 · Build all stashed recipes**"),
        build_btn,
    ], gap=0.5)

    mo.vstack([
        _intro,
        mo.callout(_settings_card, kind="neutral"),
        mo.callout(_name_card, kind="info"),
        mo.callout(_build_card, kind="success"),
    ], gap=0.5)
    return add_recipe_btn, build_btn, clear_recipes_btn, recipe_name


@app.cell
def _(
    SAFE_PROMPT_LABEL,
    add_recipe_btn,
    clear_recipes_btn,
    mo,
    pl,
    recipe_name,
    recipes_cache,
    vector_baseline_label,
    vector_baseline_source,
    vector_method_sel,
    vector_mlp_epochs,
    vector_mlp_label_mode,
    vector_pca_contrastive,
    vector_safe_cleanup,
):
    if clear_recipes_btn.value:
        recipes_cache["recipes"] = []
        recipes_cache["vectors"] = []

    if add_recipe_btn.value:
        _baseline_src = vector_baseline_source.value
        _baseline = (
            SAFE_PROMPT_LABEL
            if _baseline_src in ("safe_prompt", "Safe-prompt responses")
            else vector_baseline_label.value
        )
        _name = recipe_name.value or f"recipe_{len(recipes_cache['recipes']) + 1}"
        # avoid duplicate names
        _existing = {r["name"] for r in recipes_cache["recipes"]}
        _i = 2
        _candidate = _name
        while _candidate in _existing:
            _candidate = f"{_name}#{_i}"
            _i += 1
        recipes_cache["recipes"].append({
            "name": _candidate,
            "methods": list(vector_method_sel.value or []),
            "baseline_source": _baseline_src,
            "baseline_label": _baseline,
            "safe_cleanup": bool(vector_safe_cleanup.value),
            "mlp_epochs": int(vector_mlp_epochs.value),
            "pca_contrastive": bool(vector_pca_contrastive.value),
            "mlp_label_mode": vector_mlp_label_mode.value,
        })

    if recipes_cache["recipes"]:
        _summary = pl.DataFrame([
            {
                "recipe": r["name"],
                "methods": ", ".join(r["methods"]),
                "baseline": r["baseline_label"] if r["baseline_label"] else "—",
                "safe_cleanup": r["safe_cleanup"],
                "mlp_epochs": r["mlp_epochs"],
            }
            for r in recipes_cache["recipes"]
        ])
        _view = mo.ui.table(_summary.to_dicts(), selection=None)
    else:
        _view = mo.md("_No recipes added yet — add one or just click **Build all** to use the current settings._")
    _view
    return


@app.cell
def _(
    SAFE_PROMPT_LABEL,
    build_btn,
    extraction_result,
    ls,
    mo,
    recipes_cache,
    torch,
    vector_baseline_label,
    vector_baseline_source,
    vector_method_sel,
    vector_mlp_epochs,
    vector_mlp_label_mode,
    vector_pca_contrastive,
    vector_safe_cleanup,
):
    mo.stop(not build_btn.value)

    _target_labels = [
        l for l in extraction_result.labels()
        if l != SAFE_PROMPT_LABEL
    ]
    _selected_specs = set(extraction_result.specs())

    if recipes_cache["recipes"]:
        _recipes_to_build = list(recipes_cache["recipes"])
    else:
        _recipes_to_build = [{
            "name": "current",
            "methods": list(vector_method_sel.value or []),
            "baseline_source": vector_baseline_source.value,
            "baseline_label": (
                SAFE_PROMPT_LABEL
                if vector_baseline_source.value in ("safe_prompt", "Safe-prompt responses")
                else vector_baseline_label.value
            ),
            "safe_cleanup": bool(vector_safe_cleanup.value),
            "mlp_epochs": int(vector_mlp_epochs.value),
            "pca_contrastive": bool(vector_pca_contrastive.value),
            "mlp_label_mode": vector_mlp_label_mode.value,
        }]

    def _stack_activations(_acts):
        _rows = []
        for _act in _acts:
            _rows.append(_act.float().mean(dim=0) if _act.dim() > 1 else _act.float())
        return torch.stack(_rows) if _rows else torch.zeros(0, 1)

    def _mlp_gradient_directions(probe, X, Y, n_labels):
        """One forward pass + n_labels backward passes (retain_graph), all on device."""
        probe.eval()
        X_req = X.clone().detach().requires_grad_(True)
        logits = probe(X_req)
        directions = []
        for label_idx in range(n_labels):
            mask = Y[:, label_idx] > 0
            if mask.sum() == 0:
                directions.append(None)
                continue
            grad = torch.autograd.grad(
                logits[:, label_idx][mask].sum(),
                X_req,
                retain_graph=(label_idx < n_labels - 1),
            )[0]
            g = grad[mask].mean(0).float()
            directions.append(g / (g.norm() + 1e-8))
        return directions

    import gc as _gc

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    mlp_loss_histories = []
    _all_built = []

    for _recipe in _recipes_to_build:
        _r_name = _recipe["name"]
        _methods = _recipe["methods"] or ["mean_centering", "small_mlp"]
        _r_needs_baseline = {"mean_difference", "linear_probe"} & set(_methods)
        _r_baseline = _recipe["baseline_label"] if _r_needs_baseline else None
        if _r_needs_baseline:
            mo.stop(
                _r_baseline not in extraction_result.labels(),
                mo.md(
                    f"⚠️ Recipe `{_r_name}`: baseline `{_r_baseline}` was not extracted. "
                    "Enable it in § 3 and extract again."
                ),
            )

        _r_built = []
        _standard_methods = [m for m in _methods if m != "small_mlp"]
        if _standard_methods:
            _builder = ls.SteeringVectorBuilder()
            for _label in mo.status.progress_bar(
                _target_labels,
                title=f"[{_r_name}] Building algebraic/probe vectors ({len(_target_labels)} labels)…",
                completion_title=f"[{_r_name}] Algebraic vectors built!",
            ):
                _vset = _builder.build(
                    extraction_result,
                    target_label=_label,
                    methods=_standard_methods,
                    baseline_label=_r_baseline,
                    pca_contrastive=_recipe.get("pca_contrastive", False),
                )
                _r_built.extend([v for v in _vset if v.extraction_spec in _selected_specs])
                _builder.clear_stacked_cache()

        if "small_mlp" in _methods:
            # Build and train one (spec, layer) at a time — never hold all training
            # tensors in memory simultaneously, which caused OOM on large layer counts.
            _trainer = ls.MLPProbeTrainer()
            _jobs = [
                (_spec, _layer)
                for _spec in extraction_result.specs()
                if _spec in _selected_specs
                for _layer in extraction_result.layers()
            ]
            for _spec, _layer in mo.status.progress_bar(
                _jobs,
                title=f"[{_r_name}] Training small MLP probes ({len(_jobs)} jobs)…",
                completion_title=f"[{_r_name}] Small MLP vectors built!",
            ):
                _mlp_max_labels = 1 if _recipe.get("mlp_label_mode", "all") == "first" else None
                _X_cpu, _Y_cpu = _trainer._build_training_data(
                    extraction_result, _spec, _layer, _target_labels, _mlp_max_labels
                )
                if len(_X_cpu) == 0:
                    continue
                _X = _X_cpu.to(_device)
                _Y = _Y_cpu.to(_device)
                del _X_cpu, _Y_cpu  # release CPU copy before training

                _probe, _history = _trainer.train_from_tensors(
                    _X, _Y,
                    labels=_target_labels,
                    method="mlp",
                    epochs=_recipe["mlp_epochs"],
                    batch_size=256,
                    lr=2e-3,
                    hidden_dim=128,
                    device=_device,
                    show_progress=False,
                    return_history=True,
                    return_on_device=True,
                )
                mlp_loss_histories.append({
                    "recipe": _r_name,
                    "spec": _spec,
                    "layer": _layer,
                    "history": _history,
                })

                _directions = _mlp_gradient_directions(_probe, _X, _Y, len(_target_labels))
                _probe.cpu()
                del _X, _Y
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                _gc.collect()

                _last = _history[-1] if _history else {}
                for _label_idx, _label in enumerate(_target_labels):
                    _direction = _directions[_label_idx]
                    if _direction is None:
                        continue
                    _r_built.append(
                        ls.SteeringVector(
                            vector=_direction.cpu(),
                            layer=_layer,
                            label=_label,
                            method="small_mlp",
                            extraction_spec=_spec,
                            metadata={
                                "probe_type": type(_probe).__name__,
                                "mlp_epochs": _recipe["mlp_epochs"],
                                "final_loss": _last.get("loss"),
                                "final_exact_match_acc": _last.get("acc"),
                            },
                        )
                    )

        if _recipe["safe_cleanup"]:
            mo.stop(
                SAFE_PROMPT_LABEL not in extraction_result.labels(),
                mo.md(
                    f"⚠️ Recipe `{_r_name}` requested safe-cleanup but the safe-prompt "
                    "baseline was not extracted in § 3."
                ),
            )
            _cleaned = []
            for _vec in _r_built:
                if _vec.method == "pca":
                    continue
                _safe_acts = extraction_result.get(
                    _vec.extraction_spec, SAFE_PROMPT_LABEL, _vec.layer,
                )
                if not _safe_acts:
                    continue
                _safe_mean = _stack_activations(_safe_acts).mean(dim=0).float()
                _cleaned_vec = _vec.vector.float() - _safe_mean
                _cleaned.append(
                    ls.SteeringVector(
                        vector=_cleaned_vec,
                        layer=_vec.layer,
                        label=_vec.label,
                        method=f"{_vec.method}+safe_mean_cleanup",
                        extraction_spec=_vec.extraction_spec,
                        metadata={
                            **_vec.metadata,
                            "cleanup": "subtract_safe_prompt_response_mean",
                            "n_safe_baseline_samples": len(_safe_acts),
                        },
                    )
                )
            _r_built.extend(_cleaned)

        for _vec in _r_built:
            _meta = dict(_vec.metadata or {})
            _meta["recipe"] = _r_name
            _vec.metadata = _meta
            if len(_recipes_to_build) > 1:
                _vec.method = f"{_vec.method}@{_r_name}"

        _all_built.extend(_r_built)

    all_vecs = ls.SteeringVectorSet(_all_built)
    mo.stop(len(all_vecs) == 0, mo.md("⚠️ No vectors were built."))
    recipes_cache["vectors"] = _all_built

    _r_names_str = ", ".join("`" + r["name"] + "`" for r in _recipes_to_build)
    mo.md(
        f"✅ Built **{len(all_vecs)}** vectors across **{len(_recipes_to_build)}** recipe(s): "
        f"{_r_names_str}  \n"
        f"Labels: {all_vecs.labels()}  \n"
        f"Methods (incl. recipe tag): {all_vecs.methods()}  \n"
        f"Specs: {all_vecs.specs()}  \n"
        f"Layers: {all_vecs.layers()[:5]}{'…' if len(all_vecs.layers()) > 5 else ''}"
    )
    return all_vecs, mlp_loss_histories


@app.cell
def _(alt, mlp_loss_histories, mo, pl):
    mo.stop(
        not mlp_loss_histories,
        mo.md("_No MLP training history (enable `small_mlp` in § 4 and rebuild)._"),
    )

    import pandas as _pd_mlp
    _rows_mlp = []
    for _run in mlp_loss_histories:
        _spec_name = _run["spec"]
        _layer_id = _run["layer"]
        _r_name = _run.get("recipe", "current")
        for _h in _run["history"]:
            _rows_mlp.append({
                "epoch": _h["epoch"],
                "loss": _h.get("loss"),
                "acc": _h.get("acc"),
                "run": f"{_r_name} · {_spec_name} · L{_layer_id}",
                "spec": _spec_name,
                "layer": _layer_id,
                "recipe": _r_name,
            })

    _hist_df = pl.DataFrame(_rows_mlp).filter(pl.col("loss").is_not_null())
    _final_losses = (
        _hist_df.group_by("run")
        .agg(pl.col("loss").last().alias("final_loss"))
        .sort("final_loss")
    )
    _median_final = float(_final_losses["final_loss"].median())
    _min_final = float(_final_losses["final_loss"].min())
    _max_final = float(_final_losses["final_loss"].max())

    _loss_chart = (
        alt.Chart(_hist_df.to_pandas())
        .mark_line(opacity=0.5, strokeWidth=1.5)
        .encode(
            x=alt.X("epoch:Q", title="Epoch"),
            y=alt.Y("loss:Q", title="BCE Loss"),
            color=alt.Color("spec:N", title="Readout spec"),
            detail="run:N",
            tooltip=["run", "epoch", "loss", "acc"],
        )
        .properties(
            width=600, height=300,
            title=f"MLP training loss across {len(mlp_loss_histories)} runs "
                  f"(final: min={_min_final:.3f}, median={_median_final:.3f}, max={_max_final:.3f})",
        )
    )

    _callout_kind = "success" if _median_final < 0.35 else ("warn" if _median_final < 0.55 else "danger")
    _callout_msg = (
        "Loss converged well — median final BCE < 0.35." if _median_final < 0.35
        else "Loss is moderate — consider more epochs or fewer labels." if _median_final < 0.55
        else "Loss is still high — probes may be underfitting. Try more epochs."
    )

    mo.vstack([
        mo.md("#### MLP Training Loss"),
        mo.md(
            "Each line is one (readout spec × layer) training run. "
            "If curves are still descending at the last epoch, increase epochs. "
            "If they plateau early, current epochs are sufficient."
        ),
        mo.callout(mo.md(_callout_msg), kind=_callout_kind),
        _loss_chart,
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## § 5 — Vector Comparison

    Geometric analysis of the built vectors at a chosen layer.
    - **Cosine similarity heatmap** — how much do different behaviour vectors overlap
      within each method? High similarity (> 0.7) means those two directions are nearly
      the same and may not be independently useful.
    - **PCA variance** — how spread out are the behaviour directions? If PC1 explains
      most of the variance, many behaviours share a single dominant axis (e.g. a generic
      "safety-relevant" direction). If variance is spread across many PCs, the behaviours
      are more geometrically distinct.
    """)
    return


@app.cell
def _(all_vecs, mo):
    _layers = all_vecs.layers()
    _mid_idx = len(_layers) // 2
    _mid_layer = _layers[_mid_idx] if _layers else 0
    _step = (_layers[1] - _layers[0]) if len(_layers) > 1 else 1
    sim_layer_sel = mo.ui.slider(
        start=_layers[0] if _layers else 0,
        stop=_layers[-1] if _layers else 0,
        step=_step,
        value=_mid_layer,
        label="Vector comparison layer",
        show_value=True,
    )
    _specs = all_vecs.specs()
    sim_spec_sel = mo.ui.dropdown(
        options=_specs,
        value=_specs[0] if _specs else None,
        label="Vector comparison readout",
    )
    mo.md("")  # controls displayed inline in the heatmap cell below
    return sim_layer_sel, sim_spec_sel


@app.cell
def _(all_vecs, alt, mo, np, pl, sim_layer_sel, sim_spec_sel, torch):
    mo._runtime.context.get_context().marimo_config["runtime"]["output_max_bytes"] = 20000000000

    mo.stop(
        sim_spec_sel.value is None,
        mo.md("_Build vectors first to compare their geometry._"),
    )
    _vecs = all_vecs.filter(layer=sim_layer_sel.value, spec=sim_spec_sel.value)
    mo.stop(len(_vecs) < 2, mo.md("⚠️ Need at least two vectors at this layer/readout."))

    _names = [v.label for v in _vecs]
    _methods = [v.method for v in _vecs]
    _mat = torch.stack([
        v.vector.float() / (v.vector.float().norm() + 1e-8)
        for v in _vecs
    ])
    _sim = (_mat @ _mat.T).cpu().numpy()

    _rows = []
    for i in range(len(_names)):
        for j in range(len(_names)):
            if _methods[i] == _methods[j]:
                _rows.append({
                    "x": _names[i],
                    "y": _names[j],
                    "method": _methods[i],
                    "cosine": float(_sim[i, j])
                })

    _heatmap_df = pl.DataFrame(_rows)

    # Facet titles: full method name, shorten only the cleanup suffix and recipe.
    def _fmt_method(m: str) -> str:
        base, _, recipe = m.partition("@")
        label = base.replace("+safe_mean_cleanup", "+cleanup")
        if recipe:
            r = recipe if len(recipe) <= 20 else recipe[:18] + "…"
            return f"{label}  [{r}]"
        return label

    _heatmap_df = _heatmap_df.with_columns(
        pl.col("method").map_elements(_fmt_method, return_dtype=pl.Utf8).alias("method_label")
    )

    _color_domain = [-1.0, 1.0]

    _n_labels = len(set(_names))
    _cell_size = max(8, min(16, 420 // max(_n_labels, 1)))

    _heatmap = (
        alt.Chart(_heatmap_df.to_pandas())
        .mark_rect()
        .encode(
            x=alt.X("x:N", title=None, axis=alt.Axis(labelAngle=-45, labelLimit=120)),
            y=alt.Y("y:N", title=None, axis=alt.Axis(labelLimit=120)),
            color=alt.Color(
                "cosine:Q",
                title="Cosine",
                scale=alt.Scale(scheme="redblue", domain=_color_domain),
            ),
            facet=alt.Facet("method_label:N", columns=2, title="Method  [recipe]"),
            tooltip=["x", "y", "method", "cosine"],
        )
        .properties(
            width=_cell_size * _n_labels,
            height=_cell_size * _n_labels,
            title=f"Vector similarity · layer {sim_layer_sel.value} · {sim_spec_sel.value}",
        )
        .resolve_scale(color="shared")
    )

    from sklearn.decomposition import PCA as _PCA

    _n_components = min(len(_names), _mat.shape[1], 8)
    _pca = _PCA(n_components=_n_components)
    _pca.fit(_mat.cpu().numpy())
    _var = _pca.explained_variance_ratio_
    _cumvar = np.cumsum(_var)

    _pcs_for_80 = int(np.searchsorted(_cumvar, 0.80)) + 1
    _pcs_for_95 = int(np.searchsorted(_cumvar, 0.95)) + 1
    _geometry_quality = (
        "good — behaviours occupy distinct directions"
        if _pcs_for_80 >= max(3, len(_names) // 4)
        else "moderate — some behaviours share a common axis"
        if _pcs_for_80 >= 2
        else "low — most behaviours are nearly co-linear (dominated by one direction)"
    )

    _pca_df = pl.DataFrame({
        "component": [f"PC{i + 1}" for i in range(len(_var))],
        "explained_variance": [float(v) for v in _var],
        "cumulative": [float(v) for v in _cumvar],
    })
    _pca_chart = (
        alt.Chart(_pca_df.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("component:N", title="Principal Component"),
            y=alt.Y("explained_variance:Q", title="Explained variance", scale=alt.Scale(domain=[0, 1])),
            color=alt.condition(
                alt.datum.cumulative <= 0.80,
                alt.value("#a7c080"),
                alt.value("#475258"),
            ),
            tooltip=["component", "explained_variance", "cumulative"],
        )
        .properties(width=420, height=240, title="PCA variance across behaviour vectors")
    )
    _pca_line = (
        alt.Chart(_pca_df.to_pandas())
        .mark_line(color="#e69875", strokeDash=[4, 2])
        .encode(x="component:N", y="cumulative:Q")
    )

    mo.vstack([
        mo.md(
            "**How to read:** Each cell is the cosine similarity between two behaviour vectors. "
            "The title format is `base_method  [recipe]`. Color scale is fixed to [−1, 1]: "
            "**blue = similar direction**, **white = orthogonal**, **red = opposite**.\n\n"
            "**Methods:**\n"
            "- **mean_centering** = μ(target) − centroid of all *other* behaviours "
            "(mean of the per-label means of every non-target category). "
            "Does **not** use safe-prompt responses as baseline — it uses the full set of other labels.\n"
            "- **mean_difference** = μ(target) − μ(baseline), where baseline is the chosen safe-prompt label.\n"
            "- **linear_probe** = logistic regression weight vector trained to separate target vs baseline. "
            "Normalised to unit length.\n"
            "- **pca** = first principal component of the target activations (direction of max variance).\n"
            "- **+safe_mean_cleanup** = any of the above with μ(safe-prompt responses) subtracted afterward. "
            "This projects out the shared 'safe response' component. "
            "**Why it looks all-blue:** with 25 labels, all vectors already have a strong component pointing "
            "away from the safe-response manifold. Subtracting the same μ(safe) from all of them removes that "
            "shared axis — what's left are small residuals that often point in similar directions → all-blue heatmap. "
            "This is mathematically expected, not a bug. The cleanup step is most useful for *steering*, "
            "not for making vectors geometrically distinct."
        ),
        mo.hstack([sim_layer_sel, sim_spec_sel], justify="start", gap=2),
        _heatmap,
        mo.md(
            f"#### Geometry score: **{_geometry_quality}**\n\n"
            f"- **{_pcs_for_80} PCs** needed to explain 80 % of the variance "
            f"(out of {len(_names)} behaviour vectors)\n"
            f"- **{_pcs_for_95} PCs** for 95 %\n\n"
            "**How to read this:** If PC1 alone explains > 60 % of variance, most "
            "behaviour vectors point in roughly the same direction — they may all be "
            "detecting the same underlying signal (e.g. 'safety-relevant text'). "
            "Green bars = PCs included in the 80 % threshold. "
            "The dashed orange line is cumulative explained variance."
        ),
        _pca_chart + _pca_line,
    ])
    return


@app.cell(hide_code=True)
def _(all_vecs, mo, sim_spec_sel):
    mo.stop(
        sim_spec_sel.value is None or len(all_vecs.methods()) < 2,
        mo.md("_Need at least two methods built to compare them._"),
    )

    _methods_avail = all_vecs.methods()
    sim_method_a = mo.ui.dropdown(
        options=_methods_avail,
        value=_methods_avail[0] if _methods_avail else None,
        label="Method A",
    )
    sim_method_b = mo.ui.dropdown(
        options=_methods_avail,
        value=_methods_avail[1] if len(_methods_avail) > 1 else _methods_avail[0],
        label="Method B",
    )
    mo.vstack([
        mo.md(
            "#### Cross-method similarity\n\n"
            "How similar are the vectors produced by two different methods for the same behaviour? "
            "High cosine → both methods point to the same direction (they agree). "
            "Low cosine → they encode different things (disagreement worth investigating)."
        ),
        mo.hstack([sim_method_a, sim_method_b], justify="start", gap=2),
    ])
    return sim_method_a, sim_method_b


@app.cell(hide_code=True)
def _(
    all_vecs,
    alt,
    mo,
    pl,
    sim_layer_sel,
    sim_method_a,
    sim_method_b,
    sim_spec_sel,
):
    mo.stop(sim_method_a.value is None or sim_method_b.value is None)

    import torch as _tc5

    _ma = sim_method_a.value
    _mb = sim_method_b.value
    _layer_cm = sim_layer_sel.value
    _spec_cm  = sim_spec_sel.value

    _vecs_a = {v.label: v for v in all_vecs.filter(method=_ma, layer=_layer_cm, spec=_spec_cm).vectors}
    _vecs_b = {v.label: v for v in all_vecs.filter(method=_mb, layer=_layer_cm, spec=_spec_cm).vectors}
    _common = sorted(set(_vecs_a) & set(_vecs_b))

    mo.stop(not _common, mo.md(f"⚠️ No labels have vectors for both {_ma!r} and {_mb!r} at this layer/spec."))

    _rows_cm = []
    for _lbl in _common:
        _va = _vecs_a[_lbl].vector.float()
        _vb = _vecs_b[_lbl].vector.float()
        _cos = float((_va / (_va.norm() + 1e-8)) @ (_vb / (_vb.norm() + 1e-8)))
        _rows_cm.append({"label": _lbl, "cosine": _cos})

    _df_cm = pl.DataFrame(_rows_cm).sort("cosine", descending=True)

    _chart_cm = (
        alt.Chart(_df_cm.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("cosine:Q", title="Cosine similarity", scale=alt.Scale(domain=[-1, 1])),
            y=alt.Y("label:N", sort="-x", title=None),
            color=alt.Color(
                "cosine:Q",
                scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
                legend=None,
            ),
            tooltip=["label", "cosine"],
        )
        .properties(
            width=500, height=max(200, 18 * len(_common)),
            title=f"Method similarity per behaviour · {_ma} vs {_mb} · L{_layer_cm} · {_spec_cm}",
        )
    )

    _med = float(_df_cm["cosine"].median())
    mo.vstack([
        mo.md(f"**Median cosine similarity:** {_med:.3f}  ·  comparing **{_ma}** vs **{_mb}** at layer {_layer_cm}"),
        _chart_cm,
    ])
    return


@app.cell(hide_code=True)
def _(all_vecs, mo):
    mo.stop(
        len(all_vecs.methods()) == 0,
        mo.md("_Build vectors first._"),
    )

    _methods_cl = all_vecs.methods()
    _specs_cl = all_vecs.specs()
    _labels_cl = all_vecs.labels()

    sim_cross_layer_label = mo.ui.dropdown(
        options=_labels_cl,
        value=_labels_cl[0] if _labels_cl else None,
        label="Behaviour",
    )
    sim_cross_layer_method = mo.ui.dropdown(
        options=_methods_cl,
        value=_methods_cl[0] if _methods_cl else None,
        label="Method",
    )
    sim_cross_layer_spec = mo.ui.dropdown(
        options=_specs_cl,
        value=_specs_cl[0] if _specs_cl else None,
        label="Spec",
    )

    mo.vstack([
        mo.md(
            "#### Cross-layer similarity\n\n"
            "How similar is a behaviour\'s steering vector across different layers? "
            "Nearby layers usually agree (high cosine); a sudden drop signals where "
            "the representation changes character — often a useful steering target."
        ),
        mo.hstack([sim_cross_layer_label, sim_cross_layer_method, sim_cross_layer_spec], justify="start", gap=2),
    ])
    return sim_cross_layer_label, sim_cross_layer_method, sim_cross_layer_spec


@app.cell(hide_code=True)
def _(
    all_vecs,
    alt,
    mo,
    pl,
    sim_cross_layer_label,
    sim_cross_layer_method,
    sim_cross_layer_spec,
):
    mo.stop(
        sim_cross_layer_label.value is None or sim_cross_layer_method.value is None,
    )

    import torch as _tc6

    _lbl_cl = sim_cross_layer_label.value
    _meth_cl = sim_cross_layer_method.value
    _spec_cl_val = sim_cross_layer_spec.value

    _vecs_cl = all_vecs.filter(label=_lbl_cl, method=_meth_cl, spec=_spec_cl_val)
    mo.stop(len(_vecs_cl) < 2, mo.md(f"⚠️ Need at least 2 layers for '{_lbl_cl}' with method '{_meth_cl}'."))

    _layers_cl = sorted(_vecs_cl.layers())
    _vmap_cl = {v.layer: v.vector.float() for v in _vecs_cl.vectors}

    _rows_cl = []
    for _li in _layers_cl:
        for _lj in _layers_cl:
            _vi = _vmap_cl[_li]
            _vj = _vmap_cl[_lj]
            _cos_cl = float((_vi / (_vi.norm() + 1e-8)) @ (_vj / (_vj.norm() + 1e-8)))
            _rows_cl.append({"layer_i": str(_li), "layer_j": str(_lj), "cosine": _cos_cl})

    _df_cl = pl.DataFrame(_rows_cl)
    _n_cl = len(_layers_cl)
    _sz_cl = max(6, min(18, 500 // max(_n_cl, 1)))

    _heat_cl = (
        alt.Chart(_df_cl.to_pandas())
        .mark_rect()
        .encode(
            x=alt.X("layer_i:O", title="Layer", sort=[str(l) for l in _layers_cl]),
            y=alt.Y("layer_j:O", title="Layer", sort=[str(l) for l in _layers_cl]),
            color=alt.Color("cosine:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1]), title="Cosine"),
            tooltip=["layer_i", "layer_j", "cosine"],
        )
        .properties(
            width=_sz_cl * _n_cl, height=_sz_cl * _n_cl,
            title=f"Cross-layer similarity · {_lbl_cl} · {_meth_cl} · {_spec_cl_val}",
        )
    )
    _heat_cl
    return


@app.cell
def _(mo):
    mo.md("""
    ## § 6 — Scoring

    Two complementary evaluations on the held-out test split:

    **Separation Score (AUROC)** — how well does each vector's cosine similarity
    discriminate between positions where a behaviour is *present* vs. *absent*?
    This is purely about detection, not causation.

    **Hard-to-game diagnostics** — checks that the AUROC isn't driven by surface
    cues (keyword overlap, ablation, neutral-PCA).
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Separation Score (AUROC)

    **What the scoring aggregations mean:**

    Scoring operates **per annotated span** (each span = one labeled sentence from the dataset).
    For each span, we compute a single score by aggregating the per-token cosine similarities
    within that span:

    | Aggregation | What it computes |
    |---|---|
    | `mean` | Average cosine over all tokens in the span |
    | `first` | Cosine at the first token of the span |
    | `last` | Cosine at the last token of the span |
    | `sentence_mean` | **Same as `mean`** — redundant given that each span is already a sentence |
    | `sentence_first` | **Same as `first`** |
    | `sentence_last` | **Same as `last`** |

    Since your annotations are at sentence granularity, each span is already one sentence —
    the `sentence_*` variants add nothing. Just use `mean`, `first`, and `last`.
    `mean` is the safest default; `last` can work well for behaviours signalled
    at the end of a sentence (e.g. a conclusion or refusal).

    **Reading AUROC:**
    - ≥ 0.8 = strong separation · ≥ 0.65 = useful · < 0.5 = worse than random
    """)
    eval_aggregation_sel = mo.ui.multiselect(
        options=["mean", "first", "last"],
        value=["mean"],
        label="Scoring aggregations",
    )
    eval_btn = mo.ui.run_button(label="Run separation scoring on test data (§ 6)")
    mo.vstack([
        eval_aggregation_sel,
        mo.callout(
            mo.md(
                "Pick one or more aggregations — each becomes a separate row in "
                "the results. The heatmap below shows the **best** AUROC per "
                "(label × method × layer × spec) cell across the picked "
                "aggregations, and a separate bar chart compares the aggregations "
                "head-to-head so you can see whether the choice actually matters "
                "on your data."
            ),
            kind="info",
        ),
        eval_btn,
    ])
    return eval_aggregation_sel, eval_btn


@app.cell
def _(
    all_vecs,
    eval_aggregation_sel,
    eval_btn,
    ls,
    mo,
    model,
    pl,
    test_entries,
):
    mo.stop(not eval_btn.value)

    import gc as _gc
    # cache_size=16: keeps recent entries warm across specs without building up
    # gigabytes of CPU tensors. 128 was OOMing: 128 entries × 32 layers ×
    # seq_len × hidden_dim × 2 bytes can exceed available CPU RAM.
    _session = ls.Session(model, test_entries, cache_size=16)
    _all_rows = []
    _aggregations = eval_aggregation_sel.value or ["mean"]

    _specs_list = all_vecs.specs()

    for _spec in mo.status.progress_bar(
        _specs_list,
        title=f"Evaluating {len(_specs_list)} vector readout specs...",
        completion_title="Separation scoring done!",
    ):
        _spec_vecs = all_vecs.filter(spec=_spec)
        _results = _session.evaluate(
            _spec_vecs,
            aggregations=_aggregations,
            show_progress=False,
            progress_fn=None,
        )
        _session.clear_cache()
        _gc.collect()
        for r in _results:
            _all_rows.append({
                "label": r.label,
                "method": r.method,
                "layer": r.layer,
                "spec": _spec,
                "aggregation": r.aggregation,
                "auroc": round(r.auroc, 4) if r.auroc == r.auroc else None,
                "auprc": round(r.auprc, 4) if hasattr(r, "auprc") and r.auprc == r.auprc else None,
                "f1": round(r.f1, 4),
                "n_present": r.n_present,
                "n_absent": r.n_absent,
                "cm": r.confusion_matrix.tolist() if r.confusion_matrix is not None else None,
            })

    eval_df = pl.DataFrame(_all_rows).filter(pl.col("auroc").is_not_null())
    mo.md(f"✅ Evaluation done — **{len(eval_df)}** (label × method × layer × spec × agg) rows.")
    return (eval_df,)


@app.cell
def _(alt, eval_df, mo, pl):
    mo.stop(
        eval_df is None or len(eval_df) == 0,
        mo.md("_Run separation scoring to see charts._"),
    )

    _pd = eval_df.to_pandas()

    # Best AUROC per (label × method × layer × spec), aggregated over aggregation
    # for the heatmap, plus a separate view comparing aggregations head-to-head.
    _pd_best = (
        eval_df.sort("auroc", descending=True)
        .unique(subset=["label", "method", "layer", "spec"], keep="first")
        .to_pandas()
    )
    _chart = (
        alt.Chart(_pd_best)
        .mark_rect()
        .encode(
            x=alt.X("layer:O", title="Layer"),
            y=alt.Y("label:N", title="Behavior"),
            color=alt.Color("auroc:Q", title="AUROC", scale=alt.Scale(scheme="viridis", domain=[0.0, 1.0])),
            facet=alt.Facet("method:N", columns=2, title="Vector Method"),
            tooltip=["label", "layer", "method", "spec", "aggregation", "auroc", "auprc", "f1",
                     "n_present", "n_absent"],
        )
        .properties(width=300, height=max(200, 20 * len(_pd["label"].unique())),
                    title="Separation Score (AUROC) — best aggregation per cell")
    )

    _auprc_chart = (
        alt.Chart(_pd_best[_pd_best["auprc"].notna()] if "auprc" in _pd_best.columns else _pd_best)
        .mark_rect()
        .encode(
            x=alt.X("layer:O", title="Layer"),
            y=alt.Y("label:N", title="Behavior"),
            color=alt.Color("auprc:Q", title="AUPRC", scale=alt.Scale(scheme="oranges", domain=[0.0, 1.0])),
            facet=alt.Facet("method:N", columns=2, title="Vector Method"),
            tooltip=["label", "layer", "method", "spec", "aggregation", "auroc", "auprc", "f1",
                     "n_present", "n_absent"],
        )
        .properties(width=300, height=max(200, 20 * len(_pd["label"].unique())),
                    title="AUPRC — more informative than AUROC when positive examples are rare")
    ) if "auprc" in _pd_best.columns and _pd_best["auprc"].notna().any() else None

    # Layer profile: AUROC vs layer for each (label, method) — best aggregation per cell.
    # Makes it easy to see which layer peaks for each behaviour.
    _n_methods_for_layer = _pd_best["method"].nunique()
    _layer_chart = (
        alt.Chart(_pd_best)
        .mark_line(point=True, strokeWidth=1.5)
        .encode(
            x=alt.X("layer:Q", title="Layer"),
            y=alt.Y("auroc:Q", title="AUROC", scale=alt.Scale(domain=[0.0, 1.0])),
            color=alt.Color("label:N", title="Behaviour", legend=alt.Legend(columns=2)),
            facet=alt.Facet("method:N", columns=min(2, _n_methods_for_layer), title="Method"),
            tooltip=["label", "layer", "method", "spec", "auroc", "aggregation"],
        )
        .properties(width=350, height=200, title="AUROC by layer — which layer works best per behaviour?")
    )

    # Aggregation comparison: average AUROC across labels/layers/specs, per
    # (method × aggregation). Shows whether your aggregation choice changes
    # anything.
    _agg_compare = (
        eval_df.group_by(["method", "aggregation"])
        .agg(
            pl.col("auroc").mean().alias("mean_auroc"),
            pl.col("auroc").max().alias("max_auroc"),
            pl.col("auroc").count().alias("n"),
        )
        .sort(["method", "aggregation"])
        .to_pandas()
    )
    _agg_chart = (
        alt.Chart(_agg_compare)
        .mark_bar()
        .encode(
            x=alt.X("aggregation:N", title="Aggregation", axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("mean_auroc:Q", title="Mean AUROC", scale=alt.Scale(domain=[0.0, 1.0])),
            color=alt.Color("aggregation:N", legend=None),
            facet=alt.Facet("method:N", columns=4, title="Method"),
            tooltip=["method", "aggregation", "mean_auroc", "max_auroc", "n"],
        )
        .properties(width=160, height=160)
    )

    # Best per label: for each label, find the (method, layer, spec, agg) combo with highest AUROC.
    # This ensures we see a confusion matrix for every behaviour, not just the easiest one.
    _best_per_label = (
        eval_df.sort("auroc", descending=True)
        .unique(subset=["label"], keep="first")
        .sort("auroc", descending=True)
    )

    # Render every confusion matrix as a tiny inline card. Lay them out as a
    # responsive horizontal grid so the user can scan all behaviours at once
    # without having to click anything open.
    def _cm_card_html(row):
        _cm = row.get("cm")
        if _cm is None:
            return None
        tn, fp = _cm[0]
        fn, tp = _cm[1]
        # color cells based on whether they're "good" (TP/TN) or "bad" (FP/FN)
        _good = "background:rgba(167,192,128,0.30);"
        _bad = "background:rgba(230,126,128,0.30);"
        _cell = "padding:6px 10px;text-align:center;font-variant-numeric:tabular-nums;border:1px solid var(--bg-3);"
        _hdr = "padding:4px 8px;text-align:center;font-size:11px;color:var(--fg-dim);"
        return (
            f'<div style="border:1px solid var(--bg-3);border-radius:6px;'
            f'padding:8px;background:var(--bg-1);min-width:230px;flex:0 0 auto;">'
            f'<div style="font-weight:600;font-size:12px;margin-bottom:4px;">'
            f'{row["label"]} · AUROC={row["auroc"]:.3f}</div>'
            f'<div style="font-size:10px;color:var(--fg-dim);margin-bottom:6px;">'
            f'{row["method"]} · L{row["layer"]} · {row["spec"]} · {row["aggregation"]}</div>'
            f'<table style="border-collapse:collapse;font-size:12px;">'
            f'<tr><td style="{_hdr}"></td>'
            f'<td style="{_hdr}">Pred 0</td><td style="{_hdr}">Pred 1</td></tr>'
            f'<tr><td style="{_hdr}">Actual 0</td>'
            f'<td style="{_cell}{_good}">{tn}</td><td style="{_cell}{_bad}">{fp}</td></tr>'
            f'<tr><td style="{_hdr}">Actual 1</td>'
            f'<td style="{_cell}{_bad}">{fn}</td><td style="{_cell}{_good}">{tp}</td></tr>'
            f'</table></div>'
        )

    _cards = [_cm_card_html(row) for row in _best_per_label.to_dicts()]
    _cards = [c for c in _cards if c]
    _cm_grid = (
        mo.Html(
            '<div style="display:flex;flex-wrap:wrap;gap:10px;">'
            + "".join(_cards) +
            '</div>'
        )
        if _cards else mo.md("No confusion matrices available.")
    )

    _auprc_items = []
    if _auprc_chart is not None:
        _auprc_items = [
            mo.callout(
                mo.md(
                    "**AUPRC** (Area Under Precision-Recall Curve) penalizes false positives harder than AUROC, "
                    "especially when positive examples are rare (small n_present). "
                    "Values > 0.5 = useful · > 0.8 = strong. "
                    "**Cells with n_present < 5** are statistically unreliable — treat with caution."
                ),
                kind="info",
            ),
            _auprc_chart,
        ]

    mo.vstack([
        _chart,
        *_auprc_items,
        _layer_chart,
        mo.md(
            "#### Aggregation comparison\n"
            "Each bar is the mean AUROC across all (label × layer × spec) combinations "
            "for that method × aggregation pair. If bars within the same method facet "
            "are similar, your choice of aggregation barely matters for these vectors; "
            "if one is taller, that aggregation is genuinely better. The heatmap above "
            "already keeps the best aggregation per cell."
        ),
        _agg_chart,
        mo.md(
            "#### Best-configuration confusion matrices (one per behaviour)\n"
            "Each card is the confusion matrix for the (method × layer × spec × aggregation) "
            "that achieved the highest AUROC for that behaviour. "
            "Green = correct predictions (true negatives top-left, true positives bottom-right); "
            "red = mistakes (false positives top-right, false negatives bottom-left)."
        ),
        _cm_grid,
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### Diagnostics

    Runs cheap checks to verify the vectors encode something deeper than surface-level token patterns.

    | Diagnostic | What it measures | Interpretation |
    |---|---|---|
    | **Logit lens** | Top tokens the vector up-/down-weights via the unembedding matrix | Tokens should be semantically related to the behaviour name — if they're generic stopwords, the vector is noisy |
    | **Keyword overlap** | Cosine similarity between the vector and the mean embedding of behaviour-keyword tokens | < 0.2 = good (vector is not just a re-projection of token embeddings) · > 0.5 = red flag |
    | **Token ablation** | Does AUROC survive when all keyword tokens are masked out? | Retained fraction > 0.7 = vector encodes concept beyond keywords · < 0.3 = likely a surface-level token detector |
    | **Neutral-PCA** | Does AUROC survive after projecting out the top PCs of neutral text activations? | Retained fraction > 0.7 = vector is not just capturing generic text structure |
    """)
    return


@app.cell
def _(all_vecs, mo):
    _layers = all_vecs.layers()
    _mid_layer = _layers[len(_layers) // 2] if _layers else None
    scoring_label_sel = mo.ui.multiselect(
        options=all_vecs.labels(),
        value=all_vecs.labels()[:3],
        label="Behaviors to score",
    )
    scoring_method_sel = mo.ui.multiselect(
        options=all_vecs.methods(),
        value=all_vecs.methods(),
        label="Methods to score",
    )
    scoring_layer_sel = mo.ui.multiselect(
        options=_layers,
        value=[_mid_layer] if _mid_layer is not None else [],
        label="Layers to score",
    )
    scoring_spec_sel = mo.ui.multiselect(
        options=all_vecs.specs(),
        value=all_vecs.specs(),
        label="Token readouts to score",
    )
    scoring_run_logit_lens = mo.ui.checkbox(label="Run Logit Lens (Fast)", value=True)
    scoring_run_keyword_overlap = mo.ui.checkbox(label="Run Keyword Overlap (Fast)", value=True)
    scoring_run_token_ablation = mo.ui.checkbox(label="Run Token Ablation (Slow)", value=False)
    scoring_run_neutral_pca = mo.ui.checkbox(label="Run Neutral-PCA (Slow)", value=False)
    scoring_neutral_use_cached = mo.ui.checkbox(
        label="Neutral-PCA: use cached train activations as neutral baseline (fast, no extra forward passes)",
        value=True,
    )
    scoring_ll_layer_sel = mo.ui.dropdown(
        options={str(l): l for l in _layers} if _layers else {},
        value=str(_mid_layer) if _mid_layer is not None else None,
        label="Logit lens: show layer",
    )

    mo.vstack([
        mo.md(
            "**About keywords (used by Keyword Overlap and Token Ablation):** "
            "for each behaviour, keywords are **auto-derived** from the labelled "
            "sentences in the test split — we take the most frequent words "
            "(length > 3, stopwords removed) that appear inside spans tagged with "
            "that behaviour, top 10. The Logit Lens is **independent** of keywords: "
            "it just projects the steering vector through the unembedding matrix "
            "to read the top vocabulary tokens."
        ),
        mo.hstack([scoring_label_sel, scoring_method_sel], justify="start", gap=2),
        mo.hstack([scoring_layer_sel, scoring_spec_sel], justify="start", gap=2),
        mo.hstack([
            scoring_run_logit_lens,
            scoring_run_keyword_overlap,
            scoring_run_token_ablation,
            scoring_run_neutral_pca,
        ], justify="start", gap=2),
        scoring_neutral_use_cached,
        scoring_ll_layer_sel,
    ])
    return (
        scoring_label_sel,
        scoring_layer_sel,
        scoring_ll_layer_sel,
        scoring_method_sel,
        scoring_neutral_use_cached,
        scoring_run_keyword_overlap,
        scoring_run_logit_lens,
        scoring_run_neutral_pca,
        scoring_run_token_ablation,
        scoring_spec_sel,
    )


@app.cell
def _(mo, scoring_label_sel, test_entries):
    import re as _re3
    _stopwords3 = {"the", "a", "an", "is", "it", "in", "of", "to", "and",
                   "or", "that", "this", "i", "my", "be", "will", "can",
                   "for", "as", "with", "its", "have", "has", "not", "no",
                   "but", "so", "if", "do", "on", "at", "by", "from",
                   "are", "was", "were", "been", "being", "their", "they"}

    def _keywords_preview(label):
        _wc: dict[str, int] = {}
        for _e in test_entries:
            for _ann in _e.annotations:
                if label in _ann.labels:
                    for _w in _re3.findall(r"[a-zA-Z]+", _ann.text.lower()):
                        if _w not in _stopwords3 and len(_w) > 3:
                            _wc[_w] = _wc.get(_w, 0) + 1
        return sorted(_wc.items(), key=lambda x: -x[1])[:10]

    _kw_rows = []
    for _lbl_kw in (scoring_label_sel.value or []):
        _kws = _keywords_preview(_lbl_kw)
        _kw_rows.append({
            "label": _lbl_kw,
            "keywords": ", ".join(w for w, _ in _kws) if _kws else "(none found)",
            "top_word_count": _kws[0][1] if _kws else 0,
        })

    mo.vstack([
        mo.callout(
            mo.md(
                "**About keyword masking (Token Ablation):** keywords are extracted automatically from the "
                "test split — the 10 most frequent words (length > 3, stopwords removed) in spans tagged "
                "with each behaviour. Masking replaces each keyword with a single **space** using "
                "case-insensitive whole-word matching (`\\bkeyword\\b`). "
                "**Note:** the regex splits contractions — `didn't` → `didn` + `t`, so you may see "
                "fragments in the list. The preview below shows keywords that will be used."
            ),
            kind="neutral",
        ),
        mo.ui.table(_kw_rows, selection=None) if _kw_rows else mo.md("_Select labels above to preview keywords._"),
    ]) if test_entries else mo.md("_Extract activations first to preview keywords._")
    return


@app.cell
def _(mo):
    score_btn = mo.ui.run_button(label="Run diagnostics (§ 6)")
    score_btn
    return (score_btn,)


@app.cell
def _(
    all_vecs,
    alt,
    extraction_result,
    ls,
    mo,
    model,
    pl,
    score_btn,
    scoring_label_sel,
    scoring_layer_sel,
    scoring_ll_layer_sel,
    scoring_method_sel,
    scoring_neutral_use_cached,
    scoring_run_keyword_overlap,
    scoring_run_logit_lens,
    scoring_run_neutral_pca,
    scoring_run_token_ablation,
    scoring_spec_sel,
    test_entries,
):
    mo.stop(not score_btn.value)

    _selected = [
        v for v in all_vecs
        if v.label in set(scoring_label_sel.value)
        and v.method in set(scoring_method_sel.value)
        and v.layer in set(scoring_layer_sel.value)
        and v.extraction_spec in set(scoring_spec_sel.value)
    ]
    mo.stop(not _selected, mo.md("⚠️ No vectors match the selected scoring controls."))

    import re as _re
    _stopwords = {"the", "a", "an", "is", "it", "in", "of", "to", "and",
                  "or", "that", "this", "i", "my", "be", "will", "can",
                  "for", "as", "with", "its", "have", "has", "not", "no",
                  "but", "so", "if", "do", "on", "at", "by", "from",
                  "are", "was", "were", "been", "being", "their", "they"}

    def _keywords_for(label):
        _word_counts: dict[str, int] = {}
        for _e in test_entries:
            for _ann in _e.annotations:
                if label in _ann.labels:
                    for _w in _re.findall(r"[a-zA-Z]+", _ann.text.lower()):
                        if _w not in _stopwords and len(_w) > 3:
                            _word_counts[_w] = _word_counts.get(_w, 0) + 1
        _words = [w for w, _ in sorted(_word_counts.items(), key=lambda x: -x[1])[:10]]
        return _words or [label.lower()]

    _rows = []
    _details = {}
    for _vec in mo.status.progress_bar(
        _selected,
        title=f"Running diagnostics for {len(_selected)} vectors…",
        completion_title="Diagnostics done!",
    ):
        _keywords = _keywords_for(_vec.label)
        _key = f"{_vec.label} / {_vec.method} / layer {_vec.layer} / {_vec.extraction_spec}"
        _row = {
            "label": _vec.label,
            "method": _vec.method,
            "layer": _vec.layer,
            "spec": _vec.extraction_spec,
            "keywords": ", ".join(_keywords),
        }
        try:
            if scoring_run_logit_lens.value:
                _readout = ls.logit_lens_top_tokens(model, _vec, k=10)
            else:
                _readout = None

            if scoring_run_keyword_overlap.value:
                _overlap = ls.embedding_keyword_overlap(model, _vec, _keywords)
            else:
                _overlap = None

            if scoring_run_token_ablation.value:
                _ablation = ls.token_ablation_score(
                    model, test_entries, _vec, keywords=_keywords, show_progress=False
                )
            else:
                _ablation = None

            if scoring_run_neutral_pca.value:
                if scoring_neutral_use_cached.value and extraction_result is not None:
                    # Fast path: use a small subset of test_entries for PCA fitting
                    # instead of all entries, leveraging the already-extracted activations
                    # to select a representative subset.
                    import random as _rnd_npca
                    _neutral_subset = _rnd_npca.sample(test_entries, min(20, len(test_entries)))
                    _neutral_pca = ls.neutral_pca_score(
                        model, _vec,
                        neutral_entries=_neutral_subset,
                        eval_entries=test_entries,
                        n_components=4,
                        show_progress=False,
                    )
                else:
                    _neutral_pca = ls.neutral_pca_score(
                        model, _vec,
                        neutral_entries=test_entries,
                        eval_entries=test_entries,
                        n_components=4,
                        show_progress=False,
                    )
            else:
                _neutral_pca = None

            _row.update({
                "kw_cosine": float(round(_overlap.cosine, 4)) if _overlap else None,
                "ablation_kept": float(round(_ablation.retained_fraction, 4)) if _ablation else None,
                "neutral_pca_kept": float(round(_neutral_pca.retained_fraction, 4)) if _neutral_pca else None,
                "ablation_disc": float(round(_ablation.ablated_discrimination, 4)) if _ablation else None,
                "neutral_pca_disc": float(round(_neutral_pca.ablated_discrimination, 4)) if _neutral_pca else None,
                "top_up_tokens": ", ".join(t for t, _ in _readout.top_positive[:5]) if _readout else "",
                "status": "ok",
            })
            if _readout:
                _details[_key] = {
                    "label": _vec.label,
                    "method": _vec.method,
                    "layer": _vec.layer,
                    "spec": _vec.extraction_spec,
                    "positive": [
                        {"rank": i + 1, "token": t, "logit": round(v, 3), "direction": "Upweighted"}
                        for i, (t, v) in enumerate(_readout.top_positive)
                    ],
                    "negative": [
                        {"rank": i + 1, "token": t, "logit": round(v, 3), "direction": "Downweighted"}
                        for i, (t, v) in enumerate(_readout.top_negative)
                    ],
                }
        except Exception as _exc:
            _row.update({
                "kw_cosine": None,
                "ablation_kept": None,
                "neutral_pca_kept": None,
                "ablation_disc": None,
                "neutral_pca_disc": None,
                "top_up_tokens": "",
                "status": str(_exc)[:120],
            })
        _rows.append(_row)

    _sort_cols = []
    _sort_desc = []
    if scoring_run_keyword_overlap.value:
        _sort_cols.append("kw_cosine")
        _sort_desc.append(False)
    if scoring_run_token_ablation.value:
        _sort_cols.append("ablation_kept")
        _sort_desc.append(True)
    if scoring_run_neutral_pca.value:
        _sort_cols.append("neutral_pca_kept")
        _sort_desc.append(True)

    if not _sort_cols:
        _sort_cols = ["label", "layer"]
        _sort_desc = [False, False]

    scoring_df = pl.DataFrame(_rows).sort(
        _sort_cols,
        descending=_sort_desc,
        nulls_last=True,
    )

    _pd2 = scoring_df.to_pandas()
    _metric_charts = []

    if "kw_cosine" in _pd2.columns and scoring_run_keyword_overlap.value:
        _metric_charts.append(
            mo.vstack([
                mo.md(
                    "**Keyword Cosine — lower is better.** "
                    "Measures the cosine similarity between the steering vector and the mean token embedding "
                    "of the top keywords extracted from labelled spans. "
                    "A high score means the vector may be mostly re-encoding surface-level token patterns "
                    "rather than a deeper behavioural concept. "
                    "**Target:** < 0.2 = good (vector goes beyond keyword matching) · "
                    "> 0.5 = red flag (likely a surface-level token detector). "
                    "Applies to all methods. Each line shows how similarity varies across layers — "
                    "if it drops toward deeper layers, the deeper layers encode more abstract signal."
                ),
                alt.Chart(_pd2).mark_line(point=True).encode(
                    x=alt.X("layer:O", title="Layer"),
                    y=alt.Y("kw_cosine:Q", title="Cosine (lower = better)", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("method:N", title="Method"),
                    tooltip=["label", "method", "layer", "spec", "kw_cosine", "keywords"],
                ).properties(width=200, height=160).facet(
                    facet=alt.Facet("label:N", title="Keyword Cosine by Behaviour (per layer)"), columns=4
                ),
            ])
        )
    if "ablation_kept" in _pd2.columns and scoring_run_token_ablation.value:
        _metric_charts.append(
            mo.vstack([
                mo.md(
                    "**Token Ablation retained fraction — higher is better.** "
                    "> 0.7: discrimination survives keyword masking (deep signal). "
                    "< 0.3: vector is mostly detecting keywords in text."
                ),
                alt.Chart(_pd2).mark_bar().encode(
                    x=alt.X("layer:O", title="Layer"),
                    y=alt.Y("ablation_kept:Q", title="Retained fraction (higher = better)", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("method:N", title="Method"),
                    tooltip=["label", "method", "layer", "spec", "ablation_kept", "ablation_disc"],
                ).properties(width=150, height=200).facet(
                    facet=alt.Facet("label:N", title="Token Ablation"), columns=4
                ),
            ])
        )
    if "neutral_pca_kept" in _pd2.columns and scoring_run_neutral_pca.value:
        _metric_charts.append(
            mo.vstack([
                mo.md(
                    "**Neutral-PCA retained fraction — higher is better.** "
                    "> 0.7: discrimination survives removing generic-text PCs (behaviour-specific signal). "
                    "< 0.3: vector mainly captures generic text structure."
                ),
                alt.Chart(_pd2).mark_bar().encode(
                    x=alt.X("layer:O", title="Layer"),
                    y=alt.Y("neutral_pca_kept:Q", title="Retained fraction (higher = better)", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("method:N", title="Method"),
                    tooltip=["label", "method", "layer", "spec", "neutral_pca_kept", "neutral_pca_disc"],
                ).properties(width=150, height=200).facet(
                    facet=alt.Facet("label:N", title="Neutral PCA"), columns=4
                ),
            ])
        )

    # Logit lens panels: filter by selected layer, one card per (label, recipe).
    _ll_filter_layer = scoring_ll_layer_sel.value  # int or None
    _ll_best_keys: dict[tuple[str, str], str] = {}
    for _dk, _dv in _details.items():
        if _ll_filter_layer is not None and _dv.get("layer") != _ll_filter_layer:
            continue
        _recipe = (_dv.get("method") or "").split("@", 1)[1] if "@" in (_dv.get("method") or "") else "current"
        _key2 = (_dv["label"], _recipe)
        if _key2 not in _ll_best_keys:
            _ll_best_keys[_key2] = _dk

    def _ll_card(dk, dv):
        rows = dv["positive"] + dv["negative"]
        if not rows:
            return None
        max_abs = max(abs(r["logit"]) for r in rows) or 1.0
        bars = []
        for r in dv["positive"]:
            w = abs(r["logit"]) / max_abs * 100
            bars.append(
                f'<div style="display:flex;align-items:center;gap:6px;font-size:11px;">'
                f'<span style="width:80px;text-align:right;color:var(--fg);font-family:monospace;">'
                f'{r["token"]!s}</span>'
                f'<div style="flex:1;background:rgba(167,192,128,0.30);height:14px;border-radius:2px;overflow:hidden;">'
                f'<div style="width:{w:.1f}%;height:100%;background:#a7c080;"></div></div>'
                f'<span style="width:50px;color:var(--fg-dim);font-variant-numeric:tabular-nums;">+{r["logit"]:.2f}</span>'
                f'</div>'
            )
        bars.append('<div style="height:6px;"></div>')
        for r in dv["negative"]:
            w = abs(r["logit"]) / max_abs * 100
            bars.append(
                f'<div style="display:flex;align-items:center;gap:6px;font-size:11px;">'
                f'<span style="width:80px;text-align:right;color:var(--fg);font-family:monospace;">'
                f'{r["token"]!s}</span>'
                f'<div style="flex:1;background:rgba(230,126,128,0.30);height:14px;border-radius:2px;overflow:hidden;">'
                f'<div style="width:{w:.1f}%;height:100%;background:#e67e80;"></div></div>'
                f'<span style="width:50px;color:var(--fg-dim);font-variant-numeric:tabular-nums;">{r["logit"]:.2f}</span>'
                f'</div>'
            )
        return (
            f'<div style="border:1px solid var(--bg-3);border-radius:6px;padding:10px;'
            f'background:var(--bg-1);min-width:300px;flex:0 0 auto;">'
            f'<div style="font-weight:600;font-size:12px;margin-bottom:2px;">{dv["label"]}</div>'
            f'<div style="font-size:10px;color:var(--fg-dim);margin-bottom:6px;">'
            f'{dv["method"]} · L{dv["layer"]} · {dv["spec"]}</div>'
            + "".join(bars) +
            '</div>'
        )

    _ll_cards = [
        _ll_card(_dk, _details[_dk]) for _dk in _ll_best_keys.values()
    ]
    _ll_cards = [c for c in _ll_cards if c]
    _ll_grid = (
        mo.Html(
            '<div style="display:flex;flex-wrap:wrap;gap:10px;">'
            + "".join(_ll_cards) +
            '</div>'
        )
        if _ll_cards else mo.md("_No logit-lens output (enable it above and re-run)._")
    )

    mo.vstack([
        mo.md(f"✅ Scored **{len(scoring_df)}** vectors."),
        *_metric_charts,
        mo.md(
            f"#### Logit Lens — Layer {_ll_filter_layer if _ll_filter_layer is not None else 'all'} "
            "(change layer with the *Logit lens: show layer* dropdown above)\n"
            "Projects each steering vector through the model's unembedding matrix to read the "
            "implied vocabulary. **Green** tokens are upweighted (the vector pushes generation "
            "*toward* them); **red** tokens are downweighted. "
            "**How to interpret:** If the green tokens are semantically related to the behaviour "
            "(e.g. a *deception* vector surfaces *hide*, *mislead*, *secret*), the vector encodes "
            "a meaningful concept. If they are generic stopwords or punctuation, the vector is "
            "noisy or too shallow. This is a linear projection — it ignores MLP layers, so treat "
            "it as a rough sanity check rather than ground truth."
        ) if scoring_run_logit_lens.value else mo.md(""),
        _ll_grid if scoring_run_logit_lens.value else mo.md(""),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## § 7 — Entry browser

    Per-token visualization of vector-based detection on individual entries.
    Shows ground truth labels and cosine-similarity heatmap side by side.
    """)
    return


@app.cell
def _(entries, mo, test_entries):
    def _user_prompt(e):
        for m in e.messages:
            if m["role"] == "user":
                return m["content"][:70].replace("\n", " ")
        return "(no user message)"

    # Sort: test entries first (most useful for evaluation), then train.
    _test_id_set = {id(e) for e in test_entries}
    _ordered = sorted(
        list(enumerate(entries)),
        key=lambda iv: (0 if id(iv[1]) in _test_id_set else 1, iv[0]),
    )
    _seen: dict[str, int] = {}
    _options: dict[str, int] = {}
    for _i, _e in _ordered:
        _split = "test" if id(_e) in _test_id_set else "train"
        _base = f"[{_split}] {_user_prompt(_e)} · {_e.model.split('/')[-1][:20]}"
        _cnt = _seen.get(_base, 0)
        _seen[_base] = _cnt + 1
        _k = _base if _cnt == 0 else f"{_base} #{_cnt + 1}"
        _options[_k] = _i

    viz_entry_sel = mo.ui.dropdown(options=_options, label="Select entry")
    viz_entry_sel
    return (viz_entry_sel,)


@app.cell
def _(all_vecs, mo):
    _layer_opts = {str(l): l for l in all_vecs.layers()}
    viz_layer_sel = mo.ui.dropdown(
        options=_layer_opts,
        value=list(_layer_opts.keys())[len(_layer_opts) // 2] if _layer_opts else None,
        label="Layer",
    )
    viz_spec_sel = mo.ui.dropdown(
        options=all_vecs.specs(),
        value=all_vecs.specs()[0] if all_vecs.specs() else None,
        label="Spec",
    )
    viz_method_sel = mo.ui.dropdown(
        options=all_vecs.methods(),
        value=all_vecs.methods()[0] if all_vecs.methods() else None,
        label="Method",
    )
    viz_threshold = mo.ui.slider(
        start=0.05, stop=0.95, step=0.05, value=0.30,
        label="Cosine threshold", show_value=True,
    )
    viz_mode = mo.ui.radio(
        options={
            "Token scores": "token",
            "Sentence-wise scores": "sentence",
        },
        value="Token scores",
        label="Display mode",
    )
    viz_skip_prefix = mo.ui.checkbox(label="Hide system/user prefix", value=True)
    mo.vstack([
        mo.hstack([viz_layer_sel, viz_spec_sel, viz_method_sel], justify="start", gap=2),
        mo.hstack([viz_threshold, viz_mode], justify="start", gap=2),
        viz_skip_prefix,
    ])
    return (
        viz_layer_sel,
        viz_method_sel,
        viz_mode,
        viz_skip_prefix,
        viz_spec_sel,
        viz_threshold,
    )


@app.cell
def _(
    all_vecs,
    entries,
    ls,
    mo,
    model,
    np,
    viz_entry_sel,
    viz_layer_sel,
    viz_method_sel,
    viz_mode,
    viz_skip_prefix,
    viz_spec_sel,
    viz_threshold,
):
    import importlib as _il
    import little_steer.visualization.probe_view as _pv
    _il.reload(_pv)

    mo.stop(viz_entry_sel.value is None or viz_layer_sel.value is None)

    _entry = entries[viz_entry_sel.value]
    _layer = viz_layer_sel.value
    _spec = viz_spec_sel.value
    _method = viz_method_sel.value

    _vecs_at_layer = all_vecs.filter(layer=_layer, spec=_spec, method=_method)
    mo.stop(not _vecs_at_layer.vectors, mo.md("⚠️ No vectors for this combination."))

    _det = ls.get_multilabel_token_scores(model, _entry, _vecs_at_layer, layer=_layer)
    mo.stop(_det is None, mo.md("⚠️ Could not extract token scores for this entry."))

    _, _msg_offsets = model.format_messages_with_offsets(_entry.messages)
    _role_names = {
        "system": "System", "user": "User",
        "reasoning": "Reasoning", "assistant": "Response",
    }
    _section_markers: dict[int, str] = {}
    for _msg_idx, _msg in enumerate(_entry.messages):
        _char_offset = _msg_offsets.get(_msg_idx)
        if _char_offset is None:
            continue
        for _tok_i, (_cs, _ce) in enumerate(_det.token_char_spans):
            if _ce > 0 and _cs >= _char_offset:
                _section_markers[_tok_i] = _role_names.get(_msg["role"], _msg["role"])
                break

    _start_tok = 0
    if viz_skip_prefix.value:
        _reasoning_keys = [k for k, v in _section_markers.items()
                           if v in ("Reasoning", "Response")]
        if _reasoning_keys:
            _start_tok = min(_reasoning_keys)

    def _get_sentence_spans(tokens):
        _spans = []
        _start = 0
        for _i, _tok in enumerate(tokens):
            if "\n" in _tok or (_tok.strip() and _tok.strip()[-1] in (".", "!", "?")):
                _spans.append(ls.TokenSpan(token_start=_start, token_end=_i + 1, labels=[]))
                _start = _i + 1
        if _start < len(tokens):
            _spans.append(ls.TokenSpan(token_start=_start, token_end=len(tokens), labels=[]))
        return _spans

    def _dedup(det_obj):
        _seen_lbls: set[str] = set()
        _u_labels: list[str] = []
        _col_idxs: list[int] = []
        for _ii, _lbl2 in enumerate(det_obj.labels):
            if _lbl2 not in _seen_lbls:
                _seen_lbls.add(_lbl2)
                _u_labels.append(_lbl2)
                _col_idxs.append(_ii)
        return _u_labels, det_obj.scores[:, _col_idxs]

    _u_labels, _scores = _dedup(_det)

    _gt_scores = np.zeros((len(_det.tokens), len(_u_labels)), dtype=np.float32)
    _lbl_to_idx = {l: i for i, l in enumerate(_u_labels)}
    for _ts in _det.token_spans:
        for _lbl3 in _ts.labels:
            if _lbl3 in _lbl_to_idx:
                _gt_scores[
                    _ts.token_start:min(_ts.token_end, len(_det.tokens)),
                    _lbl_to_idx[_lbl3],
                ] = 1.0

    def _mk_html(det_obj, labels, scores_arr, norm=False):
        _toks = det_obj.tokens[_start_tok:]
        _cspans = det_obj.token_char_spans[_start_tok:]
        _sc = scores_arr[_start_tok:]
        _markers = {k - _start_tok: v for k, v in _section_markers.items() if k >= _start_tok}
        _spans = (
            _get_sentence_spans(_toks)
            if viz_mode.value == "sentence"
            else [
                ls.TokenSpan(
                    token_start=max(0, ts.token_start - _start_tok),
                    token_end=ts.token_end - _start_tok,
                    labels=ts.labels,
                )
                for ts in det_obj.token_spans
                if ts.token_end > _start_tok and ts.token_start - _start_tok < len(_toks)
            ]
        )
        return _pv.render_probe_detection_html(
            tokens=_toks, token_char_spans=_cspans,
            scores=_sc, labels=labels,
            formatted_text=det_obj.formatted_text, token_spans=_spans,
            threshold=viz_threshold.value, mode=viz_mode.value,
            show_ground_truth=False, normalize_scores=norm,
            section_markers=_markers, show_legend=False, show_header=False,
        )

    _html_gt = _mk_html(_det, _u_labels, _gt_scores, norm=False)
    _html_vec = _mk_html(_det, _u_labels, _scores, norm=True)
    _legend = mo.Html(_pv.legend_html(_u_labels, viz_threshold.value))
    _user_msg = next((m["content"] for m in _entry.messages if m["role"] == "user"), "")
    _panels = [mo.vstack([mo.md("#### Ground truth"), mo.Html(_html_gt)])]
    _panels.append(mo.vstack([mo.md("#### Vector detection"), mo.Html(_html_vec)]))

    mo.vstack([
        mo.md(f"### {_user_msg[:100].replace(chr(10), ' ')}{'…' if len(_user_msg) > 100 else ''}"),
        mo.md(f"**Model:** `{_entry.model.split('/')[-1]}` · **Layer:** `{_layer}` · "
              f"**Spec:** `{_spec}` · **Method:** `{_method}` · **Mode:** `{viz_mode.value}`"),
        _legend,
        mo.hstack(_panels, justify="start", gap=4),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## § 8 — Steering playground

    Inject a chosen steering vector during generation and compare to the baseline.
    **Alpha sweep** generates at multiple strengths in one run so you can see the
    effect curve without needing an LLM judge.
    """)
    return


@app.cell
def _(all_vecs, mo):
    _layers = all_vecs.layers()
    _mid_layer = _layers[len(_layers) // 2] if _layers else None

    steer_label_sel = mo.ui.dropdown(
        options=all_vecs.labels(),
        value=all_vecs.labels()[0] if all_vecs.labels() else None,
        label="Behavior",
    )
    steer_method_sel = mo.ui.dropdown(
        options=all_vecs.methods(),
        value=all_vecs.methods()[0] if all_vecs.methods() else None,
        label="Method",
    )
    steer_layer_sel = mo.ui.dropdown(
        options={str(l): l for l in _layers},
        value=str(_mid_layer) if _mid_layer is not None else None,
        label="Layer",
    )
    steer_spec_sel = mo.ui.dropdown(
        options=all_vecs.specs(),
        value=all_vecs.specs()[0] if all_vecs.specs() else None,
        label="Readout spec (which token positions were used to build the vector)",
    )
    steer_alpha = mo.ui.slider(start=-30.0, stop=30.0, step=0.1, value=10.0, label="Alpha (negative = steer away)", show_value=True)
    steer_max_tokens = mo.ui.slider(start=20, stop=300, step=10, value=120, label="Max new tokens", show_value=True)

    steer_multi_mode = mo.ui.checkbox(value=False, label="Steer towards multiple behaviours simultaneously")
    steer_label_2 = mo.ui.dropdown(options=all_vecs.labels(), value=all_vecs.labels()[1] if len(all_vecs.labels()) > 1 else None, label="Behavior 2")
    steer_alpha_2 = mo.ui.slider(start=-30.0, stop=30.0, step=0.1, value=10.0, label="Alpha 2", show_value=True)
    steer_label_3 = mo.ui.dropdown(options=all_vecs.labels(), value=all_vecs.labels()[2] if len(all_vecs.labels()) > 2 else None, label="Behavior 3")
    steer_alpha_3 = mo.ui.slider(start=-30.0, stop=30.0, step=0.1, value=10.0, label="Alpha 3", show_value=True)
    steer_normalize_vecs = mo.ui.checkbox(value=False, label="Normalize vectors to unit length before combining (equalizes scale across behaviours)")

    # Keys = displayed label, values = code string returned by .value
    steer_phase = mo.ui.dropdown(
        options={
            "Response only":        "response_only",
            "All tokens":           "all",
            "Prompt only (KV cache effect)":     "prompt_only",
            "First N tokens":       "n_tokens",
            "First N sentences":    "n_sentences",
        },
        value="Response only",
        label="Steering phase",
    )
    steer_n_tokens = mo.ui.slider(start=1, stop=200, step=1, value=20, label="N (tokens or sentences — for First N phases)", show_value=True)

    steer_bleed = mo.ui.slider(start=0, stop=5, step=1, value=0, label="Layer bleed ±", show_value=True)

    steer_compare_layers = mo.ui.multiselect(
        options={str(l): l for l in _layers},
        value=[],
        label="Compare: additional layers to steer at (side-by-side with selected layer above)",
    )

    steer_prompt = mo.ui.text(
        value="How can I make sure no one finds out what I did?",
        label="User prompt",
        full_width=True,
    )
    steer_btn = mo.ui.run_button(label="Generate baseline + steered")
    steer_sweep_btn = mo.ui.run_button(label="Alpha sweep (5 values from 0 to alpha)")
    steer_compare_btn = mo.ui.run_button(label="Generate all selected layers for comparison")

    mo.vstack([
        mo.hstack([steer_label_sel, steer_method_sel, steer_layer_sel, steer_spec_sel], justify="start", gap=2),
        mo.hstack([steer_multi_mode], justify="start"),
        steer_prompt,
        mo.hstack([steer_alpha, steer_max_tokens], justify="start", gap=2),
        mo.hstack([steer_phase, steer_n_tokens, steer_bleed], justify="start", gap=2),
        mo.hstack([steer_btn, steer_sweep_btn], justify="start", gap=2),
        mo.md("---"),
        mo.md("**Layer comparison** — select additional layers, then click *Generate all selected layers* to see the same steering at different layers side-by-side:"),
        mo.hstack([steer_compare_layers, steer_compare_btn], justify="start", gap=2),
    ])
    return (
        steer_alpha,
        steer_alpha_2,
        steer_alpha_3,
        steer_bleed,
        steer_btn,
        steer_compare_btn,
        steer_compare_layers,
        steer_label_2,
        steer_label_3,
        steer_label_sel,
        steer_layer_sel,
        steer_max_tokens,
        steer_method_sel,
        steer_multi_mode,
        steer_n_tokens,
        steer_normalize_vecs,
        steer_phase,
        steer_prompt,
        steer_spec_sel,
        steer_sweep_btn,
    )


@app.cell(hide_code=True)
def _(
    mo,
    steer_alpha_2,
    steer_alpha_3,
    steer_label_2,
    steer_label_3,
    steer_multi_mode,
    steer_normalize_vecs,
):
    if steer_multi_mode.value:
        mo.vstack([
            mo.md("**Additional behaviours to steer simultaneously:**"),
            mo.hstack([steer_label_2, steer_alpha_2], justify="start", gap=2),
            mo.hstack([steer_label_3, steer_alpha_3], justify="start", gap=2),
            steer_normalize_vecs,
        ])
    return


@app.cell(hide_code=True)
def _(all_vecs):
    # Mutable cache: resets whenever all_vecs changes (new vectors built).
    # Stores the last baseline text so we skip re-generating it when only alpha changes.
    _av_sig = id(all_vecs)
    steer_baseline_memo: dict = {}
    return (steer_baseline_memo,)


@app.cell
def _(
    all_vecs,
    ls,
    mo,
    model,
    steer_alpha,
    steer_alpha_2,
    steer_alpha_3,
    steer_baseline_memo: dict,
    steer_bleed,
    steer_btn,
    steer_label_2,
    steer_label_3,
    steer_label_sel,
    steer_layer_sel,
    steer_max_tokens,
    steer_method_sel,
    steer_multi_mode,
    steer_n_tokens,
    steer_normalize_vecs,
    steer_phase,
    steer_prompt,
    steer_spec_sel,
):
    mo.stop(not steer_btn.value or steer_label_sel.value is None)

    import torch as _tc

    _method  = steer_method_sel.value
    _spec    = steer_spec_sel.value
    _c_layer = steer_layer_sel.value
    _alpha   = steer_alpha.value
    _phase   = steer_phase.value
    _bleed   = steer_bleed.value

    def _get_vec(label, layer):
        _vs = all_vecs.filter(label=label, method=_method, layer=layer, spec=_spec)
        return _vs.vectors[0] if _vs.vectors else None

    def _nearest_vec(label, target_layer):
        _ls2 = all_vecs.filter(label=label, method=_method, spec=_spec).layers()
        if not _ls2:
            return None
        _near = min(_ls2, key=lambda l: abs(l - target_layer))
        return _get_vec(label, _near)

    _vec_primary = _get_vec(steer_label_sel.value, _c_layer)
    mo.stop(_vec_primary is None, mo.md("⚠️ No steering vector for that behavior / method / layer / readout."))

    _labels_alphas = [(steer_label_sel.value, _alpha)]
    if steer_multi_mode.value:
        if steer_label_2.value:
            _labels_alphas.append((steer_label_2.value, steer_alpha_2.value))
        if steer_label_3.value:
            _labels_alphas.append((steer_label_3.value, steer_alpha_3.value))

    _all_avail = sorted(all_vecs.layers())
    _steer_layers = sorted({_c_layer} | {l for l in _all_avail if 0 < abs(l - _c_layer) <= _bleed}) if _bleed > 0 else [_c_layer]

    _specs_list = []
    for _lbl, _a in _labels_alphas:
        for _sl in _steer_layers:
            _v = _nearest_vec(_lbl, _sl) if _bleed > 0 else _get_vec(_lbl, _sl)
            if _v is None:
                continue
            _raw = _v.vector.float()
            if steer_normalize_vecs.value:
                _raw = _raw / (_raw.norm() + 1e-8)
            _eff_alpha = _a * (1.0 - 0.5 * abs(_sl - _c_layer) / max(_bleed, 1)) if _sl != _c_layer else _a
            _specs_list.append((_raw, _sl, _eff_alpha))

    mo.stop(not _specs_list, mo.md("⚠️ No valid steering vectors found after expansion."))

    _response_only = _phase == "response_only"
    _prompt_only = _phase == "prompt_only"
    _n_tokens = steer_n_tokens.value if _phase == "n_tokens" else None
    _n_sentences = steer_n_tokens.value if _phase == "n_sentences" else None
    _msgs = [{"role": "user", "content": steer_prompt.value}]

    _bkey = (steer_prompt.value, steer_max_tokens.value)
    if steer_baseline_memo.get("key") != _bkey:
        with mo.status.spinner("Generating baseline…"):
            _bt = ls.steered_generate(model, _msgs, max_new_tokens=steer_max_tokens.value, do_sample=False)
        steer_baseline_memo["key"] = _bkey
        steer_baseline_memo["text"] = _bt
    _baseline_text = steer_baseline_memo["text"]

    _steer_label = " + ".join(f"{lbl} (α={a:.1f})" for lbl, a in _labels_alphas)
    if _bleed > 0:
        _steer_label += f" [bleed ±{_bleed}]"
    _phase_notes = {
        "response_only": "response only",
        "all": "all tokens",
        "prompt_only": "prompt only (KV cache effect)",
        "n_tokens": f"first {steer_n_tokens.value} tokens then off",
        "n_sentences": f"first {steer_n_tokens.value} sentences then off",
    }
    _phase_note = _phase_notes.get(_phase, _phase)

    _needs_mask = _n_tokens is not None or _n_sentences is not None

    def _highlight_steered(text: str, mask: list[bool]) -> "mo.Html":
        """Render token-level steering highlights for N-phase modes."""
        _tids = model.tokenizer(text, add_special_tokens=False)["input_ids"]
        _toks = [model.tokenizer.decode([t]) for t in _tids]
        _parts = []
        for _i, _tok in enumerate(_toks):
            _active = mask[_i] if _i < len(mask) else False
            _escaped = _tok.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            if _active:
                _parts.append(
                    f'<span style="background:rgba(230,200,100,0.45);border-radius:2px;">'
                    f'{_escaped}</span>'
                )
            else:
                _parts.append(_escaped)
        return mo.Html(
            '<div style="font-family:monospace;white-space:pre-wrap;font-size:13px;'
            'line-height:1.6;padding:8px;">'
            + "".join(_parts)
            + "</div>"
        )

    with mo.status.spinner("Generating steered…"):
        if len(_specs_list) == 1:
            _raw1, _sl1, _sa1 = _specs_list[0]
            _steer_result = ls.steered_generate(
                model, _msgs, steering_vec=_raw1, layer=_sl1,
                alpha=_sa1, response_only=_response_only,
                prompt_only=_prompt_only,
                n_steered_tokens=_n_tokens,
                n_steered_sentences=_n_sentences,
                max_new_tokens=steer_max_tokens.value, do_sample=False,
                return_steered_mask=_needs_mask,
            )
        else:
            _steer_result = ls.multi_steered_generate(
                model, _msgs, steering_specs=_specs_list,
                response_only=_response_only,
                max_new_tokens=steer_max_tokens.value, do_sample=False,
            )

    if _needs_mask and isinstance(_steer_result, tuple):
        _steered_text, _steered_mask = _steer_result
        _steered_display = _highlight_steered(_steered_text, _steered_mask)
    else:
        _steered_text = _steer_result if isinstance(_steer_result, str) else _steer_result[0]
        _steered_display = mo.callout(mo.md(_steered_text), kind="warn")

    mo.vstack([
        mo.md(f"**Prompt:** {steer_prompt.value}  ·  method={_method}  ·  layer={_c_layer}  ·  spec={_spec}"),
        mo.md(f"**Steering:** {_steer_label}  ·  phase={_phase_note}"),
        mo.md("_Yellow highlight = tokens generated while steering was active_") if _needs_mask else mo.md(""),
        mo.hstack([
            mo.vstack([mo.md("#### Baseline"), mo.callout(mo.md(_baseline_text), kind="neutral")]),
            mo.vstack([mo.md("#### Steered"), _steered_display]),
        ]),
    ])
    return


@app.cell
def _(
    all_vecs,
    ls,
    mo,
    model,
    np,
    steer_alpha,
    steer_label_sel,
    steer_layer_sel,
    steer_max_tokens,
    steer_method_sel,
    steer_n_tokens,
    steer_normalize_vecs,
    steer_phase,
    steer_prompt,
    steer_spec_sel,
    steer_sweep_btn,
):
    mo.stop(not steer_sweep_btn.value or steer_label_sel.value is None)

    import torch as _tc2

    _method3  = steer_method_sel.value
    _spec3    = steer_spec_sel.value
    _c_layer3 = steer_layer_sel.value
    _phase3   = steer_phase.value

    _vs3 = all_vecs.filter(label=steer_label_sel.value, method=_method3, layer=_c_layer3, spec=_spec3)
    mo.stop(not _vs3.vectors, mo.md("⚠️ No steering vector for that behavior/method/layer/readout."))
    _vec3 = _vs3.vectors[0]
    _raw3 = _vec3.vector.float()
    if steer_normalize_vecs.value:
        _raw3 = _raw3 / (_raw3.norm() + 1e-8)

    _response_only3 = _phase3 == "response_only"
    _prompt_only3 = _phase3 == "prompt_only"
    _n_tokens3 = steer_n_tokens.value if _phase3 == "n_tokens" else None
    _n_sentences3 = steer_n_tokens.value if _phase3 == "n_sentences" else None
    _msgs3 = [{"role": "user", "content": steer_prompt.value}]
    _alphas = [round(float(a), 1) for a in np.linspace(0.0, steer_alpha.value, 5)]
    _outputs = {}

    for _a in mo.status.progress_bar(_alphas, title="Alpha sweep…", completion_title="Done!"):
        _out = ls.steered_generate(
            model, _msgs3,
            steering_vec=_raw3 if _a > 0 else None,
            layer=_c_layer3 if _a > 0 else None,
            alpha=_a,
            response_only=_response_only3,
            prompt_only=_prompt_only3,
            n_steered_tokens=_n_tokens3,
            n_steered_sentences=_n_sentences3,
            max_new_tokens=steer_max_tokens.value, do_sample=False,
        )
        _outputs[_a] = _out

    _items = [
        mo.vstack([
            mo.md(f"**α = {_a}**"),
            mo.callout(mo.md(_outputs[_a]), kind="neutral" if _a == 0 else "warn"),
        ])
        for _a in _alphas
    ]
    _rows = []
    for _i in range(0, len(_items), 3):
        _rows.append(mo.hstack(_items[_i:_i+3], gap=2))

    mo.vstack([
        mo.md(f"**Alpha sweep** · {steer_label_sel.value} · {_method3} · L{_c_layer3} · {_spec3}"),
        *_rows,
    ])
    return


@app.cell(hide_code=True)
def _(
    all_vecs,
    ls,
    mo,
    model,
    steer_alpha,
    steer_baseline_memo: dict,
    steer_compare_btn,
    steer_compare_layers,
    steer_label_sel,
    steer_layer_sel,
    steer_max_tokens,
    steer_method_sel,
    steer_n_tokens,
    steer_normalize_vecs,
    steer_phase,
    steer_prompt,
    steer_spec_sel,
):
    mo.stop(not steer_compare_btn.value or steer_label_sel.value is None)
    mo.stop(len(steer_compare_layers.value) == 0, mo.md("⚠️ Select at least one additional layer above."))

    import torch as _tc4

    _method4  = steer_method_sel.value
    _spec4    = steer_spec_sel.value
    _phase4   = steer_phase.value
    _response_only4 = _phase4 == "response_only"
    _prompt_only4 = _phase4 == "prompt_only"
    _n_tokens4 = steer_n_tokens.value if _phase4 == "n_tokens" else None
    _n_sentences4 = steer_n_tokens.value if _phase4 == "n_sentences" else None
    _msgs4 = [{"role": "user", "content": steer_prompt.value}]

    _compare_layers = sorted({steer_layer_sel.value} | set(steer_compare_layers.value))

    _bkey4 = (steer_prompt.value, steer_max_tokens.value)
    if steer_baseline_memo.get("key") != _bkey4:
        with mo.status.spinner("Generating baseline…"):
            _bt4 = ls.steered_generate(model, _msgs4, max_new_tokens=steer_max_tokens.value, do_sample=False)
        steer_baseline_memo["key"] = _bkey4
        steer_baseline_memo["text"] = _bt4
    _baseline_text4 = steer_baseline_memo["text"]

    _layer_outputs = {}
    for _cl in mo.status.progress_bar(_compare_layers, title="Generating per layer…", completion_title="Done!"):
        _vs4 = all_vecs.filter(label=steer_label_sel.value, method=_method4, layer=_cl, spec=_spec4)
        if not _vs4.vectors:
            _layer_outputs[_cl] = f"_(no vector at layer {_cl})_"
            continue
        _r4 = _vs4.vectors[0].vector.float()
        if steer_normalize_vecs.value:
            _r4 = _r4 / (_r4.norm() + 1e-8)
        _layer_outputs[_cl] = ls.steered_generate(
            model, _msgs4, steering_vec=_r4, layer=_cl,
            alpha=steer_alpha.value, response_only=_response_only4,
            prompt_only=_prompt_only4,
            n_steered_tokens=_n_tokens4,
            n_steered_sentences=_n_sentences4,
            max_new_tokens=steer_max_tokens.value, do_sample=False,
        )

    _cols4 = [mo.vstack([mo.md("**Baseline**"), mo.callout(mo.md(_baseline_text4), kind="neutral")])]
    for _cl in _compare_layers:
        _cols4.append(mo.vstack([
            mo.md(f"**Layer {_cl}** (α={steer_alpha.value:.1f})"),
            mo.callout(mo.md(_layer_outputs[_cl]), kind="warn"),
        ]))

    _rows4 = []
    for _i in range(0, len(_cols4), 3):
        _rows4.append(mo.hstack(_cols4[_i:_i+3], gap=2))

    mo.vstack([
        mo.md(f"**Layer comparison** · {steer_label_sel.value} · {_method4} · {_spec4}"),
        *_rows4,
    ])
    return


if __name__ == "__main__":
    app.run()
