# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.23.2",
# ]
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium", css_file="")


@app.cell
def _():
    import sys
    from pathlib import Path
    import json
    import marimo as mo
    import polars as pl
    import altair as alt

    @alt.theme.register("everforest_dark", enable=True)
    def _ef_dark():
        _bg = "#232a2e"
        _card = "#343f44"
        _border = "#475258"
        _fg = "#d3c6aa"
        return {
            "background": _bg,
            "config": {
                "view": {"fill": _card, "stroke": _border},
                "title": {"color": _fg, "subtitleColor": _fg, "fontSize": 14, "anchor": "start"},
                "axis": {
                    "labelColor": _fg, "titleColor": _fg,
                    "gridColor": _border, "tickColor": _border, "domainColor": _border,
                },
                "header": {"labelColor": _fg, "titleColor": _fg},
                "legend": {"labelColor": _fg, "titleColor": _fg, "padding": 10},
                "range": {
                    "category": ["#a7c080", "#7fbbb3", "#dbbc7f", "#e67e80", "#d699b6", "#83c092", "#e69875"],
                },
                "mark": {"color": "#a7c080"},
                "text": {"color": _fg},
            },
        }

    _nb_dir = Path(__file__).parent
    _sentence_dir = _nb_dir.parent
    if str(_sentence_dir) not in sys.path:
        sys.path.insert(0, str(_sentence_dir))

    from iaa.compute import (
        MODES,
        _interval_intersect,
        compute_agreement_matrix,
        compute_krippendorff_alpha,
        compute_substitution_matrix,
        filter_stable_labels,
        load_iaa_data,
        normalize_labels,
    )


    return (
        MODES,
        Path,
        alt,
        compute_agreement_matrix,
        compute_krippendorff_alpha,
        compute_substitution_matrix,
        filter_stable_labels,
        json,
        load_iaa_data,
        mo,
        pl,
    )


@app.cell
def _(mo):
    mo.md("""
    # IAA — Sentence-Level Annotation Agreement

    How consistently do annotators (human and AI) label the same reasoning traces?
    Use the controls below to select annotators and comparison mode.
    """)
    return


@app.cell
def _(Path, json, load_iaa_data):
    _data_path = Path(__file__).parents[3] / "data" / "dataset.jsonl"
    _wo_path = Path(__file__).parents[1] / "work_order_iaa.json"

    entry_data, annotator_counts = load_iaa_data(_data_path)

    with open(_wo_path) as _f:
        _wo = json.load(_f)

    iaa_ids = frozenset(e["id"] for e in _wo["flat_order"])
    iaa_meta = {k: v for k, v in _wo.items() if k not in ("flat_order", "per_file", "pairs")}
    return annotator_counts, entry_data, iaa_ids, iaa_meta


@app.cell(hide_code=True)
def _(Path, iaa_ids, json):
    _texts_path = Path(__file__).parents[3] / "data" / "dataset.jsonl"
    entry_texts: dict[str, str] = {}
    with open(_texts_path) as _ft:
        for _tline in _ft:
            _tline = _tline.strip()
            if not _tline:
                continue
            _te = json.loads(_tline)
            if _te.get("id") not in iaa_ids:
                continue
            _tmsgs = _te.get("messages") or []
            _tridx = next((i for i, m in enumerate(_tmsgs) if m.get("role") == "reasoning"), None)
            if _tridx is None:
                continue
            _tgh = (_te.get("metadata") or {}).get("generation_hash", "")
            _tk = f"{_te.get('id', '')}::{_tgh}"
            entry_texts[_tk] = _tmsgs[_tridx].get("content", "")
    return (entry_texts,)


@app.cell
def _(annotator_counts, entry_data, iaa_ids, pl):
    iaa_entry_data = {
        key: by_ann
        for key, by_ann in entry_data.items()
        if key.split("::")[0] in iaa_ids
    }

    _rows = []
    for _name, _total in annotator_counts.items():
        _iaa_count = sum(1 for by_ann in iaa_entry_data.values() if _name in by_ann)
        _rows.append({"annotator": _name, "total_entries": _total, "iaa_entries": _iaa_count})

    annotator_df = pl.DataFrame(_rows).sort("iaa_entries", descending=True)
    annotator_names = annotator_df["annotator"].to_list()
    return annotator_df, annotator_names, iaa_entry_data


@app.cell
def _(annotator_df, iaa_entry_data, iaa_ids, iaa_meta, mo):
    mo.md(f"""
    ## Dataset Overview

    | Metric | Value |
    |--------|-------|
    | IAA work order size | {len(iaa_ids)} entries |
    | IAA entries with ≥1 label run | {len(iaa_entry_data)} |
    | Total annotators (full dataset) | {len(annotator_df)} |
    | Strategy | `{iaa_meta.get("strategy", "?")}` |
    | Generated at | `{iaa_meta.get("generated_at", "?")}` |
    | Datasets | {", ".join(iaa_meta.get("datasets", []))} |
    """)
    return


@app.cell
def _(annotator_df, mo):
    _rows = "\n".join(
        f"| `{r['annotator']}` | **{r['iaa_entries']}** / {r['total_entries']} | {'█' * min(20, max(1, int(20 * r['iaa_entries'] / max(1, r['total_entries'])))) if r['total_entries'] > 0 else ''} |"
        for r in annotator_df.iter_rows(named=True)
    )
    mo.md(f"""
    ### Annotator Coverage
    *Showing counts and relative coverage of entries assigned for IAA.*

    | Annotator | Entries (IAA / Total) | Coverage |
    |:----------|:----------------------|:---------|
    {_rows}
    """)
    return


@app.cell
def _(MODES, annotator_names, mo):
    _defaults = [
        name for name in annotator_names
        if name in {
            "gpt-5.4-mini",
            "gpt-5.4-mini_pass2",
            "gemini-3.1-flash-lite-preview",
            "gemini-3.1-flash-lite-preview_pass2",
            "human_antstudent",
        }
    ]

    annotator_multiselect = mo.ui.multiselect(
        options=annotator_names,
        value=_defaults,
        label="Annotators (≥ 2 required)",
        full_width=True,
    )

    mode_selector = mo.ui.radio(
        options={m.replace("_", " "): m for m in MODES},
        value="score",
        label="Comparison mode",
    )

    iaa_only_toggle = mo.ui.checkbox(label="IAA entries only", value=True)
    stable_filter_toggle = mo.ui.checkbox(label="Stable labels only (pass1∩pass2)", value=False)
    return (
        annotator_multiselect,
        iaa_only_toggle,
        mode_selector,
        stable_filter_toggle,
    )


@app.cell
def _(
    annotator_multiselect,
    iaa_only_toggle,
    mo,
    mode_selector,
    stable_filter_toggle,
):
    mo.vstack([
        mo.md("## Controls"),
        mo.hstack([mode_selector, iaa_only_toggle, stable_filter_toggle], gap="2rem"),
        mo.md(f"**Current Comparison Mode:** `{mode_selector.value.replace('_', ' ')}`"),
        mo.md(
            "*Changes what is compared: **score** (−1/0/1), "
            "**primary label** (first label only), or **label set** (all labels). "
            "All charts below update automatically.*"
        ),
        annotator_multiselect,
    ])
    return


@app.cell
def _(annotator_multiselect):
    selected_annotators = list(annotator_multiselect.value)
    return (selected_annotators,)


@app.cell(hide_code=True)
def _(
    entry_data,
    filter_stable_labels,
    iaa_entry_data,
    iaa_only_toggle,
    stable_filter_toggle,
):
    _base = iaa_entry_data if iaa_only_toggle.value else entry_data
    active_data = filter_stable_labels(_base) if stable_filter_toggle.value else _base
    return (active_data,)


@app.cell(hide_code=True)
def _(
    entry_data,
    filter_stable_labels,
    iaa_entry_data,
    iaa_only_toggle,
    mo,
    stable_filter_toggle,
):
    mo.stop(not stable_filter_toggle.value)

    _base = iaa_entry_data if iaa_only_toggle.value else entry_data
    _filtered = filter_stable_labels(_base)

    _all_anns = set()
    for _by_ann in _base.values():
        _all_anns.update(_by_ann.keys())

    _pairs = [(j, j + "_pass2") for j in sorted(_all_anns) if j + "_pass2" in _all_anns]

    _rows_md = []
    for _j1, _j2 in _pairs:
        _before = sum(len(_by_ann[_j1]) for _by_ann in _base.values() if _j1 in _by_ann)
        _after  = sum(len(_by_ann[_j1]) for _by_ann in _filtered.values() if _j1 in _by_ann)
        _pct = 100 * _after / _before if _before else 0
        _rows_md.append(f"| `{_j1}` | {_before:,} | {_after:,} | **{_pct:.1f}%** |")

    mo.callout(mo.md(
        "**Stable-label filter active** — spans retained per model (score + label consistency):\n\n"
        "| Annotator | Spans before | Spans after | Retained |\n"
        "|:----------|-------------:|------------:|---------:|\n"
        + "\n".join(_rows_md)
    ), kind="info")
    return


@app.cell
def _(
    active_data,
    compute_agreement_matrix,
    mode_selector,
    selected_annotators,
):
    if len(selected_annotators) >= 2:
        agreement_result = compute_agreement_matrix(
            active_data, selected_annotators, mode_selector.value
        )
    else:
        agreement_result = None
    return (agreement_result,)


@app.cell
def _(agreement_result, alt, mo, pl):
    mo.stop(
        agreement_result is None,
        mo.md("⚠️ Select at least **2 annotators** to compute the agreement matrix."),
    )

    _rows = []
    for _key, _val in agreement_result["pairs"].items():
        _a, _b = _key.split("||")
        _kappa = _val.get("kappa")
        _rows.append({
            "annotator_a": _a,
            "annotator_b": _b,
            "kappa": float(_kappa) if _kappa is not None else float("nan"),
            "pct": float(_val["pct"]) if _val.get("pct") is not None else float("nan"),
            "shared": _val.get("shared"),
        })

    _df = pl.DataFrame(_rows)

    _base = alt.Chart(_df).encode(
        x=alt.X("annotator_a:N", title=None, axis=alt.Axis(labelAngle=-45, labelLimit=200)),
        y=alt.Y("annotator_b:N", title=None, axis=alt.Axis(labelLimit=200)),
    )

    _heat = _base.mark_rect().encode(
        color=alt.Color(
            "kappa:Q",
            scale=alt.Scale(scheme="redyellowgreen", domain=[-1, 1]),
            legend=alt.Legend(title="κ"),
        ),
        tooltip=[
            alt.Tooltip("annotator_a:N", title="A"),
            alt.Tooltip("annotator_b:N", title="B"),
            alt.Tooltip("kappa:Q", title="Cohen's κ", format=".3f"),
            alt.Tooltip("pct:Q", title="% agree", format=".2%"),
            alt.Tooltip("shared:Q", title="Shared entries"),
        ],
    )

    _text = _base.mark_text(fontSize=10, fontWeight="bold").encode(
        text=alt.Text("kappa:Q", format=".2f"),
        color=alt.condition(
            (alt.datum.kappa > 0.3) & (alt.datum.kappa < 0.8),
            alt.value("#2b3339"),
            alt.value("#d3c6aa"),
        ),
    )

    kappa_chart = mo.ui.altair_chart(
        (_heat + _text).properties(
            title=f"Cohen's κ — {agreement_result['mode']} mode",
            width=500,
            height=420,
        )
    )

    mo.vstack([
        mo.md(
            "### Pairwise agreement (Cohen's κ)\n"
            "κ = 1 is perfect agreement; κ = 0 is chance; κ < 0 is systematic disagreement.\n\n"
            "*Click a cell to explore that pair's annotation disagreements in real text below.*"
        ),
        kappa_chart,
    ])
    return (kappa_chart,)


@app.cell(hide_code=True)
def _(kappa_chart, mo):
    mo.stop(
        len(kappa_chart.value) == 0,
        mo.callout(
            mo.md("**Click a cell** in the κ heatmap above to explore that pair's real annotation disagreements."),
            kind="info",
        ),
    )

    _v = kappa_chart.value
    try:
        _row = dict(zip(_v.columns, _v.row(0)))  # polars
    except AttributeError:
        _row = _v.iloc[0].to_dict()              # pandas

    span_pair_a = _row["annotator_a"]
    span_pair_b = _row["annotator_b"]
    return span_pair_a, span_pair_b


@app.cell(hide_code=True)
def _(active_data, entry_texts: dict[str, str], mo, span_pair_a, span_pair_b):
    _ep_entries = []
    for _ep_key, _ep_by_ann in active_data.items():
        if span_pair_a not in _ep_by_ann or span_pair_b not in _ep_by_ann:
            continue
        if _ep_key not in entry_texts:
            continue
        _ep_pairs = list(_interval_intersect(_ep_by_ann[span_pair_a], _ep_by_ann[span_pair_b]))
        _ep_ndis = sum(1 for _, _, _sa, _sb in _ep_pairs if set(_sa[3]) != set(_sb[3]))
        _ep_ntot = len(_ep_pairs)
        if _ep_ntot == 0:
            continue
        _ep_entries.append({
            "key": _ep_key,
            "label": f"{_ep_key.split('::')[0]}  ({_ep_ndis}/{_ep_ntot} spans differ)",
            "n_dis": _ep_ndis,
        })

    _ep_entries.sort(key=lambda x: -x["n_dis"])

    entry_picker = mo.ui.dropdown(
        options={e["label"]: e["key"] for e in _ep_entries},
        label=f"Entry — {len(_ep_entries)} shared between `{span_pair_a}` & `{span_pair_b}`",
        full_width=True,
    )

    mo.vstack([
        mo.md(f"### Span browser: `{span_pair_a}` vs `{span_pair_b}`\nSorted by number of disagreeing spans (most first). Hover highlights to see labels."),
        entry_picker,
    ])
    return (entry_picker,)


@app.cell(hide_code=True)
def _(
    active_data,
    entry_picker,
    entry_texts: dict[str, str],
    mo,
    span_pair_a,
    span_pair_b,
):
    mo.stop(entry_picker.value is None, mo.md("Select an entry above."))

    import html as _html_mod

    _br_key = entry_picker.value
    _br_text = entry_texts[_br_key]
    _br_sa = active_data[_br_key].get(span_pair_a, [])
    _br_sb = active_data[_br_key].get(span_pair_b, [])

    _bounds = sorted(set(
        [0, len(_br_text)]
        + [s[0] for s in _br_sa] + [s[1] for s in _br_sa]
        + [s[0] for s in _br_sb] + [s[1] for s in _br_sb]
    ))

    _html_parts = []
    for _bi in range(len(_bounds) - 1):
        _lo, _hi = _bounds[_bi], _bounds[_bi + 1]
        _seg = _html_mod.escape(_br_text[_lo:_hi])
        _a_here = [s for s in _br_sa if s[0] <= _lo and s[1] >= _hi]
        _b_here = [s for s in _br_sb if s[0] <= _lo and s[1] >= _hi]
        if not _a_here and not _b_here:
            _html_parts.append(_seg)
        else:
            _la = set(l for s in _a_here for l in s[3])
            _lb = set(l for s in _b_here for l in s[3])
            if _a_here and _b_here:
                _shared = _la & _lb
                _col = "rgba(219,188,127,0.45)" if _shared else "rgba(230,126,128,0.45)"
                _tip = f"A ({span_pair_a}): {', '.join(sorted(_la))}  |  B ({span_pair_b}): {', '.join(sorted(_lb))}"
            elif _a_here:
                _col = "rgba(127,187,179,0.45)"
                _tip = f"{span_pair_a}: {', '.join(sorted(_la))}"
            else:
                _col = "rgba(167,192,128,0.45)"
                _tip = f"{span_pair_b}: {', '.join(sorted(_lb))}"
            _html_parts.append(
                f'<mark style="background:{_col};border-radius:3px;padding:0 1px;cursor:default"'
                f' title="{_html_mod.escape(_tip)}">{_seg}</mark>'
            )

    _legend_html = (
        "<div style='display:flex;gap:1.2rem;flex-wrap:wrap;margin-bottom:0.7rem;"
        "font-size:0.78rem;font-family:sans-serif;color:#d3c6aa;align-items:center'>"
        "<span><mark style='background:rgba(219,188,127,0.5);padding:1px 6px;border-radius:3px'>agreed (shared label)</mark></span>"
        "<span><mark style='background:rgba(230,126,128,0.5);padding:1px 6px;border-radius:3px'>disagreed (no shared label)</mark></span>"
        f"<span><mark style='background:rgba(127,187,179,0.5);padding:1px 6px;border-radius:3px'>only {span_pair_a}</mark></span>"
        f"<span><mark style='background:rgba(167,192,128,0.5);padding:1px 6px;border-radius:3px'>only {span_pair_b}</mark></span>"
        "</div>"
    )

    mo.Html(
        "<div style='font-family:sans-serif'>"
        + _legend_html
        + "<div style='font-family:monospace;font-size:0.82rem;line-height:1.8;white-space:pre-wrap;"
        "color:#d3c6aa;background:#2d3b41;padding:1.2rem;border-radius:8px;"
        "max-height:540px;overflow-y:auto;border:1px solid #475258'>"
        + "".join(_html_parts)
        + "</div></div>"
    )
    return


@app.cell
def _(
    active_data,
    compute_krippendorff_alpha,
    mode_selector,
    selected_annotators,
):
    if len(selected_annotators) >= 2:
        alpha_result = compute_krippendorff_alpha(
            active_data, selected_annotators, mode_selector.value
        )
    else:
        alpha_result = None
    return (alpha_result,)


@app.cell
def _(alpha_result, mo, mode_selector, selected_annotators):
    mo.stop(alpha_result is None, mo.md("⚠️ Select at least **2 annotators** to compute α."))

    _alpha = alpha_result.get("alpha")
    _n = alpha_result.get("n_coincidences", 0)
    _chars = alpha_result.get("total_chars", 0)
    _alpha_str = f"{_alpha:.4f}" if _alpha is not None else "N/A"

    if _alpha is None:
        _interp = "insufficient data"
    elif _alpha >= 0.8:
        _interp = "substantial"
    elif _alpha >= 0.6:
        _interp = "moderate"
    elif _alpha >= 0.4:
        _interp = "fair"
    elif _alpha >= 0.2:
        _interp = "slight"
    else:
        _interp = "poor / chance"

    mo.md(f"""
    ## Overall agreement across all annotators (Krippendorff's α)

    Krippendorff's α is a single number that summarizes how much **all selected annotators agree**,
    corrected for chance. Unlike Cohen's κ, it handles any number of annotators and missing data.
    Range: −1 (systematic disagreement) → 0 (chance) → 1 (perfect agreement).

    | Metric | Value |
    |--------|-------|
    | **α** | **{_alpha_str}** ({_interp}) |
    | Mode | `{mode_selector.value}` |
    | Annotators | {len(selected_annotators)} |
    | Coincidence pairs | {_n:,} |

    Thresholds: **α ≥ 0.8** = strong agreement, **≥ 0.667** = acceptable for research use.
    With few human annotations, treat α as indicative — report alongside n.
    """)
    return


@app.cell
def _(active_data, pl, selected_annotators):
    _rows = []
    for _key, _by_ann in active_data.items():
        for _ann in selected_annotators:
            if _ann not in _by_ann:
                continue
            for _span in _by_ann[_ann]:
                _s, _e, _score, _labels = _span
                for _lbl in _labels:
                    _rows.append({
                        "annotator": _ann,
                        "label": _lbl,
                        "score": _score,
                        "chars": _e - _s,
                    })

    label_dist_df = (
        pl.DataFrame(_rows)
        if _rows
        else pl.DataFrame({
            "annotator": pl.Series([], dtype=pl.Utf8),
            "label": pl.Series([], dtype=pl.Utf8),
            "score": pl.Series([], dtype=pl.Int64),
            "chars": pl.Series([], dtype=pl.Int64),
        })
    )
    return (label_dist_df,)


@app.cell
def _(alt, label_dist_df, mo, pl):
    mo.stop(label_dist_df.is_empty(), mo.md("No label data for selected annotators."))

    _top20 = (
        label_dist_df
        .group_by("label")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(20)["label"]
        .to_list()
    )

    _dist = (
        label_dist_df
        .filter(pl.col("label").is_in(_top20))
        .group_by(["annotator", "label"])
        .agg(pl.len().alias("count"))
    )

    mo.vstack([
        mo.md(
            "### Label usage by annotator\n"
            "Which labels does each annotator use most? Large differences in bar lengths "
            "between annotators for the same label suggest one is more trigger-happy with it — "
            "a potential calibration issue."
        ),
        alt.Chart(_dist).mark_bar().encode(
            x=alt.X("count:Q", title="Span count"),
            y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
            color=alt.Color("annotator:N"),
            tooltip=["annotator:N", "label:N", "count:Q"],
        ).properties(title="Label distribution (top 20) by annotator", width=620, height=500),
    ])
    return


@app.cell
def _(alt, label_dist_df, mo, pl):
    mo.stop(label_dist_df.is_empty(), mo.md("No score data."))

    _score_dist = (
        label_dist_df
        .group_by(["annotator", "score"])
        .agg(pl.len().alias("count"))
        .with_columns(pl.col("score").cast(pl.Utf8))
    )

    alt.Chart(_score_dist).mark_bar().encode(
        x=alt.X("annotator:N", title=None, axis=alt.Axis(labelAngle=-30, labelLimit=200)),
        y=alt.Y("count:Q", title="Span count"),
        color=alt.Color(
            "score:N",
            scale=alt.Scale(
                domain=["-1", "0", "1"],
                range=["#e67e80", "#7fbbb3", "#a7c080"]  # red, blue, green (Everforest)
            ),
            legend=alt.Legend(title="Score"),
        ),
        tooltip=["annotator:N", "score:N", "count:Q"],
    ).properties(title="Score distribution by annotator", width=520, height=300)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## How confident is the judge? (Logprob analysis)

    `gpt-5.4-mini` outputs a confidence score for each label token it assigns.
    Values near **0** = very certain; more negative = uncertain.
    A label the model consistently second-guesses is a candidate for merging or clarifying.

    *Only `gpt-5.4-mini` and `gpt-5.4-mini_pass2` record these scores.*
    """)
    return


@app.cell
def _(Path, iaa_ids, json):
    # Load logprobs for every IAA entry (no annotator filter — filter later).
    # Runs once; subsequent cells filter downstream.
    import math as _math

    _data_path = Path(__file__).parents[3] / "data" / "dataset.jsonl"
    _lp_rows = []

    with open(_data_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line:
                continue
            _e = json.loads(_line)
            if _e.get("id") not in iaa_ids:
                continue
            for _run in _e.get("label_runs", []):
                _judge = _run.get("judge_name", "")
                for _sa in _run.get("sentence_annotations", []):
                    _lp = _sa.get("label_logprobs")
                    if not _lp:
                        continue
                    for _lk, _vals in _lp.items():
                        if not _vals:
                            continue
                        _mean_lp = sum(_vals) / len(_vals)
                        _lp_rows.append({
                            "judge": _judge,
                            "label": _lk,
                            "mean_logprob": round(_mean_lp, 4),
                            "confidence": round(_math.exp(max(_mean_lp, -15)), 6),
                        })

    import polars as _pl2
    logprob_all_df = (
        _pl2.DataFrame(_lp_rows) if _lp_rows
        else _pl2.DataFrame({
            "judge": _pl2.Series([], dtype=_pl2.Utf8),
            "label": _pl2.Series([], dtype=_pl2.Utf8),
            "mean_logprob": _pl2.Series([], dtype=_pl2.Float64),
            "confidence": _pl2.Series([], dtype=_pl2.Float64),
        })
    )
    return (logprob_all_df,)


@app.cell
def _(logprob_all_df, pl, selected_annotators):
    _sel = set(selected_annotators)
    logprob_df = logprob_all_df.filter(pl.col("judge").is_in(_sel))
    return (logprob_df,)


@app.cell
def _(alt, logprob_df, mo, pl):
    mo.stop(logprob_df.is_empty(), mo.md("No logprob data for selected annotators. Only `gpt-5.4-mini` records logprobs."))

    # Aggregate: median and IQR of mean_logprob per (judge, label)
    _agg = (
        logprob_df
        .group_by(["judge", "label"])
        .agg([
            pl.col("mean_logprob").median().alias("median_lp"),
            pl.col("mean_logprob").quantile(0.25).alias("q25"),
            pl.col("mean_logprob").quantile(0.75).alias("q75"),
            pl.col("mean_logprob").min().alias("min_lp"),
            pl.col("mean_logprob").count().alias("n"),
        ])
        .sort("median_lp")
    )

    _label_order = _agg.group_by("label").agg(pl.col("median_lp").median()).sort("median_lp")["label"].to_list()

    _bars = alt.Chart(_agg).mark_bar(size=10).encode(
        y=alt.Y("label:N", sort=_label_order, title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X("q25:Q", title="Mean logprob"),
        x2=alt.X2("q75:Q"),
        color=alt.Color("judge:N", legend=alt.Legend(title="Judge")),
        tooltip=[
            "judge:N", "label:N",
            alt.Tooltip("median_lp:Q", title="Median logprob", format=".3f"),
            alt.Tooltip("q25:Q", title="Q25", format=".3f"),
            alt.Tooltip("q75:Q", title="Q75", format=".3f"),
            alt.Tooltip("n:Q", title="N sentences"),
        ],
    )

    _medians = alt.Chart(_agg).mark_tick(thickness=2, size=14, color="black").encode(
        y=alt.Y("label:N", sort=_label_order),
        x=alt.X("median_lp:Q"),
        tooltip=["judge:N", "label:N", alt.Tooltip("median_lp:Q", format=".3f")],
    )

    mo.vstack([
        mo.md(
            "### Confidence per label\n"
            "Each bar spans the middle 50% of scores (IQR); the tick is the median. "
            "Labels with bars far left (very negative) = the judge is consistently uncertain. "
            "Labels near 0 = the judge always assigns them with high certainty."
        ),
        (_bars + _medians).properties(width=560, height=520),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Pass1 vs Pass2: Label Stability

    Any judge that ran twice (`X` and `X_pass2`) is automatically detected and compared here —
    no configuration needed when you add new labelers.

    This checks how consistent a judge is **with itself** — a measure of label clarity.

    **Jaccard** = overlap / union of labels across both passes, per label.
    - Jaccard = 1: judge always assigned the same label → unambiguous
    - Jaccard = 0: judge never agreed with itself → label definition is unclear

    The **swap matrix** (below the chart) is the most useful part: which labels replace each other
    across passes? High counts at (A → B) means those two labels are being used interchangeably.
    """)
    return


@app.cell
def _(iaa_entry_data, pl):
    # Inline sweep-line interval intersection (mirrors iaa/compute.py)
    def _intersect(a, b):
        result = []
        ia = ib = 0
        while ia < len(a) and ib < len(b):
            lo = max(a[ia][0], b[ib][0])
            hi = min(a[ia][1], b[ib][1])
            if lo < hi:
                result.append((lo, hi, a[ia], b[ib]))
            if a[ia][1] <= b[ib][1]:
                ia += 1
            else:
                ib += 1
        return result

    # Detect all pass-pairs present in the IAA data (any annotator, not just selected)
    _all_anns = set()
    for _by_ann in iaa_entry_data.values():
        _all_anns.update(_by_ann.keys())

    _all_pass_pairs = [
        (j, j + "_pass2")
        for j in sorted(_all_anns)
        if j + "_pass2" in _all_anns
    ]

    _rows = []
    _swap_rows = []

    for _j1, _j2 in _all_pass_pairs:
        for _key, _by_ann in iaa_entry_data.items():
            if _j1 not in _by_ann or _j2 not in _by_ann:
                continue
            for _lo, _hi, _sp1, _sp2 in _intersect(_by_ann[_j1], _by_ann[_j2]):
                _overlap = _hi - _lo
                _lbls1 = set(_sp1[3])
                _lbls2 = set(_sp2[3])
                _union = _lbls1 | _lbls2
                for _lbl in _union:
                    _rows.append({
                        "judge_pair": f"{_j1} → {_j2}",
                        "judge": _j1,
                        "label": _lbl,
                        "in_pass1": _lbl in _lbls1,
                        "in_pass2": _lbl in _lbls2,
                        "overlap_chars": _overlap,
                    })
                # Record swaps: for each dropped label, pair with each added label
                _dropped = _lbls1 - _lbls2
                _added = _lbls2 - _lbls1
                for _d in _dropped:
                    for _a in _added:
                        _swap_rows.append({
                            "judge_pair": f"{_j1} → {_j2}",
                            "judge": _j1,
                            "dropped": _d,
                            "added": _a,
                            "overlap_chars": _overlap,
                        })

    _empty_lbl = pl.DataFrame({
        "judge_pair": pl.Series([], dtype=pl.Utf8), "judge": pl.Series([], dtype=pl.Utf8),
        "label": pl.Series([], dtype=pl.Utf8),
        "in_pass1": pl.Series([], dtype=pl.Boolean), "in_pass2": pl.Series([], dtype=pl.Boolean),
        "overlap_chars": pl.Series([], dtype=pl.Int64),
    })
    _empty_swap = pl.DataFrame({
        "judge_pair": pl.Series([], dtype=pl.Utf8),
        "judge": pl.Series([], dtype=pl.Utf8),
        "dropped": pl.Series([], dtype=pl.Utf8), "added": pl.Series([], dtype=pl.Utf8),
        "overlap_chars": pl.Series([], dtype=pl.Int64),
    })

    pass2_label_df = pl.DataFrame(_rows) if _rows else _empty_lbl
    pass2_swap_df = pl.DataFrame(_swap_rows) if _swap_rows else _empty_swap
    return pass2_label_df, pass2_swap_df


@app.cell
def _(alt, mo, pass2_label_df, pl, selected_annotators):
    mo.stop(pass2_label_df.is_empty(), mo.md("No pass2 data found."))

    _sel = set(selected_annotators)
    # Filter to judge pairs where at least one of the pair is selected
    _pairs_visible = pass2_label_df.filter(pl.col("judge").is_in(_sel))

    mo.stop(_pairs_visible.is_empty(), mo.md("None of the selected annotators have a `_pass2` counterpart."))

    # Jaccard per (judge_pair, label)
    _jaccard = (
        _pairs_visible
        .group_by(["judge_pair", "label"])
        .agg([
            pl.col("in_pass1").sum().alias("n_pass1"),
            pl.col("in_pass2").sum().alias("n_pass2"),
            (pl.col("in_pass1") & pl.col("in_pass2")).sum().alias("n_both"),
            pl.len().alias("n_either"),
        ])
        .with_columns([
            (pl.col("n_both") / pl.col("n_either")).alias("jaccard"),
            (pl.col("n_both") / pl.col("n_pass1").clip(lower_bound=1)).alias("recall_pass1"),
        ])
        .sort("jaccard")
    )

    _label_order2 = _jaccard.group_by("label").agg(pl.col("jaccard").median()).sort("jaccard")["label"].to_list()

    _chart = alt.Chart(_jaccard).mark_bar().encode(
        y=alt.Y("label:N", sort=_label_order2, title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X("jaccard:Q", title="Jaccard (pass1 ∩ pass2) / (pass1 ∪ pass2)", scale=alt.Scale(domain=[0, 1]), stack=False),
        yOffset=alt.YOffset("judge_pair:N"),
        color=alt.Color("judge_pair:N"),
        tooltip=[
            "judge_pair:N", "label:N",
            alt.Tooltip("jaccard:Q", format=".3f"),
            alt.Tooltip("recall_pass1:Q", title="Recall (P1→P2)", format=".3f"),
            alt.Tooltip("n_both:Q", title="Both"),
            alt.Tooltip("n_pass1:Q", title="Pass1"),
            alt.Tooltip("n_pass2:Q", title="Pass2"),
        ],
    ).properties(
        title="Per-label pass1↔pass2 Jaccard — closer to 1 = more stable",
        width=520, height=520,
    )

    _rule = alt.Chart(pl.DataFrame({"x": [0.67]})).mark_rule(color="orange", strokeDash=[4, 2]).encode(
        x="x:Q",
        tooltip=[alt.Tooltip("x:Q", title="Krippendorff α=0.667 threshold")],
    )

    _chart + _rule
    return


@app.cell
def _(alt, mo, pass2_swap_df, pl, selected_annotators):
    mo.stop(pass2_swap_df.is_empty(), mo.md("No swap data."))

    _sel2 = set(selected_annotators)
    # Only keep swaps from selected judge pairs
    _swap_filtered = pass2_swap_df.filter(pl.col("judge").is_in(_sel2))
    mo.stop(_swap_filtered.is_empty(), mo.md("No label swaps found for selected annotators."))

    _swap_agg = (
        _swap_filtered
        .group_by(["dropped", "added"])
        .agg(pl.len().alias("swaps"))
        .sort("swaps", descending=True)
    )

    # Only show labels that appear in at least 2 swaps
    _active_labels = (
        _swap_agg.filter(pl.col("swaps") >= 2)
        .select(["dropped", "added"])
        .unpivot(value_name="label")["label"]
        .unique()
        .to_list()
    )
    _swap_plot = _swap_agg.filter(
        pl.col("dropped").is_in(_active_labels) & pl.col("added").is_in(_active_labels)
    )

    mo.stop(_swap_plot.is_empty(), mo.md("Too few swaps to plot (need ≥2 per pair)."))

    mo.vstack([
        mo.md("### Label swap matrix — which labels replace each other across passes?\n*High count at (row, col): the judge dropped the row label and added the col label (or vice versa)*"),
        alt.Chart(_swap_plot).mark_rect().encode(
            x=alt.X("added:N", title="Added in pass2", axis=alt.Axis(labelAngle=-45, labelLimit=180)),
            y=alt.Y("dropped:N", title="Dropped from pass1", axis=alt.Axis(labelLimit=180)),
            color=alt.Color("swaps:Q", scale=alt.Scale(scheme="orangered"), legend=alt.Legend(title="Swaps")),
            tooltip=["dropped:N", "added:N", alt.Tooltip("swaps:Q", title="# swaps")],
        ).properties(title="Label swap heatmap (pass1 → pass2)", width=480, height=440),
        mo.md("**Top Label Swaps:**"),
        mo.ui.table(
            _swap_agg.head(15).rename({
                "dropped": "Dropped (P1)",
                "added": "Added (P2)",
                "swaps": "Count"
            }),
            selection=None,
        )
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Label Co-occurrence

    Across all selected annotators and IAA entries, which pairs of labels appear
    together on the **same span**? High Jaccard similarity between two labels means
    they are rarely used independently — candidate for merging or clarifying.
    """)
    return


@app.cell
def _(active_data, pl, selected_annotators):
    from collections import defaultdict as _dd

    _sel3 = set(selected_annotators)

    _label_count = _dd(int)
    _cooc_count = _dd(int)

    for _key, _by_ann in active_data.items():
        for _ann, _spans in _by_ann.items():
            if _ann not in _sel3:
                continue
            for _span in _spans:
                _lbls = list(_span[3])
                for _l in _lbls:
                    _label_count[_l] += 1
                for _i in range(len(_lbls)):
                    for _j in range(_i + 1, len(_lbls)):
                        _pair = tuple(sorted([_lbls[_i], _lbls[_j]]))
                        _cooc_count[_pair] += 1

    _rows = []
    for (_la, _lb), _cnt in _cooc_count.items():
        _union = _label_count[_la] + _label_count[_lb] - _cnt
        _jaccard_c = _cnt / _union if _union > 0 else 0.0
        _rows.append({
            "label_a": _la, "label_b": _lb,
            "co_occurrences": _cnt,
            "jaccard": round(_jaccard_c, 4),
            "pct_of_a": round(_cnt / _label_count[_la], 4) if _label_count[_la] else 0.0,
            "pct_of_b": round(_cnt / _label_count[_lb], 4) if _label_count[_lb] else 0.0,
        })

    cooc_df = (
        pl.DataFrame(_rows).sort("jaccard", descending=True)
        if _rows
        else pl.DataFrame({
            "label_a": pl.Series([], dtype=pl.Utf8), "label_b": pl.Series([], dtype=pl.Utf8),
            "co_occurrences": pl.Series([], dtype=pl.Int64), "jaccard": pl.Series([], dtype=pl.Float64),
            "pct_of_a": pl.Series([], dtype=pl.Float64), "pct_of_b": pl.Series([], dtype=pl.Float64),
        })
    )
    return (cooc_df,)


@app.cell
def _(alt, cooc_df, mo, pl):
    mo.stop(cooc_df.is_empty(), mo.md("No co-occurrence data."))

    # Build symmetric matrix: add mirrored rows so the heatmap is symmetric
    _mirror = cooc_df.rename({"label_a": "label_b", "label_b": "label_a"}).select(cooc_df.columns)
    _all_labels = list(set(cooc_df["label_a"].to_list() + cooc_df["label_b"].to_list()))
    _self_df = pl.DataFrame({
        "label_a": _all_labels,
        "label_b": _all_labels,
        "co_occurrences": [0] * len(_all_labels),
        "jaccard": [1.0] * len(_all_labels),
        "pct_of_a": [1.0] * len(_all_labels),
        "pct_of_b": [1.0] * len(_all_labels),
    }).select(cooc_df.columns)
    _sym = pl.concat([cooc_df, _mirror, _self_df]).unique(subset=["label_a", "label_b"], keep="first")

    # Filter to labels with at least 2 co-occurrences (keep the heatmap readable)
    _active = cooc_df.filter(pl.col("co_occurrences") >= 2)
    _active_lbls = (
        _active.select(["label_a", "label_b"]).unpivot(value_name="label")["label"].unique().to_list()
    )
    _sym_filt = _sym.filter(
        pl.col("label_a").is_in(_active_lbls) & pl.col("label_b").is_in(_active_lbls)
    )

    mo.vstack([
        mo.md("### Label co-occurrence Jaccard heatmap\n*Diagonal = 1. High off-diagonal = labels frequently appear on the same span.*"),
        alt.Chart(_sym_filt).mark_rect().encode(
            x=alt.X("label_a:N", title=None, axis=alt.Axis(labelAngle=-45, labelLimit=180)),
            y=alt.Y("label_b:N", title=None, axis=alt.Axis(labelLimit=180)),
            color=alt.Color(
                "jaccard:Q",
                scale=alt.Scale(scheme="yelloworangered", domain=[0, 0.5]),
                legend=alt.Legend(title="Jaccard"),
            ),
            tooltip=[
                "label_a:N", "label_b:N",
                alt.Tooltip("jaccard:Q", format=".3f"),
                alt.Tooltip("co_occurrences:Q"),
                alt.Tooltip("pct_of_a:Q", title="% of label_a spans", format=".1%"),
                alt.Tooltip("pct_of_b:Q", title="% of label_b spans", format=".1%"),
            ],
        ).properties(title="Label co-occurrence Jaccard", width=520, height=500),
        mo.md("**Top co-occurring pairs:**"),
        mo.ui.table(
            cooc_df.head(15).rename({
                "label_a": "Label A",
                "label_b": "Label B",
                "jaccard": "Jaccard",
                "co_occurrences": "Count"
            }),
            selection=None,
        )
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Cross-Annotator Label Substitution

    When annotators disagree on a span, which label pairs tend to be swapped for each other?
    High counts mean those two labels are used interchangeably in practice.

    Computed symmetrically across all selected annotator pairs — hover cells for exact counts.
    Use the table to identify candidates for merging before re-running IAA.
    """)
    return


@app.cell(hide_code=True)
def _(active_data, compute_substitution_matrix, pl, selected_annotators):
    _sub_raw = compute_substitution_matrix(active_data, selected_annotators)
    _sub_rows = [
        {"label_a": la, "label_b": lb, "swaps": cnt}
        for (la, lb), cnt in _sub_raw.items()
    ]
    subst_df = (
        pl.DataFrame(_sub_rows).sort("swaps", descending=True)
        if _sub_rows
        else pl.DataFrame({
            "label_a": pl.Series([], dtype=pl.Utf8),
            "label_b": pl.Series([], dtype=pl.Utf8),
            "swaps": pl.Series([], dtype=pl.Int64),
        })
    )
    return (subst_df,)


@app.cell(hide_code=True)
def _(alt, mo, pl, subst_df):
    mo.stop(subst_df.is_empty(), mo.md("No substitution data for selected annotators."))

    _sub_mirror = subst_df.rename({"label_a": "label_b", "label_b": "label_a"}).select(subst_df.columns)
    _sub_sym = pl.concat([subst_df, _sub_mirror]).unique(subset=["label_a", "label_b"])

    _sub_active = (
        subst_df.filter(pl.col("swaps") >= 2)
        .select(["label_a", "label_b"]).unpivot(value_name="label")["label"]
        .unique().to_list()
    )
    _sub_plot = _sub_sym.filter(
        pl.col("label_a").is_in(_sub_active) & pl.col("label_b").is_in(_sub_active)
    )

    _sub_chart = alt.Chart(_sub_plot).mark_rect().encode(
        x=alt.X("label_a:N", title=None, axis=alt.Axis(labelAngle=-45, labelLimit=200)),
        y=alt.Y("label_b:N", title=None, axis=alt.Axis(labelLimit=200)),
        color=alt.Color(
            "swaps:Q",
            scale=alt.Scale(scheme="orangered"),
            legend=alt.Legend(title="Swaps"),
        ),
        tooltip=[
            "label_a:N", "label_b:N",
            alt.Tooltip("swaps:Q", title="# times swapped between annotators"),
        ],
    ).properties(title="Cross-annotator label substitution frequency", width=520, height=500)

    mo.vstack([
        mo.md("### Label substitution heatmap\n*Each cell = how many times those two labels appeared as substitutes for each other across annotators. Diagonal = self-swaps (impossible, so empty). Labels with < 2 swaps are hidden.*"),
        _sub_chart,
        mo.md("**Top substitution pairs (merge candidates):**"),
        mo.ui.table(
            subst_df.head(20).rename({"label_a": "Label A", "label_b": "Label B", "swaps": "Swaps"}),
            selection=None,
        ),
    ])
    return


if __name__ == "__main__":
    app.run()
