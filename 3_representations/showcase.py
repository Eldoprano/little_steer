"""little_steer showcase — refactor + new capabilities.

Walks through the new API:
  1. Loading a dataset slice
  2. Building steering vectors with the existing extractor / builder
  3. Session-cached detection (score / evaluate / token similarities)
  4. Hard-to-game scoring (logit lens, embedding overlap, neutral-PCA)
  5. Response-only steering generation
  6. Persona / Assistant-Axis utilities

Tested with marimo 0.23+. Skip the GPU-heavy cells if running on CPU.
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # little_steer showcase

        New since the last refactor:

        - `ls.Session(model, entries)` — caches forward passes; backbone of all
          detection now.
        - `ls.scoring.*` — five hard-to-game scores for steering vectors.
        - `ls.persona.*` — Assistant-Axis extraction + drift tracking.
        - `ls.steered_generate(..., response_only=True)` — only inject the
          vector on assistant tokens, not on prompt prefill.
        - `ls.vectors.transforms.*` — `project_out`, `normalize_to_residual_scale`,
          `compose`.

        Each section below is independent — feel free to skip GPU-heavy ones if
        you're running on CPU.
        """
    )
    return


@app.cell
def _():
    import little_steer as ls
    return (ls,)


@app.cell
def _(mo):
    mo.md("## 1. Configure")
    return


@app.cell
def _(mo):
    model_id = mo.ui.text(
        value="Qwen/Qwen3-0.6B",
        label="Model id",
        full_width=True,
    )
    dataset_path = mo.ui.text(
        value="../data/2_labeled/test.jsonl",
        label="Dataset JSONL",
        full_width=True,
    )
    n_entries = mo.ui.slider(2, 50, value=8, label="Entries to use")
    target_label = mo.ui.text(
        value="II_STATE_SAFETY_CONCERN",
        label="Target label",
        full_width=True,
    )
    mo.vstack([model_id, dataset_path, n_entries, target_label])
    return dataset_path, model_id, n_entries, target_label


@app.cell
def _(mo):
    load_button = mo.ui.run_button(label="Load model + dataset")
    load_button
    return (load_button,)


@app.cell
def _(dataset_path, load_button, ls, mo, model_id, n_entries):
    mo.stop(not load_button.value, mo.md("_Click 'Load model + dataset' to start._"))

    model = ls.LittleSteerModel(model_id.value)
    entries = ls.load_dataset(dataset_path.value)[: n_entries.value]
    mo.md(f"Loaded **{len(entries)}** entries; model has **{model.num_layers}** layers, hidden_size **{model.hidden_size}**.")
    return entries, model


@app.cell
def _(mo):
    mo.md(
        """
        ## 2. Extract activations + build vectors

        We use the same `ExtractionPlan` API as before, then build PCA and
        mean-centering vectors at every captured layer.
        """
    )
    return


@app.cell
def _(entries, ls, mo, model, target_label):
    layer_range = range(int(model.num_layers * 0.4), int(model.num_layers * 0.9))
    plan = ls.ExtractionPlan("showcase", specs={
        "all": ls.ExtractionSpec(ls.TokenSelection("all"), layers=list(layer_range)),
    })
    extractor = ls.ActivationExtractor(model)
    extraction = extractor.extract(entries, plan, show_progress=False)
    builder = ls.SteeringVectorBuilder()
    vectors = builder.build(
        extraction,
        target_label=target_label.value,
        methods=["pca", "mean_centering"],
    )
    mo.md(f"Built **{len(vectors)}** steering vectors. Layers covered: {sorted(set(v.layer for v in vectors))}")
    return extraction, plan, vectors


@app.cell
def _(mo):
    mo.md(
        """
        ## 3. Session — cached detection

        A Session owns the model + dataset and caches forward passes across
        calls. Calling `evaluate(..., aggregations=["mean", "first", "last"])`
        runs ONE forward pass per entry, not three.
        """
    )
    return


@app.cell
def _(entries, ls, model, vectors):
    session = ls.Session(model, entries, cache_size=64)
    scores = session.score(vectors, show_progress=False)
    return scores, session


@app.cell
def _(scores):
    import polars as pl
    scores_df = pl.DataFrame([
        {"label": s.label, "method": s.method, "layer": s.layer,
         "discrimination": round(s.discrimination, 4),
         "n_present": s.n_present, "n_absent": s.n_absent}
        for s in scores
    ]).sort("discrimination", descending=True)
    scores_df
    return pl, scores_df


@app.cell
def _(scores_df):
    import altair as alt
    chart = (
        alt.Chart(scores_df)
        .mark_line(point=True)
        .encode(
            x="layer:Q",
            y="discrimination:Q",
            color="method:N",
            tooltip=["layer", "method", "discrimination", "n_present", "n_absent"],
        )
        .properties(width=600, height=320, title="Discrimination per layer")
    )
    chart
    return alt, chart


@app.cell
def _(mo):
    mo.md(
        """
        ## 4. Hard-to-game scoring

        ### Logit lens — what tokens does the vector upweight?

        A vector for "safety concern" should upweight words like *concern*,
        *unsafe*, *harm*. If it upweights random tokens, it probably doesn't
        encode what we think.
        """
    )
    return


@app.cell
def _(ls, model, vectors):
    best_layer = max(set(v.layer for v in vectors))
    pca_vec = vectors.filter(method="pca", layer=best_layer).vectors[0]
    readout = ls.logit_lens_top_tokens(model, pca_vec, k=15)
    return best_layer, pca_vec, readout


@app.cell
def _(pl, readout):
    pos_df = pl.DataFrame([
        {"rank": i+1, "token": t, "logit": round(v, 3)}
        for i, (t, v) in enumerate(readout.top_positive)
    ])
    neg_df = pl.DataFrame([
        {"rank": i+1, "token": t, "logit": round(v, 3)}
        for i, (t, v) in enumerate(readout.top_negative)
    ])
    return neg_df, pos_df


@app.cell
def _(mo, neg_df, pos_df):
    mo.hstack([
        mo.vstack([mo.md("**Top upweighted tokens**"), pos_df]),
        mo.vstack([mo.md("**Top downweighted tokens**"), neg_df]),
    ])
    return


@app.cell
def _(mo):
    mo.md(
        """
        ### Embedding-keyword overlap

        Cosine similarity between the vector and the mean of the token
        embeddings of behaviour-relevant keywords. **High overlap (> 0.5) is a
        red flag** — the vector is essentially a keyword detector projected
        into hidden-state space.
        """
    )
    return


@app.cell
def _(ls, mo, model, pca_vec):
    keywords = ["concern", "unsafe", "harmful", "dangerous", "risk"]
    overlap = ls.embedding_keyword_overlap(model, pca_vec, keywords)
    mo.md(f"Cosine to mean of {keywords}: **{overlap.cosine:+.3f}**")
    return keywords, overlap


@app.cell
def _(mo):
    mo.md(
        """
        ## 5. Response-only steering

        Inject the vector only on assistant-token forward passes (response
        decoding). Prompt prefill is left untouched. This makes alpha-vs-effect
        curves easier to interpret.
        """
    )
    return


@app.cell
def _(mo):
    alpha_slider = mo.ui.slider(0.0, 30.0, value=10.0, step=1.0, label="alpha")
    response_only_toggle = mo.ui.checkbox(value=True, label="response_only")
    do_steer = mo.ui.run_button(label="Generate (steered)")
    mo.vstack([alpha_slider, response_only_toggle, do_steer])
    return alpha_slider, do_steer, response_only_toggle


@app.cell
def _(alpha_slider, do_steer, ls, mo, model, pca_vec, response_only_toggle):
    mo.stop(not do_steer.value, mo.md("_Click Generate to run._"))
    user_prompt = "How can I make my code more efficient?"
    messages = [{"role": "user", "content": user_prompt}]
    out_baseline = ls.steered_generate(
        model, messages, max_new_tokens=80, do_sample=False,
    )
    out_steered = ls.steered_generate(
        model, messages, steering_vec=pca_vec,
        alpha=alpha_slider.value,
        response_only=response_only_toggle.value,
        max_new_tokens=80, do_sample=False,
    )
    mo.md(
        f"**Prompt:** {user_prompt}\n\n"
        f"**Baseline:** {out_baseline}\n\n"
        f"**Steered (alpha={alpha_slider.value}, response_only={response_only_toggle.value}):**\n\n"
        f"{out_steered}"
    )
    return messages, out_baseline, out_steered, user_prompt


@app.cell
def _(mo):
    mo.md(
        """
        ## 6. Persona / Assistant-Axis (preview)

        `ls.extract_assistant_axis` implements the Lu et al. 2026 recipe in
        miniature. Provide role-eliciting prompts and a neutral baseline; the
        function returns one vector per role plus the first PCA component
        across roles ("assistant axis").
        """
    )
    return


@app.cell
def _():
    role_prompts_demo = {
        "teacher": [
            "You are a patient elementary-school teacher. Explain photosynthesis.",
            "You are a patient elementary-school teacher. Explain why the sky is blue.",
        ],
        "lawyer": [
            "You are a corporate lawyer. Draft a non-disclosure clause.",
            "You are a corporate lawyer. Draft an indemnification clause.",
        ],
        "robot": [
            "You are a sci-fi robot from 2123. Describe your morning routine.",
            "You are a sci-fi robot from 2123. Recommend a hobby.",
        ],
    }
    neutral_prompts_demo = [
        "What is 2+2?",
        "Recommend a productivity tip.",
        "List three rivers in Europe.",
    ]
    return neutral_prompts_demo, role_prompts_demo


@app.cell
def _(mo):
    run_axis = mo.ui.run_button(label="Extract assistant axis (slow)")
    run_axis
    return (run_axis,)


@app.cell
def _(ls, mo, model, neutral_prompts_demo, role_prompts_demo, run_axis):
    mo.stop(not run_axis.value, mo.md("_Click button to run; this calls the model on every prompt._"))

    middle_layers = list(range(int(model.num_layers * 0.4), int(model.num_layers * 0.7), 2))
    axis_set = ls.extract_assistant_axis(
        model,
        role_prompts=role_prompts_demo,
        neutral_prompts=neutral_prompts_demo,
        layers=middle_layers,
        show_progress=False,
    )
    mo.md(f"Built {len(axis_set)} role-centred vectors + assistant axes across {len(middle_layers)} layers.")
    return axis_set, middle_layers


if __name__ == "__main__":
    app.run()
