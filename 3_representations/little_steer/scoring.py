"""Hard-to-game scoring of steering vectors.

A vector with high AUROC on a held-out classification task can still be a
*token detector* in disguise: the words "illegal", "unsafe", "refuse" are
strong predictors of the label "I_INTEND_REFUSAL" without the vector
encoding any deeper concept. The functions in this module make those
shortcut behaviours measurable.

Five complementary scores:

============================ ==================================== ============
function                     measures                             cost
============================ ==================================== ============
``causal_steering_score``    Does the vector *cause* the          High
                             behaviour to appear when injected?
``token_ablation_score``     Is discrimination preserved when     Low
                             keyword tokens are removed?
``neutral_pca_score``        Does discrimination survive          Medium
                             projecting out neutral-text PCs?
``logit_lens_top_tokens``    Diagnostic: top tokens the           ~free
                             vector upweights via unembedding.
``embedding_keyword_overlap`` Diagnostic: cosine similarity to    ~free
                             the mean of behaviour-keyword
                             token embeddings.
============================ ==================================== ============

The "causal" score is the headline metric — it requires actual generation
and a behaviour judge (callable that returns 0/1 or a float in [0, 1]),
but it cannot be gamed by any amount of token co-occurrence in the
training set. A vector that doesn't actually cause the behaviour will
have a flat alpha-vs-judge curve.

Combine them via :func:`scoring_report`, which renders a Rich table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from ._internal.forward import run_forward_pass
from .vectors.transforms import project_out

if TYPE_CHECKING:
    from .models.model import LittleSteerModel
    from .vectors.steering_vector import SteeringVector, SteeringVectorSet
    from thesis_schema import ConversationEntry


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class CausalSteeringScore:
    """Generation-based causal score for a single vector.

    Attributes:
        label, layer, method:  identifying tuple.
        alphas:                Alphas evaluated (sorted asc).
        judge_scores:          Mean judge score per alpha (in [0, 1]).
        slope:                 Linear-regression slope of judge_scores vs.
                               alpha. Positive = vector amplifies behaviour.
        baseline_at_zero:      judge_scores at alpha=0 (un-steered baseline).
        peak:                  Max judge_score across alphas.
        n_per_alpha:           Generations per alpha.
    """

    label: str
    layer: int
    method: str
    alphas: list[float]
    judge_scores: list[float]
    slope: float
    baseline_at_zero: float
    peak: float
    n_per_alpha: int


@dataclass
class TokenAblationScore:
    """Discrimination drop when behaviour-keyword tokens are masked.

    Attributes:
        label, layer, method:    identifying tuple.
        keywords:                The tokens that were masked.
        original_discrimination: mean_present − mean_absent on raw text.
        ablated_discrimination:  mean_present − mean_absent on masked text.
        retained_fraction:       ablated / original. <0.3 = vector is
                                 mostly token-matching; >0.7 = vector
                                 carries deeper signal.
    """

    label: str
    layer: int
    method: str
    keywords: list[str]
    original_discrimination: float
    ablated_discrimination: float
    retained_fraction: float


@dataclass
class NeutralPCAScore:
    """Discrimination drop after projecting out top neutral-text PCs.

    A vector that mostly captures generic text-processing structure will
    collapse here. A vector that captures behaviour-specific signal
    survives.
    """

    label: str
    layer: int
    method: str
    n_components_removed: int
    original_discrimination: float
    ablated_discrimination: float
    retained_fraction: float


@dataclass
class LogitLensReadout:
    """Top tokens a vector upweights / downweights via the unembedding."""

    label: str
    layer: int
    method: str
    top_positive: list[tuple[str, float]]   # (token, logit)
    top_negative: list[tuple[str, float]]


@dataclass
class EmbeddingKeywordOverlap:
    """Cosine similarity of vector to mean keyword token embedding.

    High overlap (>0.5) is a red flag — vector might be a re-projection
    of token embeddings rather than a deep behaviour direction.
    """

    label: str
    layer: int
    method: str
    keywords: list[str]
    cosine: float


# ---------------------------------------------------------------------------
# Causal score (the headline metric)
# ---------------------------------------------------------------------------

JudgeFn = Callable[[str, str, str], float]
"""A judge takes (prompt, generated_response, target_label) and returns the
probability or 0/1 indicator that the target behaviour is present in the
response. Callers typically wrap their LLM-judge of choice."""


def causal_steering_score(
    model: "LittleSteerModel",
    vector: "SteeringVector",
    *,
    neutral_prompts: list[str | list[dict[str, str]]],
    judge: JudgeFn,
    alphas: Iterable[float] = (0.0, 5.0, 10.0, 15.0, 20.0),
    n_per_alpha: int | None = None,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    response_only: bool = True,
    show_progress: bool = True,
) -> CausalSteeringScore:
    """Generate with the vector at increasing alphas, ask the judge, regress.

    A vector that genuinely encodes the target behaviour will produce a
    monotonically increasing judge score as ``alpha`` goes up. A vector
    that's just a token-matching artefact will produce a flat curve. The
    slope of that curve is the score — and you can't game it without
    actually inducing the behaviour in generated text.

    Args:
        vector:          SteeringVector to evaluate.
        neutral_prompts: Mostly-benign prompts. Format: either pre-formatted
                         strings, or chat-message lists. Behaviour should
                         NOT obviously appear in the un-steered output —
                         otherwise the baseline is already saturated.
        judge:           ``JudgeFn`` — see type alias above.
        alphas:          Steering strengths. Include 0.0 to anchor a
                         baseline.
        n_per_alpha:     If set, randomly subsample neutral_prompts to this
                         size at every alpha (for cost control).
        response_only:   Pass-through to ``steered_generate``. Default True
                         so we measure the effect on the response, not on
                         prompt processing.
    """
    from .steering import steered_generate

    alpha_list = sorted({float(a) for a in alphas})
    prompts = list(neutral_prompts)
    rng = np.random.default_rng(seed=0)

    judge_means: list[float] = []
    iterator = (
        tqdm(alpha_list, desc=f"alpha sweep ({vector.label})")
        if show_progress else alpha_list
    )

    for alpha in iterator:
        sub = prompts
        if n_per_alpha is not None and n_per_alpha < len(prompts):
            idx = rng.choice(len(prompts), size=n_per_alpha, replace=False)
            sub = [prompts[i] for i in idx]

        scores: list[float] = []
        for p in sub:
            out = steered_generate(
                model, p, steering_vec=vector,
                alpha=alpha,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                response_only=response_only,
            )
            prompt_str = (
                p if isinstance(p, str)
                else model.format_messages(p, add_generation_prompt=True)
            )
            scores.append(float(judge(prompt_str, out, vector.label)))
        judge_means.append(float(np.mean(scores)) if scores else float("nan"))

    # Linear slope of judge vs alpha
    a_arr = np.array(alpha_list, dtype=np.float64)
    j_arr = np.array(judge_means, dtype=np.float64)
    if len(a_arr) >= 2 and a_arr.std() > 0:
        slope = float(np.polyfit(a_arr, j_arr, 1)[0])
    else:
        slope = float("nan")

    baseline = float(j_arr[a_arr == 0.0][0]) if (a_arr == 0.0).any() else float("nan")

    return CausalSteeringScore(
        label=vector.label,
        layer=vector.layer,
        method=vector.method,
        alphas=alpha_list,
        judge_scores=judge_means,
        slope=slope,
        baseline_at_zero=baseline,
        peak=float(np.nanmax(j_arr)) if j_arr.size else float("nan"),
        n_per_alpha=n_per_alpha if n_per_alpha is not None else len(prompts),
    )


# ---------------------------------------------------------------------------
# Token-ablation score
# ---------------------------------------------------------------------------


def _mask_keywords(text: str, keywords: list[str], replacement: str = " ") -> str:
    """Case-insensitive whole-word replacement of every keyword in ``text``."""
    import re
    out = text
    for kw in keywords:
        if not kw.strip():
            continue
        # \b doesn't work for non-ASCII — fall back to a simple non-letter
        # boundary by allowing word chars on either side to NOT match.
        pattern = re.compile(rf"(?i)\b{re.escape(kw)}\b")
        out = pattern.sub(replacement, out)
    return out


def token_ablation_score(
    model: "LittleSteerModel",
    entries: list["ConversationEntry"],
    vector: "SteeringVector",
    *,
    keywords: list[str],
    show_progress: bool = True,
) -> TokenAblationScore:
    """Discrimination retained after masking ``keywords`` from the dataset.

    Implementation: for each entry we apply the chat template, replace every
    case-insensitive whole-word match of ``keywords`` with a space, then run
    the model and compute mean cosine similarity at ``vector.layer`` for
    spans WITH and WITHOUT the vector's label. Compare to the same metric on
    un-masked text.

    Caller supplies ``keywords`` because the right list depends on the
    behaviour. For "I_INTEND_REFUSAL" you might pass
    ``["refuse", "refusal", "cannot", "won't", "decline"]``.
    """
    def _disc(get_text):
        present, absent = [], []
        sv_norm = F.normalize(vector.vector.float().unsqueeze(0), dim=-1).squeeze(0)
        iterator = tqdm(entries, desc="ablation") if show_progress else entries
        for entry in iterator:
            text = get_text(entry)
            token_ids = model.tokenize(text)["input_ids"][0]
            if token_ids.numel() == 0:
                continue
            fwd = run_forward_pass(model, token_ids, [vector.layer])
            la = fwd.layer_acts.get(vector.layer)
            if la is None:
                continue
            # Use mean over the whole sequence as a coarse proxy.
            act = F.normalize(la.float().mean(dim=0).unsqueeze(0), dim=-1).squeeze(0)
            sim = float((act * sv_norm).sum().item())
            label_set = set()
            for ann in entry.annotations:
                label_set.update(ann.labels)
            (present if vector.label in label_set else absent).append(sim)
        if not present or not absent:
            return float("nan")
        return float(np.mean(present) - np.mean(absent))

    original = _disc(lambda e: model.format_messages(e.messages))
    ablated = _disc(
        lambda e: _mask_keywords(model.format_messages(e.messages), keywords)
    )
    retained = (ablated / original) if (original and not np.isnan(original)) else float("nan")

    return TokenAblationScore(
        label=vector.label,
        layer=vector.layer,
        method=vector.method,
        keywords=list(keywords),
        original_discrimination=original,
        ablated_discrimination=ablated,
        retained_fraction=retained,
    )


# ---------------------------------------------------------------------------
# Neutral-PCA score
# ---------------------------------------------------------------------------


def neutral_pca_score(
    model: "LittleSteerModel",
    vector: "SteeringVector",
    *,
    neutral_entries: list["ConversationEntry"] | list[str],
    eval_entries: list["ConversationEntry"],
    n_components: int = 8,
    max_seq_len: int = 4096,
    show_progress: bool = True,
) -> NeutralPCAScore:
    """Project out top PCs of neutral-text activations, re-score discrimination.

    The Anthropic emotion-vector recipe: collect activations on benign
    prompts, fit a PCA, project the top components out of ``vector``, and
    measure how much discrimination survives.
    """
    layer = vector.layer

    # 1. Collect neutral activations at the vector's layer.
    neutral_acts: list[torch.Tensor] = []
    iterator = (
        tqdm(neutral_entries, desc="neutral PCA fit")
        if show_progress else neutral_entries
    )
    for item in iterator:
        text = (
            item if isinstance(item, str)
            else model.format_messages(item.messages)
        )
        token_ids = model.tokenize(text)["input_ids"][0][:max_seq_len]
        if token_ids.numel() == 0:
            continue
        fwd = run_forward_pass(model, token_ids, [layer])
        la = fwd.layer_acts.get(layer)
        if la is None:
            continue
        neutral_acts.append(la.float().mean(dim=0))   # one vector per entry

    if len(neutral_acts) < 2:
        raise ValueError("Need at least 2 neutral activations for PCA")

    X = torch.stack(neutral_acts).numpy().astype(np.float32)
    k = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=k)
    pca.fit(X)
    basis = torch.from_numpy(pca.components_.copy()).float()    # (k, D)

    cleaned = project_out(vector.vector, basis)

    # 2. Discrimination on eval_entries with original vs cleaned vector.
    def _disc(v: torch.Tensor) -> float:
        v_norm = F.normalize(v.unsqueeze(0), dim=-1).squeeze(0)
        present, absent = [], []
        for entry in eval_entries:
            text = model.format_messages(entry.messages)
            token_ids = model.tokenize(text)["input_ids"][0][:max_seq_len]
            if token_ids.numel() == 0:
                continue
            fwd = run_forward_pass(model, token_ids, [layer])
            la = fwd.layer_acts.get(layer)
            if la is None:
                continue
            act = F.normalize(la.float().mean(dim=0).unsqueeze(0), dim=-1).squeeze(0)
            sim = float((act * v_norm).sum().item())
            label_set: set[str] = set()
            for ann in entry.annotations:
                label_set.update(ann.labels)
            (present if vector.label in label_set else absent).append(sim)
        if not present or not absent:
            return float("nan")
        return float(np.mean(present) - np.mean(absent))

    original = _disc(vector.vector.float())
    ablated = _disc(cleaned)
    retained = ablated / original if (original and not np.isnan(original)) else float("nan")

    return NeutralPCAScore(
        label=vector.label, layer=vector.layer, method=vector.method,
        n_components_removed=k,
        original_discrimination=original,
        ablated_discrimination=ablated,
        retained_fraction=retained,
    )


# ---------------------------------------------------------------------------
# Diagnostics: logit lens & embedding overlap
# ---------------------------------------------------------------------------


def logit_lens_top_tokens(
    model: "LittleSteerModel",
    vector: "SteeringVector",
    *,
    k: int = 20,
) -> LogitLensReadout:
    """Project ``vector`` through the unembedding matrix; return top/bottom tokens.

    A vector for "II_STATE_SAFETY_CONCERN" should up-weight tokens like
    "harmful", "unsafe", "concern". If the top tokens are unrelated, the
    vector probably doesn't encode what you think it does.

    Lightweight and fast — computes ``W_U @ vector`` once.
    """
    # nnterp exposes the LM head as model.st.lm_head or model.st._model.lm_head.
    # Fall back to whatever has a `.weight` of shape (vocab_size, hidden_size).
    W_U = None
    for attr_path in ("st.lm_head", "st._model.lm_head"):
        cur = model
        ok = True
        for part in attr_path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and hasattr(cur, "weight"):
            W_U = cur.weight
            break
    if W_U is None:
        raise RuntimeError(
            "Could not locate the LM-head weight on the model. Pass it "
            "manually if your architecture is unusual."
        )

    v = vector.vector.float().to(W_U.device).to(W_U.dtype)
    with torch.no_grad():
        logits = (W_U @ v).float().cpu()        # (vocab_size,)
    top_idx = torch.topk(logits, k).indices.tolist()
    bot_idx = torch.topk(-logits, k).indices.tolist()
    decode = model.tokenizer.decode

    return LogitLensReadout(
        label=vector.label, layer=vector.layer, method=vector.method,
        top_positive=[(decode([i]), float(logits[i])) for i in top_idx],
        top_negative=[(decode([i]), float(logits[i])) for i in bot_idx],
    )


def embedding_keyword_overlap(
    model: "LittleSteerModel",
    vector: "SteeringVector",
    keywords: list[str],
) -> EmbeddingKeywordOverlap:
    """Cosine similarity of ``vector`` to the mean of keyword token embeddings.

    A vector with cosine > 0.5 here is essentially a "keyword detector
    living in token-embedding space" rather than a deep behaviour direction.
    """
    if not keywords:
        raise ValueError("keywords must be non-empty")

    embed = None
    for attr_path in ("st.embed_tokens.weight", "st._model.model.embed_tokens.weight"):
        cur = model
        ok = True
        for part in attr_path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok:
            embed = cur
            break
    if embed is None:
        # Generic fallback: try lm_head's input embedding (often tied)
        embed = model.tokenizer.get_vocab  # placeholder so the next line errors clearly
        raise RuntimeError("Could not locate token embedding matrix on the model")

    tok_ids: list[int] = []
    for kw in keywords:
        ids = model.tokenizer(kw, add_special_tokens=False)["input_ids"]
        if ids:
            tok_ids.extend(ids)
    if not tok_ids:
        return EmbeddingKeywordOverlap(
            label=vector.label, layer=vector.layer, method=vector.method,
            keywords=keywords, cosine=float("nan"),
        )

    rows = embed[tok_ids].float().cpu()                 # (n_tokens, D)
    direction = rows.mean(dim=0)
    direction = F.normalize(direction.unsqueeze(0), dim=-1).squeeze(0)
    v_norm = F.normalize(vector.vector.float().unsqueeze(0), dim=-1).squeeze(0)
    cos = float((direction * v_norm).sum().item())

    return EmbeddingKeywordOverlap(
        label=vector.label, layer=vector.layer, method=vector.method,
        keywords=keywords, cosine=cos,
    )


# ---------------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------------


@dataclass
class ScoringReport:
    """All available scores for a SteeringVectorSet, plus the rendered table."""

    causal: list[CausalSteeringScore] = field(default_factory=list)
    ablation: list[TokenAblationScore] = field(default_factory=list)
    neutral_pca: list[NeutralPCAScore] = field(default_factory=list)
    logit_lens: list[LogitLensReadout] = field(default_factory=list)
    embedding_overlap: list[EmbeddingKeywordOverlap] = field(default_factory=list)

    def render(self, console: Console | None = None) -> None:
        """Print a colourful Rich table summarising every score collected."""
        c = console or Console()
        table = Table(
            title="Steering vector scoring report", show_lines=True,
        )
        table.add_column("label", style="cyan")
        table.add_column("layer", justify="right")
        table.add_column("method", style="magenta")
        table.add_column("causal slope", justify="right")
        table.add_column("ablation kept", justify="right")
        table.add_column("PCA kept", justify="right")
        table.add_column("kw cosine", justify="right")

        # Index everything by (label, layer, method) for the join.
        keys: set[tuple[str, int, str]] = set()
        for source in (self.causal, self.ablation, self.neutral_pca,
                       self.embedding_overlap):
            for s in source:
                keys.add((s.label, s.layer, s.method))

        c_map = {(s.label, s.layer, s.method): s for s in self.causal}
        a_map = {(s.label, s.layer, s.method): s for s in self.ablation}
        p_map = {(s.label, s.layer, s.method): s for s in self.neutral_pca}
        k_map = {(s.label, s.layer, s.method): s for s in self.embedding_overlap}

        for k in sorted(keys):
            cs = c_map.get(k)
            asc = a_map.get(k)
            ps = p_map.get(k)
            ks = k_map.get(k)
            table.add_row(
                k[0], str(k[1]), k[2],
                f"{cs.slope:+.3f}" if cs else "—",
                f"{asc.retained_fraction:.2f}" if asc else "—",
                f"{ps.retained_fraction:.2f}" if ps else "—",
                f"{ks.cosine:+.2f}" if ks else "—",
            )
        c.print(table)
