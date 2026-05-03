"""
little_steer — A library for steering vector extraction, detection, and
intervention on transformer language models.

Two ways to use the library:

1. Loose functional style (one-shot, drop-in):

    import little_steer as ls

    model    = ls.LittleSteerModel("Qwen/Qwen3-8B")
    entries  = ls.load_dataset("dataset.jsonl")
    plan     = ls.ExtractionPlan("v1", specs={
        "last": ls.ExtractionSpec(ls.TokenSelection("last"), layers=range(10, 32)),
    })
    result   = ls.ActivationExtractor(model).extract(entries, plan)
    vectors  = ls.SteeringVectorBuilder().build(
        result, target_label="II_STATE_SAFETY_CONCERN", methods=["pca"],
    )
    scores   = ls.score_dataset(model, entries, vectors)
    out      = ls.steered_generate(model, messages, vectors[0], alpha=15)

2. Session-backed style (cached, fast for repeated queries):

    session  = ls.Session(model, entries)          # caches forward passes
    scores   = session.score(vectors)
    results  = session.evaluate(vectors, aggregations=["mean", "first"])
    ts       = session.token_similarities(entries[0], vectors[0],
                                          layers=[10, 20, 30])

The two styles produce identical results — the loose functions construct a
temporary Session under the hood. Use Session whenever you'll be calling
multiple read-side operations on the same dataset.

New modules:

* ``little_steer.scoring``    — hard-to-game scores for steering vectors
                                (causal, ablation, neutral-PCA, logit lens,
                                embedding overlap).
* ``little_steer.persona``    — Lu et al. 2026 Assistant-Axis utilities
                                (extract_assistant_axis, persona_drift,
                                residual_norm).
* ``little_steer.vectors.transforms`` — project_out, normalize_to_residual_scale,
                                compose.

Steering generation now supports ``response_only=True`` to inject the vector
only on assistant tokens, leaving the prompt-prefill untouched.
"""

from . import data
from . import extraction
from . import vectors
from . import probing
from . import steering
from . import persona
from . import scoring

from .data import (
    AnnotatedSpan,
    ConversationEntry,
    convert_file,
    load_dataset,
    save_dataset,
    iter_dataset,
    TokenPositionMapper,
    TokenSpan,
)
from .extraction import (
    TokenSelection,
    ExtractionSpec,
    ExtractionPlan,
    ActivationExtractor,
    ExtractionResult,
)
from .vectors import (
    MeanDifference,
    MeanCentering,
    PCADirection,
    LinearProbe,
    SteeringVector,
    SteeringVectorSet,
    SteeringVectorBuilder,
)
from .vectors.transforms import (
    project_out,
    normalize_to_residual_scale,
    compose,
)
from .session import Session
from .probing import (
    BehaviorScore,
    TokenSimilarities,
    EvaluationResult,
    ProbeDetectionResult,
    probe_text,
    score_dataset,
    get_token_similarities,
    evaluate_dataset,
    vector_similarity_matrix,
    get_probe_predictions,
    get_multilabel_token_scores,
)
from .steering import (
    steered_generate,
    multi_steered_generate,
)
from .mlp_probe import (
    MLPProbe,
    LinearProbeMultilabel,
    MLPProbeTrainer,
    load_probe,
)
from .k_steering import (
    k_steered_generate,
    projection_removal_generate,
)
from .persona import (
    extract_assistant_axis,
    persona_drift,
    residual_norm,
)
from .scoring import (
    causal_steering_score,
    token_ablation_score,
    neutral_pca_score,
    logit_lens_top_tokens,
    embedding_keyword_overlap,
    CausalSteeringScore,
    TokenAblationScore,
    NeutralPCAScore,
    LogitLensReadout,
    EmbeddingKeywordOverlap,
    ScoringReport,
)

try:
    from . import visualization
    from .visualization import (
        render_token_similarity_html,
        render_multilayer_html,
        show_token_similarity,
        plot_layer_discrimination,
        plot_layer_metrics,
        plot_confusion_matrix,
        plot_vector_similarity,
        render_probe_detection_html,
        legend_html,
    )
except ImportError:
    pass

# Lazy import: models requires nnterp (optional dependency for extraction)
def __getattr__(name: str):
    if name in ("LittleSteerModel", "VRAMManager", "BatchConfig", "models"):
        import importlib
        _models = importlib.import_module("little_steer.models")
        globals()["models"] = _models
        globals()["LittleSteerModel"] = _models.LittleSteerModel
        globals()["VRAMManager"] = _models.VRAMManager
        globals()["BatchConfig"] = _models.BatchConfig
        return globals()[name]
    raise AttributeError(f"module 'little_steer' has no attribute {name!r}")

__version__ = "0.2.0"

__all__ = [
    # Sub-modules
    "data", "models", "extraction", "vectors", "visualization",
    "persona", "scoring",
    # Data
    "AnnotatedSpan", "ConversationEntry", "convert_file",
    "load_dataset", "save_dataset", "iter_dataset",
    "TokenPositionMapper", "TokenSpan",
    # Models
    "LittleSteerModel", "VRAMManager", "BatchConfig",
    # Extraction
    "TokenSelection", "ExtractionSpec", "ExtractionPlan",
    "ActivationExtractor", "ExtractionResult",
    # Vectors + transforms
    "MeanDifference", "MeanCentering", "PCADirection", "LinearProbe",
    "SteeringVector", "SteeringVectorSet", "SteeringVectorBuilder",
    "project_out", "normalize_to_residual_scale", "compose",
    # Session
    "Session",
    # Probing (reading)
    "BehaviorScore", "TokenSimilarities", "EvaluationResult",
    "ProbeDetectionResult",
    "probe_text", "score_dataset", "get_token_similarities",
    "evaluate_dataset", "vector_similarity_matrix",
    "get_probe_predictions", "get_multilabel_token_scores",
    # Steering (writing)
    "steered_generate", "multi_steered_generate",
    # MLP / linear multi-label probes (K-Steering paper)
    "MLPProbe", "LinearProbeMultilabel", "MLPProbeTrainer", "load_probe",
    # Gradient-based steering (K-Steering paper)
    "k_steered_generate", "projection_removal_generate",
    # Persona / Assistant axis
    "extract_assistant_axis", "persona_drift", "residual_norm",
    # Scoring (hard-to-game)
    "causal_steering_score", "token_ablation_score", "neutral_pca_score",
    "logit_lens_top_tokens", "embedding_keyword_overlap",
    "CausalSteeringScore", "TokenAblationScore", "NeutralPCAScore",
    "LogitLensReadout", "EmbeddingKeywordOverlap", "ScoringReport",
    # Visualization
    "render_token_similarity_html", "render_multilayer_html",
    "show_token_similarity",
    "plot_layer_discrimination", "plot_layer_metrics",
    "plot_confusion_matrix", "plot_vector_similarity",
    "render_probe_detection_html", "legend_html",
]
