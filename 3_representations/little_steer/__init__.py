"""
little_steer — A library for steering vector extraction and creation.

Quick start:
    import little_steer as ls

    # Load/convert dataset
    entries = ls.data.convert_file("data.json", "dataset.jsonl")
    # or
    entries = ls.data.load_dataset("dataset.jsonl")

    # Load model (requires nnterp)
    model = ls.LittleSteerModel("Qwen/Qwen3-8B")

    # Define extraction plan
    plan = ls.ExtractionPlan("my_plan", specs={
        "last_token": ls.ExtractionSpec(ls.TokenSelection("last"), layers=[20, 25]),
        "whole_sentence": ls.ExtractionSpec(ls.TokenSelection("all"), layers=[20, 25]),
        "bleed_5": ls.ExtractionSpec(
            ls.TokenSelection("all", bleed_before=5, bleed_after=5), layers=[20, 25]
        ),
    })

    # Extract activations (model runs ONCE per conversation)
    extractor = ls.ActivationExtractor(model)
    result = extractor.extract(entries, plan)
    print(result.summary())

    # Build steering vectors
    builder = ls.SteeringVectorBuilder()
    vectors = builder.build(
        result,
        target_label="I_REPHRASE_PROMPT",
        methods=["mean_centering", "pca", "linear_probe"],
        baseline_label="IV_INTEND_REFUSAL_OR_SAFE_ACTION",
    )
    print(vectors.summary())

    # Filter and save
    pca_last = vectors.filter(method="pca", spec="last_token")
    vectors.save("my_vectors.pt")
"""

from . import data
from . import extraction
from . import vectors
from . import probing
from . import steering

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
        # Use importlib to avoid triggering __getattr__ recursively
        _models = importlib.import_module("little_steer.models")
        globals()["models"] = _models
        globals()["LittleSteerModel"] = _models.LittleSteerModel
        globals()["VRAMManager"] = _models.VRAMManager
        globals()["BatchConfig"] = _models.BatchConfig
        return globals()[name]
    raise AttributeError(f"module 'little_steer' has no attribute {name!r}")

__version__ = "0.1.0"

__all__ = [
    # Sub-modules
    "data",
    "models",
    "extraction",
    "vectors",
    "visualization",
    # Data
    "AnnotatedSpan",
    "ConversationEntry",
    "convert_file",
    "load_dataset",
    "save_dataset",
    "iter_dataset",
    "TokenPositionMapper",
    "TokenSpan",
    # Models
    "LittleSteerModel",
    "VRAMManager",
    "BatchConfig",
    # Extraction
    "TokenSelection",
    "ExtractionSpec",
    "ExtractionPlan",
    "ActivationExtractor",
    "ExtractionResult",
    # Vectors
    "MeanDifference",
    "MeanCentering",
    "PCADirection",
    "LinearProbe",
    "SteeringVector",
    "SteeringVectorSet",
    "SteeringVectorBuilder",
    # Probing (reading)
    "BehaviorScore",
    "TokenSimilarities",
    "EvaluationResult",
    "ProbeDetectionResult",
    "probe_text",
    "score_dataset",
    "get_token_similarities",
    "evaluate_dataset",
    "vector_similarity_matrix",
    "get_probe_predictions",
    "get_multilabel_token_scores",
    # Steering (writing)
    "steered_generate",
    "multi_steered_generate",
    # MLP / linear multi-label probes (K-Steering paper)
    "MLPProbe",
    "LinearProbeMultilabel",
    "MLPProbeTrainer",
    "load_probe",
    # Gradient-based steering (K-Steering paper)
    "k_steered_generate",
    "projection_removal_generate",
    # Visualization
    "render_token_similarity_html",
    "render_multilayer_html",
    "show_token_similarity",
    "plot_layer_discrimination",
    "plot_layer_metrics",
    "plot_confusion_matrix",
    "plot_vector_similarity",
    "render_probe_detection_html",
    "legend_html",
]
