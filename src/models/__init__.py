"""
CanvasIQ Art Valuation Modeling & Inference Subsystem.

Contains regression training, logarithmic price back-transformation,
confidence interval estimation, comparative ablation, and persistence.
"""

from src.models.evaluate import (
    compute_metrics,
    compute_valuation_metrics,
    format_ablation_table,
    run_ablation_study,
)
from src.models.io import (
    load_model_pipeline,
    save_model_pipeline,
)
from src.models.predict import (
    predict_valuation,
)
from src.models.train import (
    ArtValuationModel,
    train_valuation_model,
)

__all__ = [
    "ArtValuationModel",
    "train_valuation_model",
    "predict_valuation",
    "compute_metrics",
    "compute_valuation_metrics",
    "run_ablation_study",
    "format_ablation_table",
    "save_model_pipeline",
    "load_model_pipeline",
]
