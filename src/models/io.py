"""
Atomic model pipeline persistence and artifact deserialization for CanvasIQ.

Serializes valuation model, tabular transformer, and JSON metadata to disk,
guaranteeing reload prediction invariance and sub-second cold starts.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Union

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_FILE_NAME = "valuation_model.joblib"
TRANSFORMER_FILE_NAME = "tabular_transformer.joblib"
METADATA_FILE_NAME = "metadata.json"


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively converts NumPy and custom scalar types to standard JSON-compatible types."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)) or (pd is not None and isinstance(obj, (pd.Index, pd.Series))):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.complexfloating)):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif obj is None or isinstance(obj, (int, float, str, bool)):
        return obj
    return str(obj)


def save_model_pipeline(
    model: Any,
    tabular_transformer: Any,
    metadata: Dict[str, Any],
    output_dir: Union[str, Path] = "models",
) -> None:
    """Serializes valuation model, tabular transformer, and metadata to disk.

    Parameters:
        model: Trained ArtValuationModel or underlying estimator.
        tabular_transformer: Fitted TabularTransformer instance or state dict.
        metadata: Dictionary containing training parameters, feature names, metrics.
        output_dir: Target destination directory (default: 'models').
    """
    target_path = Path(output_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    model_path = target_path / MODEL_FILE_NAME
    transformer_path = target_path / TRANSFORMER_FILE_NAME
    metadata_path = target_path / METADATA_FILE_NAME

    # 1. Serialize model
    joblib.dump(model, model_path, compress=3)
    logger.info("Saved model artifact to %s", model_path)

    # 2. Serialize tabular transformer
    joblib.dump(tabular_transformer, transformer_path, compress=3)
    logger.info("Saved tabular transformer to %s", transformer_path)

    # 3. Serialize metadata JSON with type sanitization and Unicode support
    clean_meta = _sanitize_for_json(metadata if metadata is not None else {})
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(clean_meta, f, indent=2, ensure_ascii=False)
    logger.info("Saved pipeline metadata to %s", metadata_path)


def load_model_pipeline(
    model_dir: Union[str, Path] = "models",
) -> Dict[str, Any]:
    """Deserializes saved model artifacts, tabular transformer, and metadata.

    Parameters:
        model_dir: Path to directory containing saved artifacts.

    Returns:
        Dict[str, Any] containing keys: 'model', 'tabular_transformer', 'metadata'.

    Raises:
        FileNotFoundError: If model_dir does not exist or required files are missing.
        ValueError / NotADirectoryError: If model_dir is not a valid directory.
        json.JSONDecodeError: If metadata.json is corrupted.
    """
    dir_path = Path(model_dir)

    if not dir_path.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Specified model path is not a directory: {model_dir}")

    metadata_path = dir_path / METADATA_FILE_NAME
    model_path = dir_path / MODEL_FILE_NAME
    transformer_path = dir_path / TRANSFORMER_FILE_NAME

    # Check and parse metadata first
    if not metadata_path.exists():
        raise FileNotFoundError(f"Required metadata file missing: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not model_path.exists():
        raise FileNotFoundError(f"Required model artifact missing: {model_path}")
    if not transformer_path.exists():
        raise FileNotFoundError(f"Required transformer artifact missing: {transformer_path}")

    # Load serialized artifacts
    model = joblib.load(model_path)
    transformer = joblib.load(transformer_path)

    return {
        "model": model,
        "tabular_transformer": transformer,
        "metadata": metadata,
    }
