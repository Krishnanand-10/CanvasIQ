"""
CanvasIQ Artwork Price Valuation Inference Engine.

Performs vectorized logarithmic price back-transformation, asymmetric log-normal
residual confidence interval estimation, and edge-case sanitization.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple, Union

import numpy as np

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    stats = None
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


def _compute_z_score(alpha: float) -> float:
    """Computes critical z-score for a two-sided significance level alpha in (0.0, 1.0]."""
    if alpha >= 1.0:
        return 0.0

    if SCIPY_AVAILABLE and stats is not None:
        p = 1.0 - (alpha / 2.0)
        return float(stats.norm.ppf(p))

    # Fallback lookup table for standard confidence intervals
    conf = 1.0 - alpha
    if conf <= 0.0:
        return 0.0
    elif conf >= 0.99:
        return 2.576
    elif conf >= 0.95:
        return 1.960
    elif conf >= 0.90:
        return 1.645
    elif conf >= 0.85:
        return 1.440
    elif conf >= 0.80:
        return 1.282
    elif conf >= 0.50:
        return 0.674
    else:
        return float(1.282 * (conf / 0.80))


def predict_valuation(
    model: Any,
    X: np.ndarray,
    return_bounds: bool = True,
    alpha: float = 0.05,
    residual_std: Optional[float] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Predicts artwork fair market value in US dollars with optional confidence bounds.

    Args:
        model: Trained model (ArtValuationModel or any regressor exposing predict/predict_log).
        X: Multimodal feature matrix of shape (N, D) or single feature vector (D,).
        return_bounds: Whether to compute lower and upper valuation bounds.
        alpha: Two-sided significance level in (0.0, 1.0]. Defaults to 0.05 (95% confidence).
        residual_std: Standard deviation of log residuals. If None, resolves from model.

    Returns:
        If return_bounds=False:
            np.ndarray: 1D array of dollar valuations (N,), clamped >= 0.0.
        If return_bounds=True:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (lower_bounds, point_estimates, upper_bounds).

    Raises:
        ValueError: If alpha is not in (0.0, 1.0].
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"Significance level alpha must be in (0.0, 1.0], got {alpha}")

    X_arr = np.asarray(X, dtype=np.float32)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)

    # Empty batch handling
    if X_arr.shape[0] == 0:
        empty = np.empty((0,), dtype=np.float32)
        return (empty, empty, empty) if return_bounds else empty

    # Sanitize features against NaNs / Infs
    X_clean = np.nan_to_num(X_arr, nan=0.0, posinf=1e6, neginf=-1e6)

    # Obtain log predictions
    if hasattr(model, "predict_log"):
        log_preds = model.predict_log(X_clean)
    elif hasattr(model, "predict"):
        log_preds = model.predict(X_clean)
    else:
        raise TypeError(f"Model object {type(model).__name__} does not implement predict() or predict_log().")

    log_preds = np.asarray(log_preds, dtype=np.float32).ravel()
    log_preds = np.clip(log_preds, -50.0, 50.0)

    # Invert log price: P = max(0.0, expm1(y))
    point_price = np.maximum(0.0, np.expm1(log_preds)).astype(np.float32)

    if not return_bounds:
        return point_price

    # Resolve residual std
    if residual_std is not None:
        sigma = max(0.0, float(residual_std))
    elif hasattr(model, "residual_std_") and model.residual_std_ is not None:
        sigma = max(0.0, float(model.residual_std_))
    else:
        sigma = 0.20

    z = _compute_z_score(alpha)

    # Log bounds
    log_lower = np.clip(log_preds - (z * sigma), -50.0, 50.0)
    log_upper = np.clip(log_preds + (z * sigma), -50.0, 50.0)

    # Dollar bounds back-transformation
    lower_price = np.maximum(0.0, np.expm1(log_lower)).astype(np.float32)
    upper_price = np.maximum(0.0, np.expm1(log_upper)).astype(np.float32)

    # Enforce strict ordering and non-negativity: 0.0 <= lower <= point <= upper
    lower_price = np.minimum(lower_price, point_price)
    upper_price = np.maximum(upper_price, point_price)
    lower_price = np.maximum(0.0, lower_price)

    return (lower_price, point_price, upper_price)
