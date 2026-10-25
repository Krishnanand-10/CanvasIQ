"""
Comparative ablation runner and econometric valuation metrics for CanvasIQ.

Evaluates regression performance across Tabular-only, Vision-only, and Multimodal
feature representations using R², MAE, RMSE, and MAPE.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class AblationDataFrame(pd.DataFrame):
    """DataFrame subclass supporting case-insensitive column access."""

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str) and key not in self.columns:
            for col in self.columns:
                if col.lower() == key.lower():
                    return super().__getitem__(col)
        return super().__getitem__(key)


def compute_metrics(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Computes valuation regression metrics: R², MAE, RMSE, and MAPE.

    Parameters:
        y_true: Ground truth price array (native currency).
        y_pred: Predicted price array (native currency).
        eps: Denominator floor for MAPE to prevent division by zero.

    Returns:
        Dict[str, float] containing keys: 'r2', 'mae', 'rmse', 'mape'.
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()

    if y_t.size == 0 or y_p.size == 0:
        raise ValueError("Input arrays must not be empty.")
    if y_t.shape != y_p.shape:
        raise ValueError(f"Shape mismatch: y_true shape {y_t.shape} != y_pred shape {y_p.shape}.")

    # Absolute and squared errors
    diff = y_t - y_p
    abs_diff = np.abs(diff)
    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    # MAPE with numerical safeguard
    denom = np.maximum(np.abs(y_t), eps)
    mape = float(np.mean(abs_diff / denom) * 100.0)

    # R2 Score calculation with edge case protections
    if y_t.size < 2:
        r2 = float("nan")
    else:
        var_true = float(np.var(y_t))
        if var_true == 0.0:
            r2 = 1.0 if np.allclose(y_t, y_p) else 0.0
        else:
            r2 = float(r2_score(y_t, y_p))

    return {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }


def compute_valuation_metrics(
    y_true_dollar: Union[np.ndarray, List[float]],
    y_pred_dollar: Union[np.ndarray, List[float]],
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Semantic alias for compute_metrics."""
    return compute_metrics(y_true_dollar, y_pred_dollar, eps=eps)


def run_ablation_study(
    arg1: Optional[np.ndarray] = None,
    arg2: Optional[np.ndarray] = None,
    arg3: Optional[Union[np.ndarray, int]] = None,
    X_tab: Optional[np.ndarray] = None,
    X_vis: Optional[np.ndarray] = None,
    y_price: Optional[np.ndarray] = None,
    X_features: Optional[np.ndarray] = None,
    y_dollar: Optional[np.ndarray] = None,
    tabular_dim: Optional[int] = None,
    model_type: str = "lightgbm",
    test_size: float = 0.2,
    random_state: int = 42,
    **model_kwargs: Any,
) -> pd.DataFrame:
    """Runs automated comparative ablation study across modalities:

    Tabular-only, Vision-only, and Multimodal.

    Supports dual calling conventions:
    1. run_ablation_study(X_tab, X_vis, y_price, ...)
    2. run_ablation_study(X_features, y_dollar, tabular_dim=..., ...)
    """
    from src.models.train import ArtValuationModel

    # Resolve keyword or positional conventions
    if X_tab is not None and X_vis is not None and y_price is not None:
        _X_tab = np.asarray(X_tab, dtype=np.float32)
        _X_vis = np.asarray(X_vis, dtype=np.float32)
        _y = np.asarray(y_price, dtype=np.float32).ravel()
        t_dim = _X_tab.shape[1]
        _X_mm = np.hstack([_X_tab, _X_vis])
    elif X_features is not None and y_dollar is not None:
        _X_mm = np.asarray(X_features, dtype=np.float32)
        _y = np.asarray(y_dollar, dtype=np.float32).ravel()
        t_dim = int(tabular_dim if tabular_dim is not None else arg3 or 0)
        if t_dim <= 0 or t_dim >= _X_mm.shape[1]:
            raise ValueError(f"Invalid tabular_dim {t_dim} for feature matrix of shape {_X_mm.shape}.")
        _X_tab = _X_mm[:, :t_dim]
        _X_vis = _X_mm[:, t_dim:]
    else:
        # Positional convention dispatch
        a1 = np.asarray(arg1, dtype=np.float32)
        a2 = np.asarray(arg2, dtype=np.float32)
        if a2.ndim == 2:
            # Convention 1: X_tab, X_vis, y_price
            _X_tab = a1
            _X_vis = a2
            _y = np.asarray(arg3, dtype=np.float32).ravel()
            t_dim = _X_tab.shape[1]
            _X_mm = np.hstack([_X_tab, _X_vis])
        else:
            # Convention 2: X_features, y_dollar, tabular_dim
            _X_mm = a1
            _y = a2.ravel()
            t_dim = int(arg3) if isinstance(arg3, (int, np.integer)) else int(tabular_dim or 0)
            if t_dim <= 0 or t_dim >= _X_mm.shape[1]:
                raise ValueError(f"Invalid tabular_dim {t_dim} for feature matrix of shape {_X_mm.shape}.")
            _X_tab = _X_mm[:, :t_dim]
            _X_vis = _X_mm[:, t_dim:]

    n_samples = len(_y)
    if n_samples < 5:
        raise ValueError(f"Insufficient samples for ablation study: N={n_samples} < 5.")

    # Target in log space for training
    y_log = np.log1p(np.maximum(0.0, _y))

    # Split train/test deterministically
    indices = np.arange(n_samples)
    idx_train, idx_test = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
    )

    y_train_log = y_log[idx_train]
    y_test_dollar = _y[idx_test]
    y_test_log = y_log[idx_test]

    configurations = [
        ("Tabular-only", _X_tab, t_dim),
        ("Vision-only", _X_vis, _X_vis.shape[1]),
        ("Multimodal", _X_mm, _X_mm.shape[1]),
    ]

    records = []
    for mod_name, feat_matrix, dim_count in configurations:
        X_tr = feat_matrix[idx_train]
        X_te = feat_matrix[idx_test]

        model = ArtValuationModel(
            model_type=model_type,
            random_state=random_state,
            **model_kwargs,
        )
        model.fit(X_tr, y_train_log)

        pred_dollar = model.predict_price(X_te)
        pred_log = model.predict_log(X_te)

        m_dollar = compute_metrics(y_test_dollar, pred_dollar)
        if len(y_test_log) >= 2:
            var_log = float(np.var(y_test_log))
            if var_log == 0.0:
                r2_log = 1.0 if np.allclose(y_test_log, pred_log) else 0.0
            else:
                r2_log = float(r2_score(y_test_log, pred_log))
        else:
            r2_log = float("nan")

        records.append({
            "modality": mod_name,
            "n_features": dim_count,
            "r2": m_dollar["r2"],
            "mae": m_dollar["mae"],
            "rmse": m_dollar["rmse"],
            "mape": m_dollar["mape"],
            "r2_log": r2_log,
        })

    report_df = AblationDataFrame(records)
    return report_df


def format_ablation_table(df: pd.DataFrame) -> str:
    """Formats ablation summary dataframe into readable markdown without external dependencies."""
    formatted = df.copy()
    if "r2" in formatted.columns:
        formatted["r2"] = formatted["r2"].map(lambda v: f"{v:.4f}" if pd.notnull(v) else "N/A")
    if "mae" in formatted.columns:
        formatted["mae"] = formatted["mae"].map(lambda v: f"${v:,.2f}" if pd.notnull(v) else "N/A")
    if "rmse" in formatted.columns:
        formatted["rmse"] = formatted["rmse"].map(lambda v: f"${v:,.2f}" if pd.notnull(v) else "N/A")
    if "mape" in formatted.columns:
        formatted["mape"] = formatted["mape"].map(lambda v: f"{v:.2f}%" if pd.notnull(v) else "N/A")
    if "r2_log" in formatted.columns:
        formatted["r2_log"] = formatted["r2_log"].map(lambda v: f"{v:.4f}" if pd.notnull(v) else "N/A")

    cols = list(formatted.columns)
    headers = [str(c).title().replace("_", " ") for c in cols]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in formatted.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)

