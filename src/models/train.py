"""
CanvasIQ Art Valuation Model Training Subsystem.

Provides ArtValuationModel and train_valuation_model supporting LightGBM,
XGBoost, and Scikit-Learn GradientBoosting regressors on log1p price targets.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Dynamic imports with availability flags
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    lgb = None
    LGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    xgb = None
    XGB_AVAILABLE = False

try:
    from sklearn.ensemble import GradientBoostingRegressor
    SKLEARN_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    GradientBoostingRegressor = None
    SKLEARN_AVAILABLE = False


class ArtValuationModel:
    """Gradient-boosted multimodal art price valuation model."""

    def __init__(
        self,
        model_type: str = "lightgbm",
        objective: str = "regression",
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.model_type = model_type.strip().lower()
        self.objective = objective
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.random_state = int(random_state)
        self.kwargs = kwargs

        self.estimator_: Any = None
        self.residual_std_: float = 0.20
        self.is_fitted_: bool = False

    def _create_estimator(self, n_samples: int) -> Any:
        """Instantiate underlying tree regressor with dynamic small-sample adjustments."""
        min_child = min(20, max(1, n_samples // 2))

        if self.model_type in ("lightgbm", "lgb") and LGBM_AVAILABLE:
            num_leaves = min(31, 2 ** self.max_depth) if self.max_depth > 0 else 31
            params = {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "num_leaves": num_leaves,
                "learning_rate": self.learning_rate,
                "random_state": self.random_state,
                "min_child_samples": min_child,
                "objective": self.objective,
                "verbose": -1,
                "n_jobs": -1,
            }
            params.update(self.kwargs)
            return lgb.LGBMRegressor(**params)

        if self.model_type in ("xgboost", "xgb") and XGB_AVAILABLE:
            params = {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "random_state": self.random_state,
                "n_jobs": 1,
            }
            params.update(self.kwargs)
            return xgb.XGBRegressor(**params)

        if SKLEARN_AVAILABLE and GradientBoostingRegressor is not None:
            params = {
                "n_estimators": self.n_estimators,
                "max_depth": min(self.max_depth, 5),
                "learning_rate": self.learning_rate,
                "random_state": self.random_state,
                "min_samples_leaf": min_child,
            }
            params.update(self.kwargs)
            return GradientBoostingRegressor(**params)

        raise RuntimeError("No supported gradient boosting library available (LightGBM, XGBoost, or Scikit-Learn).")

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        """Expose feature importances of the underlying estimator if fitted."""
        if self.estimator_ is not None and hasattr(self.estimator_, "feature_importances_"):
            return np.asarray(self.estimator_.feature_importances_)
        return None

    def fit(self, X: np.ndarray, y_log: np.ndarray) -> "ArtValuationModel":
        """Fits the model on multimodal feature matrix X and log-prices y_log."""
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y_log, dtype=np.float32).ravel()

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        if len(X_arr) == 0:
            raise ValueError("Cannot fit ArtValuationModel on empty dataset.")

        if len(X_arr) != len(y_arr):
            raise ValueError(f"Sample count mismatch: X has {len(X_arr)}, y has {len(y_arr)}.")

        X_clean = np.nan_to_num(X_arr, nan=0.0, posinf=1e6, neginf=-1e6)
        if len(X_clean) < 2:
            raise ValueError(f"ArtValuationModel requires at least 2 samples to fit, got N={len(X_clean)}.")
        self.estimator_ = self._create_estimator(n_samples=len(X_clean))
        self.estimator_.fit(X_clean, y_arr)

        # Estimate residual standard error
        preds = self.estimator_.predict(X_clean)
        residuals = y_arr - preds
        if len(residuals) > 1:
            res_std = float(np.std(residuals, ddof=1))
            self.residual_std_ = res_std if res_std > 1e-6 else 0.15
        else:
            self.residual_std_ = 0.15

        self.is_fitted_ = True
        return self

    def predict_log(self, X: np.ndarray) -> np.ndarray:
        """Predicts log-transformed price."""
        if not self.is_fitted_ or self.estimator_ is None:
            raise RuntimeError("ArtValuationModel must be fitted before predict_log.")

        X_arr = np.asarray(X, dtype=np.float32)

        if X_arr.size == 0 or X_arr.shape[0] == 0:
            return np.empty((0,), dtype=np.float32)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        X_clean = np.nan_to_num(X_arr, nan=0.0, posinf=1e6, neginf=-1e6)
        preds = self.estimator_.predict(X_clean)
        # Clip log predictions to prevent numerical overflow in expm1
        preds = np.clip(preds, -50.0, 50.0)
        return np.asarray(preds, dtype=np.float32).ravel()

    def predict_price(self, X: np.ndarray) -> np.ndarray:
        """Predicts native dollar valuation."""
        from src.models.predict import predict_valuation
        return predict_valuation(self, X, return_bounds=False)

    def predict_with_confidence(
        self,
        X: np.ndarray,
        confidence: float = 0.80,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicts native dollar valuation with confidence bounds."""
        if not (0.0 <= confidence < 1.0):
            raise ValueError(f"Confidence parameter must be in [0.0, 1.0), got {confidence}")

        from src.models.predict import predict_valuation
        alpha = 1.0 - confidence
        return predict_valuation(
            self,
            X,
            return_bounds=True,
            alpha=alpha,
            residual_std=self.residual_std_,
        )


def train_valuation_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "lightgbm",
    val_fraction: float = 0.2,
    random_state: int = 42,
    is_log_target: bool = False,
    **hyperparams: Any,
) -> Tuple[ArtValuationModel, Dict[str, float]]:
    """Trains an ArtValuationModel and computes training/validation metrics."""
    X_arr = np.asarray(X, dtype=np.float32)
    y_raw = np.asarray(y, dtype=np.float32).ravel()

    n_samples = len(X_arr)
    if n_samples == 0:
        raise ValueError("Cannot train on empty feature matrix.")

    # Target conversion: default is_log_target=False (raw prices in dollars) unless explicitly set to True
    if is_log_target is True:
        y_arr = y_raw
    else:
        y_arr = np.log1p(np.maximum(0.0, y_raw))

    # Train-val split if dataset size permits
    if n_samples >= 10 and 0.0 < val_fraction < 1.0:
        rng = np.random.RandomState(random_state)
        indices = rng.permutation(n_samples)
        val_size = max(1, int(n_samples * val_fraction))
        train_idx, val_idx = indices[val_size:], indices[:val_size]
        X_train, y_train = X_arr[train_idx], y_arr[train_idx]
        X_val, y_val = X_arr[val_idx], y_arr[val_idx]
    else:
        X_train, y_train = X_arr, y_arr
        X_val, y_val = X_arr, y_arr

    model = ArtValuationModel(model_type=model_type, random_state=random_state, **hyperparams)
    model.fit(X_train, y_train)

    val_preds = model.predict_log(X_val)
    val_residuals = y_val - val_preds
    val_std = float(np.std(val_residuals))
    model.residual_std_ = val_std if val_std > 1e-6 else 0.15

    mse = float(np.mean(val_residuals ** 2))
    mae = float(np.mean(np.abs(val_residuals)))
    rmse = float(np.sqrt(mse))

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "residual_std": model.residual_std_,
    }
    return model, metrics
