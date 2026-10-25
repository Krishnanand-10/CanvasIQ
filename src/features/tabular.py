"""
Tabular feature transformation pipeline for CanvasIQ artwork valuation.

Encodes:
1. Artist pedigree via Bayesian smoothed target encoding on training log prices.
2. Physical and temporal dimensions via StandardScaler (surface_area, aspect_ratio, height, width, age).
3. Artwork medium via OneHotEncoder with handle_unknown='ignore'.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

NUMERICAL_FEATURES: List[str] = [
    "surface_area",
    "aspect_ratio",
    "height",
    "width",
    "age",
]


class TabularTransformer:
    """Transforms raw or cleaned artwork metadata into an encoded, scaled tabular feature matrix."""

    def __init__(
        self,
        smoothing: float = 10.0,
        reference_year: int = 2026,
    ) -> None:
        """
        Initialize TabularTransformer.

        Parameters:
            smoothing: Weight m given to prior mean in empirical Bayes target encoding.
            reference_year: Reference year used to calculate artwork age (default: 2026).
        """
        self.smoothing = float(smoothing)
        self.reference_year = int(reference_year)
        self.one_hot_encoder: Optional[OneHotEncoder] = None
        self.scaler: Optional[StandardScaler] = None
        self.artist_target_map_: Dict[str, float] = {}
        self.global_target_mean_: float = 0.0
        self.feature_names_: List[str] = []
        self._is_fitted: bool = False

    def _prepare_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        """Sanitizes categorical columns and computes derived geometric/age features if absent."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}")

        df = X.copy()

        # Sanitize categorical medium
        if "medium" in df.columns:
            df["medium"] = (
                df["medium"]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
                .replace(r"^\s*$", "Unknown", regex=True)
            )
        else:
            df["medium"] = "Unknown"

        # Sanitize categorical artist
        if "artist" in df.columns:
            df["artist"] = (
                df["artist"]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
                .replace(r"^\s*$", "Unknown", regex=True)
            )
        else:
            df["artist"] = "Unknown"

        # Coerce dimensions and creation year
        for col in ["height", "width", "year"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            else:
                df[col] = 0.0

        # Derived geometric & temporal features
        if "surface_area" not in df.columns or df["surface_area"].isna().any():
            df["surface_area"] = (df["height"] * df["width"]).fillna(0.0)
        else:
            df["surface_area"] = pd.to_numeric(df["surface_area"], errors="coerce").fillna(0.0)

        if "aspect_ratio" not in df.columns or df["aspect_ratio"].isna().any():
            with np.errstate(divide="ignore", invalid="ignore"):
                safe_h = np.maximum(df["height"].values, 1e-6)
                ar = df["width"].values / safe_h
            df["aspect_ratio"] = pd.Series(ar, index=df.index).fillna(1.0)
        else:
            df["aspect_ratio"] = pd.to_numeric(df["aspect_ratio"], errors="coerce").fillna(1.0)

        if "age" not in df.columns or df["age"].isna().any():
            age_vals = np.maximum(0.0, self.reference_year - df["year"].values)
            df["age"] = pd.Series(age_vals, index=df.index).fillna(0.0)
        else:
            df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(0.0)

        return df

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[np.ndarray, pd.Series, Sequence[float]]] = None,
    ) -> "TabularTransformer":
        """
        Fits artist target encoding, medium OneHotEncoder, and numerical StandardScaler.

        Parameters:
            X: Artwork DataFrame.
            y: Optional continuous target (log-prices or prices). If None, searches X for
               'log_price' or 'price'.
        """
        df = self._prepare_dataframe(X)
        n_samples = len(df)

        # 1. Target vector extraction for artist pedigree encoding
        if y is not None:
            y_arr = np.asarray(y, dtype=float).ravel()
        elif "log_price" in df.columns and df["log_price"].notna().any():
            y_arr = df["log_price"].astype(float).values
        elif "price" in df.columns and df["price"].notna().any():
            y_arr = np.log1p(np.maximum(0.0, df["price"].astype(float).values))
        else:
            y_arr = None

        if y_arr is not None and len(y_arr) == n_samples and n_samples > 0:
            finite_mask = np.isfinite(y_arr)
            if np.any(finite_mask):
                self.global_target_mean_ = float(np.mean(y_arr[finite_mask]))
            else:
                self.global_target_mean_ = 0.0

            df_y = pd.DataFrame({"artist": df["artist"], "y": y_arr})
            grouped = df_y.groupby("artist")["y"].agg(["count", "mean"])
            counts = grouped["count"].values
            means = grouped["mean"].values
            smoothed = (counts * means + self.smoothing * self.global_target_mean_) / (counts + self.smoothing)
            self.artist_target_map_ = dict(zip(grouped.index, smoothed))
        else:
            self.global_target_mean_ = 0.0
            self.artist_target_map_ = {}

        # 2. Fit numerical StandardScaler
        num_data = df[NUMERICAL_FEATURES].values.astype(np.float64)
        self.scaler = StandardScaler()
        if n_samples > 0:
            self.scaler.fit(num_data)
            # Clamp zero variance features to unit scale to prevent divide-by-zero / NaN
            if hasattr(self.scaler, "scale_") and self.scaler.scale_ is not None:
                zero_scale = np.abs(self.scaler.scale_) < 1e-12
                self.scaler.scale_[zero_scale] = 1.0
        else:
            self.scaler.mean_ = np.zeros(len(NUMERICAL_FEATURES), dtype=np.float64)
            self.scaler.scale_ = np.ones(len(NUMERICAL_FEATURES), dtype=np.float64)

        # 3. Fit categorical OneHotEncoder
        med_data = df[["medium"]].astype(str)
        self.one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        if n_samples > 0:
            self.one_hot_encoder.fit(med_data)
        else:
            self.one_hot_encoder.categories_ = [np.array([], dtype=object)]

        # 4. Assemble feature names tracking
        artist_names = ["artist_target_enc"]
        num_names = list(NUMERICAL_FEATURES)
        if n_samples > 0 and hasattr(self.one_hot_encoder, "get_feature_names_out"):
            med_names = list(self.one_hot_encoder.get_feature_names_out(["medium"]))
        else:
            med_names = []

        self.feature_names_ = artist_names + num_names + med_names
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforms input DataFrame into encoded float32 feature array.

        Parameters:
            X: Artwork DataFrame.

        Returns:
            np.ndarray: Transformed tabular feature matrix of shape (N, D_t), dtype float32.
        """
        if not self._is_fitted:
            raise RuntimeError("TabularTransformer must be fitted before calling transform().")

        if X.empty:
            return np.empty((0, len(self.feature_names_)), dtype=np.float32)

        df = self._prepare_dataframe(X)

        # 1. Artist pedigree feature
        artist_enc = (
            df["artist"]
            .map(self.artist_target_map_)
            .fillna(self.global_target_mean_)
            .values.astype(np.float32)
            .reshape(-1, 1)
        )

        # 2. Scaled numerical features
        num_data = df[NUMERICAL_FEATURES].values.astype(np.float64)
        scaled_nums = self.scaler.transform(num_data).astype(np.float32)
        # Extra safety check against NaNs in numerical scaling
        scaled_nums = np.nan_to_num(scaled_nums, nan=0.0, posinf=0.0, neginf=0.0)

        # 3. One-hot medium features
        med_data = df[["medium"]].astype(str)
        med_enc = self.one_hot_encoder.transform(med_data).astype(np.float32)

        # Concatenate horizontally
        out = np.hstack([artist_enc, scaled_nums, med_enc]).astype(np.float32)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[Union[np.ndarray, pd.Series, Sequence[float]]] = None,
    ) -> np.ndarray:
        """Fits transformer and returns encoded feature array deterministically."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> List[str]:
        """Returns list of all output feature names."""
        return list(self.feature_names_)
