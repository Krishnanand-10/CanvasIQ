"""
Multimodal late fusion and feature coordination pipeline for CanvasIQ.

Coordinates tabular metadata transformation and visual feature extraction, providing
horizontal late fusion ([X_tab || X_vis]) and functional modality slicing for ablation studies.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VALID_TABULAR_ALIASES = {"tabular", "tabular_only", "tab"}
VALID_VISION_ALIASES = {"vision", "vision_only", "vis"}
VALID_MULTIMODAL_ALIASES = {"multimodal", "both", "mm", "combined"}


class MultimodalFeaturePipeline:
    """
    Coordinates tabular feature transformation and visual feature extraction,
    providing late fusion and modality slicing.
    """

    def __init__(
        self,
        vision_extractor: Optional[Any] = None,
        tabular_transformer: Optional[Any] = None,
    ) -> None:
        """
        Initialize the MultimodalFeaturePipeline.

        Parameters:
            vision_extractor: Optional instance of ResNet50FeatureExtractor.
            tabular_transformer: Optional instance of TabularTransformer.
        """
        self.vision_extractor = vision_extractor
        self.tabular_transformer = tabular_transformer

    @property
    def tabular_dim(self) -> int:
        """Returns the number of tabular features if the transformer is fitted, else 0."""
        if self.tabular_transformer is not None and hasattr(self.tabular_transformer, "feature_names_"):
            return len(self.tabular_transformer.feature_names_)
        return 0

    def fuse(
        self,
        tabular_features: np.ndarray,
        visual_features: np.ndarray,
    ) -> np.ndarray:
        """
        Horizontally concatenates tabular and visual feature matrices [X_tab || X_vis].

        Parameters:
            tabular_features: Array of shape (N, D_tab) or (D_tab,).
            visual_features: Array of shape (N, D_vis) or (D_vis,).

        Returns:
            np.ndarray: Fused feature matrix of shape (N, D_tab + D_vis) and dtype float32.
        """
        tab = np.asarray(tabular_features, dtype=np.float32)
        vis = np.asarray(visual_features, dtype=np.float32)

        # Handle 1D inputs
        if tab.ndim == 1 and vis.ndim == 1:
            tab = tab.reshape(1, -1)
            vis = vis.reshape(1, -1)
        elif tab.ndim == 1:
            tab = tab.reshape(1, -1)
        elif vis.ndim == 1:
            vis = vis.reshape(1, -1)

        if tab.shape[0] != vis.shape[0]:
            raise ValueError(
                f"Sample count mismatch in multimodal fusion: "
                f"tabular has {tab.shape[0]} samples, visual has {vis.shape[0]} samples."
            )

        fused = np.hstack([tab, vis])
        return np.ascontiguousarray(fused, dtype=np.float32)

    def slice_features(
        self,
        features: np.ndarray,
        modality: str = "multimodal",
        tabular_dim: Optional[int] = None,
    ) -> np.ndarray:
        """
        Slices a fused feature matrix by modality for ablation studies.

        Parameters:
            features: 2D (or 1D) fused feature array.
            modality: Modality to extract ('tabular', 'vision', or 'multimodal'). Case-insensitive.
            tabular_dim: Optional column dimension of tabular slice. If omitted, inferred from transformer.

        Returns:
            np.ndarray: Sliced feature array.
        """
        feat_arr = np.asarray(features, dtype=np.float32)
        if feat_arr.ndim == 1:
            feat_arr = feat_arr.reshape(1, -1)

        t_dim = tabular_dim if tabular_dim is not None else self.tabular_dim
        mod = modality.strip().lower()

        if mod in VALID_TABULAR_ALIASES:
            if t_dim <= 0:
                raise ValueError(
                    f"tabular_dim must be provided or tabular_transformer must be fitted to slice by '{modality}'."
                )
            return feat_arr[:, :t_dim]
        elif mod in VALID_VISION_ALIASES:
            if t_dim <= 0:
                raise ValueError(
                    f"tabular_dim must be provided or tabular_transformer must be fitted to slice by '{modality}'."
                )
            return feat_arr[:, t_dim:]
        elif mod in VALID_MULTIMODAL_ALIASES:
            return feat_arr
        else:
            raise ValueError(
                f"Unknown modality: '{modality}'. Expected one of: "
                f"'multimodal', 'tabular_only', or 'vision_only'."
            )

    def get_modality_slices(
        self,
        tabular_dim: Optional[int] = None,
        vision_dim: int = 2048,
    ) -> Dict[str, slice]:
        """
        Returns column slice objects for tabular, vision, and multimodal features.

        Parameters:
            tabular_dim: Column count of tabular features.
            vision_dim: Column count of vision features (default: 2048).

        Returns:
            dict: Mapping modality name to slice object.
        """
        t_dim = tabular_dim if tabular_dim is not None else self.tabular_dim
        return {
            "tabular": slice(0, t_dim),
            "vision": slice(t_dim, t_dim + vision_dim),
            "multimodal": slice(0, t_dim + vision_dim),
        }

    def get_features(
        self,
        df: pd.DataFrame,
        image_inputs: Optional[Sequence[Any]] = None,
        modality: str = "multimodal",
    ) -> np.ndarray:
        """
        Unified extraction API returning tabular, visual, or multimodal representations.

        Parameters:
            df: Artwork DataFrame.
            image_inputs: Sequence of images, file paths, or bytes (optional).
            modality: Modality selection ('multimodal', 'tabular', or 'vision').

        Returns:
            np.ndarray: Feature matrix of dtype float32.
        """
        mod = modality.strip().lower()

        # 1. Tabular Only
        if mod in VALID_TABULAR_ALIASES:
            if self.tabular_transformer is None:
                raise RuntimeError("tabular_transformer must be initialized to extract tabular features.")
            return self.tabular_transformer.transform(df)

        # 2. Vision Only
        if mod in VALID_VISION_ALIASES:
            if self.vision_extractor is None:
                raise RuntimeError("vision_extractor must be initialized to extract visual features.")
            inputs = image_inputs
            if inputs is None and "image_path" in df.columns:
                inputs = df["image_path"].tolist()
            if inputs is None:
                raise ValueError("image_inputs must be provided or 'image_path' present in DataFrame.")
            return self.vision_extractor.extract_batch(inputs)

        # 3. Multimodal
        if mod in VALID_MULTIMODAL_ALIASES:
            if self.tabular_transformer is None or self.vision_extractor is None:
                raise RuntimeError("Both tabular_transformer and vision_extractor required for multimodal extraction.")
            X_tab = self.tabular_transformer.transform(df)
            inputs = image_inputs
            if inputs is None and "image_path" in df.columns:
                inputs = df["image_path"].tolist()
            if inputs is None:
                inputs = [None] * len(df)
            X_vis = self.vision_extractor.extract_batch(inputs)
            return self.fuse(X_tab, X_vis)

        raise ValueError(f"Unknown modality: '{modality}'. Expected 'multimodal', 'tabular_only', or 'vision_only'.")
