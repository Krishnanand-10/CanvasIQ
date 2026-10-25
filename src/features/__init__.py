"""
CanvasIQ Features Package.

Provides multimodal feature extraction components:
- ResNet50FeatureExtractor: Extracts dense visual aesthetic representations.
- TabularTransformer: Encodes and scales tabular physical, temporal, and pedigree metadata.
- MultimodalFeaturePipeline: Coordinates late horizontal feature fusion and ablation slicing.
"""

from src.features.multimodal import MultimodalFeaturePipeline
from src.features.tabular import NUMERICAL_FEATURES, TabularTransformer
from src.features.vision import VISION_FEATURE_DIM, ResNet50FeatureExtractor

__all__ = [
    "ResNet50FeatureExtractor",
    "TabularTransformer",
    "MultimodalFeaturePipeline",
    "VISION_FEATURE_DIM",
    "NUMERICAL_FEATURES",
]
