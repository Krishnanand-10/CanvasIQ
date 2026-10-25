"""
CanvasIQ Deep Visual Aesthetic Feature Extraction Module.

Extracts dense 2048-dimensional aesthetic representations from artwork images using
a frozen PyTorch ResNet50 convolutional backbone truncated at the penultimate global
average pooling layer (fc replaced with nn.Identity()).

Features:
- Fixed feature dimensionality: 2048 float32 values.
- Frozen parameters (requires_grad = False) evaluated under torch.inference_mode().
- Flexible multi-format input ingestion (file path string, Path, PIL Image, raw bytes).
- Standard ImageNet preprocessing (Resize 256, CenterCrop 224, Normalize).
- Canonical zero-vector fallback for corrupt, missing, or unreadable images.
- Automatic CUDA/CPU device selection with robust fallback.
- Batch processing API with chunked vectorized inference.
- Offline deterministic mock backbone for rapid test execution without network downloads.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

# Feature dimension constant
VISION_FEATURE_DIM: int = 2048
DEFAULT_IMAGE_SIZE = (224, 224)
RESIZE_SIZE: int = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Lazy-loaded / optional PyTorch imports
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from torchvision.models import ResNet50_Weights, resnet50
    TORCH_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as err:
    torch = None
    nn = None
    transforms = None
    ResNet50_Weights = None
    resnet50 = None
    TORCH_AVAILABLE = False
    logger.warning("PyTorch/Torchvision not available in environment: %s", err)


class _MockResNetBackbone:
    """Lightweight deterministic mock backbone for fast offline testing."""

    def __init__(self) -> None:
        if TORCH_AVAILABLE and nn is not None:
            self._module = nn.Linear(1, 1, bias=False)
            self._module.weight.requires_grad = False
            self._module.eval()
        else:
            self._module = None
        self.training = False

    def parameters(self):
        if self._module is not None:
            return self._module.parameters()
        return []

    def eval(self):
        self.training = False
        return self

    def __call__(self, x: Any) -> Any:
        return self.forward(x)

    def forward(self, x: Any) -> Any:
        if TORCH_AVAILABLE and torch is not None and isinstance(x, torch.Tensor):
            batch_size = x.size(0)
            means = x.mean(dim=(2, 3))
            stds = x.std(dim=(2, 3))
            stats = torch.cat([means, stds], dim=1)
            repeats = (VISION_FEATURE_DIM // stats.size(1)) + 1
            expanded = stats.repeat(1, repeats)[:, :VISION_FEATURE_DIM]
            return expanded.to(torch.float32)
        else:
            return np.ones((1, VISION_FEATURE_DIM), dtype=np.float32)


class ResNet50FeatureExtractor:
    """
    Deep visual feature extractor utilizing an ImageNet-pretrained ResNet50 backbone.

    Parameters:
        device: Target execution device ('cuda', 'cpu', torch.device, or None for auto-detect).
        weights: Torchvision weights specification ('DEFAULT', None, or ResNet50_Weights).
        use_mock: If True, uses lightweight deterministic mock backbone without downloading weights.
    """

    def __init__(
        self,
        device: Optional[Union[str, Any]] = None,
        weights: Optional[Union[str, Any]] = "DEFAULT",
        use_mock: Optional[bool] = None,
    ) -> None:
        # Resolve mock mode
        if use_mock is None:
            env_mock = os.environ.get("CANVASIQ_MOCK_VISION", "").lower()
            self.use_mock = env_mock in ("1", "true", "yes")
        else:
            self.use_mock = bool(use_mock)

        # Resolve device
        self.device = self._resolve_device(device)

        # Build transform pipeline
        if TORCH_AVAILABLE and transforms is not None:
            self.transform = transforms.Compose([
                transforms.Resize(RESIZE_SIZE),
                transforms.CenterCrop(DEFAULT_IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform = None

        # Build model backbone
        self.model = self._build_backbone(weights)

    def _resolve_device(self, device: Optional[Union[str, Any]]) -> Any:
        """Resolve and validate computation device with graceful CPU fallback."""
        if not TORCH_AVAILABLE or torch is None:
            return "cpu"

        if device is None or (isinstance(device, str) and not device.strip()):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(device, str):
            dev_str = device.strip().lower()
            if dev_str.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA device '%s' requested but not available. Falling back to CPU.", device)
                return torch.device("cpu")
            try:
                return torch.device(dev_str)
            except Exception as exc:
                logger.warning("Invalid device identifier '%s' (%s). Falling back to CPU.", device, exc)
                return torch.device("cpu")

        return device

    def _build_backbone(self, weights: Optional[Union[str, Any]]) -> Any:
        """Initialize, truncate, freeze, and position the ResNet50 backbone."""
        if self.use_mock or not TORCH_AVAILABLE:
            mock_model = _MockResNetBackbone()
            if TORCH_AVAILABLE and isinstance(self.device, torch.device) and hasattr(mock_model, "_module") and mock_model._module is not None:
                mock_model._module.to(self.device)
            return mock_model

        # Attempt to load pretrained model
        model = None
        if weights is not None:
            try:
                weights_spec = ResNet50_Weights.DEFAULT if (isinstance(weights, str) and weights.upper() == "DEFAULT") else weights
                model = resnet50(weights=weights_spec)
            except Exception as exc:
                logger.warning("Failed to load pretrained ResNet50 weights (%s). Falling back to uninitialized model.", exc)
                model = None

        if model is None:
            model = resnet50(weights=None)

        # Truncate at penultimate layer
        model.fc = nn.Identity()

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Set evaluation mode
        model.eval()

        # Move to target device
        model.to(self.device)
        return model

    @staticmethod
    def _load_pil_image(image_input: Any) -> Optional[Image.Image]:
        """
        Safely load, decode, and normalize an image input into an RGB PIL Image.
        Returns None if the image is missing, corrupt, empty, or unreadable.
        """
        if image_input is None:
            return None

        try:
            # Case 1: In-memory PIL Image
            if isinstance(image_input, Image.Image):
                img = image_input.copy()
                img.load()
                return img.convert("RGB")

            # Case 2: In-memory raw bytes
            if isinstance(image_input, (bytes, bytearray)):
                if len(image_input) == 0:
                    return None
                bio = io.BytesIO(image_input)
                img = Image.open(bio)
                img.load()
                return img.convert("RGB")

            # Case 3: File path string or Path object
            if isinstance(image_input, (str, Path)):
                str_path = str(image_input).strip()
                if not str_path:
                    return None
                path_obj = Path(str_path)
                if not path_obj.is_file():
                    return None
                if path_obj.stat().st_size == 0:
                    return None
                img = Image.open(path_obj)
                img.load()
                return img.convert("RGB")

            # Unsupported input type
            return None

        except Exception as exc:
            logger.warning("Could not decode image input (%s): %s", type(image_input).__name__, exc)
            return None

    def _extract_mock(self, pil_img: Image.Image) -> np.ndarray:
        """Generate deterministic pseudo-embedding from PIL image statistics."""
        arr = np.array(pil_img.resize((16, 16)), dtype=np.float32) / 255.0
        flat = arr.flatten()
        repeats = (VISION_FEATURE_DIM // len(flat)) + 1
        tiled = np.tile(flat, repeats)[:VISION_FEATURE_DIM]
        norm = np.linalg.norm(tiled)
        if norm > 1e-6:
            tiled = tiled / norm
        return tiled.astype(np.float32)

    def extract_image(self, image_input: Any) -> np.ndarray:
        """
        Extract a 2048-dimensional aesthetic feature vector from an artwork image.

        Parameters:
            image_input: File path (str or Path), PIL.Image.Image, or raw bytes.

        Returns:
            np.ndarray: Vector of shape (2048,) and dtype np.float32.
                        Returns np.zeros(2048, dtype=np.float32) on failure.
        """
        if image_input is None:
            return np.zeros(VISION_FEATURE_DIM, dtype=np.float32)

        pil_img = self._load_pil_image(image_input)
        if pil_img is None:
            return np.zeros(VISION_FEATURE_DIM, dtype=np.float32)

        if self.use_mock or not TORCH_AVAILABLE or self.transform is None:
            return self._extract_mock(pil_img)

        try:
            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                out = self.model(tensor)
                if out.ndim > 2:
                    out = torch.flatten(out, 1)
                feats = out.squeeze(0).cpu().numpy().astype(np.float32)
            return feats
        except Exception as exc:
            logger.warning("Unexpected error during visual extraction: %s", exc)
            return np.zeros(VISION_FEATURE_DIM, dtype=np.float32)

    def extract_batch(
        self,
        image_inputs: Sequence[Any],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Extract 2048-dimensional aesthetic feature vectors for a sequence of images.

        Parameters:
            image_inputs: Sequence of file paths, PIL Images, or raw image bytes.
            batch_size: Batch size for model inference (default: 32).

        Returns:
            np.ndarray: 2D array of shape (N, 2048) and dtype np.float32.
                        Corrupt or missing items receive np.zeros(2048, dtype=np.float32).
        """
        if image_inputs is None:
            return np.empty((0, VISION_FEATURE_DIM), dtype=np.float32)

        if isinstance(image_inputs, (str, bytes, bytearray, Path, Image.Image)):
            items: List[Any] = [image_inputs]
        elif isinstance(image_inputs, (pd.Series, np.ndarray)):
            raw_list = image_inputs.tolist()
            items = raw_list if isinstance(raw_list, list) else [raw_list]
        elif isinstance(image_inputs, list):
            items = image_inputs
        else:
            try:
                items = list(image_inputs)
            except TypeError:
                items = [image_inputs]

        if len(items) == 0:
            return np.empty((0, VISION_FEATURE_DIM), dtype=np.float32)

        n_total = len(items)
        results = np.zeros((n_total, VISION_FEATURE_DIM), dtype=np.float32)

        if self.use_mock or not TORCH_AVAILABLE or self.transform is None:
            for idx, item in enumerate(items):
                results[idx] = self.extract_image(item)
            return results

        eff_batch_size = max(1, int(batch_size))
        for start_idx in range(0, n_total, eff_batch_size):
            end_idx = min(start_idx + eff_batch_size, n_total)
            chunk_items = items[start_idx:end_idx]

            valid_indices: List[int] = []
            valid_tensors: List[torch.Tensor] = []

            for offset, item in enumerate(chunk_items):
                pil_img = self._load_pil_image(item)
                if pil_img is not None:
                    tensor = self.transform(pil_img)
                    valid_tensors.append(tensor)
                    valid_indices.append(start_idx + offset)

            if valid_tensors:
                try:
                    batch_tensor = torch.stack(valid_tensors).to(self.device)
                    with torch.inference_mode():
                        out = self.model(batch_tensor)
                        if out.ndim > 2:
                            out = torch.flatten(out, 1)
                        chunk_feats = out.cpu().numpy().astype(np.float32)
                    for k, orig_idx in enumerate(valid_indices):
                        results[orig_idx] = chunk_feats[k]
                except Exception as exc:
                    logger.warning("Batch chunk inference failed: %s. Falling back to single item extraction.", exc)
                    for orig_idx in valid_indices:
                        results[orig_idx] = self.extract_image(items[orig_idx])

        return results
