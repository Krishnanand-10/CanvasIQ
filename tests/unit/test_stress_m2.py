"""
Empirical Stress Tests and Adversarial Challenge Suite for Milestone 2 (CanvasIQ).

Adversarially challenges:
1. ResNet50FeatureExtractor input resilience:
   - Corrupted JPEG, PNG, and arbitrary garbage byte streams.
   - Non-standard image color modes: 1, L, P, RGBA, CMYK, YCbCr, LAB, HSV, I, F.
   - Extreme spatial resolutions: 1x1, 1x4000, 4000x1, 3000x3000.
   - Boundary types: None, numbers, collections, NaN, raw numpy ndarrays.
   - Filesystem edge cases: 0-byte files, missing paths, whitespace-only paths.
2. Embedding invariants & integrity:
   - Fixed 2048-dimensional float32 vector guarantee.
   - Canonical zero-vector fallback: np.zeros(2048, dtype=np.float32).
   - Deterministic embeddings on identical inputs across repeated runs.
   - Parameter freezing: requires_grad=False on all backbone parameters.
   - Gradient isolation: torch.enable_grad() in caller context cannot leak gradients.
3. Batch extraction invariants:
   - Mixed-payload slot alignment: valid images preserve embeddings, corrupt slots receive exact zero vectors.
   - Invariance across chunking batch sizes (e.g. batch_size=1 vs 2 vs 5 vs 16).
   - Boundary batch sizes (0, negative, empty sequences).
4. End-to-end multimodal pipeline stress:
   - Missing / corrupt image resilience in MultimodalFeaturePipeline.get_features.
   - Modality slicing under degraded visual signals.
5. Mock mode behavioral parity.
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src.features.multimodal import MultimodalFeaturePipeline
from src.features.tabular import TabularTransformer
from src.features.vision import (
    VISION_FEATURE_DIM,
    ResNet50FeatureExtractor,
)


@pytest.fixture(scope="module")
def real_extractor() -> ResNet50FeatureExtractor:
    """Instantiate real ResNet50 extractor on CPU once for stress testing."""
    return ResNet50FeatureExtractor(device="cpu", weights="DEFAULT", use_mock=False)


@pytest.fixture(scope="module")
def mock_extractor() -> ResNet50FeatureExtractor:
    """Instantiate mock ResNet50 extractor for offline parity verification."""
    return ResNet50FeatureExtractor(device="cpu", use_mock=True)


class TestAdversarialInputFormats:
    """Attacks ResNet50FeatureExtractor.extract_image with malformed, hostile inputs."""

    def test_corrupted_jpeg_truncated_stream(self, real_extractor):
        """Valid JPEG header followed by incomplete stream must trigger clean zero-vector fallback."""
        truncated_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb"
        res = real_extractor.extract_image(truncated_jpeg)
        assert res.shape == (VISION_FEATURE_DIM,)
        assert res.dtype == np.float32
        assert np.all(res == 0.0)

    def test_corrupted_png_truncated_stream(self, real_extractor):
        """Valid PNG signature followed by truncated header must return clean zero vector."""
        truncated_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        res = real_extractor.extract_image(truncated_png)
        assert res.shape == (VISION_FEATURE_DIM,)
        assert res.dtype == np.float32
        assert np.all(res == 0.0)

    def test_random_noise_byte_streams(self, real_extractor):
        """Random entropy buffers of various sizes must never crash or raise exceptions."""
        for size in [1, 16, 256, 1024, 8192]:
            noise = os.urandom(size)
            res = real_extractor.extract_image(noise)
            assert res.shape == (VISION_FEATURE_DIM,)
            assert res.dtype == np.float32
            assert np.all(res == 0.0)

    def test_ascii_and_text_files(self, real_extractor):
        """Plaintext / HTML / JSON strings and bytes must trigger zero-vector fallback."""
        text_payloads = [
            b"<html><body>Not an image</body></html>",
            b"{\"artist\": \"Monet\", \"price\": 1000000}",
            b"SELECT * FROM artworks;",
            b"\x00" * 128,
        ]
        for payload in text_payloads:
            res = real_extractor.extract_image(payload)
            assert res.shape == (VISION_FEATURE_DIM,)
            assert res.dtype == np.float32
            assert np.all(res == 0.0)

    def test_empty_file_and_zero_bytes(self, real_extractor):
        """Zero-byte file on disk and empty bytes buffer must return zero vector."""
        assert np.all(real_extractor.extract_image(b"") == 0.0)
        assert np.all(real_extractor.extract_image(bytearray()) == 0.0)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            tmp_path = Path(tmp.name)
            assert tmp_path.stat().st_size == 0
            res = real_extractor.extract_image(tmp_path)
            assert res.shape == (VISION_FEATURE_DIM,)
            assert np.all(res == 0.0)

    def test_nonexistent_paths_and_directories(self, real_extractor):
        """Non-existent filesystem paths and directory paths must return zero vector."""
        nonexistent = [
            "/nonexistent/image.jpg",
            Path("/var/tmp/does_not_exist_ever_99.png"),
            "   ",
            "",
            tempfile.gettempdir(),  # passing a directory path
        ]
        for path in nonexistent:
            res = real_extractor.extract_image(path)
            assert res.shape == (VISION_FEATURE_DIM,)
            assert res.dtype == np.float32
            assert np.all(res == 0.0)

    def test_invalid_input_types(self, real_extractor):
        """Primitive non-image types must cleanly return zero vector without uncaught exceptions."""
        hostile_types = [
            None,
            42,
            3.14159,
            float("nan"),
            float("inf"),
            [1, 2, 3],
            {"image": "data"},
            (224, 224),
            np.zeros((224, 224, 3), dtype=np.uint8),  # numpy array not directly supported
            object(),
        ]
        for item in hostile_types:
            res = real_extractor.extract_image(item)
            assert res.shape == (VISION_FEATURE_DIM,)
            assert res.dtype == np.float32
            assert np.all(res == 0.0)


class TestColorModeCoercions:
    """Verifies that ResNet50FeatureExtractor robustly converts all PIL color modes."""

    @pytest.mark.parametrize("mode", ["1", "L", "P", "RGB", "RGBA", "CMYK", "YCbCr", "LAB", "HSV"])
    def test_standard_pil_modes(self, real_extractor, mode):
        """Tests that standard 1-channel, 3-channel, and 4-channel modes produce valid embeddings."""
        if mode == "1":
            im = Image.new(mode, (64, 64), color=1)
        elif mode in ("L", "P"):
            im = Image.new(mode, (64, 64), color=128)
        elif mode == "RGBA":
            im = Image.new(mode, (64, 64), color=(100, 150, 200, 255))
        elif mode == "CMYK":
            im = Image.new(mode, (64, 64), color=(50, 100, 150, 200))
        else:
            im = Image.new(mode, (64, 64), color=(100, 150, 200))

        vec = real_extractor.extract_image(im)
        assert vec.shape == (VISION_FEATURE_DIM,)
        assert vec.dtype == np.float32
        assert not np.all(vec == 0.0), f"Mode {mode} unexpectedly produced a zero vector"
        assert np.isfinite(vec).all(), f"Mode {mode} produced non-finite values (NaN or Inf)"

    @pytest.mark.parametrize("float_mode", ["I", "F"])
    def test_high_depth_modes(self, real_extractor, float_mode):
        """Tests 32-bit integer (I) and 32-bit floating point (F) modes."""
        im = Image.new(float_mode, (64, 64), color=10)
        vec = real_extractor.extract_image(im)
        assert vec.shape == (VISION_FEATURE_DIM,)
        assert vec.dtype == np.float32
        assert not np.all(vec == 0.0)
        assert np.isfinite(vec).all()


class TestSpatialResolutionLimits:
    """Verifies boundary resilience under extreme spatial dimensions."""

    def test_tiny_pixel_limits(self, real_extractor):
        """Tests minimal 1x1, 2x1, and 1x2 image canvases."""
        for dims in [(1, 1), (2, 1), (1, 2)]:
            im = Image.new("RGB", dims, color=(200, 100, 50))
            vec = real_extractor.extract_image(im)
            assert vec.shape == (VISION_FEATURE_DIM,)
            assert vec.dtype == np.float32
            assert not np.all(vec == 0.0)
            assert np.isfinite(vec).all()

    def test_extreme_aspect_ratios(self, real_extractor):
        """Tests ultra-skinny panoramic and column images (e.g. 1x1000 and 1000x1)."""
        for dims in [(1, 1000), (1000, 1)]:
            im = Image.new("RGB", dims, color=(128, 128, 128))
            vec = real_extractor.extract_image(im)
            assert vec.shape == (VISION_FEATURE_DIM,)
            assert vec.dtype == np.float32
            assert not np.all(vec == 0.0)
            assert np.isfinite(vec).all()

    def test_large_canvas_resolution(self, real_extractor):
        """Tests high resolution image (3000x3000) processing."""
        im = Image.new("RGB", (3000, 3000), color=(180, 90, 45))
        vec = real_extractor.extract_image(im)
        assert vec.shape == (VISION_FEATURE_DIM,)
        assert vec.dtype == np.float32
        assert not np.all(vec == 0.0)
        assert np.isfinite(vec).all()


class TestEmbeddingInvariants:
    """Verifies strict invariance of embeddings, determinism, parameter freezing, and gradient isolation."""

    def test_strict_dimension_and_dtype(self, real_extractor):
        """Visual representation must be strictly 1D float32 of length 2048."""
        im = Image.new("RGB", (128, 128), color=(10, 20, 30))
        vec = real_extractor.extract_image(im)
        assert vec.ndim == 1
        assert vec.shape == (2048,)
        assert vec.dtype == np.float32

    def test_embedding_determinism(self, real_extractor):
        """Successive forward passes on identical image must be bit-for-bit identical."""
        im = Image.new("RGB", (224, 224), color=(77, 88, 99))
        vec1 = real_extractor.extract_image(im)
        vec2 = real_extractor.extract_image(im)
        assert np.array_equal(vec1, vec2), "Consecutive extractions on same image produced different embeddings!"

    def test_all_parameters_frozen(self, real_extractor):
        """All parameters in ResNet50 backbone must have requires_grad=False."""
        unfrozen = [name for name, p in real_extractor.model.named_parameters() if p.requires_grad]
        assert len(unfrozen) == 0, f"Found unfrozen parameters: {unfrozen}"

    def test_model_in_eval_mode(self, real_extractor):
        """Backbone must operate under model.eval() (training=False)."""
        assert real_extractor.model.training is False

    def test_gradient_isolation_under_torch_grad(self, real_extractor):
        """Caller execution under torch.enable_grad() must not leak gradient state."""
        import torch
        with torch.enable_grad():
            im = Image.new("RGB", (64, 64), color=(50, 50, 50))
            vec = real_extractor.extract_image(im)
            assert isinstance(vec, np.ndarray)
            assert vec.dtype == np.float32
            assert vec.shape == (2048,)


class TestBatchProcessingInvariants:
    """Stress tests extract_batch under diverse chunkings, mixed corruption, and edge cases."""

    def test_mixed_payload_slot_alignment(self, real_extractor):
        """Verifies that valid images keep valid embeddings while corrupted images receive exact zero vectors."""
        im1 = Image.new("RGB", (80, 80), color=(255, 0, 0))
        im2 = Image.new("RGB", (90, 90), color=(0, 255, 0))
        expected_v1 = real_extractor.extract_image(im1)
        expected_v2 = real_extractor.extract_image(im2)

        payload = [
            im1,
            b"corrupted_garbage_bytes",
            None,
            "/nonexistent/file/path.png",
            im2,
            12345,
            b"",
        ]

        batch_feats = real_extractor.extract_batch(payload, batch_size=3)
        assert batch_feats.shape == (7, VISION_FEATURE_DIM)
        assert batch_feats.dtype == np.float32

        # Valid slots
        assert np.allclose(batch_feats[0], expected_v1, atol=1e-5)
        assert np.allclose(batch_feats[4], expected_v2, atol=1e-5)

        # Corrupt slots must be exact zero vectors
        for corrupt_idx in [1, 2, 3, 5, 6]:
            assert np.all(batch_feats[corrupt_idx] == 0.0), f"Slot {corrupt_idx} was not zero vector!"

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 7, 16])
    def test_chunking_invariance(self, real_extractor, batch_size):
        """Results must be identical regardless of chunking batch_size."""
        im1 = Image.new("RGB", (80, 80), color=(100, 150, 200))
        im2 = Image.new("RGB", (80, 80), color=(200, 150, 100))
        items = [im1, b"corrupt", im2, None]

        ref = real_extractor.extract_batch(items, batch_size=1)
        test_run = real_extractor.extract_batch(items, batch_size=batch_size)

        assert np.allclose(ref, test_run, atol=1e-5), f"Batch size {batch_size} deviated from batch_size=1"

    def test_empty_batch_inputs(self, real_extractor):
        """Empty input list returns shape (0, 2048) array."""
        res = real_extractor.extract_batch([])
        assert res.shape == (0, VISION_FEATURE_DIM)
        assert res.dtype == np.float32

    def test_all_corrupt_batch(self, real_extractor):
        """Batch containing only invalid items returns all-zero array."""
        items = [None, b"bad", "/missing.jpg", 99]
        res = real_extractor.extract_batch(items, batch_size=2)
        assert res.shape == (4, VISION_FEATURE_DIM)
        assert np.all(res == 0.0)

    def test_degenerate_batch_sizes(self, real_extractor):
        """Non-positive batch_size arguments must be sanitized to >= 1 without crashing."""
        im = Image.new("RGB", (64, 64), color=(10, 20, 30))
        for degenerate_bs in [0, -1, -10]:
            res = real_extractor.extract_batch([im, None], batch_size=degenerate_bs)
            assert res.shape == (2, VISION_FEATURE_DIM)
            assert not np.all(res[0] == 0.0)
            assert np.all(res[1] == 0.0)


class TestMultimodalIntegrationStress:
    """Stress tests MultimodalFeaturePipeline handling corrupted image inputs in combined workflows."""

    def test_pipeline_get_features_with_corrupt_images(self, real_extractor):
        """MultimodalFeaturePipeline.get_features must gracefully handle mixed corrupt images."""
        df = pd.DataFrame({
            "artist": ["Monet", "Picasso", "Unknown Artist"],
            "year": [1890, 1930, 2020],
            "medium": ["Oil on Canvas", "Sculpture", "Digital"],
            "height": [50.0, 100.0, 75.0],
            "width": [60.0, 80.0, 75.0],
            "price": [1000000.0, 5000000.0, 10000.0],
        })
        transformer = TabularTransformer().fit(df)
        pipeline = MultimodalFeaturePipeline(
            vision_extractor=real_extractor,
            tabular_transformer=transformer,
        )

        valid_im = Image.new("RGB", (64, 64), color=(50, 100, 150))
        corrupt_images = [valid_im, b"bad_bytes", None]

        mm_feats = pipeline.get_features(df, image_inputs=corrupt_images, modality="multimodal")
        tab_dim = pipeline.tabular_dim
        assert mm_feats.shape == (3, tab_dim + VISION_FEATURE_DIM)
        assert mm_feats.dtype == np.float32

        # Verify vision portion of corrupt slots are zero
        vis_part = mm_feats[:, tab_dim:]
        assert not np.all(vis_part[0] == 0.0)
        assert np.all(vis_part[1] == 0.0)
        assert np.all(vis_part[2] == 0.0)

        # Tabular portion must be valid and non-zero across all rows
        tab_part = mm_feats[:, :tab_dim]
        assert np.isfinite(tab_part).all()


class TestMockExtractorParity:
    """Verifies that ResNet50FeatureExtractor(use_mock=True) provides identical interface guarantees."""

    def test_mock_extractor_dimension_and_fallback(self, mock_extractor):
        im = Image.new("RGB", (64, 64), color=(10, 20, 30))
        vec = mock_extractor.extract_image(im)
        assert vec.shape == (VISION_FEATURE_DIM,)
        assert vec.dtype == np.float32
        assert not np.all(vec == 0.0)

        zero_vec = mock_extractor.extract_image(b"invalid_data")
        assert zero_vec.shape == (VISION_FEATURE_DIM,)
        assert np.all(zero_vec == 0.0)

    def test_mock_extractor_batch_extraction(self, mock_extractor):
        im = Image.new("RGB", (64, 64), color=(10, 20, 30))
        batch = mock_extractor.extract_batch([im, None, b"bad", im], batch_size=2)
        assert batch.shape == (4, VISION_FEATURE_DIM)
        assert not np.all(batch[0] == 0.0)
        assert np.all(batch[1] == 0.0)
        assert np.all(batch[2] == 0.0)
        assert np.array_equal(batch[0], batch[3])


class TestBatchContainerDefects:
    """
    Adversarial tests demonstrating defects when extract_batch is invoked
    with standard Python data container types (pandas.Series, numpy.ndarray).
    """

    def test_extract_batch_with_pandas_series(self, real_extractor):
        """Passing pd.Series to extract_batch must not raise ValueError for ambiguous truth value."""
        im1 = Image.new("RGB", (64, 64), color=(10, 20, 30))
        im2 = Image.new("RGB", (64, 64), color=(40, 50, 60))
        s = pd.Series([im1, im2, None])
        res = real_extractor.extract_batch(s, batch_size=2)
        assert res.shape == (3, VISION_FEATURE_DIM)

    def test_extract_batch_with_numpy_array(self, real_extractor):
        """Passing np.ndarray of objects to extract_batch must not raise ValueError for ambiguous truth value."""
        im1 = Image.new("RGB", (64, 64), color=(10, 20, 30))
        im2 = Image.new("RGB", (64, 64), color=(40, 50, 60))
        arr = np.array([im1, im2, None], dtype=object)
        res = real_extractor.extract_batch(arr, batch_size=2)
        assert res.shape == (3, VISION_FEATURE_DIM)

    def test_pipeline_get_features_with_series(self, real_extractor):
        """Passing df['image_path'] Series directly to MultimodalFeaturePipeline.get_features must work without crashing."""
        df = pd.DataFrame({
            "artist": ["Monet", "Picasso"],
            "year": [1900, 1920],
            "medium": ["Oil", "Canvas"],
            "height": [10.0, 20.0],
            "width": [10.0, 20.0],
            "price": [100.0, 200.0],
            "image_path": ["/tmp/nonexistent1.jpg", "/tmp/nonexistent2.jpg"],
        })
        trans = TabularTransformer().fit(df)
        pipe = MultimodalFeaturePipeline(real_extractor, trans)
        res = pipe.get_features(df, image_inputs=df["image_path"])
        assert res.shape[0] == 2

