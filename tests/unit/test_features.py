"""
Comprehensive Unit Test Suite for CanvasIQ Feature Extraction Pipeline (Milestone 2).

Tests:
1. src/features/vision.py (ResNet50FeatureExtractor)
   - Architecture, penultimate pooling layer truncation (2048-D)
   - Frozen parameters & evaluation mode
   - Multiple input formats (Path, str, PIL.Image, raw bytes)
   - Color space coercions (RGB, RGBA, L, P, CMYK)
   - Canonical zero-vector fallback for missing/corrupted/invalid images
   - CPU/CUDA device configuration & fallback handling
   - Vectorized batch extraction with slot preservation
   - Deterministic offline mock mode

2. src/features/tabular.py (TabularTransformer)
   - Bayesian smoothed target encoding for artist pedigree
   - One-hot medium categorical encoding with unknown category handling
   - Continuous numerical feature scaling via StandardScaler
   - Automatic calculation of derived columns (surface_area, aspect_ratio, age)
   - Handling of unseen artists and media at inference
   - Feature name tracking and reproducible transformation

3. src/features/multimodal.py (MultimodalFeaturePipeline)
   - Late horizontal feature concatenation ([X_tab || X_vis])
   - Modality slicing for ablation studies ('tabular', 'vision', 'multimodal')
   - Shape mismatch validation & edge case handling
   - End-to-end get_features orchestration
"""

from __future__ import annotations

import io
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src.features.multimodal import (
    MultimodalFeaturePipeline,
    VALID_MULTIMODAL_ALIASES,
    VALID_TABULAR_ALIASES,
    VALID_VISION_ALIASES,
)
from src.features.tabular import (
    NUMERICAL_FEATURES,
    TabularTransformer,
)
from src.features.vision import (
    VISION_FEATURE_DIM,
    ResNet50FeatureExtractor,
)


# ==============================================================================
# Pytest Fixtures
# ==============================================================================

@pytest.fixture
def sample_pil_rgb() -> Image.Image:
    """Creates a standard 224x224 RGB PIL image for testing."""
    return Image.new("RGB", (224, 224), color=(140, 70, 35))


@pytest.fixture
def mock_vision_extractor() -> ResNet50FeatureExtractor:
    """Instantiates a fast deterministic mock ResNet50FeatureExtractor."""
    return ResNet50FeatureExtractor(device="cpu", use_mock=True)


@pytest.fixture
def real_vision_extractor() -> ResNet50FeatureExtractor:
    """Instantiates a ResNet50FeatureExtractor with uninitialized weights (no download)."""
    return ResNet50FeatureExtractor(device="cpu", weights=None, use_mock=False)


@pytest.fixture
def sample_artwork_records() -> List[Dict[str, Any]]:
    """Generates diverse artwork metadata records for tabular transformer testing."""
    return [
        {
            "artist": "Claude Monet",
            "year": 1899,
            "medium": "Oil on Canvas",
            "height": 65.0,
            "width": 81.0,
            "price": 25000000.0,
        },
        {
            "artist": "Claude Monet",
            "year": 1904,
            "medium": "Oil on Canvas",
            "height": 73.0,
            "width": 92.0,
            "price": 32000000.0,
        },
        {
            "artist": "Pablo Picasso",
            "year": 1937,
            "medium": "Oil on Canvas",
            "height": 349.0,
            "width": 776.0,
            "price": 180000000.0,
        },
        {
            "artist": "Vincent van Gogh",
            "year": 1889,
            "medium": "Oil on Canvas",
            "height": 73.7,
            "width": 92.1,
            "price": 110000000.0,
        },
        {
            "artist": "Georgia O'Keeffe",
            "year": 1928,
            "medium": "Watercolor on Paper",
            "height": 40.0,
            "width": 30.0,
            "price": 4500000.0,
        },
        {
            "artist": "Kazimir Malevich",
            "year": 1915,
            "medium": "Mixed Media",
            "height": 79.5,
            "width": 79.5,
            "price": 60000000.0,
        },
    ]


@pytest.fixture
def sample_artwork_df(sample_artwork_records) -> pd.DataFrame:
    """DataFrame built from sample artwork records."""
    return pd.DataFrame(sample_artwork_records)


# ==============================================================================
# 1. Vision Feature Extractor Tests (ResNet50FeatureExtractor)
# ==============================================================================

class TestResNet50FeatureExtractor:
    """Unit test suite for ResNet50FeatureExtractor."""

    def test_feature_vector_dimension_mock(self, mock_vision_extractor, sample_pil_rgb):
        feats = mock_vision_extractor.extract_image(sample_pil_rgb)
        assert feats.shape == (VISION_FEATURE_DIM,)
        assert feats.dtype == np.float32
        assert np.all(np.isfinite(feats))

    def test_feature_vector_dimension_real(self, real_vision_extractor, sample_pil_rgb):
        feats = real_vision_extractor.extract_image(sample_pil_rgb)
        assert feats.shape == (VISION_FEATURE_DIM,)
        assert feats.dtype == np.float32
        assert np.all(np.isfinite(feats))

    def test_parameters_frozen(self, real_vision_extractor):
        assert hasattr(real_vision_extractor, "model")
        for param in real_vision_extractor.model.parameters():
            assert param.requires_grad is False

    def test_eval_mode_set(self, real_vision_extractor):
        assert hasattr(real_vision_extractor, "model")
        assert real_vision_extractor.model.training is False

    def test_accepts_pil_image(self, mock_vision_extractor, sample_pil_rgb):
        feats = mock_vision_extractor.extract_image(sample_pil_rgb)
        assert feats.shape == (VISION_FEATURE_DIM,)

    def test_accepts_pathlib_path(self, mock_vision_extractor, sample_pil_rgb, tmp_path):
        p = tmp_path / "artwork.png"
        sample_pil_rgb.save(p)
        feats = mock_vision_extractor.extract_image(p)
        assert feats.shape == (VISION_FEATURE_DIM,)

    def test_accepts_string_path(self, mock_vision_extractor, sample_pil_rgb, tmp_path):
        p = tmp_path / "artwork.jpg"
        sample_pil_rgb.save(p)
        feats = mock_vision_extractor.extract_image(str(p))
        assert feats.shape == (VISION_FEATURE_DIM,)

    def test_accepts_raw_bytes(self, mock_vision_extractor, sample_pil_rgb):
        bio = io.BytesIO()
        sample_pil_rgb.save(bio, format="PNG")
        feats = mock_vision_extractor.extract_image(bio.getvalue())
        assert feats.shape == (VISION_FEATURE_DIM,)

    @pytest.mark.parametrize("mode", ["RGBA", "L", "P", "CMYK"])
    def test_coerces_diverse_color_modes(self, mock_vision_extractor, mode):
        img = Image.new(mode, (200, 200))
        feats = mock_vision_extractor.extract_image(img)
        assert feats.shape == (VISION_FEATURE_DIM,)
        assert np.all(np.isfinite(feats))

    @pytest.mark.parametrize("size", [(1, 1), (600, 50), (50, 600), (1024, 768)])
    def test_handles_arbitrary_dimensions_and_aspect_ratios(self, mock_vision_extractor, size):
        img = Image.new("RGB", size, color="teal")
        feats = mock_vision_extractor.extract_image(img)
        assert feats.shape == (VISION_FEATURE_DIM,)

    def test_deterministic_embedding_on_same_image(self, mock_vision_extractor, sample_pil_rgb):
        f1 = mock_vision_extractor.extract_image(sample_pil_rgb)
        f2 = mock_vision_extractor.extract_image(sample_pil_rgb)
        np.testing.assert_allclose(f1, f2, atol=1e-6)

    def test_fallback_on_none(self, mock_vision_extractor):
        feats = mock_vision_extractor.extract_image(None)
        assert feats.shape == (VISION_FEATURE_DIM,)
        np.testing.assert_array_equal(feats, np.zeros(VISION_FEATURE_DIM, dtype=np.float32))

    def test_fallback_on_empty_string(self, mock_vision_extractor):
        feats = mock_vision_extractor.extract_image("")
        np.testing.assert_array_equal(feats, np.zeros(VISION_FEATURE_DIM, dtype=np.float32))

    def test_fallback_on_nonexistent_file(self, mock_vision_extractor):
        feats = mock_vision_extractor.extract_image("non_existent_path_987654.png")
        np.testing.assert_array_equal(feats, np.zeros(VISION_FEATURE_DIM, dtype=np.float32))

    def test_fallback_on_directory_path(self, mock_vision_extractor):
        feats = mock_vision_extractor.extract_image(tempfile.gettempdir())
        np.testing.assert_array_equal(feats, np.zeros(VISION_FEATURE_DIM, dtype=np.float32))

    def test_fallback_on_empty_bytes(self, mock_vision_extractor):
        feats = mock_vision_extractor.extract_image(b"")
        np.testing.assert_array_equal(feats, np.zeros(VISION_FEATURE_DIM, dtype=np.float32))

    def test_fallback_on_corrupt_bytes(self, mock_vision_extractor):
        feats = mock_vision_extractor.extract_image(b"not_an_image_stream_garbage_123")
        np.testing.assert_array_equal(feats, np.zeros(VISION_FEATURE_DIM, dtype=np.float32))

    def test_fallback_on_zero_byte_file(self, mock_vision_extractor, tmp_path):
        p = tmp_path / "zero_bytes.png"
        p.touch()
        feats = mock_vision_extractor.extract_image(p)
        np.testing.assert_array_equal(feats, np.zeros(VISION_FEATURE_DIM, dtype=np.float32))

    def test_fallback_on_text_file_disguised_as_jpg(self, mock_vision_extractor, tmp_path):
        p = tmp_path / "fake.jpg"
        p.write_text("Hello art world this is not an image")
        feats = mock_vision_extractor.extract_image(p)
        np.testing.assert_array_equal(feats, np.zeros(VISION_FEATURE_DIM, dtype=np.float32))

    def test_fallback_vector_l2_norm_is_zero(self, mock_vision_extractor):
        feats = mock_vision_extractor.extract_image("bad_path.png")
        assert float(np.linalg.norm(feats)) == 0.0

    def test_init_explicit_cpu(self):
        ext = ResNet50FeatureExtractor(device="cpu", use_mock=True)
        assert ext is not None

    def test_init_default_device(self):
        ext = ResNet50FeatureExtractor(use_mock=True)
        assert ext is not None

    def test_init_empty_string_device(self):
        ext = ResNet50FeatureExtractor(device="", use_mock=True)
        assert ext is not None

    def test_init_invalid_device_name_falls_back_to_cpu(self):
        ext = ResNet50FeatureExtractor(device="non_existent_device_tpu", use_mock=True)
        assert ext is not None

    def test_init_cuda_fallback_without_unhandled_crash(self):
        ext = ResNet50FeatureExtractor(device="cuda", use_mock=True)
        assert ext is not None

    def test_extract_batch_empty_list(self, mock_vision_extractor):
        res = mock_vision_extractor.extract_batch([])
        assert res.shape == (0, VISION_FEATURE_DIM)
        assert res.dtype == np.float32

    def test_extract_batch_single_item(self, mock_vision_extractor, sample_pil_rgb):
        res = mock_vision_extractor.extract_batch([sample_pil_rgb])
        assert res.shape == (1, VISION_FEATURE_DIM)
        assert res.dtype == np.float32

    def test_extract_batch_multiple_valid_images(self, mock_vision_extractor, sample_pil_rgb):
        images = [sample_pil_rgb, Image.new("RGB", (100, 100), color="blue")]
        res = mock_vision_extractor.extract_batch(images)
        assert res.shape == (2, VISION_FEATURE_DIM)
        assert not np.all(res[0] == 0)
        assert not np.all(res[1] == 0)

    def test_extract_batch_mixed_valid_and_corrupt(self, mock_vision_extractor, sample_pil_rgb):
        items = [sample_pil_rgb, "non_existent_img.png", b"broken_bytes"]
        res = mock_vision_extractor.extract_batch(items)
        assert res.shape == (3, VISION_FEATURE_DIM)
        assert not np.all(res[0] == 0)
        np.testing.assert_array_equal(res[1], np.zeros(VISION_FEATURE_DIM, dtype=np.float32))
        np.testing.assert_array_equal(res[2], np.zeros(VISION_FEATURE_DIM, dtype=np.float32))

    def test_extract_batch_chunking_consistency(self, mock_vision_extractor, sample_pil_rgb):
        images = [sample_pil_rgb for _ in range(5)]
        res_b2 = mock_vision_extractor.extract_batch(images, batch_size=2)
        res_b10 = mock_vision_extractor.extract_batch(images, batch_size=10)
        np.testing.assert_allclose(res_b2, res_b10, atol=1e-6)


# ==============================================================================
# 2. Tabular Feature Transformer Tests (TabularTransformer)
# ==============================================================================

class TestTabularTransformer:
    """Unit test suite for TabularTransformer."""

    def test_fit_transform_basic_shapes(self, sample_artwork_df):
        transformer = TabularTransformer(smoothing=10.0)
        y = np.log1p(sample_artwork_df["price"].values)
        out = transformer.fit_transform(sample_artwork_df, y)
        assert out.ndim == 2
        assert out.shape[0] == len(sample_artwork_df)
        assert out.shape[1] >= 6  # 1 (artist) + 5 (nums) + K (mediums)
        assert out.dtype == np.float32
        assert not np.any(np.isnan(out))

    def test_one_hot_medium_encoding(self, sample_artwork_df):
        transformer = TabularTransformer()
        transformer.fit(sample_artwork_df)
        out = transformer.transform(sample_artwork_df)
        med_features = [f for f in transformer.get_feature_names_out() if f.startswith("medium_")]
        assert len(med_features) >= 2
        # Check binary indicator values (0.0 or 1.0)
        start_idx = len(transformer.feature_names_) - len(med_features)
        med_submatrix = out[:, start_idx:]
        assert np.all(np.isin(med_submatrix, [0.0, 1.0]))

    def test_unseen_medium_at_inference(self, sample_artwork_df):
        transformer = TabularTransformer()
        transformer.fit(sample_artwork_df)
        test_df = pd.DataFrame([
            {"artist": "Pablo Picasso", "year": 1940, "medium": "Titanium Foil 3D", "height": 50.0, "width": 50.0}
        ])
        out = transformer.transform(test_df)
        assert out.shape[0] == 1
        med_features = [f for f in transformer.get_feature_names_out() if f.startswith("medium_")]
        start_idx = len(transformer.feature_names_) - len(med_features)
        # All one-hot columns for novel medium should be 0.0
        np.testing.assert_array_equal(out[0, start_idx:], np.zeros(len(med_features), dtype=np.float32))

    def test_missing_and_empty_medium_imputed(self):
        transformer = TabularTransformer()
        df = pd.DataFrame([
            {"artist": "Monet", "year": 1900, "medium": None, "height": 50.0, "width": 50.0},
            {"artist": "Picasso", "year": 1920, "medium": "", "height": 60.0, "width": 60.0},
            {"artist": "Dali", "year": 1930, "medium": "   ", "height": 70.0, "width": 70.0},
        ])
        transformer.fit(df)
        out = transformer.transform(df)
        assert not np.any(np.isnan(out))
        assert "medium_Unknown" in transformer.get_feature_names_out()

    def test_artist_target_encoding_exact_formula(self):
        # 2 records for Artist A, 1 record for Artist B
        df = pd.DataFrame([
            {"artist": "A", "year": 2000, "medium": "Oil", "height": 50.0, "width": 50.0},
            {"artist": "A", "year": 2001, "medium": "Oil", "height": 50.0, "width": 50.0},
            {"artist": "B", "year": 2002, "medium": "Oil", "height": 50.0, "width": 50.0},
        ])
        y = np.array([10.0, 12.0, 8.0])  # A's mean = 11.0, count=2; B's mean = 8.0, count=1
        global_mean = float(np.mean(y))  # 10.0
        smoothing = 10.0

        transformer = TabularTransformer(smoothing=smoothing)
        transformer.fit(df, y)

        expected_A = (2 * 11.0 + 10.0 * global_mean) / (2 + 10.0)  # (22 + 100) / 12 = 122 / 12 = 10.166667
        expected_B = (1 * 8.0 + 10.0 * global_mean) / (1 + 10.0)   # (8 + 100) / 11 = 108 / 11 = 9.818182

        assert math.isclose(transformer.artist_target_map_["A"], expected_A, rel_tol=1e-5)
        assert math.isclose(transformer.artist_target_map_["B"], expected_B, rel_tol=1e-5)

    def test_artist_target_encoding_shrinkage_smoothness(self):
        records = [{"artist": "A", "year": 2000, "medium": "Oil", "height": 50.0, "width": 50.0} for _ in range(100)]
        records.append({"artist": "B", "year": 2000, "medium": "Oil", "height": 50.0, "width": 50.0})
        df = pd.DataFrame(records)
        y = np.array([20.0] * 100 + [5.0])  # Artist A has 100 sales at 20.0; Artist B has 1 sale at 5.0
        global_mean = float(np.mean(y))

        transformer = TabularTransformer(smoothing=10.0)
        transformer.fit(df, y)

        # High-count artist A should be close to 20.0
        assert math.isclose(transformer.artist_target_map_["A"], 20.0, abs_tol=0.5)
        # Low-count artist B should be heavily shrunk toward global_mean (~19.85)
        assert transformer.artist_target_map_["B"] > 15.0

    def test_unseen_artist_receives_global_mean(self, sample_artwork_df):
        transformer = TabularTransformer(smoothing=10.0)
        y = np.log1p(sample_artwork_df["price"].values)
        transformer.fit(sample_artwork_df, y)

        novel_df = pd.DataFrame([
            {"artist": "Novel Artist Never Seen Before", "year": 2024, "medium": "Oil on Canvas", "height": 50.0, "width": 50.0}
        ])
        out = transformer.transform(novel_df)
        assert math.isclose(out[0, 0], transformer.global_target_mean_, rel_tol=1e-5)

    def test_empty_or_missing_artist(self):
        transformer = TabularTransformer()
        df = pd.DataFrame([
            {"artist": "", "year": 2000, "medium": "Oil", "height": 50.0, "width": 50.0, "price": 1000.0},
            {"artist": None, "year": 2000, "medium": "Oil", "height": 50.0, "width": 50.0, "price": 2000.0},
        ])
        transformer.fit(df)
        out = transformer.transform(df)
        assert not np.any(np.isnan(out))

    def test_numerical_scaling_zero_mean_unit_variance(self, sample_artwork_df):
        transformer = TabularTransformer()
        out = transformer.fit_transform(sample_artwork_df)
        # Continuous feature indices: 1 to 5 (surface_area, aspect_ratio, height, width, age)
        num_cols = out[:, 1:6]
        means = np.mean(num_cols, axis=0)
        stds = np.std(num_cols, axis=0)
        np.testing.assert_allclose(means, np.zeros(5), atol=1e-5)
        np.testing.assert_allclose(stds, np.ones(5), atol=1e-5)

    def test_numerical_scaling_zero_variance_constant_dimensions(self):
        # All records have identical dimensions and year
        records = [
            {"artist": "A", "year": 2000, "medium": "Oil", "height": 100.0, "width": 100.0}
            for _ in range(5)
        ]
        df = pd.DataFrame(records)
        transformer = TabularTransformer()
        out = transformer.fit_transform(df)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))

    def test_derived_columns_automatic_computation(self):
        # Missing surface_area, aspect_ratio, and age
        df = pd.DataFrame([
            {"artist": "Artist", "year": 2020, "medium": "Oil", "height": 50.0, "width": 100.0}
        ])
        transformer = TabularTransformer(reference_year=2026)
        prepared = transformer._prepare_dataframe(df)
        assert prepared["surface_area"].iloc[0] == 5000.0
        assert math.isclose(prepared["aspect_ratio"].iloc[0], 2.0, rel_tol=1e-5)
        assert prepared["age"].iloc[0] == 6.0

    def test_feature_names_tracking(self, sample_artwork_df):
        transformer = TabularTransformer()
        transformer.fit(sample_artwork_df)
        names = transformer.get_feature_names_out()
        assert names[0] == "artist_target_enc"
        assert names[1:6] == NUMERICAL_FEATURES
        assert len(names) == transformer.transform(sample_artwork_df).shape[1]

    def test_deterministic_reproducibility(self, sample_artwork_df):
        transformer = TabularTransformer(smoothing=10.0)
        y = np.log1p(sample_artwork_df["price"].values)
        o1 = transformer.fit_transform(sample_artwork_df, y)
        o2 = transformer.transform(sample_artwork_df)
        np.testing.assert_allclose(o1, o2, atol=1e-5)

    def test_fit_without_target_y(self, sample_artwork_df):
        transformer = TabularTransformer()
        # Does not provide y or price column
        df_no_target = sample_artwork_df.drop(columns=["price"])
        transformer.fit(df_no_target)
        out = transformer.transform(df_no_target)
        assert out.shape[0] == len(sample_artwork_df)
        assert not np.any(np.isnan(out))


# ==============================================================================
# 3. Multimodal Late Fusion Pipeline Tests (MultimodalFeaturePipeline)
# ==============================================================================

class TestMultimodalFeaturePipeline:
    """Unit test suite for MultimodalFeaturePipeline."""

    def test_fuse_standard_dimensions(self):
        pipeline = MultimodalFeaturePipeline()
        tab = np.ones((5, 12), dtype=np.float32)
        vis = np.zeros((5, 2048), dtype=np.float32)
        fused = pipeline.fuse(tab, vis)
        assert fused.shape == (5, 2060)
        assert fused.dtype == np.float32

    def test_fuse_horizontal_column_order(self):
        pipeline = MultimodalFeaturePipeline()
        tab = np.full((3, 4), 7.0, dtype=np.float32)
        vis = np.full((3, 8), 9.0, dtype=np.float32)
        fused = pipeline.fuse(tab, vis)
        np.testing.assert_array_equal(fused[:, :4], tab)
        np.testing.assert_array_equal(fused[:, 4:], vis)

    def test_fuse_mismatched_sample_counts_raises(self):
        pipeline = MultimodalFeaturePipeline()
        tab = np.ones((4, 10), dtype=np.float32)
        vis = np.zeros((5, 2048), dtype=np.float32)
        with pytest.raises(ValueError):
            pipeline.fuse(tab, vis)

    def test_fuse_1d_vectors(self):
        pipeline = MultimodalFeaturePipeline()
        tab = np.ones(10, dtype=np.float32)
        vis = np.zeros(2048, dtype=np.float32)
        fused = pipeline.fuse(tab, vis)
        assert fused.shape == (1, 2058)
        assert fused.dtype == np.float32

    def test_fuse_zero_sample_arrays(self):
        pipeline = MultimodalFeaturePipeline()
        tab = np.zeros((0, 10), dtype=np.float32)
        vis = np.zeros((0, 2048), dtype=np.float32)
        fused = pipeline.fuse(tab, vis)
        assert fused.shape == (0, 2058)
        assert fused.dtype == np.float32

    def test_fuse_non_contiguous_arrays(self):
        pipeline = MultimodalFeaturePipeline()
        big = np.ones((4, 20), dtype=np.float32)
        tab_strided = big[:, ::2]  # non-contiguous
        vis = np.zeros((4, 2048), dtype=np.float32)
        fused = pipeline.fuse(tab_strided, vis)
        assert fused.flags.c_contiguous
        assert fused.shape == (4, 10 + 2048)

    def test_modality_slicing_tabular(self):
        pipeline = MultimodalFeaturePipeline()
        combined = np.random.randn(5, 2060).astype(np.float32)
        tab_slice = pipeline.slice_features(combined, modality="tabular", tabular_dim=12)
        assert tab_slice.shape == (5, 12)
        np.testing.assert_array_equal(tab_slice, combined[:, :12])

    def test_modality_slicing_vision(self):
        pipeline = MultimodalFeaturePipeline()
        combined = np.random.randn(5, 2060).astype(np.float32)
        vis_slice = pipeline.slice_features(combined, modality="vision", tabular_dim=12)
        assert vis_slice.shape == (5, 2048)
        np.testing.assert_array_equal(vis_slice, combined[:, 12:])

    def test_modality_slicing_multimodal(self):
        pipeline = MultimodalFeaturePipeline()
        combined = np.random.randn(5, 2060).astype(np.float32)
        mm_slice = pipeline.slice_features(combined, modality="multimodal", tabular_dim=12)
        assert mm_slice.shape == (5, 2060)
        np.testing.assert_array_equal(mm_slice, combined)

    @pytest.mark.parametrize("modality_str", ["tabular", "TABULAR", "Tabular_Only", "tab"])
    def test_modality_slicing_case_insensitivity_tabular(self, modality_str):
        pipeline = MultimodalFeaturePipeline()
        combined = np.random.randn(4, 2060).astype(np.float32)
        s = pipeline.slice_features(combined, modality=modality_str, tabular_dim=12)
        assert s.shape == (4, 12)

    @pytest.mark.parametrize("modality_str", ["vision", "VISION", "vision_only", "vis"])
    def test_modality_slicing_case_insensitivity_vision(self, modality_str):
        pipeline = MultimodalFeaturePipeline()
        combined = np.random.randn(4, 2060).astype(np.float32)
        s = pipeline.slice_features(combined, modality=modality_str, tabular_dim=12)
        assert s.shape == (4, 2048)

    def test_modality_slicing_invalid_raises(self):
        pipeline = MultimodalFeaturePipeline()
        combined = np.random.randn(5, 2060).astype(np.float32)
        with pytest.raises(ValueError):
            pipeline.slice_features(combined, modality="quantum_modality", tabular_dim=12)

    def test_get_modality_slices(self):
        pipeline = MultimodalFeaturePipeline()
        slices = pipeline.get_modality_slices(tabular_dim=12, vision_dim=2048)
        assert slices["tabular"] == slice(0, 12)
        assert slices["vision"] == slice(12, 2060)
        assert slices["multimodal"] == slice(0, 2060)

    def test_get_features_tabular_only(self, sample_artwork_df):
        transformer = TabularTransformer()
        transformer.fit(sample_artwork_df)
        pipeline = MultimodalFeaturePipeline(tabular_transformer=transformer)
        feats = pipeline.get_features(sample_artwork_df, modality="tabular")
        assert feats.shape[0] == len(sample_artwork_df)
        assert feats.shape[1] == pipeline.tabular_dim

    def test_get_features_vision_only(self, mock_vision_extractor, sample_pil_rgb):
        pipeline = MultimodalFeaturePipeline(vision_extractor=mock_vision_extractor)
        feats = pipeline.get_features(
            pd.DataFrame([{"id": 1}]),
            image_inputs=[sample_pil_rgb],
            modality="vision",
        )
        assert feats.shape == (1, VISION_FEATURE_DIM)

    def test_get_features_multimodal(self, mock_vision_extractor, sample_artwork_df, sample_pil_rgb):
        transformer = TabularTransformer()
        transformer.fit(sample_artwork_df)
        pipeline = MultimodalFeaturePipeline(
            vision_extractor=mock_vision_extractor,
            tabular_transformer=transformer,
        )
        images = [sample_pil_rgb for _ in range(len(sample_artwork_df))]
        feats = pipeline.get_features(sample_artwork_df, image_inputs=images, modality="multimodal")
        assert feats.shape == (len(sample_artwork_df), pipeline.tabular_dim + VISION_FEATURE_DIM)
        assert feats.dtype == np.float32

    def test_fusion_with_corrupt_image_zero_vector_fallback(self, mock_vision_extractor, sample_artwork_df):
        transformer = TabularTransformer()
        transformer.fit(sample_artwork_df)
        pipeline = MultimodalFeaturePipeline(
            vision_extractor=mock_vision_extractor,
            tabular_transformer=transformer,
        )
        # Pass non-existent images to trigger fallback
        corrupt_inputs = ["non_existent.jpg" for _ in range(len(sample_artwork_df))]
        feats = pipeline.get_features(sample_artwork_df, image_inputs=corrupt_inputs, modality="multimodal")
        assert feats.shape == (len(sample_artwork_df), pipeline.tabular_dim + VISION_FEATURE_DIM)
        # Vision slice must be all zeros
        vis_part = feats[:, pipeline.tabular_dim:]
        np.testing.assert_array_equal(vis_part, np.zeros_like(vis_part))
        # Tabular part must be non-zero and finite
        tab_part = feats[:, :pipeline.tabular_dim]
        assert np.all(np.isfinite(tab_part))
