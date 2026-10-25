"""
CanvasIQ Milestone 2 Empirical Adversarial Stress Test Suite.

Adversarially challenges:
1. TabularTransformer:
   - Bayesian shrinkage mathematical identity against theoretical oracle
   - Asymptotic sample size behavior (n=1 vs n=1000)
   - 100% unseen artists and extreme outlier targets (0 to 10^14)
   - One-hot medium encoding edge cases (unseen media, empty strings, case mismatches)
   - Continuous feature scaling (zero-variance/uniform data, negative/NaN dimensions, single-row inference)
   - State preservation under pickle and joblib serialization roundtrips
2. MultimodalFeaturePipeline:
   - Late concatenation shape invariance and contiguous float32 preservation
   - Zero-vector visual fallback integration
   - Case-insensitive modality slicing and alias variations
   - Mathematical slice equality identities
   - Invalid modality string and missing tabular dimension rejections
   - End-to-end get_features orchestration
"""

from __future__ import annotations

import io
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import pytest
from PIL import Image

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
# 1. TabularTransformer Adversarial Stress Tests
# ==============================================================================

class TestBayesianShrinkageAdversarial:
    """Stress tests on Bayesian smoothed target encoding and mathematical shrinkage."""

    @pytest.mark.parametrize("smoothing", [0.1, 1.0, 5.0, 10.0, 25.0, 100.0])
    def test_bayesian_shrinkage_mathematical_oracle(self, smoothing: float):
        """
        Verify that TabularTransformer's empirical target encoding strictly matches
        the theoretical formula S(a) = (n_a * y_bar_a + m * mu) / (n_a + m)
        within floating point tolerance across arbitrary sample sizes and smoothing factors.
        """
        # Create dataset with diverse artist frequencies: 1, 3, 10, 50, 200
        artist_counts = {"Solitary": 1, "Trio": 3, "Decade": 10, "Fifty": 50, "Centennial": 200}
        rows = []
        rng = np.random.RandomState(42)

        for artist, count in artist_counts.items():
            base_log_price = float(rng.uniform(5.0, 15.0))
            for _ in range(count):
                price = float(np.expm1(base_log_price + rng.normal(0, 0.5)))
                rows.append({
                    "artist": artist,
                    "medium": "Oil on Canvas",
                    "height": 50.0,
                    "width": 60.0,
                    "year": 1950,
                    "price": max(10.0, price),
                })

        df = pd.DataFrame(rows)
        log_prices = np.log1p(df["price"].values)
        oracle_mu = float(np.mean(log_prices))

        transformer = TabularTransformer(smoothing=smoothing)
        transformer.fit(df, y=log_prices)

        # Check global mean match
        assert np.isclose(transformer.global_target_mean_, oracle_mu, rtol=1e-6)

        # Verify each artist against theoretical oracle
        for artist, count in artist_counts.items():
            artist_mask = df["artist"] == artist
            artist_log_prices = log_prices[artist_mask]
            oracle_y_bar = float(np.mean(artist_log_prices))
            oracle_expected = (count * oracle_y_bar + smoothing * oracle_mu) / (count + smoothing)

            actual_encoded = transformer.artist_target_map_[artist]
            np.testing.assert_allclose(
                actual_encoded,
                oracle_expected,
                rtol=1e-5,
                err_msg=f"Shrinkage mismatch for artist {artist} with count={count}, smoothing={smoothing}",
            )

    def test_asymptotic_sample_size_behavior(self):
        """
        Verify asymptotic properties:
        - For a rare artist with n=1, estimate S(a) is dominated by the prior mean mu.
        - For a prolific master with n=1000, estimate S(a) is dominated by the empirical mean y_bar.
        """
        m = 10.0
        rare_artist = "Rare Newcomer"
        master_artist = "Prolific Master"

        # Global average log price ~ 10.0
        # Rare artist sells 1 painting for log price 20.0
        # Master artist sells 1000 paintings for average log price 20.0
        rows = []
        for _ in range(100):
            rows.append({"artist": "Baseline", "medium": "Oil", "height": 50.0, "width": 50.0, "year": 2000, "price": np.expm1(10.0)})
        rows.append({"artist": rare_artist, "medium": "Oil", "height": 50.0, "width": 50.0, "year": 2000, "price": np.expm1(20.0)})
        for _ in range(1000):
            rows.append({"artist": master_artist, "medium": "Oil", "height": 50.0, "width": 50.0, "year": 2000, "price": np.expm1(20.0)})

        df = pd.DataFrame(rows)
        transformer = TabularTransformer(smoothing=m).fit(df)

        mu = transformer.global_target_mean_
        rare_val = transformer.artist_target_map_[rare_artist]
        master_val = transformer.artist_target_map_[master_artist]

        # Rare artist estimate should be pulled heavily towards mu:
        # Distance |rare - mu| should be much smaller than |rare - 20.0|
        assert abs(rare_val - mu) < abs(rare_val - 20.0)
        # Master artist estimate should be within 1% of 20.0:
        assert abs(master_val - 20.0) / 20.0 < 0.01

    def test_100_percent_unseen_artists_at_inference(self):
        """
        At inference time, if all artists in the batch are previously unseen,
        every row must receive the exact global target mean mu without error or NaN.
        """
        train_df = pd.DataFrame({
            "artist": ["Monet", "Picasso", "Da Vinci"],
            "medium": ["Oil", "Oil", "Fresco"],
            "height": [50.0, 60.0, 70.0],
            "width": [50.0, 60.0, 70.0],
            "year": [1900, 1930, 1500],
            "price": [10000.0, 50000.0, 100000.0],
        })
        transformer = TabularTransformer().fit(train_df)
        mu = transformer.global_target_mean_

        inference_df = pd.DataFrame({
            "artist": ["Anonymous Contemporary", "Digital Creator X", "Unknown Studio", "Random Persona 99"],
            "medium": ["Oil", "Oil", "Oil", "Oil"],
            "height": [40.0, 45.0, 50.0, 55.0],
            "width": [40.0, 45.0, 50.0, 55.0],
            "year": [2020, 2021, 2022, 2023],
        })

        out = transformer.transform(inference_df)
        assert out.shape[0] == 4
        # Column 0 is artist target encoding
        np.testing.assert_allclose(out[:, 0], mu, rtol=1e-5)
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()

    def test_extreme_outlier_prices_stability(self):
        """
        Verify mathematical stability under extreme prices:
        - price near zero ($0.01 -> log1p near zero)
        - astronomical price ($10^14 -> log1p ~ 32.23)
        """
        extreme_df = pd.DataFrame({
            "artist": ["Budget Artist", "Astronomical Billionaire", "Standard Artist"],
            "medium": ["Sketch", "Sculpture", "Oil"],
            "height": [10.0, 500.0, 50.0],
            "width": [10.0, 500.0, 50.0],
            "year": [2020, 2010, 1980],
            "price": [0.01, 1e14, 50000.0],
        })

        transformer = TabularTransformer().fit(extreme_df)
        out = transformer.transform(extreme_df)

        assert not np.isnan(out).any()
        assert not np.isinf(out).any()
        assert np.isfinite(transformer.global_target_mean_)
        for artist, val in transformer.artist_target_map_.items():
            assert np.isfinite(val)

    def test_smoothing_boundary_extremes(self):
        """
        Verify behavior at theoretical smoothing boundaries:
        - m = 0.0: No Bayesian shrinkage; each artist maps to pure empirical mean.
        - m = 1e8: Extreme shrinkage; every artist collapses to global mean mu.
        """
        df = pd.DataFrame({
            "artist": ["ArtistA"] * 5 + ["ArtistB"] * 5,
            "medium": ["Oil"] * 10,
            "height": [20.0] * 10,
            "width": [20.0] * 10,
            "year": [1990] * 10,
            "price": [100.0] * 5 + [10000.0] * 5,
        })
        log_p = np.log1p(df["price"].values)
        mean_A = float(np.mean(log_p[:5]))
        mean_B = float(np.mean(log_p[5:]))
        global_mu = float(np.mean(log_p))

        # m = 0.0
        trans_m0 = TabularTransformer(smoothing=0.0).fit(df)
        assert np.isclose(trans_m0.artist_target_map_["ArtistA"], mean_A, rtol=1e-5)
        assert np.isclose(trans_m0.artist_target_map_["ArtistB"], mean_B, rtol=1e-5)

        # m = 1e8
        trans_m_inf = TabularTransformer(smoothing=1e8).fit(df)
        assert np.isclose(trans_m_inf.artist_target_map_["ArtistA"], global_mu, rtol=1e-4)
        assert np.isclose(trans_m_inf.artist_target_map_["ArtistB"], global_mu, rtol=1e-4)


# ==============================================================================
# 2. Categorical Medium Encoding Adversarial Stress Tests
# ==============================================================================

class TestCategoricalMediumAdversarial:
    """Stress tests on one-hot categorical medium encoding and edge cases."""

    def test_unseen_mediums_map_to_all_zeros(self):
        """
        Unseen artwork mediums encountered during inference must map to all zeros
        across one-hot columns without raising errors or altering column dimensions.
        """
        train_df = pd.DataFrame({
            "artist": ["A", "B", "C"],
            "medium": ["Oil on Canvas", "Watercolor", "Bronze"],
            "height": [50.0, 30.0, 100.0],
            "width": [40.0, 20.0, 80.0],
            "year": [1900, 1950, 2000],
            "price": [5000.0, 1500.0, 25000.0],
        })
        transformer = TabularTransformer().fit(train_df)
        n_features = len(transformer.get_feature_names_out())

        # Inference contains 100% unseen mediums
        infer_df = pd.DataFrame({
            "artist": ["A", "B"],
            "medium": ["Digital Generative NFT", "Neon Hologram Light Installation"],
            "height": [50.0, 60.0],
            "width": [50.0, 60.0],
            "year": [2024, 2025],
        })

        out = transformer.transform(infer_df)
        assert out.shape == (2, n_features)

        # Identify medium columns (indices 6 through 8: 1 artist + 5 numerical + 3 mediums)
        med_start = 1 + len(NUMERICAL_FEATURES)
        med_slice = out[:, med_start:]
        assert med_slice.shape == (2, 3)
        assert np.all(med_slice == 0.0), f"Expected all zeros for unseen media, got: {med_slice}"

    def test_empty_string_and_whitespace_medium_sanitization(self):
        """
        Empty strings, whitespace-only strings, None, and NaN mediums must be
        sanitized to 'Unknown' and transformed reliably.
        """
        train_df = pd.DataFrame({
            "artist": ["A", "B", "C"],
            "medium": ["Unknown", "Oil", "Acrylic"],
            "height": [50.0, 50.0, 50.0],
            "width": [50.0, 50.0, 50.0],
            "year": [2000, 2000, 2000],
            "price": [1000.0, 2000.0, 3000.0],
        })
        transformer = TabularTransformer().fit(train_df)

        adversarial_df = pd.DataFrame({
            "artist": ["A", "A", "A", "A"],
            "medium": ["", "   ", "\t\n", None],
            "height": [50.0, 50.0, 50.0, 50.0],
            "width": [50.0, 50.0, 50.0, 50.0],
            "year": [2000, 2000, 2000, 2000],
        })

        out = transformer.transform(adversarial_df)
        assert out.shape[0] == 4
        assert not np.isnan(out).any()

        # All 4 rows should have identical representations because they sanitize to 'Unknown'
        for i in range(1, 4):
            np.testing.assert_array_equal(out[i], out[0])

    def test_case_variation_medium_handling(self):
        """
        Case variations (e.g. 'oil on canvas' vs 'Oil on Canvas') should be treated
        gracefully as unknown categories without crashing.
        """
        train_df = pd.DataFrame({
            "artist": ["A"],
            "medium": ["Oil on Canvas"],
            "height": [50.0],
            "width": [50.0],
            "year": [2000],
            "price": [1000.0],
        })
        transformer = TabularTransformer().fit(train_df)

        case_variants_df = pd.DataFrame({
            "artist": ["A", "A", "A"],
            "medium": ["Oil on Canvas", "oil on canvas", "OIL ON CANVAS"],
            "height": [50.0, 50.0, 50.0],
            "width": [50.0, 50.0, 50.0],
            "year": [2000, 2000, 2000],
        })

        out = transformer.transform(case_variants_df)
        assert out.shape == (3, len(transformer.get_feature_names_out()))
        # Row 0 exact match: medium column should be 1.0
        assert out[0, -1] == 1.0
        # Rows 1 and 2 case mismatches: handled gracefully by OneHotEncoder as 0.0
        assert out[1, -1] == 0.0
        assert out[2, -1] == 0.0


# ==============================================================================
# 3. Continuous Feature Scaling Adversarial Stress Tests
# ==============================================================================

class TestContinuousScalingAdversarial:
    """Stress tests on continuous numerical feature scaling and degenerate matrices."""

    def test_zero_variance_uniform_columns_handling(self):
        """
        When all continuous features are perfectly uniform (variance = 0),
        StandardScaler scale_ clamping must prevent divide-by-zero, returning
        zeros for training data and finite non-NaN outputs for unseen records.
        """
        uniform_df = pd.DataFrame({
            "artist": ["A", "B", "C"],
            "medium": ["Oil", "Oil", "Oil"],
            "height": [50.0, 50.0, 50.0],
            "width": [50.0, 50.0, 50.0],
            "year": [2000, 2000, 2000],
            "price": [100.0, 200.0, 300.0],
        })

        transformer = TabularTransformer().fit(uniform_df)

        # Scale array should be clamped to 1.0 for zero variance features
        assert hasattr(transformer.scaler, "scale_")
        assert np.all(transformer.scaler.scale_ == 1.0)

        # Transform on uniform training data should yield exact 0.0 for continuous features
        out_train = transformer.transform(uniform_df)
        num_slice = out_train[:, 1:6]  # 5 numerical features
        np.testing.assert_array_equal(num_slice, np.zeros((3, 5), dtype=np.float32))
        assert not np.isnan(out_train).any()

        # Transform on new data with different dimensions must not produce NaNs or Infs
        novel_df = pd.DataFrame({
            "artist": ["A"],
            "medium": ["Oil"],
            "height": [120.0],
            "width": [80.0],
            "year": [1980],
        })
        out_novel = transformer.transform(novel_df)
        assert not np.isnan(out_novel).any()
        assert not np.isinf(out_novel).any()
        assert np.all(np.isfinite(out_novel))

    def test_single_row_fit_and_inference(self):
        """
        TabularTransformer must support single-row fitting (N=1) and single-row inference
        without shape collapses or variance crashes.
        """
        single_row = pd.DataFrame({
            "artist": ["Solo Master"],
            "medium": ["Bronze"],
            "height": [150.0],
            "width": [75.0],
            "year": [1975],
            "price": [75000.0],
        })

        transformer = TabularTransformer()
        out_fit = transformer.fit_transform(single_row)
        assert out_fit.shape == (1, 7)  # 1 artist + 5 num + 1 medium
        assert not np.isnan(out_fit).any()

        # Single row inference on unseen artwork
        out_infer = transformer.transform(pd.DataFrame({
            "artist": ["Other"],
            "medium": ["Bronze"],
            "height": [100.0],
            "width": [50.0],
            "year": [2010],
        }))
        assert out_infer.shape == (1, 7)
        assert not np.isnan(out_infer).any()

    def test_negative_dimensions_and_nan_inputs(self):
        """
        Defensive sanitization of negative dimensions, future years, and NaN inputs.
        Transformer must coerce and bound inputs, producing finite float32 arrays.
        """
        train_df = pd.DataFrame({
            "artist": ["A", "B"],
            "medium": ["Oil", "Acrylic"],
            "height": [10.0, 20.0],
            "width": [10.0, 20.0],
            "year": [1990, 2010],
            "price": [500.0, 1000.0],
        })
        transformer = TabularTransformer().fit(train_df)

        corrupt_df = pd.DataFrame({
            "artist": ["A", None, "Unseen"],
            "medium": ["Oil", None, ""],
            "height": [-50.0, np.nan, 0.0],
            "width": [np.nan, -25.0, 0.0],
            "year": [np.nan, 3050, -100],
        })

        out = transformer.transform(corrupt_df)
        assert out.shape == (3, len(transformer.get_feature_names_out()))
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()
        assert out.dtype == np.float32

    def test_empty_dataframe_transform(self):
        """Transforming an empty DataFrame must return an empty array with matching column width."""
        train_df = pd.DataFrame({
            "artist": ["A"],
            "medium": ["Oil"],
            "height": [10.0],
            "width": [10.0],
            "year": [2000],
            "price": [500.0],
        })
        transformer = TabularTransformer().fit(train_df)
        n_features = len(transformer.get_feature_names_out())

        empty_df = pd.DataFrame(columns=["artist", "medium", "height", "width", "year"])
        out = transformer.transform(empty_df)
        assert out.shape == (0, n_features)
        assert out.dtype == np.float32


# ==============================================================================
# 4. Serialization Roundtrip Preservation Tests
# ==============================================================================

class TestSerializationRoundtripAdversarial:
    """Stress tests on state persistence across pickle and joblib roundtrips."""

    @pytest.fixture
    def fitted_transformer(self) -> TabularTransformer:
        df = pd.DataFrame({
            "artist": ["Artist A", "Artist B", "Artist C"],
            "medium": ["Oil", "Watercolor", "Bronze"],
            "height": [50.0, 60.0, 70.0],
            "width": [40.0, 50.0, 60.0],
            "year": [1900, 1950, 2000],
            "price": [10000.0, 5000.0, 20000.0],
        })
        return TabularTransformer(smoothing=15.0, reference_year=2026).fit(df)

    def test_pickle_roundtrip_preservation(self, fitted_transformer: TabularTransformer):
        """Verify TabularTransformer state and transform outputs are identical after pickle roundtrip."""
        test_df = pd.DataFrame({
            "artist": ["Artist A", "Unseen Artist"],
            "medium": ["Oil", "Unknown Medium"],
            "height": [55.0, 20.0],
            "width": [45.0, 15.0],
            "year": [1920, 2024],
        })

        orig_out = fitted_transformer.transform(test_df)

        pickled = pickle.dumps(fitted_transformer)
        reloaded: TabularTransformer = pickle.loads(pickled)

        assert reloaded.smoothing == fitted_transformer.smoothing
        assert reloaded.reference_year == fitted_transformer.reference_year
        assert reloaded.global_target_mean_ == fitted_transformer.global_target_mean_
        assert reloaded.feature_names_ == fitted_transformer.feature_names_

        reloaded_out = reloaded.transform(test_df)
        np.testing.assert_array_equal(orig_out, reloaded_out)

    def test_joblib_roundtrip_preservation(self, fitted_transformer: TabularTransformer):
        """Verify TabularTransformer state and transform outputs are identical after joblib disk roundtrip."""
        test_df = pd.DataFrame({
            "artist": ["Artist B", "Unseen Artist 2"],
            "medium": ["Watercolor", "Neon"],
            "height": [65.0, 10.0],
            "width": [55.0, 10.0],
            "year": [1960, 2020],
        })

        orig_out = fitted_transformer.transform(test_df)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            joblib.dump(fitted_transformer, tmp_path)
            reloaded: TabularTransformer = joblib.load(tmp_path)

            assert reloaded.feature_names_ == fitted_transformer.feature_names_
            reloaded_out = reloaded.transform(test_df)
            np.testing.assert_array_equal(orig_out, reloaded_out)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_multimodal_pipeline_serialization_roundtrip(self, fitted_transformer: TabularTransformer):
        """Verify MultimodalFeaturePipeline can be pickled and serialized via joblib."""
        mock_vision = ResNet50FeatureExtractor(device="cpu", use_mock=True)
        pipeline = MultimodalFeaturePipeline(
            vision_extractor=mock_vision,
            tabular_transformer=fitted_transformer,
        )

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            joblib.dump(pipeline, tmp_path)
            reloaded: MultimodalFeaturePipeline = joblib.load(tmp_path)

            assert reloaded.tabular_dim == pipeline.tabular_dim
            assert reloaded.tabular_transformer is not None
            assert reloaded.vision_extractor is not None
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


# ==============================================================================
# 5. Multimodal Late Fusion & Modality Slicing Stress Tests
# ==============================================================================

class TestMultimodalLateFusionAdversarial:
    """Stress tests on multimodal horizontal late fusion and functional modality slicing."""

    @pytest.fixture
    def pipeline(self) -> MultimodalFeaturePipeline:
        return MultimodalFeaturePipeline()

    def test_late_concatenation_dimensions_and_types(self, pipeline: MultimodalFeaturePipeline):
        """Verify [X_tab || X_vis] horizontal concatenation preserves shape and contiguous float32."""
        N = 25
        D_t = 12
        D_v = VISION_FEATURE_DIM

        X_tab = np.random.randn(N, D_t).astype(np.float64)  # Test upcast conversion
        X_vis = np.random.randn(N, D_v).astype(np.float32)

        fused = pipeline.fuse(X_tab, X_vis)

        assert fused.shape == (N, D_t + D_v)
        assert fused.dtype == np.float32
        assert fused.flags["C_CONTIGUOUS"]
        np.testing.assert_allclose(fused[:, :D_t], X_tab.astype(np.float32), rtol=1e-6)
        np.testing.assert_allclose(fused[:, D_t:], X_vis, rtol=1e-6)

    def test_single_vector_1d_inputs_fusion(self, pipeline: MultimodalFeaturePipeline):
        """Verify 1D tabular vector and 1D vision vector fuse into a (1, D_t + D_v) 2D array."""
        D_t = 8
        D_v = VISION_FEATURE_DIM
        x_tab_1d = np.ones(D_t, dtype=np.float32) * 5.0
        x_vis_1d = np.ones(D_v, dtype=np.float32) * 3.0

        fused = pipeline.fuse(x_tab_1d, x_vis_1d)
        assert fused.shape == (1, D_t + D_v)
        assert np.all(fused[0, :D_t] == 5.0)
        assert np.all(fused[0, D_t:] == 3.0)

    def test_sample_count_mismatch_rejection(self, pipeline: MultimodalFeaturePipeline):
        """Verify ValueError is raised when tabular and visual arrays have mismatched sample counts."""
        X_tab = np.zeros((10, 5), dtype=np.float32)
        X_vis = np.zeros((9, 2048), dtype=np.float32)

        with pytest.raises(ValueError, match="Sample count mismatch"):
            pipeline.fuse(X_tab, X_vis)

    def test_zero_vector_fallback_integration(self, pipeline: MultimodalFeaturePipeline):
        """
        Verify that all-zero fallback visual vectors integrate cleanly into the fused representation
        without producing NaNs and maintaining strict slice identity.
        """
        N = 10
        D_t = 6
        D_v = 2048

        X_tab = np.random.uniform(1.0, 100.0, size=(N, D_t)).astype(np.float32)
        X_vis_fallback = np.zeros((N, D_v), dtype=np.float32)

        fused = pipeline.fuse(X_tab, X_vis_fallback)

        assert fused.shape == (N, D_t + D_v)
        assert not np.isnan(fused).any()
        assert not np.isinf(fused).any()

        # Tabular slice matches exactly
        np.testing.assert_array_equal(fused[:, :D_t], X_tab)
        # Visual slice is purely zeros
        assert np.all(fused[:, D_t:] == 0.0)
        assert np.linalg.norm(fused[:, D_t:]) == 0.0

    @pytest.mark.parametrize("tabular_alias", [
        "tabular", "TABULAR", "Tabular", "tabular_only", "TABULAR_ONLY", "tab", "TAB", "  tabular  "
    ])
    def test_tabular_modality_slicing_aliases(self, pipeline: MultimodalFeaturePipeline, tabular_alias: str):
        """Verify case-insensitive slicing for all tabular modality aliases."""
        N, D_t, D_v = 8, 9, 2048
        X_tab = np.random.randn(N, D_t).astype(np.float32)
        X_vis = np.random.randn(N, D_v).astype(np.float32)
        fused = pipeline.fuse(X_tab, X_vis)

        sliced = pipeline.slice_features(fused, modality=tabular_alias, tabular_dim=D_t)
        assert sliced.shape == (N, D_t)
        np.testing.assert_array_equal(sliced, X_tab)

    @pytest.mark.parametrize("vision_alias", [
        "vision", "VISION", "Vision", "vision_only", "VISION_ONLY", "vis", "VIS", " vision "
    ])
    def test_vision_modality_slicing_aliases(self, pipeline: MultimodalFeaturePipeline, vision_alias: str):
        """Verify case-insensitive slicing for all vision modality aliases."""
        N, D_t, D_v = 8, 9, 2048
        X_tab = np.random.randn(N, D_t).astype(np.float32)
        X_vis = np.random.randn(N, D_v).astype(np.float32)
        fused = pipeline.fuse(X_tab, X_vis)

        sliced = pipeline.slice_features(fused, modality=vision_alias, tabular_dim=D_t)
        assert sliced.shape == (N, D_v)
        np.testing.assert_array_equal(sliced, X_vis)

    @pytest.mark.parametrize("multimodal_alias", [
        "multimodal", "MULTIMODAL", "Multimodal", "both", "BOTH", "mm", "MM", "combined", "Combined", " multimodal "
    ])
    def test_multimodal_modality_slicing_aliases(self, pipeline: MultimodalFeaturePipeline, multimodal_alias: str):
        """Verify case-insensitive slicing for all multimodal/combined aliases."""
        N, D_t, D_v = 8, 9, 2048
        X_tab = np.random.randn(N, D_t).astype(np.float32)
        X_vis = np.random.randn(N, D_v).astype(np.float32)
        fused = pipeline.fuse(X_tab, X_vis)

        sliced = pipeline.slice_features(fused, modality=multimodal_alias, tabular_dim=D_t)
        assert sliced.shape == (N, D_t + D_v)
        np.testing.assert_array_equal(sliced, fused)

    def test_invalid_modality_string_rejection(self, pipeline: MultimodalFeaturePipeline):
        """Verify ValueError is raised on invalid or unsupported modality strings."""
        fused = np.zeros((4, 2056), dtype=np.float32)
        for invalid_mod in ["audio", "text", "graph", "unknown", "", "   "]:
            with pytest.raises(ValueError, match="Unknown modality"):
                pipeline.slice_features(fused, modality=invalid_mod, tabular_dim=8)

    def test_missing_tabular_dim_rejection(self, pipeline: MultimodalFeaturePipeline):
        """Verify ValueError is raised when tabular_dim is omitted and transformer is unfitted."""
        fused = np.zeros((4, 2056), dtype=np.float32)
        with pytest.raises(ValueError, match="tabular_dim must be provided"):
            pipeline.slice_features(fused, modality="tabular")

    def test_modality_slices_dictionary(self, pipeline: MultimodalFeaturePipeline):
        """Verify get_modality_slices returns exact slice boundaries matching array partitions."""
        D_t = 11
        D_v = 2048
        slices = pipeline.get_modality_slices(tabular_dim=D_t, vision_dim=D_v)

        assert slices["tabular"] == slice(0, D_t)
        assert slices["vision"] == slice(D_t, D_t + D_v)
        assert slices["multimodal"] == slice(0, D_t + D_v)

    def test_end_to_end_orchestration_with_fallbacks(self):
        """
        Verify end-to-end get_features orchestration:
        - When image inputs are missing or None, pipeline seamlessly falls back to zero-vectors.
        - Fused output aligns tabular metadata and visual representations without NaNs.
        """
        train_df = pd.DataFrame({
            "artist": ["Monet", "Picasso"],
            "medium": ["Oil on Canvas", "Sculpture"],
            "height": [60.0, 120.0],
            "width": [80.0, 40.0],
            "year": [1905, 1935],
            "price": [25000000.0, 12000000.0],
        })

        transformer = TabularTransformer().fit(train_df)
        vision_mock = ResNet50FeatureExtractor(device="cpu", use_mock=True)
        pipeline = MultimodalFeaturePipeline(vision_extractor=vision_mock, tabular_transformer=transformer)

        # Call with no image inputs provided (must fallback to zero vectors)
        infer_df = pd.DataFrame({
            "artist": ["Monet", "Unknown"],
            "medium": ["Oil on Canvas", "Bronze"],
            "height": [65.0, 50.0],
            "width": [85.0, 50.0],
            "year": [1908, 2000],
        })

        fused = pipeline.get_features(infer_df, image_inputs=None, modality="multimodal")
        assert fused.shape == (2, pipeline.tabular_dim + 2048)
        assert not np.isnan(fused).any()

        # Since no images were provided, visual component must be all-zeros
        assert np.all(fused[:, pipeline.tabular_dim:] == 0.0)

        # Extracting tabular only via pipeline
        tab_only = pipeline.get_features(infer_df, modality="tabular")
        assert tab_only.shape == (2, pipeline.tabular_dim)
        np.testing.assert_array_equal(tab_only, fused[:, :pipeline.tabular_dim])
