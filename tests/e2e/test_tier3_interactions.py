"""Tier 3: Cross-Feature Combinations & Pairwise Interactions E2E Test Suite for CanvasIQ.

Tests multi-component pipelines, cross-boundary contracts, ablation synergies,
and system-level feature interactions.
"""

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Repo root setup
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.conftest import (
    get_component,
    skip_test,
    make_sample_artwork_dict,
    make_sample_pil_image,
    make_diverse_artwork_records,
)

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None


class TestTier3CrossFeatureInteractions(unittest.TestCase):
    """Tier 3: Pairwise and Cross-Feature Integration Scenarios."""

    def test_t3_01_synthetic_benchmark_to_schema_validator(self):
        """Cross-feature: Procedural synthetic benchmark -> Schema validator."""
        GenClass = get_component("src.data.synthetic", "SyntheticArtDatasetGenerator", milestone="M1")
        validate_fn = get_component("src.data.validator", "validate_artwork_schema", milestone="M1")
        tmp_dir = tempfile.mkdtemp(prefix="caniq_t3_bench_")
        try:
            gen = GenClass(seed=42)
            df, _ = gen.generate_benchmark(n_samples=5, output_dir=tmp_dir)
            validated = validate_fn(df, is_training=True)
            self.assertEqual(len(validated), 5)
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_t3_02_schema_validator_to_cleaner(self):
        """Cross-feature: Validated records -> Data cleaner transformations."""
        validate_fn = get_component("src.data.validator", "validate_artwork_schema", milestone="M1")
        CleanerClass = get_component("src.data.cleaner", "ArtDataCleaner", milestone="M1")
        cleaner = CleanerClass()
        record = make_sample_artwork_dict()
        val_df = validate_fn(record, is_training=True)
        cleaned_df = cleaner.clean(val_df, is_training=True)
        self.assertIn("surface_area", cleaned_df.columns)
        self.assertIn("aspect_ratio", cleaned_df.columns)
        self.assertAlmostEqual(cleaned_df.iloc[0]["surface_area"], 65.0 * 81.0, places=2)

    def test_t3_03_cleaner_to_tabular_transformer(self):
        """Cross-feature: Cleaned records -> TabularTransformer encoding."""
        if pd is None or np is None:
            skip_test("pandas or numpy not installed")
        CleanerClass = get_component("src.data.cleaner", "ArtDataCleaner", milestone="M1")
        TransClass = get_component("src.features.tabular", "TabularTransformer", milestone="M2")
        cleaner = CleanerClass()
        transformer = TransClass()
        df = pd.DataFrame(make_diverse_artwork_records(6))
        cleaned = cleaner.clean(df, is_training=True)
        y = np.log1p(cleaned["price"].values)
        transformed = transformer.fit_transform(cleaned, y)
        self.assertEqual(transformed.shape[0], 6)
        self.assertTrue(np.all(np.isfinite(transformed)))

    def test_t3_04_procedural_images_to_vision_extractor(self):
        """Cross-feature: Procedural image generator -> ResNet50 extractor."""
        if np is None:
            skip_test("numpy not installed")
        GenClass = get_component("src.data.synthetic", "SyntheticArtDatasetGenerator", milestone="M1")
        ExtClass = get_component("src.features.vision", "ResNet50FeatureExtractor", milestone="M2")
        gen = GenClass(seed=42)
        extractor = ExtClass(device="cpu")
        img = gen.generate_image(style="colorfield", palette="oceanic", size=(224, 224))
        feats = extractor.extract_image(img)
        self.assertEqual(feats.shape, (2048,))
        self.assertEqual(feats.dtype, np.float32)

    def test_t3_05_multimodal_late_fusion_layer(self):
        """Cross-feature: Tabular features + Vision features -> Fused representation."""
        if np is None:
            skip_test("numpy not installed")
        PipeClass = get_component("src.features.multimodal", "MultimodalFeaturePipeline", milestone="M2")
        pipeline = PipeClass(vision_extractor=None, tabular_transformer=None)
        N = 4
        tab_feats = np.random.randn(N, 14).astype(np.float32)
        vis_feats = np.random.randn(N, 2048).astype(np.float32)
        fused = pipeline.fuse(tab_feats, vis_feats)
        self.assertEqual(fused.shape, (N, 14 + 2048))

    def test_t3_06_multimodal_features_to_model_training(self):
        """Cross-feature: Fused multimodal features -> ArtValuationModel fitting."""
        if np is None:
            skip_test("numpy not installed")
        ModelClass = get_component("src.models.train", "ArtValuationModel", milestone="M3")
        model = ModelClass()
        N = 25
        X = np.random.randn(N, 64).astype(np.float32)
        y = np.linspace(8.0, 14.0, N).astype(np.float32)
        model.fit(X, y)
        preds = model.predict_log(X)
        self.assertEqual(preds.shape, (N,))

    def test_t3_07_regressor_prediction_to_inverse_log_price(self):
        """Cross-feature: Model log predictions -> Native dollar inversion."""
        if np is None:
            skip_test("numpy not installed")
        ModelClass = get_component("src.models.train", "ArtValuationModel", milestone="M3")
        CleanerClass = get_component("src.data.cleaner", "ArtDataCleaner", milestone="M1")
        model = ModelClass()
        cleaner = CleanerClass()
        N = 10
        X = np.random.randn(N, 32).astype(np.float32)
        y = np.linspace(8.0, 13.0, N).astype(np.float32)
        model.fit(X, y)
        log_preds = model.predict_log(X)
        dollars = cleaner.inverse_log_price(log_preds)
        self.assertTrue(all(d >= 0.0 for d in dollars))

    def test_t3_08_prediction_with_confidence_bounds(self):
        """Cross-feature: Model prediction -> Lower and upper valuation bounds."""
        if np is None:
            skip_test("numpy not installed")
        ModelClass = get_component("src.models.train", "ArtValuationModel", milestone="M3")
        model = ModelClass()
        N = 20
        X = np.random.randn(N, 16).astype(np.float32)
        y = np.linspace(9.0, 14.0, N).astype(np.float32)
        model.fit(X, y)
        lower, point, upper = model.predict_with_confidence(X[:5], confidence=0.85)
        self.assertTrue(all(lower <= point + 1e-4))
        self.assertTrue(all(point <= upper + 1e-4))
        self.assertTrue(all(lower >= 0.0))

    def test_t3_09_ablation_synergy_multimodal_vs_tabular(self):
        """Cross-feature: Ablation comparison of Multimodal vs Tabular-only baselines."""
        if np is None:
            skip_test("numpy not installed")
        ablation_fn = get_component("src.models.evaluate", "run_ablation_study", milestone="M3")
        N = 40
        X_tab = np.random.randn(N, 8).astype(np.float32)
        X_vis = np.random.randn(N, 16).astype(np.float32)
        y_price = np.random.uniform(5000, 50000, N).astype(np.float32)
        df = ablation_fn(X_tab, X_vis, y_price)
        self.assertGreaterEqual(len(df), 3)

    def test_t3_10_ablation_synergy_multimodal_vs_vision(self):
        """Cross-feature: Ablation comparison of Multimodal vs Vision-only baselines."""
        if np is None:
            skip_test("numpy not installed")
        ablation_fn = get_component("src.models.evaluate", "run_ablation_study", milestone="M3")
        N = 40
        X_tab = np.random.randn(N, 8).astype(np.float32)
        X_vis = np.random.randn(N, 16).astype(np.float32)
        y_price = np.random.uniform(5000, 50000, N).astype(np.float32)
        df = ablation_fn(X_tab, X_vis, y_price)
        self.assertIn("rmse", [c.lower() for c in df.columns])

    def test_t3_11_missing_image_fallback_in_multimodal_pipeline(self):
        """Cross-feature: Zero-vector image fallback during end-to-end multimodal inference."""
        if np is None:
            skip_test("numpy not installed")
        ExtClass = get_component("src.features.vision", "ResNet50FeatureExtractor", milestone="M2")
        PipeClass = get_component("src.features.multimodal", "MultimodalFeaturePipeline", milestone="M2")
        ModelClass = get_component("src.models.train", "ArtValuationModel", milestone="M3")
        extractor = ExtClass(device="cpu")
        pipeline = PipeClass(vision_extractor=extractor, tabular_transformer=None)
        model = ModelClass()
        # Extract features for missing image
        vis_fallback = extractor.extract_image("non_existent_corrupted_artwork.png")
        tab_mock = np.ones((1, 10), dtype=np.float32)
        fused = pipeline.fuse(tab_mock, vis_fallback.reshape(1, -1))
        # Train and predict
        X_dummy = np.random.randn(15, 2058).astype(np.float32)
        y_dummy = np.linspace(8.0, 12.0, 15).astype(np.float32)
        model.fit(X_dummy, y_dummy)
        pred_price = model.predict_price(fused)
        self.assertGreaterEqual(pred_price[0], 0.0)

    def test_t3_12_unseen_artist_and_medium_in_inference(self):
        """Cross-feature: Novel artist and novel medium in end-to-end inference."""
        if pd is None or np is None:
            skip_test("pandas or numpy not installed")
        TransClass = get_component("src.features.tabular", "TabularTransformer", milestone="M2")
        ModelClass = get_component("src.models.train", "ArtValuationModel", milestone="M3")
        train_df = pd.DataFrame(make_diverse_artwork_records(6))
        transformer = TransClass()
        X_tab = transformer.fit_transform(train_df, np.log1p(train_df["price"].values))
        model = ModelClass()
        model.fit(X_tab, np.log1p(train_df["price"].values))
        novel_artwork = pd.DataFrame([{"artist": "Novel Unknown Creator", "year": 2026, "medium": "Quantum Glass", "height": 45.0, "width": 60.0}])
        X_novel = transformer.transform(novel_artwork)
        pred = model.predict_price(X_novel)
        self.assertGreaterEqual(pred[0], 0.0)

    def test_t3_13_pipeline_save_and_reload_invariance(self):
        """Cross-feature: Serialization -> Reload -> Prediction equality check."""
        if np is None:
            skip_test("numpy not installed")
        save_fn = get_component("src.models.io", "save_model_pipeline", milestone="M3")
        load_fn = get_component("src.models.io", "load_model_pipeline", milestone="M3")
        ModelClass = get_component("src.models.train", "ArtValuationModel", milestone="M3")
        model = ModelClass()
        X = np.random.randn(20, 16).astype(np.float32)
        y = np.linspace(8.0, 12.0, 20).astype(np.float32)
        model.fit(X, y)
        tmp_dir = tempfile.mkdtemp(prefix="caniq_t3_io_")
        try:
            save_fn(model=model, tabular_transformer={"dummy": 1}, metadata={"name": "test"}, output_dir=tmp_dir)
            reloaded = load_fn(tmp_dir)
            p1 = model.predict_price(X[:5])
            p2 = reloaded["model"].predict_price(X[:5])
            np.testing.assert_allclose(p1, p2, atol=1e-4)
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_t3_14_single_sample_vs_batch_consistency(self):
        """Cross-feature: predict(X[i:i+1]) matches predict(X)[i]."""
        if np is None:
            skip_test("numpy not installed")
        ModelClass = get_component("src.models.train", "ArtValuationModel", milestone="M3")
        model = ModelClass()
        X = np.random.randn(15, 16).astype(np.float32)
        y = np.linspace(8.0, 13.0, 15).astype(np.float32)
        model.fit(X, y)
        batch_preds = model.predict_price(X)
        for i in range(3):
            single_pred = model.predict_price(X[i:i+1])
            self.assertAlmostEqual(batch_preds[i], single_pred[0], places=3)

    def test_t3_15_dimension_scale_invariance(self):
        """Cross-feature: Proportional dimension scaling robustness."""
        CleanerClass = get_component("src.data.cleaner", "ArtDataCleaner", milestone="M1")
        cleaner = CleanerClass()
        ar1 = cleaner.calculate_aspect_ratio(50.0, 100.0)
        ar2 = cleaner.calculate_aspect_ratio(500.0, 1000.0)
        self.assertAlmostEqual(ar1, ar2, places=5)


if __name__ == "__main__":
    unittest.main()
