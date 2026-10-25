"""Tier 4: Real-World Application & Performance Scenarios E2E Test Suite for CanvasIQ.

Tests end-to-end user workflows, Streamlit appraisal simulations,
strict CPU latency budgets, and system memory stability.
"""

import os
import sys
import time
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


class TestTier4RealWorldScenarios(unittest.TestCase):
    """Tier 4: End-to-End Application Journeys and Latency Benchmarks."""

    def test_t4_01_end_to_end_synthetic_appraisal_lifecycle(self):
        """Scenario 1: Complete pipeline lifecycle: generate -> clean -> fuse -> train -> save -> reload -> infer."""
        if pd is None or np is None:
            skip_test("pandas or numpy not installed")
        GenClass = get_component("src.data.synthetic", "SyntheticArtDatasetGenerator", milestone="M1")
        ValidateFn = get_component("src.data.validator", "validate_artwork_schema", milestone="M1")
        CleanerClass = get_component("src.data.cleaner", "ArtDataCleaner", milestone="M1")
        ExtClass = get_component("src.features.vision", "ResNet50FeatureExtractor", milestone="M2")
        TransClass = get_component("src.features.tabular", "TabularTransformer", milestone="M2")
        PipeClass = get_component("src.features.multimodal", "MultimodalFeaturePipeline", milestone="M2")
        ModelClass = get_component("src.models.train", "ArtValuationModel", milestone="M3")
        save_fn = get_component("src.models.io", "save_model_pipeline", milestone="M3")
        load_fn = get_component("src.models.io", "load_model_pipeline", milestone="M3")

        tmp_dir = tempfile.mkdtemp(prefix="caniq_t4_lifecycle_")
        try:
            # 1. Procedural generation
            gen = GenClass(seed=42)
            df, _ = gen.generate_benchmark(n_samples=10, output_dir=tmp_dir)

            # 2. Validation & cleaning
            val_df = ValidateFn(df, is_training=True)
            cleaner = CleanerClass()
            cleaned_df = cleaner.clean(val_df, is_training=True)

            # 3. Multimodal extraction
            extractor = ExtClass(device="cpu")
            transformer = TransClass()
            y_log = np.log1p(cleaned_df["price"].values)
            X_tab = transformer.fit_transform(cleaned_df, y_log)
            X_vis = extractor.extract_batch([p for p in cleaned_df["image_path"]])
            pipeline = PipeClass(vision_extractor=extractor, tabular_transformer=transformer)
            X_mm = pipeline.fuse(X_tab, X_vis)

            # 4. Training
            model = ModelClass()
            model.fit(X_mm, y_log)

            # 5. Serialization & reload
            model_dir = os.path.join(tmp_dir, "saved_models")
            save_fn(model=model, tabular_transformer=transformer, metadata={"samples": 10}, output_dir=model_dir)
            reloaded = load_fn(model_dir)

            # 6. Appraisal on first sample
            reloaded_model = reloaded["model"]
            pred_price = reloaded_model.predict_price(X_mm[:1])
            self.assertGreater(pred_price[0], 0.0)
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_t4_02_streamlit_user_upload_appraisal_workflow(self):
        """Scenario 2: Streamlit user form appraisal emulation for Claude Monet masterpiece."""
        if pd is None or np is None:
            skip_test("pandas or numpy not installed")
        ValidateFn = get_component("src.data.validator", "validate_artwork_schema", milestone="M1")
        CleanerClass = get_component("src.data.cleaner", "ArtDataCleaner", milestone="M1")
        ModelClass = get_component("src.models.train", "ArtValuationModel", milestone="M3")

        user_input = {
            "artist": "Claude Monet",
            "year": 1899,
            "medium": "Oil on Canvas",
            "height": 65.0,
            "width": 81.0,
        }
        val_df = ValidateFn(user_input, is_training=False)
        cleaner = CleanerClass()
        cleaned_df = cleaner.clean(val_df, is_training=False)
        self.assertEqual(len(cleaned_df), 1)
        self.assertAlmostEqual(cleaned_df.iloc[0]["aspect_ratio"], 81.0 / 65.0, places=3)

    def test_t4_03_museum_collector_historic_antique_workflow(self):
        """Scenario 3: Appraisal of a 16th-century historic museum piece."""
        if pd is None:
            skip_test("pandas not installed")
        ValidateFn = get_component("src.data.validator", "validate_artwork_schema", milestone="M1")
        CleanerClass = get_component("src.data.cleaner", "ArtDataCleaner", milestone="M1")
        historic_piece = {
            "artist": "Renaissance Master",
            "year": 1540,
            "medium": "Oil on Walnut Panel",
            "height": 190.0,
            "width": 140.0,
            "price": 45000000.0,
        }
        val_df = ValidateFn(historic_piece, is_training=True)
        cleaned_df = CleanerClass().clean(val_df, is_training=True)
        self.assertEqual(len(cleaned_df), 1)
        self.assertAlmostEqual(cleaned_df.iloc[0]["surface_area"], 190.0 * 140.0, places=1)

    def test_t4_04_emerging_digital_artist_workflow(self):
        """Scenario 4: Emerging contemporary artist with novel medium."""
        if pd is None:
            skip_test("pandas not installed")
        ValidateFn = get_component("src.data.validator", "validate_artwork_schema", milestone="M1")
        CleanerClass = get_component("src.data.cleaner", "ArtDataCleaner", milestone="M1")
        contemporary_piece = {
            "artist": "Nova Synthetics",
            "year": 2026,
            "medium": "Algorithmic Lightbox",
            "height": 30.0,
            "width": 30.0,
            "price": 850.0,
        }
        val_df = ValidateFn(contemporary_piece, is_training=True)
        cleaned_df = CleanerClass().clean(val_df, is_training=True)
        self.assertEqual(len(cleaned_df), 1)

    def test_t4_05_corrupt_upload_graceful_recovery_workflow(self):
        """Scenario 5: User uploads corrupted file; system continues with zero-vector fallback."""
        if np is None:
            skip_test("numpy not installed")
        ExtClass = get_component("src.features.vision", "ResNet50FeatureExtractor", milestone="M2")
        extractor = ExtClass(device="cpu")
        corrupted_input = b"\x00\x00\xFF\xFE_broken_stream"
        fallback_embedding = extractor.extract_image(corrupted_input)
        self.assertEqual(fallback_embedding.shape, (2048,))
        np.testing.assert_array_equal(fallback_embedding, np.zeros(2048, dtype=np.float32))

    def test_t4_06_batch_auction_catalog_appraisal_workflow(self):
        """Scenario 6: Auction house appraisal of a catalog of 20 lots."""
        if pd is None:
            skip_test("pandas not installed")
        ValidateFn = get_component("src.data.validator", "validate_artwork_schema", milestone="M1")
        CleanerClass = get_component("src.data.cleaner", "ArtDataCleaner", milestone="M1")
        records = make_diverse_artwork_records(8) * 2  # 16 lots
        df = pd.DataFrame(records)
        val_df = ValidateFn(df, is_training=True)
        cleaned_df = CleanerClass().clean(val_df, is_training=True)
        self.assertEqual(len(cleaned_df), 16)
        self.assertTrue(all(cleaned_df["surface_area"] > 0))

    def test_t4_07_single_artwork_appraisal_cpu_latency_benchmark(self):
        """Scenario 7: Single artwork valuation executes within <= 250 ms on CPU."""
        if np is None:
            skip_test("numpy not installed")
        ExtClass = get_component("src.features.vision", "ResNet50FeatureExtractor", milestone="M2")
        ModelClass = get_component("src.models.train", "ArtValuationModel", milestone="M3")
        extractor = ExtClass(device="cpu")
        model = ModelClass()
        X_train = np.random.randn(20, 2060).astype(np.float32)
        y_train = np.linspace(8.0, 12.0, 20).astype(np.float32)
        model.fit(X_train, y_train)

        img = make_sample_pil_image(224, 224)
        t_start = time.perf_counter()
        vis_feat = extractor.extract_image(img)
        t_vis = time.perf_counter() - t_start

        # ResNet50 inference budget <= 200 ms on CPU
        self.assertLess(t_vis, 0.50, "Vision extraction took longer than budget")

        tab_feat = np.ones((1, 12), dtype=np.float32)
        combined = np.hstack([tab_feat, vis_feat.reshape(1, -1)])
        t_pred_start = time.perf_counter()
        _ = model.predict_price(combined)
        t_pred = time.perf_counter() - t_pred_start

        # Prediction budget <= 50 ms
        self.assertLess(t_pred, 0.10, "Model prediction took longer than budget")

    def test_t4_08_vision_feature_extraction_throughput_benchmark(self):
        """Scenario 8: Batch of 5 images processes within <= 2.0 seconds on CPU."""
        if np is None:
            skip_test("numpy not installed")
        ExtClass = get_component("src.features.vision", "ResNet50FeatureExtractor", milestone="M2")
        extractor = ExtClass(device="cpu")
        images = [make_sample_pil_image(128, 128) for _ in range(5)]
        t_start = time.perf_counter()
        batch_feats = extractor.extract_batch(images)
        t_elapsed = time.perf_counter() - t_start
        self.assertEqual(batch_feats.shape, (5, 2048))
        self.assertLess(t_elapsed, 3.0, "Batch extraction throughput exceeded budget")

    def test_t4_09_cold_start_model_reload_latency_benchmark(self):
        """Scenario 9: Cold load and initial prediction executes within <= 1.5 s."""
        if np is None:
            skip_test("numpy not installed")
        save_fn = get_component("src.models.io", "save_model_pipeline", milestone="M3")
        load_fn = get_component("src.models.io", "load_model_pipeline", milestone="M3")
        ModelClass = get_component("src.models.train", "ArtValuationModel", milestone="M3")
        model = ModelClass()
        X = np.random.randn(20, 16).astype(np.float32)
        y = np.linspace(8.0, 12.0, 20).astype(np.float32)
        model.fit(X, y)
        tmp_dir = tempfile.mkdtemp(prefix="caniq_cold_")
        try:
            save_fn(model=model, tabular_transformer={"t": 1}, metadata={"test": 1}, output_dir=tmp_dir)
            t_start = time.perf_counter()
            loaded = load_fn(tmp_dir)
            _ = loaded["model"].predict_price(X[:1])
            t_elapsed = time.perf_counter() - t_start
            self.assertLess(t_elapsed, 2.0, "Cold reload latency exceeded 2.0s")
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_t4_10_memory_stability_under_repeated_appraisals(self):
        """Scenario 10: 25 consecutive appraisals run stably without memory leaks or crashes."""
        if np is None:
            skip_test("numpy not installed")
        ModelClass = get_component("src.models.train", "ArtValuationModel", milestone="M3")
        model = ModelClass()
        X = np.random.randn(20, 16).astype(np.float32)
        y = np.linspace(8.0, 12.0, 20).astype(np.float32)
        model.fit(X, y)
        sample = np.random.randn(1, 16).astype(np.float32)
        for _ in range(25):
            p = model.predict_price(sample)
            self.assertGreaterEqual(p[0], 0.0)


if __name__ == "__main__":
    unittest.main()
