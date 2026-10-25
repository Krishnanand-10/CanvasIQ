"""
Unit test suite for Milestone 3: Valuation Engine, Comparative Ablation & Model Persistence.

Tests Features 18 through 23 in complete isolation:
- Feature 18: LightGBM / XGBoost Regressor Training (ArtValuationModel, train_valuation_model)
- Feature 19: Price Inference Pipeline (predict_price, predict_log, back-transformation)
- Feature 20: Confidence Interval Estimation (predict_with_confidence, predict_valuation)
- Feature 21: Ablation Comparison Protocol & Metrics (compute_metrics, run_ablation_study)
- Feature 22: Model Pipeline Serialization (save_model_pipeline, JSON sanitization)
- Feature 23: Reload Verification Test (load_model_pipeline, prediction invariance)
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
import unittest

import numpy as np
import pandas as pd

from src.models.evaluate import (
    compute_metrics,
    compute_valuation_metrics,
    format_ablation_table,
    run_ablation_study,
)
from src.models.io import (
    load_model_pipeline,
    save_model_pipeline,
)
from src.models.predict import (
    predict_valuation,
)
from src.models.train import (
    ArtValuationModel,
    train_valuation_model,
)


class TestFeature18ArtValuationModelTraining(unittest.TestCase):
    """Unit tests for Feature 18: Regressor model training."""

    def test_model_init_defaults(self):
        model = ArtValuationModel()
        self.assertEqual(model.model_type, "lightgbm")
        self.assertEqual(model.n_estimators, 100)
        self.assertEqual(model.max_depth, 6)
        self.assertAlmostEqual(model.learning_rate, 0.05)
        self.assertEqual(model.random_state, 42)
        self.assertFalse(model.is_fitted_)
        self.assertIsNone(model.estimator_)

    def test_fit_lightgbm_success(self):
        np.random.seed(42)
        X = np.random.randn(25, 16).astype(np.float32)
        y = np.linspace(8.0, 14.0, 25).astype(np.float32)
        model = ArtValuationModel(model_type="lightgbm", n_estimators=20)
        model.fit(X, y)
        self.assertTrue(model.is_fitted_)
        self.assertIsNotNone(model.estimator_)
        self.assertGreater(model.residual_std_, 0.0)

    def test_fit_xgboost_or_fallback(self):
        np.random.seed(42)
        X = np.random.randn(20, 8).astype(np.float32)
        y = np.linspace(7.0, 12.0, 20).astype(np.float32)
        model = ArtValuationModel(model_type="xgboost", n_estimators=10)
        model.fit(X, y)
        self.assertTrue(model.is_fitted_)
        preds = model.predict_log(X)
        self.assertEqual(preds.shape, (20,))

    def test_tiny_dataset_training(self):
        X = np.random.randn(5, 10).astype(np.float32)
        y = np.array([8.0, 9.0, 10.0, 11.0, 12.0], dtype=np.float32)
        model = ArtValuationModel(n_estimators=10)
        model.fit(X, y)
        self.assertTrue(model.is_fitted_)
        preds = model.predict_log(X)
        self.assertEqual(len(preds), 5)

    def test_constant_target_training(self):
        X = np.random.randn(15, 8).astype(np.float32)
        y = np.full(15, 10.0, dtype=np.float32)
        model = ArtValuationModel(n_estimators=10)
        model.fit(X, y)
        preds = model.predict_log(X)
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_training_loss_finite(self):
        X = np.random.randn(25, 32).astype(np.float32)
        y = np.linspace(5.0, 12.0, 25).astype(np.float32)
        model = ArtValuationModel(n_estimators=15)
        model.fit(X, y)
        preds = model.predict_log(X)
        mse = float(np.mean((preds - y) ** 2))
        self.assertTrue(math.isfinite(mse))

    def test_feature_importances_property(self):
        X = np.random.randn(30, 8).astype(np.float32)
        y = np.linspace(8.0, 13.0, 30).astype(np.float32)
        model = ArtValuationModel(n_estimators=10)
        self.assertIsNone(model.feature_importances_)
        model.fit(X, y)
        fi = model.feature_importances_
        if fi is not None:
            self.assertEqual(len(fi), 8)

    def test_train_valuation_model_functional_interface(self):
        X = np.random.randn(30, 12).astype(np.float32)
        y = np.linspace(8.0, 14.0, 30).astype(np.float32)
        model, metrics = train_valuation_model(X, y, n_estimators=15)
        self.assertTrue(model.is_fitted_)
        self.assertIn("mse", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("residual_std", metrics)

    def test_empty_dataset_raises_error(self):
        model = ArtValuationModel()
        with self.assertRaises(ValueError):
            model.fit(np.zeros((0, 8)), np.zeros(0))

    def test_sample_count_mismatch_raises_error(self):
        model = ArtValuationModel()
        with self.assertRaises(ValueError):
            model.fit(np.zeros((10, 8)), np.zeros(5))


class TestFeature19PriceInferencePipeline(unittest.TestCase):
    """Unit tests for Feature 19: Price inference and back-transformation."""

    def setUp(self):
        np.random.seed(42)
        self.X_train = np.random.randn(25, 16).astype(np.float32)
        self.y_train = np.linspace(8.0, 14.0, 25).astype(np.float32)
        self.model = ArtValuationModel(n_estimators=15)
        self.model.fit(self.X_train, self.y_train)

    def test_predict_log_shape(self):
        X_test = np.random.randn(10, 16).astype(np.float32)
        preds = self.model.predict_log(X_test)
        self.assertEqual(preds.shape, (10,))

    def test_predict_price_native_dollars(self):
        X_test = np.random.randn(8, 16).astype(np.float32)
        prices = self.model.predict_price(X_test)
        self.assertEqual(prices.shape, (8,))
        self.assertTrue(all(p >= 0.0 for p in prices))

    def test_predict_price_strictly_non_negative(self):
        X_ext = np.full((5, 16), -500.0, dtype=np.float32)
        prices = self.model.predict_price(X_ext)
        self.assertTrue(all(p >= 0.0 for p in prices))

    def test_monotonicity_log_to_dollar(self):
        X_test = np.random.randn(12, 16).astype(np.float32)
        logs = self.model.predict_log(X_test)
        dollars = self.model.predict_price(X_test)
        np.testing.assert_array_equal(np.argsort(logs), np.argsort(dollars))

    def test_single_vector_vs_batch_consistency(self):
        X_sample = np.random.randn(5, 16).astype(np.float32)
        batch_preds = self.model.predict_price(X_sample)
        for i in range(len(X_sample)):
            single_pred = self.model.predict_price(X_sample[i : i + 1])
            self.assertAlmostEqual(batch_preds[i], single_pred[0], places=3)

    def test_empty_batch_handling(self):
        empty_X = np.zeros((0, 16), dtype=np.float32)
        prices = self.model.predict_price(empty_X)
        self.assertEqual(len(prices), 0)

    def test_extreme_features_finite_output(self):
        X_pos = np.full((3, 16), 1000.0, dtype=np.float32)
        X_neg = np.full((3, 16), -1000.0, dtype=np.float32)
        p_pos = self.model.predict_price(X_pos)
        p_neg = self.model.predict_price(X_neg)
        self.assertTrue(np.all(np.isfinite(p_pos)))
        self.assertTrue(np.all(np.isfinite(p_neg)))

    def test_unfitted_model_raises_runtime_error(self):
        unfitted = ArtValuationModel()
        with self.assertRaises(RuntimeError):
            unfitted.predict_log(np.zeros((2, 16)))


class TestFeature20ConfidenceIntervalEstimation(unittest.TestCase):
    """Unit tests for Feature 20: Confidence interval estimation."""

    def setUp(self):
        np.random.seed(42)
        self.X_train = np.random.randn(30, 16).astype(np.float32)
        self.y_train = np.linspace(9.0, 15.0, 30).astype(np.float32)
        self.model = ArtValuationModel(n_estimators=15)
        self.model.fit(self.X_train, self.y_train)

    def test_predict_with_confidence_returns_tuple(self):
        X_test = np.random.randn(6, 16).astype(np.float32)
        lower, point, upper = self.model.predict_with_confidence(X_test, confidence=0.80)
        self.assertEqual(len(lower), 6)
        self.assertEqual(len(point), 6)
        self.assertEqual(len(upper), 6)

    def test_confidence_bounds_ordering(self):
        X_test = np.random.randn(10, 16).astype(np.float32)
        lower, point, upper = self.model.predict_with_confidence(X_test, confidence=0.85)
        for l, p, u in zip(lower, point, upper):
            self.assertLessEqual(l, p + 1e-4)
            self.assertLessEqual(p, u + 1e-4)

    def test_lower_bound_non_negative(self):
        X_test = np.random.randn(12, 16).astype(np.float32)
        lower, _, _ = self.model.predict_with_confidence(X_test, confidence=0.95)
        self.assertTrue(all(l >= 0.0 for l in lower))

    def test_confidence_level_scaling(self):
        X_test = np.random.randn(5, 16).astype(np.float32)
        l80, _, u80 = self.model.predict_with_confidence(X_test, confidence=0.80)
        l95, _, u95 = self.model.predict_with_confidence(X_test, confidence=0.95)
        width80 = u80 - l80
        width95 = u95 - l95
        self.assertTrue(all(w95 >= w80 - 1e-4 for w95, w80 in zip(width95, width80)))

    def test_zero_confidence_collapsed_interval(self):
        X_test = np.random.randn(4, 16).astype(np.float32)
        lower, point, upper = self.model.predict_with_confidence(X_test, confidence=0.0)
        np.testing.assert_allclose(lower, point, atol=1e-3)
        np.testing.assert_allclose(point, upper, atol=1e-3)

    def test_single_sample_confidence_shape(self):
        X_one = np.random.randn(1, 16).astype(np.float32)
        lower, point, upper = self.model.predict_with_confidence(X_one, confidence=0.80)
        self.assertEqual(len(lower), 1)
        self.assertEqual(len(point), 1)
        self.assertEqual(len(upper), 1)

    def test_invalid_confidence_raises_error(self):
        X_test = np.random.randn(2, 16).astype(np.float32)
        with self.assertRaises(ValueError):
            self.model.predict_with_confidence(X_test, confidence=1.5)
        with self.assertRaises(ValueError):
            self.model.predict_with_confidence(X_test, confidence=-0.1)

    def test_predict_valuation_function_direct(self):
        X_test = np.random.randn(4, 16).astype(np.float32)
        pts = predict_valuation(self.model, X_test, return_bounds=False)
        self.assertEqual(len(pts), 4)

        lower, point, upper = predict_valuation(self.model, X_test, return_bounds=True, alpha=0.10)
        self.assertEqual(len(lower), 4)
        self.assertTrue(all(lower <= upper))

    def test_invalid_alpha_raises_value_error(self):
        X_test = np.random.randn(2, 16).astype(np.float32)
        with self.assertRaises(ValueError):
            predict_valuation(self.model, X_test, alpha=0.0)
        with self.assertRaises(ValueError):
            predict_valuation(self.model, X_test, alpha=1.5)


class TestFeature21AblationRunnerAndMetrics(unittest.TestCase):
    """Unit tests for Feature 21: Ablation comparison and metrics computation."""

    def test_compute_metrics_keys_present(self):
        y_true = np.array([1000.0, 5000.0, 10000.0])
        y_pred = np.array([1100.0, 4800.0, 9500.0])
        metrics = compute_metrics(y_true, y_pred)
        for k in ["r2", "mae", "rmse", "mape"]:
            self.assertIn(k, metrics)

    def test_perfect_prediction_identities(self):
        y = np.array([100.0, 200.0, 300.0])
        m = compute_metrics(y, y)
        self.assertAlmostEqual(m["r2"], 1.0, places=4)
        self.assertAlmostEqual(m["mae"], 0.0, places=4)
        self.assertAlmostEqual(m["rmse"], 0.0, places=4)
        self.assertAlmostEqual(m["mape"], 0.0, places=4)

    def test_non_negative_error_metrics(self):
        y_true = np.array([1000.0, 2000.0])
        y_pred = np.array([1500.0, 1800.0])
        m = compute_metrics(y_true, y_pred)
        self.assertGreaterEqual(m["mae"], 0.0)
        self.assertGreaterEqual(m["rmse"], 0.0)
        self.assertGreaterEqual(m["mape"], 0.0)

    def test_inverted_predictions_negative_r2(self):
        y_true = np.array([1000.0, 10000.0, 100000.0])
        y_pred = np.array([100000.0, 10000.0, 1000.0])
        m = compute_metrics(y_true, y_pred)
        self.assertLess(m["r2"], 0.0)

    def test_zero_variance_target_protection(self):
        y_true = np.full(10, 5000.0)
        y_pred = np.full(10, 5000.0)
        m = compute_metrics(y_true, y_pred)
        self.assertAlmostEqual(m["mae"], 0.0, places=4)
        self.assertAlmostEqual(m["r2"], 1.0, places=4)

    def test_single_sample_metric_calculation(self):
        y_true = np.array([1000.0])
        y_pred = np.array([1200.0])
        m = compute_metrics(y_true, y_pred)
        self.assertAlmostEqual(m["mae"], 200.0, places=2)
        self.assertTrue(math.isnan(m["r2"]))

    def test_near_zero_prices_mape_protection(self):
        y_true = np.array([0.01, 100.0])
        y_pred = np.array([0.02, 105.0])
        m = compute_metrics(y_true, y_pred)
        self.assertTrue(math.isfinite(m["mape"]))

    def test_empty_arrays_raise_error(self):
        with self.assertRaises(ValueError):
            compute_metrics(np.array([]), np.array([]))

    def test_mismatched_shapes_raise_error(self):
        with self.assertRaises(ValueError):
            compute_metrics(np.array([1.0, 2.0]), np.array([1.0]))

    def test_ablation_study_table_structure(self):
        N = 30
        X_tab = np.random.randn(N, 8).astype(np.float32)
        X_vis = np.random.randn(N, 16).astype(np.float32)
        y_price = np.random.uniform(5000, 50000, N).astype(np.float32)
        df = run_ablation_study(X_tab, X_vis, y_price, n_estimators=10)
        self.assertGreaterEqual(len(df), 3)
        self.assertIn("modality", df.columns)
        self.assertIn("r2", df.columns)
        self.assertIn("rmse", df.columns)

    def test_ablation_study_dual_calling_conventions(self):
        N = 30
        X_tab = np.random.randn(N, 6).astype(np.float32)
        X_vis = np.random.randn(N, 12).astype(np.float32)
        X_mm = np.hstack([X_tab, X_vis])
        y_price = np.random.uniform(4000, 40000, N).astype(np.float32)

        # Convention 1: X_tab, X_vis, y_price
        df1 = run_ablation_study(X_tab, X_vis, y_price, n_estimators=10)
        self.assertGreaterEqual(len(df1), 3)

        # Convention 2: X_features, y_dollar, tabular_dim
        df2 = run_ablation_study(X_mm, y_price, tabular_dim=6, n_estimators=10)
        self.assertGreaterEqual(len(df2), 3)

    def test_ablation_modalities_present(self):
        N = 25
        X_tab = np.random.randn(N, 6).astype(np.float32)
        X_vis = np.random.randn(N, 10).astype(np.float32)
        y_price = np.random.uniform(3000, 30000, N).astype(np.float32)
        df = run_ablation_study(X_tab, X_vis, y_price, n_estimators=10)
        modality_str = str(df.to_dict()).lower()
        self.assertIn("tabular", modality_str)
        self.assertIn("vision", modality_str)
        self.assertIn("multimodal", modality_str)

    def test_ablation_insufficient_samples_raises_error(self):
        with self.assertRaises(ValueError):
            run_ablation_study(np.zeros((3, 4)), np.zeros((3, 8)), np.array([100.0, 200.0, 300.0]))

    def test_format_ablation_table(self):
        N = 25
        X_tab = np.random.randn(N, 4).astype(np.float32)
        X_vis = np.random.randn(N, 8).astype(np.float32)
        y_price = np.random.uniform(1000, 10000, N).astype(np.float32)
        df = run_ablation_study(X_tab, X_vis, y_price, n_estimators=10)
        formatted = format_ablation_table(df)
        self.assertIsInstance(formatted, str)
        self.assertIn("Tabular-only", formatted)


class TestFeature22ModelPipelineSerialization(unittest.TestCase):
    """Unit tests for Feature 22: Pipeline serialization."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="caniq_u_io_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_save_creates_directory(self):
        target = os.path.join(self.tmp_dir, "nested", "models")
        save_model_pipeline(model={"m": 1}, tabular_transformer={"t": 1}, metadata={"v": 1}, output_dir=target)
        self.assertTrue(os.path.exists(target))

    def test_saved_files_created_and_non_empty(self):
        save_model_pipeline(model={"m": 1}, tabular_transformer={"t": 1}, metadata={"v": 1}, output_dir=self.tmp_dir)
        for fname in ["valuation_model.joblib", "tabular_transformer.joblib", "metadata.json"]:
            path = os.path.join(self.tmp_dir, fname)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)

    def test_metadata_json_attributes_preserved(self):
        meta = {"version": "2.0.0", "metrics": {"r2": 0.85, "mae": 1500.0}}
        save_model_pipeline(model="m", tabular_transformer="t", metadata=meta, output_dir=self.tmp_dir)
        with open(os.path.join(self.tmp_dir, "metadata.json")) as f:
            loaded = json.load(f)
        self.assertEqual(loaded["version"], "2.0.0")
        self.assertAlmostEqual(loaded["metrics"]["r2"], 0.85)

    def test_unicode_metadata_serialization(self):
        meta = {"description": "Modèle d'évaluation artistique aux enchères 🎨"}
        save_model_pipeline(model="m", tabular_transformer="t", metadata=meta, output_dir=self.tmp_dir)
        with open(os.path.join(self.tmp_dir, "metadata.json"), encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertIn("🎨", loaded["description"])

    def test_numpy_types_sanitization_in_json(self):
        meta = {
            "score_float32": np.float32(0.92),
            "count_int64": np.int64(100),
            "arr": np.array([1.0, 2.0, 3.0]),
            "flag": np.bool_(True),
        }
        # Should not raise TypeError: Object of type float32 is not JSON serializable
        save_model_pipeline(model="m", tabular_transformer="t", metadata=meta, output_dir=self.tmp_dir)
        with open(os.path.join(self.tmp_dir, "metadata.json")) as f:
            loaded = json.load(f)
        self.assertIsInstance(loaded["score_float32"], float)
        self.assertIsInstance(loaded["count_int64"], int)
        self.assertIsInstance(loaded["arr"], list)
        self.assertIsInstance(loaded["flag"], bool)

    def test_repeated_overwrites_clean(self):
        for i in range(3):
            save_model_pipeline(model={"iter": i}, tabular_transformer={"iter": i}, metadata={"iter": i}, output_dir=self.tmp_dir)
        with open(os.path.join(self.tmp_dir, "metadata.json")) as f:
            loaded = json.load(f)
        self.assertEqual(loaded["iter"], 2)


class TestFeature23ReloadVerificationTest(unittest.TestCase):
    """Unit tests for Feature 23: Reload verification and prediction invariance."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="caniq_u_reload_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_load_pipeline_returns_dict(self):
        save_model_pipeline(model="model_obj", tabular_transformer="trans_obj", metadata={"k": "v"}, output_dir=self.tmp_dir)
        loaded = load_model_pipeline(self.tmp_dir)
        self.assertIsInstance(loaded, dict)
        self.assertIn("model", loaded)
        self.assertIn("tabular_transformer", loaded)
        self.assertIn("metadata", loaded)

    def test_metadata_loaded_accurately(self):
        save_model_pipeline(model="m", tabular_transformer="t", metadata={"model_type": "lightgbm", "val": 42}, output_dir=self.tmp_dir)
        loaded = load_model_pipeline(self.tmp_dir)
        self.assertEqual(loaded["metadata"]["model_type"], "lightgbm")
        self.assertEqual(loaded["metadata"]["val"], 42)

    def test_missing_directory_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            load_model_pipeline("/non_existent_dir_99999")

    def test_empty_directory_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            load_model_pipeline(self.tmp_dir)

    def test_missing_model_file_raises_error(self):
        with open(os.path.join(self.tmp_dir, "metadata.json"), "w") as f:
            f.write("{}")
        with self.assertRaises(FileNotFoundError):
            load_model_pipeline(self.tmp_dir)

    def test_missing_transformer_file_raises_error(self):
        save_model_pipeline(model="m", tabular_transformer="t", metadata={}, output_dir=self.tmp_dir)
        os.remove(os.path.join(self.tmp_dir, "tabular_transformer.joblib"))
        with self.assertRaises(FileNotFoundError):
            load_model_pipeline(self.tmp_dir)

    def test_corrupt_json_raises_error(self):
        with open(os.path.join(self.tmp_dir, "metadata.json"), "w") as f:
            f.write("{broken json:::")
        with self.assertRaises(Exception):
            load_model_pipeline(self.tmp_dir)

    def test_file_path_instead_of_dir_raises_error(self):
        dummy_file = os.path.join(self.tmp_dir, "file.txt")
        with open(dummy_file, "w") as f:
            f.write("text")
        with self.assertRaises((FileNotFoundError, NotADirectoryError, Exception)):
            load_model_pipeline(dummy_file)

    def test_prediction_identity_after_reload(self):
        np.random.seed(42)
        X = np.random.randn(25, 12).astype(np.float32)
        y = np.linspace(8.0, 13.0, 25).astype(np.float32)
        model = ArtValuationModel(n_estimators=15)
        model.fit(X, y)

        save_model_pipeline(model=model, tabular_transformer={"transformer_mock": True}, metadata={"metric": 0.88}, output_dir=self.tmp_dir)
        loaded = load_model_pipeline(self.tmp_dir)
        reloaded_model = loaded["model"]

        p_orig_log = model.predict_log(X[:5])
        p_reloaded_log = reloaded_model.predict_log(X[:5])
        np.testing.assert_allclose(p_orig_log, p_reloaded_log, atol=1e-5)

        p_orig_dollar = model.predict_price(X[:5])
        p_reloaded_dollar = reloaded_model.predict_price(X[:5])
        np.testing.assert_allclose(p_orig_dollar, p_reloaded_dollar, atol=1e-4)

    def test_reload_latency_benchmark(self):
        save_model_pipeline(model="mock_model", tabular_transformer="mock_trans", metadata={"version": 1}, output_dir=self.tmp_dir)
        t_start = time.perf_counter()
        _ = load_model_pipeline(self.tmp_dir)
        latency = time.perf_counter() - t_start
        self.assertLess(latency, 1.5, f"Cold reload latency {latency:.3f}s exceeded 1.5s budget")


if __name__ == "__main__":
    unittest.main()
