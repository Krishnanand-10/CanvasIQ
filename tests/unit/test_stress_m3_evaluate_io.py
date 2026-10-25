"""
Adversarial Stress Test Harness for Milestone 3:
Comparative Ablation Runner (src/models/evaluate.py) & Model Persistence (src/models/io.py).

Empirically challenges:
1. compute_valuation_metrics: single sample (N=1), zero variance targets, zero prices,
   empty inputs, shape mismatches, extreme prices, multidimensional arrays.
2. run_ablation_study: polymorphic calling conventions, modality outputs,
   case-insensitive column access, sample limits, invalid feature splits, zero-price targets.
3. src/models/io.py: recursive directory creation, exact reload prediction invariance (<= 1e-5),
   multilingual Unicode metadata preservation, NumPy scalar serialization,
   missing directories/files (FileNotFoundError), malformed metadata (JSONDecodeError),
   corrupted joblib artifacts.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.evaluate import (
    AblationDataFrame,
    compute_metrics,
    compute_valuation_metrics,
    format_ablation_table,
    run_ablation_study,
)
from src.models.io import (
    load_model_pipeline,
    save_model_pipeline,
)
from src.models.train import ArtValuationModel


class TestComputeMetricsStress(unittest.TestCase):
    """Stress tests for compute_metrics and compute_valuation_metrics."""

    def test_single_sample_n1_metrics_stability(self):
        """Verify N=1 returns valid, finite MAE, RMSE, MAPE without crashing, and R2 is NaN."""
        res = compute_valuation_metrics([500.0], [520.0])
        self.assertIn("r2", res)
        self.assertIn("mae", res)
        self.assertIn("rmse", res)
        self.assertIn("mape", res)
        self.assertTrue(np.isnan(res["r2"]))
        self.assertAlmostEqual(res["mae"], 20.0, places=5)
        self.assertAlmostEqual(res["rmse"], 20.0, places=5)
        self.assertAlmostEqual(res["mape"], 4.0, places=5)

    def test_single_sample_n1_zero_price(self):
        """Verify N=1 with zero price does not divide by zero or crash."""
        res = compute_valuation_metrics([0.0], [0.0])
        self.assertTrue(np.isnan(res["r2"]))
        self.assertEqual(res["mae"], 0.0)
        self.assertEqual(res["rmse"], 0.0)
        self.assertEqual(res["mape"], 0.0)

    def test_single_sample_n1_zero_price_positive_prediction(self):
        """Verify N=1 with zero true price and positive prediction yields finite MAPE via eps floor."""
        res = compute_valuation_metrics([0.0], [100.0], eps=1e-8)
        self.assertTrue(np.isnan(res["r2"]))
        self.assertEqual(res["mae"], 100.0)
        self.assertEqual(res["rmse"], 100.0)
        self.assertTrue(np.isfinite(res["mape"]))
        self.assertGreater(res["mape"], 0.0)

    def test_zero_variance_target_perfect_predictions(self):
        """Verify R2=1.0 and error=0 when target has zero variance and predictions match exactly."""
        y_true = np.array([450.0, 450.0, 450.0, 450.0, 450.0])
        y_pred = np.array([450.0, 450.0, 450.0, 450.0, 450.0])
        res = compute_valuation_metrics(y_true, y_pred)
        self.assertEqual(res["r2"], 1.0)
        self.assertEqual(res["mae"], 0.0)
        self.assertEqual(res["rmse"], 0.0)
        self.assertEqual(res["mape"], 0.0)

    def test_zero_variance_target_imperfect_predictions(self):
        """Verify R2=0.0 when target has zero variance but predictions differ from constant target."""
        y_true = np.array([450.0, 450.0, 450.0, 450.0])
        y_pred = np.array([400.0, 460.0, 440.0, 500.0])
        res = compute_valuation_metrics(y_true, y_pred)
        self.assertEqual(res["r2"], 0.0)
        self.assertGreater(res["mae"], 0.0)
        self.assertGreater(res["rmse"], 0.0)
        self.assertGreater(res["mape"], 0.0)

    def test_zero_prices_denominator_protection(self):
        """Verify MAPE remains finite and non-infinite when ground truth contains zeros."""
        y_true = [0.0, 0.0, 50.0, 100.0]
        y_pred = [0.0, 10.0, 50.0, 110.0]
        res = compute_valuation_metrics(y_true, y_pred, eps=1e-8)
        self.assertTrue(np.isfinite(res["mape"]))
        self.assertFalse(np.isnan(res["mape"]))
        self.assertGreater(res["mape"], 0.0)

    def test_all_zero_ground_truth_and_predictions(self):
        """Verify perfect prediction of all-zero targets gives R2=1.0 and 0.0 errors."""
        y_true = [0.0, 0.0, 0.0]
        y_pred = [0.0, 0.0, 0.0]
        res = compute_valuation_metrics(y_true, y_pred)
        self.assertEqual(res["r2"], 1.0)
        self.assertEqual(res["mae"], 0.0)
        self.assertEqual(res["rmse"], 0.0)
        self.assertEqual(res["mape"], 0.0)

    def test_empty_arrays_raise_value_error(self):
        """Verify empty inputs raise explicit ValueError."""
        with self.assertRaises(ValueError):
            compute_metrics([], [])
        with self.assertRaises(ValueError):
            compute_valuation_metrics(np.array([]), np.array([]))

    def test_shape_mismatch_raises_value_error(self):
        """Verify mismatched array lengths raise explicit ValueError."""
        with self.assertRaises(ValueError):
            compute_metrics([100.0, 200.0], [100.0, 200.0, 300.0])

    def test_multidimensional_and_series_inputs(self):
        """Verify 2D column arrays and pandas Series are flattened and computed consistently."""
        y_true_2d = np.array([[100.0], [200.0], [300.0]])
        y_pred_2d = np.array([[110.0], [190.0], [310.0]])
        res_2d = compute_metrics(y_true_2d, y_pred_2d)

        y_true_s = pd.Series([100.0, 200.0, 300.0])
        y_pred_s = pd.Series([110.0, 190.0, 310.0])
        res_s = compute_metrics(y_true_s, y_pred_s)

        for k in ("r2", "mae", "rmse", "mape"):
            self.assertAlmostEqual(res_2d[k], res_s[k], places=5)

    def test_extreme_price_magnitudes(self):
        """Verify stability on sub-penny prices and nine-figure auction prices."""
        # Micro prices
        res_micro = compute_metrics([0.001, 0.002, 0.003], [0.001, 0.002, 0.003])
        self.assertEqual(res_micro["r2"], 1.0)
        self.assertAlmostEqual(res_micro["mae"], 0.0, places=5)

        # Macro nine-figure auction prices ($450M Leonardo da Vinci)
        y_huge = np.array([450_000_000.0, 250_000_000.0, 180_000_000.0])
        y_pred_huge = np.array([450_000_000.0, 250_000_000.0, 180_000_000.0])
        res_huge = compute_metrics(y_huge, y_pred_huge)
        self.assertEqual(res_huge["r2"], 1.0)
        self.assertAlmostEqual(res_huge["mae"], 0.0, places=2)

    def test_semantic_alias_exact_parity(self):
        """Verify compute_valuation_metrics is an exact semantic alias for compute_metrics."""
        y_t = [10.0, 20.0, 30.0]
        y_p = [12.0, 19.0, 31.0]
        self.assertEqual(compute_metrics(y_t, y_p), compute_valuation_metrics(y_t, y_p))


class TestAblationRunnerStress(unittest.TestCase):
    """Stress tests for run_ablation_study and AblationDataFrame."""

    def setUp(self):
        np.random.seed(42)
        self.n_samples = 25
        self.tabular_dim = 6
        self.vision_dim = 10
        self.X_tab = np.random.randn(self.n_samples, self.tabular_dim).astype(np.float32)
        self.X_vis = np.random.randn(self.n_samples, self.vision_dim).astype(np.float32)
        self.X_features = np.hstack([self.X_tab, self.X_vis])
        self.y_price = np.linspace(500.0, 15000.0, self.n_samples).astype(np.float32)

    def test_polymorphic_convention_tabular_vision_y(self):
        """Verify positional convention: (X_tab, X_vis, y_price)."""
        df = run_ablation_study(self.X_tab, self.X_vis, self.y_price, n_estimators=10)
        self.assertIsInstance(df, AblationDataFrame)
        self.assertEqual(len(df), 3)
        modalities = df["modality"].tolist()
        self.assertEqual(modalities, ["Tabular-only", "Vision-only", "Multimodal"])
        feature_counts = df["n_features"].tolist()
        self.assertEqual(feature_counts, [self.tabular_dim, self.vision_dim, self.tabular_dim + self.vision_dim])

    def test_polymorphic_convention_features_y_dim(self):
        """Verify positional convention: (X_features, y_dollar, tabular_dim) with 1D target."""
        df = run_ablation_study(self.X_features, self.y_price, self.tabular_dim, n_estimators=10)
        self.assertIsInstance(df, AblationDataFrame)
        self.assertEqual(len(df), 3)
        modalities = df["modality"].tolist()
        self.assertEqual(modalities, ["Tabular-only", "Vision-only", "Multimodal"])

    def test_keyword_calling_conventions(self):
        """Verify both keyword-based calling conventions."""
        df1 = run_ablation_study(
            X_tab=self.X_tab,
            X_vis=self.X_vis,
            y_price=self.y_price,
            n_estimators=10,
        )
        self.assertEqual(len(df1), 3)

        df2 = run_ablation_study(
            X_features=self.X_features,
            y_dollar=self.y_price,
            tabular_dim=self.tabular_dim,
            n_estimators=10,
        )
        self.assertEqual(len(df2), 3)

    def test_case_insensitive_column_access(self):
        """Verify case-insensitive access to metric columns."""
        df = run_ablation_study(self.X_tab, self.X_vis, self.y_price, n_estimators=10)
        # Verify exact and uppercase access
        pd.testing.assert_series_equal(df["r2"], df["R2"])
        pd.testing.assert_series_equal(df["mae"], df["MAE"])
        pd.testing.assert_series_equal(df["rmse"], df["RMSE"])
        pd.testing.assert_series_equal(df["mape"], df["MAPE"])
        pd.testing.assert_series_equal(df["modality"], df["MODALITY"])
        pd.testing.assert_series_equal(df["n_features"], df["N_FEATURES"])
        pd.testing.assert_series_equal(df["r2_log"], df["R2_LOG"])

        # Non-existent key raises KeyError
        with self.assertRaises(KeyError):
            _ = df["nonexistent_metric"]

    def test_insufficient_samples_raises_value_error(self):
        """Verify ValueError is raised if sample count N < 5."""
        X_tiny = np.random.randn(4, 10).astype(np.float32)
        y_tiny = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)
        with self.assertRaises(ValueError):
            run_ablation_study(X_tiny, y_tiny, tabular_dim=4)

    def test_invalid_tabular_dim_raises_value_error(self):
        """Verify invalid tabular_dim values raise ValueError."""
        # tabular_dim <= 0
        with self.assertRaises(ValueError):
            run_ablation_study(X_features=self.X_features, y_dollar=self.y_price, tabular_dim=0)
        # tabular_dim >= total_dim
        with self.assertRaises(ValueError):
            run_ablation_study(X_features=self.X_features, y_dollar=self.y_price, tabular_dim=self.X_features.shape[1])

    def test_ablation_with_zero_prices_in_target(self):
        """Verify ablation study runs cleanly when dataset includes zero-dollar hammer prices."""
        y_with_zeros = self.y_price.copy()
        y_with_zeros[0] = 0.0
        y_with_zeros[1] = 0.0
        df = run_ablation_study(self.X_tab, self.X_vis, y_with_zeros, n_estimators=10)
        self.assertEqual(len(df), 3)
        for col in ("mae", "rmse", "mape"):
            self.assertTrue(np.all(np.isfinite(df[col])))

    def test_format_ablation_table(self):
        """Verify format_ablation_table produces valid markdown table format."""
        df = run_ablation_study(self.X_tab, self.X_vis, self.y_price, n_estimators=10)
        md = format_ablation_table(df)
        self.assertIsInstance(md, str)
        lines = md.strip().split("\n")
        self.assertGreaterEqual(len(lines), 5)  # header, separator, 3 modality rows
        self.assertTrue(lines[0].startswith("|"))
        self.assertTrue(lines[1].startswith("| ---"))
        self.assertIn("Tabular-only", md)
        self.assertIn("Vision-only", md)
        self.assertIn("Multimodal", md)

    def test_multiple_model_types_in_ablation(self):
        """Verify ablation study runs across lightgbm, xgboost, and gradient_boosting."""
        for m_type in ("lightgbm", "xgboost", "gradient_boosting"):
            df = run_ablation_study(
                self.X_tab,
                self.X_vis,
                self.y_price,
                model_type=m_type,
                n_estimators=5,
            )
            self.assertEqual(len(df), 3)


class TestModelPersistenceStress(unittest.TestCase):
    """Stress tests for save_model_pipeline and load_model_pipeline in src/models/io.py."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="canvas_iq_test_io_")
        np.random.seed(42)
        self.X_train = np.random.randn(25, 8).astype(np.float32)
        self.y_train = np.linspace(8.0, 12.0, 25).astype(np.float32)
        self.model = ArtValuationModel(n_estimators=15, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        self.transformer = {"fitted_medians": {"oil": 12000.0, "acrylic": 8000.0}}
        self.metadata = {
            "model_type": "lightgbm",
            "features": ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"],
            "artists": ["Frida Kahlo", "Édouard Manet", "葛飾北斎", "Salvador Dalí"],
            "metrics": {"r2": 0.854, "mae": 450.25},
        }

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_directory_creation_deeply_nested(self):
        """Verify save_model_pipeline creates nested missing target directories automatically."""
        nested_dir = Path(self.test_dir) / "level1" / "level2" / "models"
        self.assertFalse(nested_dir.exists())

        save_model_pipeline(self.model, self.transformer, self.metadata, output_dir=nested_dir)
        self.assertTrue(nested_dir.exists())
        self.assertTrue((nested_dir / "valuation_model.joblib").exists())
        self.assertTrue((nested_dir / "tabular_transformer.joblib").exists())
        self.assertTrue((nested_dir / "metadata.json").exists())

    def test_exact_reload_prediction_invariance(self):
        """Verify reload prediction outputs match original predictions with tolerance <= 1e-5."""
        out_dir = Path(self.test_dir) / "invariance_test"
        save_model_pipeline(self.model, self.transformer, self.metadata, output_dir=out_dir)

        loaded_pipe = load_model_pipeline(out_dir)
        loaded_model = loaded_pipe["model"]

        # Test on 100 random unseen inputs
        X_test = np.random.randn(100, 8).astype(np.float32)
        orig_price = self.model.predict_price(X_test)
        loaded_price = loaded_model.predict_price(X_test)

        max_abs_diff = float(np.max(np.abs(orig_price - loaded_price)))
        self.assertLessEqual(max_abs_diff, 1e-5)
        self.assertEqual(max_abs_diff, 0.0)  # GBDT predictions are bit-for-bit identical

        # Test predict_log invariance
        orig_log = self.model.predict_log(X_test)
        loaded_log = loaded_model.predict_log(X_test)
        self.assertTrue(np.allclose(orig_log, loaded_log, atol=1e-5))

        # Test predict_with_confidence invariance
        orig_bounds = self.model.predict_with_confidence(X_test, confidence=0.85)
        loaded_bounds = loaded_model.predict_with_confidence(X_test, confidence=0.85)
        for i in range(3):
            self.assertTrue(np.allclose(orig_bounds[i], loaded_bounds[i], atol=1e-5))

    def test_unicode_metadata_preservation(self):
        """Verify non-ASCII multilingual Unicode artist names are preserved accurately in UTF-8."""
        unicode_meta = {
            "artists": [
                "Frida Kahlo",
                "Édouard Manet",
                "葛飾北斎",
                "Винсент ван Гог",
                "達芬奇",
                "الرسام",
            ]
        }
        out_dir = Path(self.test_dir) / "unicode_test"
        save_model_pipeline(self.model, self.transformer, unicode_meta, output_dir=out_dir)

        loaded_pipe = load_model_pipeline(out_dir)
        loaded_artists = loaded_pipe["metadata"]["artists"]
        self.assertEqual(loaded_artists, unicode_meta["artists"])

        # Check raw file contents on disk to ensure clean unescaped UTF-8
        with open(out_dir / "metadata.json", "r", encoding="utf-8") as f:
            raw_content = f.read()
        for artist in unicode_meta["artists"]:
            self.assertIn(artist, raw_content)

    def test_numpy_types_sanitization_in_metadata(self):
        """Verify NumPy floats, ints, booleans, and arrays in metadata serialize to clean JSON."""
        numpy_meta = {
            "np_float32": np.float32(3.14159),
            "np_float64": np.float64(2.71828),
            "np_int32": np.int32(42),
            "np_int64": np.int64(999999),
            "np_bool": np.bool_(True),
            "np_array": np.array([1, 2, 3, 4, 5]),
            "nested": {
                "inner_float": np.float32(1.618),
                "inner_list": [np.int32(10), np.float64(20.5)],
            },
        }
        out_dir = Path(self.test_dir) / "numpy_meta_test"
        # Must not raise TypeError: Object of type float32 is not JSON serializable
        save_model_pipeline(self.model, self.transformer, numpy_meta, output_dir=out_dir)

        loaded_pipe = load_model_pipeline(out_dir)
        loaded_m = loaded_pipe["metadata"]
        self.assertAlmostEqual(loaded_m["np_float32"], 3.14159, places=4)
        self.assertEqual(loaded_m["np_int32"], 42)
        self.assertEqual(loaded_m["np_bool"], True)
        self.assertEqual(loaded_m["np_array"], [1, 2, 3, 4, 5])
        self.assertAlmostEqual(loaded_m["nested"]["inner_float"], 1.618, places=3)

    def test_missing_directory_raises_file_not_found(self):
        """Verify load_model_pipeline raises FileNotFoundError when directory does not exist."""
        nonexistent = Path(self.test_dir) / "does_not_exist"
        with self.assertRaises(FileNotFoundError):
            load_model_pipeline(nonexistent)

    def test_file_instead_of_directory_raises_not_a_directory(self):
        """Verify load_model_pipeline raises NotADirectoryError when path is a file."""
        file_path = Path(self.test_dir) / "dummy_file.txt"
        file_path.write_text("not a directory")
        with self.assertRaises(NotADirectoryError):
            load_model_pipeline(file_path)

    def test_missing_metadata_raises_file_not_found(self):
        """Verify load_model_pipeline raises FileNotFoundError when metadata.json is missing."""
        out_dir = Path(self.test_dir) / "missing_meta"
        save_model_pipeline(self.model, self.transformer, self.metadata, output_dir=out_dir)
        os.remove(out_dir / "metadata.json")
        with self.assertRaises(FileNotFoundError):
            load_model_pipeline(out_dir)

    def test_missing_model_artifact_raises_file_not_found(self):
        """Verify load_model_pipeline raises FileNotFoundError when valuation_model.joblib is missing."""
        out_dir = Path(self.test_dir) / "missing_model"
        save_model_pipeline(self.model, self.transformer, self.metadata, output_dir=out_dir)
        os.remove(out_dir / "valuation_model.joblib")
        with self.assertRaises(FileNotFoundError):
            load_model_pipeline(out_dir)

    def test_missing_transformer_artifact_raises_file_not_found(self):
        """Verify load_model_pipeline raises FileNotFoundError when tabular_transformer.joblib is missing."""
        out_dir = Path(self.test_dir) / "missing_transformer"
        save_model_pipeline(self.model, self.transformer, self.metadata, output_dir=out_dir)
        os.remove(out_dir / "tabular_transformer.joblib")
        with self.assertRaises(FileNotFoundError):
            load_model_pipeline(out_dir)

    def test_malformed_metadata_raises_json_decode_error(self):
        """Verify load_model_pipeline raises json.JSONDecodeError when metadata.json is corrupted."""
        out_dir = Path(self.test_dir) / "corrupt_meta"
        save_model_pipeline(self.model, self.transformer, self.metadata, output_dir=out_dir)
        # Corrupt metadata.json
        (out_dir / "metadata.json").write_text('{"unclosed_json: true, ')
        with self.assertRaises(json.JSONDecodeError):
            load_model_pipeline(out_dir)

    def test_corrupted_joblib_artifact_raises_error(self):
        """Verify load_model_pipeline raises exception when model artifact contains invalid bytes."""
        out_dir = Path(self.test_dir) / "corrupt_model"
        save_model_pipeline(self.model, self.transformer, self.metadata, output_dir=out_dir)
        (out_dir / "valuation_model.joblib").write_bytes(b"\x00\x01\x02\x03\x04INVALID_JOB_LIB_HEADER")
        with self.assertRaises(Exception):
            load_model_pipeline(out_dir)

    def test_none_metadata_serialization(self):
        """Verify saving with metadata=None does not crash and defaults to empty dict."""
        out_dir = Path(self.test_dir) / "none_meta"
        save_model_pipeline(self.model, self.transformer, metadata=None, output_dir=out_dir)
        loaded = load_model_pipeline(out_dir)
        self.assertEqual(loaded["metadata"], {})


class TestEmpiricalDefectProbing(unittest.TestCase):
    """Empirical defect probing demonstrating subtle edge-case failure modes in evaluate.py and io.py."""

    def test_defect_1_positional_convention_with_2d_column_vector_target(self):
        """
        Defect 1: Positional dispatch in evaluate.py:136 misclassifies 2D target vector as X_vis.
        When callers provide targets as 2D column vectors `y.reshape(-1, 1)` or `df[['price']].values`,
        positional dispatch tests `if a2.ndim == 2:`, misidentifying the 2D target as `X_vis` and
        treating `arg3` (tabular_dim) as `y_price`, crashing with:
        ValueError: Insufficient samples for ablation study: N=1 < 5.
        """
        X = np.random.randn(20, 10).astype(np.float32)
        y_1d = np.random.uniform(100.0, 1000.0, 20).astype(np.float32)
        y_2d = y_1d.reshape(-1, 1)

        # 1D target passes cleanly
        res_1d = run_ablation_study(X, y_1d, 4, n_estimators=5)
        self.assertEqual(len(res_1d), 3)

        # 2D target positionally passes cleanly
        res_2d = run_ablation_study(X, y_2d, 4, n_estimators=5)
        self.assertEqual(len(res_2d), 3)
        self.assertEqual(res_2d["modality"].tolist(), ["Tabular-only", "Vision-only", "Multimodal"])

    def test_defect_2_positional_convention_with_1d_visual_vector(self):
        """
        Defect 2 (Resolved): Positional dispatch in evaluate.py correctly classifies 1D X_vis
        as vision feature and reshapes to 2D column vector.
        """
        X_tab = np.random.randn(20, 5).astype(np.float32)
        X_vis_1d = np.random.randn(20).astype(np.float32)
        y = np.linspace(100.0, 1000.0, 20).astype(np.float32)

        res = run_ablation_study(X_tab, X_vis_1d, y, n_estimators=5)
        self.assertEqual(len(res), 3)
        self.assertEqual(res["modality"].tolist(), ["Tabular-only", "Vision-only", "Multimodal"])

    def test_defect_3_mixed_positional_features_with_named_target(self):
        """
        Defect 3 (Resolved): Calling run_ablation_study with positional X_tab, X_vis and keyword y_price=...
        properly resolves the y_price keyword argument.
        """
        X_tab = np.random.randn(20, 4).astype(np.float32)
        X_vis = np.random.randn(20, 6).astype(np.float32)
        y = np.random.uniform(100.0, 1000.0, 20).astype(np.float32)

        res = run_ablation_study(X_tab, X_vis, y_price=y, n_estimators=5)
        self.assertEqual(len(res), 3)
        self.assertEqual(res["modality"].tolist(), ["Tabular-only", "Vision-only", "Multimodal"])

    def test_defect_4_ablation_dataframe_list_indexing_case_insensitivity(self):
        """
        Defect 4 (Resolved): AblationDataFrame.__getitem__ supports list and tuple keys with
        case-insensitive column matching.
        """
        df = AblationDataFrame({"r2": [0.5, 0.6], "mae": [10.0, 12.0]})
        # Scalar uppercase access works
        self.assertIsNotNone(df["R2"])
        self.assertIsNotNone(df["MAE"])

        # List of uppercase column names succeeds
        sub_df = df[["R2", "MAE"]]
        self.assertEqual(list(sub_df.columns), ["r2", "mae"])
        self.assertEqual(len(sub_df), 2)

    def test_defect_5_pandas_index_string_serialization_in_metadata(self):
        """
        Defect 5 (Resolved): _sanitize_for_json recognizes pd.Index, pd.Series, and set as
        list-like sequences and serializes them to JSON arrays.
        """
        tmp_dir = tempfile.mkdtemp()
        try:
            m = ArtValuationModel(n_estimators=5).fit([[1.0, 2.0], [3.0, 4.0]], [10.0, 11.0])
            cols = pd.Index(["artist", "year", "medium"])
            save_model_pipeline(m, {}, {"columns": cols}, output_dir=tmp_dir)

            loaded = load_model_pipeline(tmp_dir)
            reloaded_cols = loaded["metadata"]["columns"]
            # Reloaded columns are serialized as a list
            self.assertIsInstance(reloaded_cols, list)
            self.assertEqual(reloaded_cols, ["artist", "year", "medium"])
        finally:
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    unittest.main()

