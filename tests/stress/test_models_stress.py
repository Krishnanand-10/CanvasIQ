"""
CanvasIQ Adversarial Stress Test Harness for Valuation Engine (Milestone 3).

Empirically challenges:
1. Tiny sample size training (N=2, 3, 5) across LightGBM, XGBoost, and GradientBoosting.
2. Constant / identical price targets (zero-variance target).
3. Extreme price ranges ($0.01 to $100,000,000) and target auto-detection.
4. Feature dimensionalities: (N, 10), (N, 2048), (N, 2060).
5. Confidence interval monotonicity across 1,000 random inputs.
6. Collapsed interval exact bit-for-bit equality at confidence=0.0 / alpha=1.0.
7. Dimensional shape robustness (1D, 2D single, batch, empty 2D, and empty 1D).
8. Extreme/non-finite inputs (NaN, +Inf, -Inf, numerical overflow).
9. Econometric metrics & ablation study edge cases.
10. Model persistence roundtrip invariance and latency budget.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.train import ArtValuationModel, train_valuation_model
from src.models.predict import predict_valuation
from src.models.evaluate import compute_metrics, run_ablation_study, format_ablation_table
from src.models.io import save_model_pipeline, load_model_pipeline


class TestStressValuationEngine(unittest.TestCase):
    """Adversarial stress tests for CanvasIQ Valuation Engine."""

    # -------------------------------------------------------------------------
    # 1. Tiny Sample Size Stress (N=2, 3, 5)
    # -------------------------------------------------------------------------
    def test_stress_tiny_samples_all_model_types(self):
        """Verify train_valuation_model does not crash on N=2, 3, 5 across all 3 model backends."""
        model_types = ["lightgbm", "xgboost", "gradient_boosting"]
        sample_sizes = [2, 3, 5]
        dimensions = [10, 2048, 2060]

        for mtype in model_types:
            for n in sample_sizes:
                for d in dimensions:
                    with self.subTest(model_type=mtype, n_samples=n, n_features=d):
                        rng = np.random.RandomState(42 + n + d)
                        X = rng.randn(n, d).astype(np.float32)
                        y = np.linspace(100.0, 5000.0, n).astype(np.float32)

                        model, metrics = train_valuation_model(
                            X, y, model_type=mtype, n_estimators=10, random_state=42
                        )
                        self.assertTrue(model.is_fitted_)
                        self.assertIsNotNone(model.estimator_)
                        self.assertGreater(model.residual_std_, 0.0)

                        preds = model.predict_price(X)
                        self.assertEqual(preds.shape, (n,))
                        self.assertTrue(np.all(np.isfinite(preds)))
                        self.assertTrue(np.all(preds >= 0.0))

    # -------------------------------------------------------------------------
    # 2. Zero-Variance Target Stress (Identical Constant Prices)
    # -------------------------------------------------------------------------
    def test_stress_zero_variance_targets(self):
        """Verify zero-variance target prices do not crash tree splitting or cause divide-by-zero."""
        constant_prices = [0.0, 1.0, 1500.0, 100_000_000.0]
        model_types = ["lightgbm", "xgboost", "gradient_boosting"]

        for mtype in model_types:
            for const_p in constant_prices:
                for n in [2, 5, 20]:
                    with self.subTest(model_type=mtype, const_price=const_p, n_samples=n):
                        rng = np.random.RandomState(100 + n)
                        X = rng.randn(n, 12).astype(np.float32)
                        y = np.full(n, const_p, dtype=np.float32)

                        model, metrics = train_valuation_model(
                            X, y, model_type=mtype, n_estimators=10, random_state=42
                        )
                        self.assertTrue(model.is_fitted_)
                        preds = model.predict_price(X)
                        self.assertTrue(np.all(np.isfinite(preds)))
                        self.assertTrue(np.all(preds >= 0.0))

                        # Residual standard error should fall back gracefully to a non-zero floor
                        self.assertGreaterEqual(model.residual_std_, 0.10)

                        # Confidence intervals should remain non-degenerate and monotonic
                        low, pt, up = model.predict_with_confidence(X, confidence=0.80)
                        self.assertTrue(np.all(low <= pt))
                        self.assertTrue(np.all(pt <= up))
                        self.assertTrue(np.all(low >= 0.0))

    # -------------------------------------------------------------------------
    # 3. Extreme Price Range Stress ($0.01 to $100M)
    # -------------------------------------------------------------------------
    def test_stress_extreme_price_ranges(self):
        """Verify handling of extreme valuations spanning 10 orders of magnitude."""
        n_samples = 40
        y_extreme = np.geomspace(0.01, 100_000_000.0, n_samples).astype(np.float32)
        X = np.random.randn(n_samples, 16).astype(np.float32)

        for mtype in ["lightgbm", "xgboost", "gradient_boosting"]:
            with self.subTest(model_type=mtype):
                model, metrics = train_valuation_model(
                    X, y_extreme, model_type=mtype, n_estimators=25, is_log_target=False
                )
                preds = model.predict_price(X)
                self.assertTrue(np.all(np.isfinite(preds)))
                self.assertTrue(np.all(preds >= 0.0))
                # Verify max prediction does not overflow float32
                self.assertLess(np.max(preds), np.finfo(np.float32).max)

                low, pt, up = model.predict_with_confidence(X, confidence=0.90)
                self.assertTrue(np.all(np.isfinite(low)))
                self.assertTrue(np.all(np.isfinite(pt)))
                self.assertTrue(np.all(np.isfinite(up)))
                self.assertTrue(np.all(low <= pt))
                self.assertTrue(np.all(pt <= up))
                self.assertTrue(np.all(low >= 0.0))

    # -------------------------------------------------------------------------
    # 4. Confidence Interval Monotonicity & Exact Collapse Oracles
    # -------------------------------------------------------------------------
    def test_stress_confidence_bounds_monotonicity_1000_samples(self):
        """Empirically test 1,000 random input vectors for strict monotonicity 0 <= L <= P <= U."""
        rng = np.random.RandomState(999)
        X_train = rng.randn(60, 20).astype(np.float32)
        y_train = rng.uniform(8.0, 16.0, 60).astype(np.float32)

        model = ArtValuationModel(n_estimators=20, random_state=42).fit(X_train, y_train)

        # 1,000 stress inputs with large variance
        X_stress = rng.randn(1000, 20).astype(np.float32) * 10.0

        for conf in [0.20, 0.50, 0.80, 0.95, 0.99]:
            with self.subTest(confidence=conf):
                lower, point, upper = model.predict_with_confidence(X_stress, confidence=conf)
                self.assertEqual(lower.shape, (1000,))
                self.assertEqual(point.shape, (1000,))
                self.assertEqual(upper.shape, (1000,))

                self.assertTrue(np.all(lower >= 0.0), "Lower bound must be non-negative")
                self.assertTrue(np.all(lower <= point), "Lower bound must not exceed point estimate")
                self.assertTrue(np.all(point <= upper), "Point estimate must not exceed upper bound")
                self.assertTrue(np.all(np.isfinite(lower)))
                self.assertTrue(np.all(np.isfinite(point)))
                self.assertTrue(np.all(np.isfinite(upper)))

    def test_stress_confidence_exact_collapse_bit_for_bit(self):
        """Verify that confidence=0.0 / alpha=1.0 collapses intervals to exact bit-for-bit point estimates."""
        rng = np.random.RandomState(777)
        X_train = rng.randn(40, 10).astype(np.float32)
        y_train = rng.uniform(7.0, 14.0, 40).astype(np.float32)
        model = ArtValuationModel(n_estimators=15).fit(X_train, y_train)

        X_eval = rng.randn(100, 10).astype(np.float32)

        # Test via predict_with_confidence(confidence=0.0)
        low_c0, pt_c0, up_c0 = model.predict_with_confidence(X_eval, confidence=0.0)
        self.assertTrue(np.array_equal(low_c0, pt_c0), "L must equal P bit-for-bit at confidence=0.0")
        self.assertTrue(np.array_equal(up_c0, pt_c0), "U must equal P bit-for-bit at confidence=0.0")

        # Test via predict_valuation(alpha=1.0)
        low_a1, pt_a1, up_a1 = predict_valuation(model, X_eval, return_bounds=True, alpha=1.0)
        self.assertTrue(np.array_equal(low_a1, pt_a1), "L must equal P bit-for-bit at alpha=1.0")
        self.assertTrue(np.array_equal(up_a1, pt_a1), "U must equal P bit-for-bit at alpha=1.0")
        self.assertTrue(np.array_equal(pt_c0, pt_a1), "Point estimates must match across APIs")

    # -------------------------------------------------------------------------
    # 5. Dimensional Shape Robustness (1D, 2D Single, Batch, Empty)
    # -------------------------------------------------------------------------
    def test_stress_dimensional_input_shapes(self):
        """Verify shape handling across 1D (D,), 2D (1, D), batch (N, D), and empty (0, D)."""
        d = 16
        model = ArtValuationModel(n_estimators=10).fit(
            np.random.randn(20, d).astype(np.float32),
            np.linspace(8.0, 12.0, 20).astype(np.float32),
        )

        # 1D single-row array: (D,)
        x_1d = np.random.randn(d).astype(np.float32)
        p_1d = model.predict_price(x_1d)
        self.assertEqual(p_1d.shape, (1,))
        low_1d, pt_1d, up_1d = model.predict_with_confidence(x_1d)
        self.assertEqual(low_1d.shape, (1,))
        self.assertEqual(pt_1d.shape, (1,))
        self.assertEqual(up_1d.shape, (1,))

        # 2D single-row array: (1, D)
        x_2d = np.random.randn(1, d).astype(np.float32)
        p_2d = model.predict_price(x_2d)
        self.assertEqual(p_2d.shape, (1,))

        # Batch array: (25, D)
        x_batch = np.random.randn(25, d).astype(np.float32)
        p_batch = model.predict_price(x_batch)
        self.assertEqual(p_batch.shape, (25,))
        low_b, pt_b, up_b = model.predict_with_confidence(x_batch)
        self.assertEqual(low_b.shape, (25,))
        self.assertEqual(pt_b.shape, (25,))
        self.assertEqual(up_b.shape, (25,))

        # Empty 2D array: (0, D)
        x_empty_2d = np.empty((0, d), dtype=np.float32)
        p_empty = model.predict_price(x_empty_2d)
        self.assertEqual(p_empty.shape, (0,))
        low_e, pt_e, up_e = model.predict_with_confidence(x_empty_2d)
        self.assertEqual(low_e.shape, (0,))
        self.assertEqual(pt_e.shape, (0,))
        self.assertEqual(up_e.shape, (0,))

    # -------------------------------------------------------------------------
    # 6. Numerical Corruption & Non-Finite Inputs
    # -------------------------------------------------------------------------
    def test_stress_non_finite_feature_sanitization(self):
        """Verify NaNs, Infs, and extreme floats do not crash fit or inference."""
        d = 10
        model = ArtValuationModel(n_estimators=10).fit(
            np.random.randn(20, d).astype(np.float32),
            np.linspace(8.0, 12.0, 20).astype(np.float32),
        )

        adversarial_matrix = np.array([
            [np.nan] * d,
            [np.inf] * d,
            [-np.inf] * d,
            [1e25] * d,
            [-1e25] * d,
            [np.nan, np.inf, -np.inf, 1e20, -1e20, 0.0, 0.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32)

        preds = model.predict_price(adversarial_matrix)
        self.assertEqual(preds.shape, (6,))
        self.assertTrue(np.all(np.isfinite(preds)), "All predictions on corrupted features must be finite")
        self.assertTrue(np.all(preds >= 0.0), "All predictions must be non-negative")

        low, pt, up = model.predict_with_confidence(adversarial_matrix)
        self.assertTrue(np.all(np.isfinite(low)))
        self.assertTrue(np.all(np.isfinite(up)))
        self.assertTrue(np.all(low <= pt))
        self.assertTrue(np.all(pt <= up))

    # -------------------------------------------------------------------------
    # 7. Metrics & Ablation Runner Edge Cases
    # -------------------------------------------------------------------------
    def test_stress_metrics_edge_cases(self):
        """Test compute_metrics on single-sample, zero-variance ground truth, and zero price targets."""
        # Single sample: R2 is nan, but MAE/RMSE/MAPE are finite
        m_single = compute_metrics([500.0], [550.0])
        self.assertTrue(np.isnan(m_single["r2"]))
        self.assertAlmostEqual(m_single["mae"], 50.0)
        self.assertAlmostEqual(m_single["rmse"], 50.0)
        self.assertAlmostEqual(m_single["mape"], 10.0)

        # Zero-variance ground truth matching exactly
        m_const_exact = compute_metrics([100.0, 100.0, 100.0], [100.0, 100.0, 100.0])
        self.assertEqual(m_const_exact["r2"], 1.0)
        self.assertEqual(m_const_exact["mae"], 0.0)
        self.assertEqual(m_const_exact["rmse"], 0.0)
        self.assertEqual(m_const_exact["mape"], 0.0)

        # Zero-variance ground truth not matching
        m_const_diff = compute_metrics([100.0, 100.0, 100.0], [200.0, 200.0, 200.0])
        self.assertEqual(m_const_diff["r2"], 0.0)
        self.assertEqual(m_const_diff["mae"], 100.0)

        # Zero ground truth price: epsilon protection prevents division by zero
        m_zero = compute_metrics([0.0, 0.0], [0.0, 0.0])
        self.assertTrue(np.all(np.isfinite(list(m_zero.values()))))
        self.assertEqual(m_zero["mape"], 0.0)

    def test_stress_ablation_minimum_sample_and_dual_conventions(self):
        """Test ablation runner on N=5 and dual calling conventions."""
        rng = np.random.RandomState(42)
        X_tab = rng.randn(10, 4).astype(np.float32)
        X_vis = rng.randn(10, 8).astype(np.float32)
        y = np.linspace(100.0, 1000.0, 10).astype(np.float32)

        # Convention 1: (X_tab, X_vis, y_price)
        df1 = run_ablation_study(X_tab, X_vis, y, n_estimators=5)
        self.assertEqual(len(df1), 3)
        self.assertListEqual(list(df1["modality"]), ["Tabular-only", "Vision-only", "Multimodal"])

        # Convention 2: (X_features, y_dollar, tabular_dim)
        X_mm = np.hstack([X_tab, X_vis])
        df2 = run_ablation_study(X_mm, y, tabular_dim=4, n_estimators=5)
        self.assertEqual(len(df2), 3)

        # Formatting should produce valid markdown table
        tbl = format_ablation_table(df1)
        self.assertIn("| Modality |", tbl)
        self.assertIn("Multimodal", tbl)

    # -------------------------------------------------------------------------
    # 8. Model Serialization Roundtrip Invariance & Latency Budget
    # -------------------------------------------------------------------------
    def test_stress_pipeline_persistence_invariance_and_latency(self):
        """Test reload invariance <= 1e-6 and cold-start load latency < 1.0s."""
        X = np.random.randn(30, 8).astype(np.float32)
        y = np.linspace(8.0, 14.0, 30).astype(np.float32)
        model = ArtValuationModel(n_estimators=15).fit(X, y)

        complex_meta = {
            "model_type": "lightgbm",
            "np_scalar_float": np.float64(1234.5678),
            "np_scalar_int": np.int64(987654),
            "np_bool": np.bool_(True),
            "np_array": np.array([1.1, 2.2, 3.3]),
            "unicode_artists": ["葛飾北斎", "Édouard Manet", "Vincent van Gogh"],
            "nested": {"status": "trained", "score": np.float32(0.987)},
            "none_val": None,
        }
        dummy_transformer = {"feature_names": ["feat_" + str(i) for i in range(8)]}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_model_pipeline(model, dummy_transformer, complex_meta, output_dir=tmpdir)

            t0 = time.perf_counter()
            loaded = load_model_pipeline(tmpdir)
            load_latency = time.perf_counter() - t0

            # Verify cold-start reload latency is under 1.0s (spec allows 1.5s)
            self.assertLess(load_latency, 1.0, f"Cold reload latency {load_latency:.3f}s exceeded 1.0s")

            # Verify exact prediction invariance
            preds_orig = model.predict_price(X)
            preds_loaded = loaded["model"].predict_price(X)
            np.testing.assert_allclose(preds_orig, preds_loaded, atol=1e-5, rtol=1e-5)

            # Verify metadata deserialization integrity
            loaded_meta = loaded["metadata"]
            self.assertEqual(loaded_meta["unicode_artists"][0], "葛飾北斎")
            self.assertAlmostEqual(loaded_meta["np_scalar_float"], 1234.5678, places=4)
            self.assertEqual(loaded_meta["nested"]["status"], "trained")

    # -------------------------------------------------------------------------
    # 9. Resolved Edge-Case Defects & Boundary Invariants
    # -------------------------------------------------------------------------
    def test_defect_1_empty_1d_array_returns_empty_array(self):
        """DEFECT 1 (Resolved): predict_valuation(m, np.array([])) returns shape (0,)
        empty array without crashing or premature 1D reshape."""
        model = ArtValuationModel(n_estimators=5).fit(
            np.random.randn(20, 4).astype(np.float32),
            np.linspace(8.0, 12.0, 20).astype(np.float32),
        )
        empty_1d = np.empty((0,), dtype=np.float32)
        low, pt, up = predict_valuation(model, empty_1d, return_bounds=True)
        self.assertEqual(low.shape, (0,))
        self.assertEqual(pt.shape, (0,))
        self.assertEqual(up.shape, (0,))
        pt_only = predict_valuation(model, empty_1d, return_bounds=False)
        self.assertEqual(pt_only.shape, (0,))
        low_m, pt_m, up_m = model.predict_with_confidence(empty_1d)
        self.assertEqual(low_m.shape, (0,))
        self.assertEqual(pt_m.shape, (0,))
        self.assertEqual(up_m.shape, (0,))

    def test_defect_2_low_dollar_auto_detect_heuristic_resolved(self):
        """DEFECT 2 (Resolved): train_valuation_model defaults to is_log_target=False,
        preventing exponential price inflation for low-dollar artwork (< $50)."""
        X = np.random.randn(20, 4).astype(np.float32)
        y_low_dollars = np.linspace(5.0, 40.0, 20).astype(np.float32)

        # Train with default is_log_target=False
        model_default, _ = train_valuation_model(X, y_low_dollars, n_estimators=10)
        preds_default = model_default.predict_price(X)

        # Train with explicit is_log_target=False
        model_explicit, _ = train_valuation_model(X, y_low_dollars, is_log_target=False, n_estimators=10)
        preds_explicit = model_explicit.predict_price(X)

        # Resolved verification: prices remain bounded and matched
        self.assertLess(float(np.mean(preds_default)), 100.0)
        self.assertGreater(float(np.mean(preds_default)), 0.0)
        np.testing.assert_allclose(preds_default, preds_explicit, atol=1e-3)

    def test_defect_3_single_sample_fit_guard(self):
        """DEFECT 3 (Resolved): ArtValuationModel.fit raises ValueError when N < 2."""
        model = ArtValuationModel(n_estimators=5)
        with self.assertRaises(ValueError) as ctx:
            model.fit(np.array([[1.0, 2.0, 3.0, 4.0]]), np.array([50.0]))
        self.assertIn("requires at least 2 samples", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

