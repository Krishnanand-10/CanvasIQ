"""
Unit tests for CanvasIQ Appraisal Dashboard (app/app.py).

Tests helper functions, formatting, validation, dimension math,
multimodal appraisal inference, and headless Streamlit execution.
"""

from __future__ import annotations

import io
import os
import unittest
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

from app.app import (
    MODEL_DIR,
    REPO_ROOT,
    appraise_artwork,
    calculate_artwork_dimensions,
    calculate_dimensions,
    format_confidence_interval,
    format_currency,
    get_vision_extractor,
    load_or_bootstrap_pipeline,
    load_pipeline,
    validate_inputs,
)


class TestCurrencyFormatting(unittest.TestCase):
    """Tests currency formatting across extreme numerical values."""

    def test_format_currency_zero(self):
        self.assertEqual(format_currency(0.0), "$0.00")
        self.assertEqual(format_currency(0), "$0.00")

    def test_format_currency_subdollar(self):
        self.assertEqual(format_currency(0.50), "$0.50")
        self.assertEqual(format_currency(0.75), "$0.75")
        self.assertEqual(format_currency(0.01), "$0.01")

    def test_format_currency_standard(self):
        self.assertEqual(format_currency(1500.0), "$1,500.00")
        self.assertEqual(format_currency(12345.67), "$12,345.67")

    def test_format_currency_extreme_high(self):
        self.assertEqual(format_currency(50000000.0), "$50,000,000.00")
        self.assertEqual(format_currency(150000000.0), "$150,000,000.00")

    def test_format_currency_negative_clamped(self):
        self.assertEqual(format_currency(-50.0), "$0.00")
        self.assertEqual(format_currency(-1000000.0), "$0.00")


class TestArtworkDimensions(unittest.TestCase):
    """Tests geometry calculations for surface area and aspect ratio."""

    def test_calculate_artwork_dimensions_standard(self):
        area, aspect = calculate_artwork_dimensions(height=65.0, width=81.0)
        self.assertAlmostEqual(area, 65.0 * 81.0, places=4)
        self.assertAlmostEqual(aspect, 81.0 / 65.0, places=4)

    def test_calculate_dimensions_dict(self):
        dims = calculate_dimensions(height=50.0, width=100.0)
        self.assertIn("surface_area", dims)
        self.assertIn("aspect_ratio", dims)
        self.assertAlmostEqual(dims["surface_area"], 5000.0)
        self.assertAlmostEqual(dims["aspect_ratio"], 2.0)

    def test_calculate_dimensions_zero_boundary(self):
        dims = calculate_dimensions(height=0.0, width=50.0)
        self.assertEqual(dims["surface_area"], 0.0)
        self.assertEqual(dims["aspect_ratio"], 0.0)

    def test_calculate_dimensions_negative_boundary(self):
        dims = calculate_dimensions(height=-10.0, width=20.0)
        self.assertEqual(dims["aspect_ratio"], 0.0)


class TestConfidenceIntervalFormatting(unittest.TestCase):
    """Tests lower and upper bound confidence interval presentations."""

    def test_format_confidence_interval_standard(self):
        low, high = format_confidence_interval(12000.0, 45000.0)
        self.assertEqual(low, "$12,000.00")
        self.assertEqual(high, "$45,000.00")

    def test_format_confidence_interval_negative_clamped(self):
        low, high = format_confidence_interval(-500.0, 2500.0)
        self.assertEqual(low, "$0.00")
        self.assertEqual(high, "$2,500.00")

    def test_format_confidence_interval_inverted_order(self):
        low, high = format_confidence_interval(5000.0, 2000.0)
        self.assertEqual(low, "$5,000.00")
        self.assertEqual(high, "$5,000.00")


class TestInputValidation(unittest.TestCase):
    """Tests user metadata input validation and bounds checking."""

    def test_validate_inputs_valid(self):
        valid_meta = {
            "artist": "Claude Monet",
            "year": 1899,
            "medium": "Oil on Canvas",
            "height": 65.0,
            "width": 81.0,
        }
        is_valid, err = validate_inputs(valid_meta)
        self.assertTrue(is_valid)
        self.assertIsNone(err)

    def test_validate_inputs_missing_field(self):
        incomplete_meta = {
            "artist": "Claude Monet",
            "year": 1899,
            "medium": "Oil on Canvas",
        }
        is_valid, err = validate_inputs(incomplete_meta)
        self.assertFalse(is_valid)
        self.assertIn("Missing required field", str(err))

    def test_validate_inputs_empty_string(self):
        empty_meta = {
            "artist": "   ",
            "year": 1899,
            "medium": "Oil on Canvas",
            "height": 65.0,
            "width": 81.0,
        }
        is_valid, err = validate_inputs(empty_meta)
        self.assertFalse(is_valid)
        self.assertIn("cannot be empty", str(err))

    def test_validate_inputs_year_out_of_bounds(self):
        bad_year_meta = {
            "artist": "Claude Monet",
            "year": 999,
            "medium": "Oil on Canvas",
            "height": 65.0,
            "width": 81.0,
        }
        is_valid, err = validate_inputs(bad_year_meta)
        self.assertFalse(is_valid)
        self.assertIn("between 1000 and 2026", str(err))

    def test_validate_inputs_negative_dimension(self):
        bad_dim_meta = {
            "artist": "Claude Monet",
            "year": 1899,
            "medium": "Oil on Canvas",
            "height": -10.0,
            "width": 81.0,
        }
        is_valid, err = validate_inputs(bad_dim_meta)
        self.assertFalse(is_valid)
        self.assertIn("strictly positive", str(err))


class TestAppraisalInferenceEngine(unittest.TestCase):
    """Tests end-to-end multimodal valuation inference logic."""

    @classmethod
    def setUpClass(cls):
        cls.pipeline = load_or_bootstrap_pipeline()

    def test_appraise_artwork_with_image(self):
        test_img = Image.new("RGB", (224, 224), color=(100, 150, 200))
        result = appraise_artwork(
            image_input=test_img,
            artist="Claude Monet",
            year=1899,
            medium="Oil on Canvas",
            height=65.0,
            width=81.0,
            pipeline=self.pipeline,
            confidence=0.80,
        )

        self.assertIn("point_estimate", result)
        self.assertIn("lower_bound", result)
        self.assertIn("upper_bound", result)
        self.assertGreater(result["point_estimate"], 0.0)
        self.assertGreaterEqual(result["lower_bound"], 0.0)
        self.assertLessEqual(result["lower_bound"], result["point_estimate"])
        self.assertGreaterEqual(result["upper_bound"], result["point_estimate"])
        self.assertTrue(result["has_image"])
        self.assertEqual(len(result["visual_vector"]), 2048)
        self.assertGreater(result["visual_embedding_norm"], 0.0)

    def test_appraise_artwork_fallback_without_image(self):
        result = appraise_artwork(
            image_input=None,
            artist="Pablo Picasso",
            year=1912,
            medium="Oil on Canvas",
            height=46.0,
            width=38.0,
            pipeline=self.pipeline,
            confidence=0.85,
        )

        self.assertGreater(result["point_estimate"], 0.0)
        self.assertGreaterEqual(result["lower_bound"], 0.0)
        self.assertLessEqual(result["lower_bound"], result["point_estimate"])
        self.assertGreaterEqual(result["upper_bound"], result["point_estimate"])
        self.assertFalse(result["has_image"])
        self.assertEqual(result["visual_embedding_norm"], 0.0)
        self.assertEqual(result["visual_sparsity"], 100.0)


class TestHeadlessStreamlitApp(unittest.TestCase):
    """Tests headless compilation and AppTest simulation of app/app.py."""

    def test_app_source_compiles(self):
        app_path = REPO_ROOT / "app" / "app.py"
        with open(app_path, "r", encoding="utf-8") as f:
            source = f.read()
        code_obj = compile(source, str(app_path), "exec")
        self.assertIsNotNone(code_obj)

    def test_apptest_initial_render(self):
        app_path = str(REPO_ROOT / "app" / "app.py")
        at = AppTest.from_file(app_path).run()
        self.assertEqual(len(at.exception), 0, f"AppTest raised unexpected exception: {at.exception}")
        self.assertGreater(len(at.title), 0)
        self.assertIn("CanvasIQ", at.title[0].value)
        self.assertGreater(len(at.sidebar), 0)
        self.assertGreater(len(at.file_uploader), 0)
        self.assertGreater(len(at.button), 0)

    def test_apptest_click_appraise_flow(self):
        app_path = str(REPO_ROOT / "app" / "app.py")
        at = AppTest.from_file(app_path).run()
        self.assertEqual(len(at.exception), 0)
        # Find primary Appraise button and trigger click
        at.button[0].click().run()
        self.assertEqual(len(at.exception), 0, f"Exception after appraise click: {at.exception}")
        self.assertGreater(len(at.metric), 0)


if __name__ == "__main__":
    unittest.main()
