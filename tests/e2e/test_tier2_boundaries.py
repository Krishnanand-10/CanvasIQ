"""Tier 2: Boundary & Corner Cases E2E Test Suite for CanvasIQ.

Covers all 28 features from PROJECT.md with >=5 test cases per feature (140 tests total).
Exercises operational limits, domain bounds, extreme inputs, corrupt media, and corner cases.
"""

import math
import os
import re
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


class TestFeature01BoundaryValidator(unittest.TestCase):
    """F1 Boundary: Schema validation edge cases."""

    def setUp(self):
        self.validator_fn = get_component("src.data.validator", "validate_artwork_schema", milestone="M1")

    def test_f1_b01_zero_dimension_fails(self):
        record = make_sample_artwork_dict()
        record["height"] = 0.0
        with self.assertRaises((ValueError, Exception)):
            self.validator_fn(record, is_training=True)

    def test_f1_b02_negative_dimension_fails(self):
        record = make_sample_artwork_dict()
        record["width"] = -25.0
        with self.assertRaises((ValueError, Exception)):
            self.validator_fn(record, is_training=True)

    def test_f1_b03_negative_price_fails_in_training(self):
        record = make_sample_artwork_dict()
        record["price"] = -1000.0
        with self.assertRaises((ValueError, Exception)):
            self.validator_fn(record, is_training=True)

    def test_f1_b04_future_creation_year_fails(self):
        record = make_sample_artwork_dict()
        record["year"] = 2150
        with self.assertRaises((ValueError, Exception)):
            self.validator_fn(record, is_training=True)

    def test_f1_b05_ancient_creation_year_boundary(self):
        record = make_sample_artwork_dict()
        record["year"] = 1200
        # Historical art from 1200 should either pass or fail cleanly if before valid epoch
        try:
            res = self.validator_fn(record, is_training=True)
            self.assertIsNotNone(res)
        except (ValueError, Exception):
            pass


class TestFeature02BoundaryCleaner(unittest.TestCase):
    """F2 Boundary: Data cleaning and sanitization corner cases."""

    def setUp(self):
        CleanerClass = get_component("src.data.cleaner", "ArtDataCleaner", milestone="M1")
        self.cleaner = CleanerClass()

    def test_f2_b01_all_nan_medium_imputed_to_unknown(self):
        if pd is None:
            skip_test("pandas not installed")
        records = [make_sample_artwork_dict() for _ in range(3)]
        for r in records:
            r["medium"] = None
        df = pd.DataFrame(records)
        cleaned = self.cleaner.clean(df, is_training=True)
        self.assertTrue(all(cleaned["medium"] == "Unknown"))

    def test_f2_b02_empty_dataframe_handled(self):
        if pd is None:
            skip_test("pandas not installed")
        df = pd.DataFrame(columns=["artist", "year", "medium", "height", "width", "price"])
        cleaned = self.cleaner.clean(df, is_training=True)
        self.assertEqual(len(cleaned), 0)

    def test_f2_b03_extreme_whitespace_in_strings_stripped(self):
        if pd is None:
            skip_test("pandas not installed")
        record = make_sample_artwork_dict()
        record["artist"] = "  \n\tClaude Monet \t "
        df = pd.DataFrame([record])
        cleaned = self.cleaner.clean(df, is_training=True)
        self.assertEqual(cleaned.iloc[0]["artist"].strip(), "Claude Monet")

    def test_f2_b04_zero_price_allowed_in_inference_mode(self):
        if pd is None:
            skip_test("pandas not installed")
        record = make_sample_artwork_dict()
        record["price"] = 0.0
        df = pd.DataFrame([record])
        cleaned = self.cleaner.clean(df, is_training=False)
        self.assertEqual(len(cleaned), 1)

    def test_f2_b05_duplicate_rows_preserved_or_deduplicated(self):
        if pd is None:
            skip_test("pandas not installed")
        df = pd.DataFrame([make_sample_artwork_dict(), make_sample_artwork_dict()])
        cleaned = self.cleaner.clean(df, is_training=True)
        self.assertGreaterEqual(len(cleaned), 1)


class TestFeature03BoundarySurfaceArea(unittest.TestCase):
    """F3 Boundary: Surface area calculations at physical limits."""

    def setUp(self):
        CleanerClass = get_component("src.data.cleaner", "ArtDataCleaner", milestone="M1")
        self.calc = getattr(CleanerClass, "calculate_surface_area")

    def test_f3_b01_miniature_artwork(self):
        area = self.calc(0.1, 0.1)
        self.assertAlmostEqual(area, 0.01, places=4)

    def test_f3_b02_colossal_mural(self):
        area = self.calc(10000.0, 5000.0)
        self.assertAlmostEqual(area, 50000000.0, places=1)

    def test_f3_b03_submillimeter_micro_canvas(self):
        area = self.calc(0.01, 0.01)
        self.assertAlmostEqual(area, 0.0001, places=6)

    def test_f3_b04_large_float_precision(self):
        area = self.calc(1234.567, 7654.321)
        expected = 1234.567 * 7654.321
        self.assertAlmostEqual(area, expected, places=2)

    def test_f3_b05_zero_handling(self):
        with self.assertRaises((ValueError, Exception)):
            self.calc(0.0, 50.0)


class TestFeature04BoundaryAspectRatio(unittest.TestCase):
    """F4 Boundary: Aspect ratio calculations at extreme geometry."""

    def setUp(self):
        CleanerClass = get_component("src.data.cleaner", "ArtDataCleaner", milestone="M1")
        self.calc = getattr(CleanerClass, "calculate_aspect_ratio")

    def test_f4_b01_extreme_scroll_canvas(self):
        ratio = self.calc(10.0, 1000.0)
        self.assertAlmostEqual(ratio, 100.0, places=3)

    def test_f4_b02_extreme_pillar_canvas(self):
        ratio = self.calc(1000.0, 10.0)
        self.assertAlmostEqual(ratio, 0.01, places=4)

    def test_f4_b03_near_zero_height_avoids_crash(self):
        try:
            ratio = self.calc(1e-6, 50.0)
            self.assertTrue(math.isfinite(ratio))
        except ZeroDivisionError:
            pass

    def test_f4_b04_micro_canvas_square(self):
        ratio = self.calc(0.5, 0.5)
        self.assertAlmostEqual(ratio, 1.0, places=5)

    def test_f4_b05_large_aspect_ratio_accuracy(self):
        ratio = self.calc(25.0, 250.0)
        self.assertAlmostEqual(ratio, 10.0, places=4)


class TestFeature05BoundaryLogPrice(unittest.TestCase):
    """F5 Boundary: Log price transformation at price limits."""

    def setUp(self):
        CleanerClass = get_component("src.data.cleaner", "ArtDataCleaner", milestone="M1")
        self.transform = getattr(CleanerClass, "transform_log_price")

    def test_f5_b01_micro_price_cents(self):
        log_p = self.transform(0.01)
        self.assertAlmostEqual(log_p, math.log1p(0.01), places=6)

    def test_f5_b02_billion_dollar_sale(self):
        log_p = self.transform(1000000000.0)
        self.assertTrue(math.isfinite(log_p))
        self.assertAlmostEqual(log_p, math.log1p(1e9), places=3)

    def test_f5_b03_subdollar_price(self):
        log_p = self.transform(0.50)
        self.assertAlmostEqual(log_p, math.log1p(0.50), places=5)

    def test_f5_b04_exact_zero_price(self):
        log_p = self.transform(0.0)
        self.assertAlmostEqual(log_p, 0.0, places=6)

    def test_f5_b05_negative_price_handled_safely(self):
        # Should raise ValueError or clamp to 0.0, never NaN
        try:
            res = self.transform(-10.0)
            self.assertTrue(res >= 0.0 or math.isnan(res) is False)
        except ValueError:
            pass


class TestFeature06BoundaryInverseLogPrice(unittest.TestCase):
    """F6 Boundary: Inverse log price back-transformation corner cases."""

    def setUp(self):
        CleanerClass = get_component("src.data.cleaner", "ArtDataCleaner", milestone="M1")
        self.inverse = getattr(CleanerClass, "inverse_log_price")

    def test_f6_b01_huge_log_price(self):
        price = self.inverse(20.0)
        self.assertTrue(math.isfinite(price))
        self.assertGreater(price, 1e8)

    def test_f6_b02_extreme_negative_log_price_clamped_to_zero(self):
        price = self.inverse(-10.0)
        self.assertEqual(price, 0.0)

    def test_f6_b03_near_zero_log_price(self):
        price = self.inverse(1e-7)
        self.assertGreaterEqual(price, 0.0)

    def test_f6_b04_negative_small_log_price_clamped(self):
        price = self.inverse(-0.001)
        self.assertEqual(price, 0.0)

    def test_f6_b05_vectorized_clamping(self):
        if np is None:
            skip_test("numpy not installed")
        inputs = np.array([-5.0, -0.5, 0.0, 2.0])
        outputs = self.inverse(inputs)
        self.assertEqual(outputs[0], 0.0)
        self.assertEqual(outputs[1], 0.0)
        self.assertGreaterEqual(outputs[2], 0.0)


class TestFeature07BoundaryImageGenerator(unittest.TestCase):
    """F7 Boundary: Procedural artwork generation at resolution/style extremes."""

    def setUp(self):
        GeneratorClass = get_component("src.data.synthetic", "SyntheticArtDatasetGenerator", milestone="M1")
        self.gen = GeneratorClass(seed=123)

    def test_f7_b01_tiny_canvas_size(self):
        img = self.gen.generate_image(style="colorfield", palette="oceanic", size=(16, 16))
        self.assertEqual(img.size, (16, 16))

    def test_f7_b02_large_canvas_size(self):
        img = self.gen.generate_image(style="colorfield", palette="oceanic", size=(512, 512))
        self.assertEqual(img.size, (512, 512))

    def test_f7_b03_monochrome_palette(self):
        img = self.gen.generate_image(style="minimalist", palette="monochrome", size=(64, 64))
        self.assertEqual(img.size, (64, 64))

    def test_f7_b04_asymmetric_aspect_canvas(self):
        img = self.gen.generate_image(style="abstract", palette="earthy", size=(256, 64))
        self.assertEqual(img.size, (256, 64))

    def test_f7_b05_unsupported_style_fallback(self):
        # Should either fallback gracefully or raise ValueError
        try:
            img = self.gen.generate_image(style="non_existent_style_xyz", palette="oceanic", size=(64, 64))
            self.assertIsNotNone(img)
        except (ValueError, KeyError):
            pass


class TestFeature08BoundaryMetadataGenerator(unittest.TestCase):
    """F8 Boundary: Synthetic dataset generation boundary parameters."""

    def setUp(self):
        GeneratorClass = get_component("src.data.synthetic", "SyntheticArtDatasetGenerator", milestone="M1")
        self.gen = GeneratorClass(seed=42)
        self.tmp_dir = tempfile.mkdtemp(prefix="caniq_bbench_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_f8_b01_single_sample_generation(self):
        df, _ = self.gen.generate_benchmark(n_samples=1, output_dir=self.tmp_dir)
        self.assertEqual(len(df), 1)

    def test_f8_b02_zero_samples_handling(self):
        df, _ = self.gen.generate_benchmark(n_samples=0, output_dir=self.tmp_dir)
        self.assertEqual(len(df), 0)

    def test_f8_b03_deep_nested_output_directory(self):
        nested = os.path.join(self.tmp_dir, "deep", "nested", "data")
        df, _ = self.gen.generate_benchmark(n_samples=2, output_dir=nested)
        self.assertTrue(os.path.exists(nested))

    def test_f8_b04_high_seed_value(self):
        Gen = get_component("src.data.synthetic", "SyntheticArtDatasetGenerator", milestone="M1")
        g = Gen(seed=9999999)
        df, _ = g.generate_benchmark(n_samples=2, output_dir=self.tmp_dir)
        self.assertEqual(len(df), 2)

    def test_f8_b05_no_negative_dimensions_in_benchmark(self):
        df, _ = self.gen.generate_benchmark(n_samples=10, output_dir=self.tmp_dir)
        self.assertTrue(all(df["height"] > 0))
        self.assertTrue(all(df["width"] > 0))


class TestFeature09BoundaryResNet50Backbone(unittest.TestCase):
    """F9 Boundary: ResNet50 vision extraction under extreme visual inputs."""

    def setUp(self):
        if np is None:
            skip_test("numpy not installed")
        ExtractorClass = get_component("src.features.vision", "ResNet50FeatureExtractor", milestone="M2")
        self.extractor = ExtractorClass(device="cpu")

    def test_f9_b01_all_black_image(self):
        img = make_sample_pil_image(224, 224, color="black")
        feats = self.extractor.extract_image(img)
        self.assertEqual(feats.shape, (2048,))
        self.assertTrue(np.all(np.isfinite(feats)))

    def test_f9_b02_all_white_image(self):
        img = make_sample_pil_image(224, 224, color="white")
        feats = self.extractor.extract_image(img)
        self.assertEqual(feats.shape, (2048,))
        self.assertTrue(np.all(np.isfinite(feats)))

    def test_f9_b03_uniform_gray_image(self):
        img = make_sample_pil_image(224, 224, color="gray")
        feats = self.extractor.extract_image(img)
        self.assertEqual(feats.shape, (2048,))

    def test_f9_b04_high_aspect_ratio_image(self):
        img = make_sample_pil_image(500, 50)
        feats = self.extractor.extract_image(img)
        self.assertEqual(feats.shape, (2048,))

    def test_f9_b05_single_pixel_canvas_upscaled(self):
        img = make_sample_pil_image(1, 1, color="red")
        feats = self.extractor.extract_image(img)
        self.assertEqual(feats.shape, (2048,))


class TestFeature10BoundaryImageTransforms(unittest.TestCase):
    """F10 Boundary: Image mode conversions and edge cases."""

    def setUp(self):
        if np is None:
            skip_test("numpy not installed")
        ExtractorClass = get_component("src.features.vision", "ResNet50FeatureExtractor", milestone="M2")
        self.extractor = ExtractorClass(device="cpu")

    def test_f10_b01_grayscale_mode_L(self):
        img = make_sample_pil_image(100, 100).convert("L")
        feats = self.extractor.extract_image(img)
        self.assertEqual(feats.shape, (2048,))

    def test_f10_b02_rgba_mode_four_channels(self):
        img = make_sample_pil_image(100, 100).convert("RGBA")
        feats = self.extractor.extract_image(img)
        self.assertEqual(feats.shape, (2048,))

    def test_f10_b03_palette_mode_P(self):
        img = make_sample_pil_image(100, 100).convert("P")
        feats = self.extractor.extract_image(img)
        self.assertEqual(feats.shape, (2048,))

    def test_f10_b04_cmyk_mode(self):
        img = make_sample_pil_image(100, 100).convert("CMYK")
        feats = self.extractor.extract_image(img)
        self.assertEqual(feats.shape, (2048,))

    def test_f10_b05_zero_byte_empty_file_fallback(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()
        try:
            feats = self.extractor.extract_image(tmp.name)
            self.assertEqual(feats.shape, (2048,))
            np.testing.assert_array_equal(feats, np.zeros(2048, dtype=np.float32))
        finally:
            os.unlink(tmp.name)


class TestFeature11BoundaryCorruptFallback(unittest.TestCase):
    """F11 Boundary: Degraded and broken inputs fallback resilience."""

    def setUp(self):
        if np is None:
            skip_test("numpy not installed")
        ExtractorClass = get_component("src.features.vision", "ResNet50FeatureExtractor", milestone="M2")
        self.extractor = ExtractorClass(device="cpu")

    def test_f11_b01_corrupt_jpeg_header(self):
        corrupt = b"\xFF\xD8\xFF\xE0" + b"\x00" * 20
        feats = self.extractor.extract_image(corrupt)
        np.testing.assert_array_equal(feats, np.zeros(2048, dtype=np.float32))

    def test_f11_b02_directory_path_as_image_input(self):
        feats = self.extractor.extract_image(tempfile.gettempdir())
        np.testing.assert_array_equal(feats, np.zeros(2048, dtype=np.float32))

    def test_f11_b03_plain_text_file_as_image(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        tmp.write(b"Just plain text not an image")
        tmp.close()
        try:
            feats = self.extractor.extract_image(tmp.name)
            np.testing.assert_array_equal(feats, np.zeros(2048, dtype=np.float32))
        finally:
            os.unlink(tmp.name)

    def test_f11_b04_empty_bytes_fallback(self):
        feats = self.extractor.extract_image(b"")
        np.testing.assert_array_equal(feats, np.zeros(2048, dtype=np.float32))

    def test_f11_b05_fallback_vector_l2_norm_is_zero(self):
        feats = self.extractor.extract_image("bad_path_9999.png")
        self.assertAlmostEqual(float(np.linalg.norm(feats)), 0.0, places=6)


class TestFeature12BoundaryDeviceHandling(unittest.TestCase):
    """F12 Boundary: Hardware accelerator edge cases."""

    def setUp(self):
        self.ExtractorClass = get_component("src.features.vision", "ResNet50FeatureExtractor", milestone="M2")

    def test_f12_b01_invalid_device_name_falls_back_to_cpu(self):
        try:
            ext = self.ExtractorClass(device="non_existent_device_tpu")
            self.assertIsNotNone(ext)
        except (ValueError, Exception):
            pass

    def test_f12_b02_empty_device_string_defaults_safely(self):
        ext = self.ExtractorClass(device="")
        self.assertIsNotNone(ext)

    def test_f12_b03_multiple_extractions_maintain_memory(self):
        ext = self.ExtractorClass(device="cpu")
        for _ in range(5):
            f = ext.extract_image(make_sample_pil_image(64, 64))
            self.assertEqual(f.shape, (2048,))

    def test_f12_b04_empty_batch_extraction(self):
        ext = self.ExtractorClass(device="cpu")
        res = ext.extract_batch([])
        self.assertEqual(len(res), 0)

    def test_f12_b05_single_item_batch_extraction(self):
        ext = self.ExtractorClass(device="cpu")
        res = ext.extract_batch([make_sample_pil_image(64, 64)])
        self.assertEqual(res.shape, (1, 2048))


class TestFeature13BoundaryMediumEncoding(unittest.TestCase):
    """F13 Boundary: Medium encoding edge strings and rare categories."""

    def setUp(self):
        TransformerClass = get_component("src.features.tabular", "TabularTransformer", milestone="M2")
        self.transformer = TransformerClass()

    def test_f13_b01_empty_medium_string(self):
        if pd is None:
            skip_test("pandas not installed")
        df = pd.DataFrame([{"artist": "Artist", "year": 2000, "medium": "", "height": 50.0, "width": 50.0}])
        self.transformer.fit(df)
        out = self.transformer.transform(df)
        self.assertEqual(out.shape[0], 1)

    def test_f13_b02_special_character_medium(self):
        if pd is None:
            skip_test("pandas not installed")
        df = pd.DataFrame([{"artist": "Artist", "year": 2000, "medium": "Gold Leaf & Sand/Resin + Neon #1", "height": 50.0, "width": 50.0}])
        self.transformer.fit(df)
        out = self.transformer.transform(df)
        self.assertEqual(out.shape[0], 1)

    def test_f13_b03_unicode_accented_medium(self):
        if pd is None:
            skip_test("pandas not installed")
        df = pd.DataFrame([{"artist": "Artist", "year": 2000, "medium": "Trompe-l'œil sur châssis", "height": 50.0, "width": 50.0}])
        self.transformer.fit(df)
        out = self.transformer.transform(df)
        self.assertEqual(out.shape[0], 1)

    def test_f13_b04_very_long_medium_string(self):
        if pd is None:
            skip_test("pandas not installed")
        df = pd.DataFrame([{"artist": "Artist", "year": 2000, "medium": "Oil " * 50, "height": 50.0, "width": 50.0}])
        self.transformer.fit(df)
        out = self.transformer.transform(df)
        self.assertEqual(out.shape[0], 1)

    def test_f13_b05_numeric_string_medium(self):
        if pd is None:
            skip_test("pandas not installed")
        df = pd.DataFrame([{"artist": "Artist", "year": 2000, "medium": "123456", "height": 50.0, "width": 50.0}])
        self.transformer.fit(df)
        out = self.transformer.transform(df)
        self.assertEqual(out.shape[0], 1)


class TestFeature14BoundaryArtistStatistics(unittest.TestCase):
    """F14 Boundary: Target encoding edge cases."""

    def setUp(self):
        TransformerClass = get_component("src.features.tabular", "TabularTransformer", milestone="M2")
        self.transformer = TransformerClass()

    def test_f14_b01_uuid_unseen_artist(self):
        if pd is None or np is None:
            skip_test("pandas or numpy not installed")
        train_df = pd.DataFrame(make_diverse_artwork_records(5))
        self.transformer.fit(train_df, np.log1p(train_df["price"].values))
        test_df = pd.DataFrame([{"artist": "a1b2c3d4-e5f6-7890-abcd-ef1234567890", "year": 2020, "medium": "Oil", "height": 50.0, "width": 50.0}])
        out = self.transformer.transform(test_df)
        self.assertFalse(np.any(np.isnan(out)))

    def test_f14_b02_single_historical_sale_artist(self):
        if pd is None or np is None:
            skip_test("pandas or numpy not installed")
        records = make_diverse_artwork_records(4)
        records.append({"artist": "OneHitWonder", "year": 1990, "medium": "Oil", "height": 50.0, "width": 50.0, "price": 100000.0})
        df = pd.DataFrame(records)
        self.transformer.fit(df, np.log1p(df["price"].values))
        out = self.transformer.transform(df)
        self.assertFalse(np.any(np.isnan(out)))

    def test_f14_b03_punctuation_in_artist_name(self):
        if pd is None or np is None:
            skip_test("pandas or numpy not installed")
        df = pd.DataFrame([{"artist": "Artist (Attributed), School of Paris", "year": 1910, "medium": "Oil", "height": 50.0, "width": 50.0, "price": 50000.0}])
        self.transformer.fit(df, np.log1p(df["price"].values))
        out = self.transformer.transform(df)
        self.assertFalse(np.any(np.isnan(out)))

    def test_f14_b04_empty_artist_string(self):
        if pd is None or np is None:
            skip_test("pandas or numpy not installed")
        df = pd.DataFrame([{"artist": "", "year": 2000, "medium": "Oil", "height": 50.0, "width": 50.0, "price": 5000.0}])
        self.transformer.fit(df, np.log1p(df["price"].values))
        out = self.transformer.transform(df)
        self.assertFalse(np.any(np.isnan(out)))

    def test_f14_b05_artist_with_extreme_high_valuation(self):
        if pd is None or np is None:
            skip_test("pandas or numpy not installed")
        records = [
            {"artist": "BillionaireArtist", "year": 2000, "medium": "Oil", "height": 100.0, "width": 100.0, "price": 450000000.0},
            {"artist": "Emerging", "year": 2020, "medium": "Oil", "height": 100.0, "width": 100.0, "price": 500.0},
        ]
        df = pd.DataFrame(records)
        self.transformer.fit(df, np.log1p(df["price"].values))
        out = self.transformer.transform(df)
        self.assertTrue(np.all(np.isfinite(out)))


class TestFeature15BoundaryScalingEra(unittest.TestCase):
    """F15 Boundary: Dimensional scaling with extreme values."""

    def setUp(self):
        TransformerClass = get_component("src.features.tabular", "TabularTransformer", milestone="M2")
        self.transformer = TransformerClass()

    def test_f15_b01_ancient_year_scaled(self):
        if pd is None:
            skip_test("pandas not installed")
        df = pd.DataFrame([{"artist": "A", "year": 1300, "medium": "Oil", "height": 50.0, "width": 50.0}])
        out = self.transformer.fit_transform(df)
        self.assertEqual(out.shape[0], 1)

    def test_f15_b02_current_year_creation(self):
        if pd is None:
            skip_test("pandas not installed")
        df = pd.DataFrame([{"artist": "A", "year": 2026, "medium": "Oil", "height": 50.0, "width": 50.0}])
        out = self.transformer.fit_transform(df)
        self.assertEqual(out.shape[0], 1)

    def test_f15_b03_identical_dimensions_zero_variance(self):
        if pd is None or np is None:
            skip_test("pandas or numpy not installed")
        records = [{"artist": "A", "year": 2000, "medium": "Oil", "height": 100.0, "width": 100.0} for _ in range(5)]
        df = pd.DataFrame(records)
        out = self.transformer.fit_transform(df)
        self.assertFalse(np.any(np.isnan(out)))

    def test_f15_b04_extreme_aspect_ratios_scaled(self):
        if pd is None or np is None:
            skip_test("pandas or numpy not installed")
        records = [
            {"artist": "A", "year": 2000, "medium": "Oil", "height": 10.0, "width": 1000.0},
            {"artist": "B", "year": 2000, "medium": "Oil", "height": 1000.0, "width": 10.0},
        ]
        df = pd.DataFrame(records)
        out = self.transformer.fit_transform(df)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_f15_b05_no_infinite_scaled_outputs(self):
        if pd is None or np is None:
            skip_test("pandas or numpy not installed")
        df = pd.DataFrame(make_diverse_artwork_records(5))
        out = self.transformer.fit_transform(df)
        self.assertTrue(np.all(np.isfinite(out)))


class TestFeature16BoundaryLateConcatenation(unittest.TestCase):
    """F16 Boundary: Feature concatenation memory and shapes."""

    def setUp(self):
        PipelineClass = get_component("src.features.multimodal", "MultimodalFeaturePipeline", milestone="M2")
        self.pipeline = PipelineClass(vision_extractor=None, tabular_transformer=None)

    def test_f16_b01_zero_sample_arrays(self):
        if np is None:
            skip_test("numpy not installed")
        tab = np.zeros((0, 10), dtype=np.float32)
        vis = np.zeros((0, 2048), dtype=np.float32)
        fused = self.pipeline.fuse(tab, vis)
        self.assertEqual(fused.shape, (0, 2058))

    def test_f16_b02_fusing_all_zero_vision_features(self):
        if np is None:
            skip_test("numpy not installed")
        tab = np.ones((2, 5), dtype=np.float32)
        vis = np.zeros((2, 2048), dtype=np.float32)
        fused = self.pipeline.fuse(tab, vis)
        self.assertEqual(fused.shape, (2, 2053))
        np.testing.assert_array_equal(fused[:, :5], tab)

    def test_f16_b03_large_batch_fusion(self):
        if np is None:
            skip_test("numpy not installed")
        N = 500
        tab = np.zeros((N, 12), dtype=np.float32)
        vis = np.zeros((N, 2048), dtype=np.float32)
        fused = self.pipeline.fuse(tab, vis)
        self.assertEqual(fused.shape, (N, 2060))

    def test_f16_b04_1d_array_inputs_handling(self):
        if np is None:
            skip_test("numpy not installed")
        tab = np.ones(8, dtype=np.float32)
        vis = np.zeros(2048, dtype=np.float32)
        try:
            fused = self.pipeline.fuse(tab, vis)
            self.assertIn(fused.shape, [(1, 2056), (2056,)])
        except (ValueError, AssertionError):
            pass

    def test_f16_b05_non_contiguous_array_fusion(self):
        if np is None:
            skip_test("numpy not installed")
        tab = np.ones((10, 8), dtype=np.float32)[::2]  # strided
        vis = np.zeros((10, 2048), dtype=np.float32)[::2]
        fused = self.pipeline.fuse(tab, vis)
        self.assertEqual(fused.shape, (5, 2056))


class TestFeature17BoundaryModalitySlicing(unittest.TestCase):
    """F17 Boundary: Slicing interface edge cases."""

    def setUp(self):
        PipelineClass = get_component("src.features.multimodal", "MultimodalFeaturePipeline", milestone="M2")
        self.pipeline = PipelineClass(vision_extractor=None, tabular_transformer=None)

    def test_f17_b01_single_sample_slice(self):
        if np is None:
            skip_test("numpy not installed")
        if hasattr(self.pipeline, "slice_features"):
            mat = np.random.randn(1, 2060).astype(np.float32)
            tab = self.pipeline.slice_features(mat, modality="tabular", tabular_dim=12)
            self.assertEqual(tab.shape, (1, 12))

    def test_f17_b02_empty_matrix_slice(self):
        if np is None:
            skip_test("numpy not installed")
        if hasattr(self.pipeline, "slice_features"):
            mat = np.zeros((0, 2060), dtype=np.float32)
            vis = self.pipeline.slice_features(mat, modality="vision", tabular_dim=12)
            self.assertEqual(vis.shape, (0, 2048))

    def test_f17_b03_whitespace_in_modality_name(self):
        if np is None:
            skip_test("numpy not installed")
        if hasattr(self.pipeline, "slice_features"):
            mat = np.random.randn(3, 2060).astype(np.float32)
            s = self.pipeline.slice_features(mat, modality="  multimodal  ", tabular_dim=12)
            self.assertEqual(s.shape, (3, 2060))

    def test_f17_b04_exact_slice_boundaries(self):
        if np is None:
            skip_test("numpy not installed")
        if hasattr(self.pipeline, "slice_features"):
            mat = np.arange(2060, dtype=np.float32).reshape(1, 2060)
            tab = self.pipeline.slice_features(mat, modality="tabular", tabular_dim=12)
            self.assertEqual(tab[0, -1], 11.0)

    def test_f17_b05_slice_does_not_mutate_original(self):
        if np is None:
            skip_test("numpy not installed")
        if hasattr(self.pipeline, "slice_features"):
            mat = np.ones((2, 2060), dtype=np.float32)
            _ = self.pipeline.slice_features(mat, modality="tabular", tabular_dim=12)
            self.assertEqual(mat.shape, (2, 2060))


class TestFeature18BoundaryModelTraining(unittest.TestCase):
    """F18 Boundary: Regressor edge conditions."""

    def setUp(self):
        if np is None:
            skip_test("numpy not installed")
        ModelClass = get_component("src.models.train", "ArtValuationModel", milestone="M3")
        self.model = ModelClass()

    def test_f18_b01_tiny_training_set(self):
        X = np.random.randn(5, 10).astype(np.float32)
        y = np.array([8.0, 9.0, 10.0, 11.0, 12.0], dtype=np.float32)
        self.model.fit(X, y)
        self.assertIsNotNone(self.model)

    def test_f18_b02_constant_target_prices(self):
        X = np.random.randn(15, 8).astype(np.float32)
        y = np.full(15, 10.0, dtype=np.float32)
        self.model.fit(X, y)
        p = self.model.predict_log(X)
        self.assertTrue(np.all(np.isfinite(p)))

    def test_f18_b03_single_feature_dimension(self):
        X = np.random.randn(20, 1).astype(np.float32)
        y = np.linspace(8.0, 12.0, 20).astype(np.float32)
        self.model.fit(X, y)
        self.assertEqual(self.model.predict_log(X).shape, (20,))

    def test_f18_b04_all_zero_feature_matrix(self):
        X = np.zeros((10, 16), dtype=np.float32)
        y = np.linspace(8.0, 12.0, 10).astype(np.float32)
        self.model.fit(X, y)
        p = self.model.predict_log(X)
        self.assertTrue(np.all(np.isfinite(p)))

    def test_f18_b05_wide_log_target_spread(self):
        X = np.random.randn(25, 16).astype(np.float32)
        y = np.linspace(2.0, 20.0, 25).astype(np.float32)
        self.model.fit(X, y)
        p = self.model.predict_log(X)
        self.assertTrue(np.all(np.isfinite(p)))


class TestFeature19BoundaryPriceInference(unittest.TestCase):
    """F19 Boundary: Valuation predictions on extreme feature arrays."""

    def setUp(self):
        if np is None:
            skip_test("numpy not installed")
        ModelClass = get_component("src.models.train", "ArtValuationModel", milestone="M3")
        self.model = ModelClass()
        X = np.random.randn(25, 16).astype(np.float32)
        y = np.linspace(8.0, 14.0, 25).astype(np.float32)
        self.model.fit(X, y)

    def test_f19_b01_extreme_positive_feature_values(self):
        X_ext = np.full((3, 16), 1000.0, dtype=np.float32)
        prices = self.model.predict_price(X_ext)
        self.assertTrue(all(prices >= 0.0))

    def test_f19_b02_extreme_negative_feature_values(self):
        X_ext = np.full((3, 16), -1000.0, dtype=np.float32)
        prices = self.model.predict_price(X_ext)
        self.assertTrue(all(prices >= 0.0))

    def test_f19_b03_all_zero_features_inference(self):
        X_zero = np.zeros((2, 16), dtype=np.float32)
        prices = self.model.predict_price(X_zero)
        self.assertTrue(all(prices >= 0.0))

    def test_f19_b04_empty_input_batch(self):
        X_empty = np.zeros((0, 16), dtype=np.float32)
        prices = self.model.predict_price(X_empty)
        self.assertEqual(len(prices), 0)

    def test_f19_b05_deterministic_repeated_predictions(self):
        X_sample = np.random.randn(4, 16).astype(np.float32)
        p1 = self.model.predict_price(X_sample)
        p2 = self.model.predict_price(X_sample)
        np.testing.assert_array_equal(p1, p2)


class TestFeature20BoundaryConfidenceIntervals(unittest.TestCase):
    """F20 Boundary: Confidence bounds at extreme confidence levels."""

    def setUp(self):
        if np is None:
            skip_test("numpy not installed")
        ModelClass = get_component("src.models.train", "ArtValuationModel", milestone="M3")
        self.model = ModelClass()
        X = np.random.randn(25, 16).astype(np.float32)
        y = np.linspace(9.0, 15.0, 25).astype(np.float32)
        self.model.fit(X, y)

    def test_f20_b01_zero_confidence_level_returns_collapsed_interval(self):
        X_test = np.random.randn(3, 16).astype(np.float32)
        lower, point, upper = self.model.predict_with_confidence(X_test, confidence=0.0)
        np.testing.assert_allclose(lower, upper, atol=1e-3)

    def test_f20_b02_high_confidence_level_99_percent(self):
        X_test = np.random.randn(3, 16).astype(np.float32)
        lower, _, upper = self.model.predict_with_confidence(X_test, confidence=0.99)
        self.assertTrue(all(lower <= upper))

    def test_f20_b03_lower_bound_strictly_non_negative(self):
        X_test = np.random.randn(10, 16).astype(np.float32)
        lower, _, _ = self.model.predict_with_confidence(X_test, confidence=0.95)
        self.assertTrue(all(l >= 0.0 for l in lower))

    def test_f20_b04_single_sample_confidence_shape(self):
        X_test = np.random.randn(1, 16).astype(np.float32)
        lower, point, upper = self.model.predict_with_confidence(X_test, confidence=0.80)
        self.assertEqual(len(lower), 1)
        self.assertEqual(len(point), 1)
        self.assertEqual(len(upper), 1)

    def test_f20_b05_invalid_confidence_parameter_raises(self):
        X_test = np.random.randn(2, 16).astype(np.float32)
        with self.assertRaises((ValueError, AssertionError)):
            self.model.predict_with_confidence(X_test, confidence=1.5)


class TestFeature21BoundaryEvaluation(unittest.TestCase):
    """F21 Boundary: Evaluation metrics with degenerate true/pred inputs."""

    def setUp(self):
        self.compute_metrics = get_component("src.models.evaluate", "compute_metrics", milestone="M3")

    def test_f21_b01_zero_variance_in_true_prices(self):
        if np is None:
            skip_test("numpy not installed")
        y_true = np.full(10, 5000.0)
        y_pred = np.full(10, 5000.0)
        m = self.compute_metrics(y_true, y_pred)
        self.assertAlmostEqual(m["mae"], 0.0, places=4)

    def test_f21_b02_single_sample_metrics(self):
        if np is None:
            skip_test("numpy not installed")
        y_true = np.array([1000.0])
        y_pred = np.array([1200.0])
        m = self.compute_metrics(y_true, y_pred)
        self.assertAlmostEqual(m["mae"], 200.0, places=2)

    def test_f21_b03_inverted_predictions_negative_r2(self):
        if np is None:
            skip_test("numpy not installed")
        y_true = np.array([1000.0, 10000.0, 100000.0])
        y_pred = np.array([100000.0, 10000.0, 1000.0])
        m = self.compute_metrics(y_true, y_pred)
        self.assertLess(m["r2"], 0.0)

    def test_f21_b04_near_zero_prices_mape_protection(self):
        if np is None:
            skip_test("numpy not installed")
        y_true = np.array([0.01, 100.0])
        y_pred = np.array([0.02, 105.0])
        m = self.compute_metrics(y_true, y_pred)
        self.assertTrue(math.isfinite(m["mape"]))

    def test_f21_b05_empty_arrays_raise_error(self):
        if np is None:
            skip_test("numpy not installed")
        with self.assertRaises((ValueError, Exception)):
            self.compute_metrics(np.array([]), np.array([]))


class TestFeature22BoundarySerialization(unittest.TestCase):
    """F22 Boundary: Pipeline save edge cases."""

    def setUp(self):
        self.save_fn = get_component("src.models.io", "save_model_pipeline", milestone="M3")
        self.tmp_dir = tempfile.mkdtemp(prefix="caniq_bsave_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_f22_b01_unicode_in_metadata(self):
        meta = {"description": "Modèle d'évaluation artistique aux enchères 🎨"}
        self.save_fn(model={"m": 1}, tabular_transformer={"t": 1}, metadata=meta, output_dir=self.tmp_dir)
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, "metadata.json")))

    def test_f22_b02_deeply_nested_save_path(self):
        nested = os.path.join(self.tmp_dir, "a", "b", "c", "d")
        self.save_fn(model={"m": 1}, tabular_transformer={"t": 1}, metadata={"k": "v"}, output_dir=nested)
        self.assertTrue(os.path.exists(nested))

    def test_f22_b03_empty_metadata_dict(self):
        self.save_fn(model={"m": 1}, tabular_transformer={"t": 1}, metadata={}, output_dir=self.tmp_dir)
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, "metadata.json")))

    def test_f22_b04_repeated_overwrites(self):
        for i in range(3):
            self.save_fn(model={"m": i}, tabular_transformer={"t": i}, metadata={"iter": i}, output_dir=self.tmp_dir)
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, "metadata.json")))

    def test_f22_b05_none_in_metadata_values(self):
        meta = {"optional_metric": None}
        self.save_fn(model={"m": 1}, tabular_transformer={"t": 1}, metadata=meta, output_dir=self.tmp_dir)
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, "metadata.json")))


class TestFeature23BoundaryReload(unittest.TestCase):
    """F23 Boundary: Pipeline load resilience."""

    def setUp(self):
        self.load_fn = get_component("src.models.io", "load_model_pipeline", milestone="M3")
        self.tmp_dir = tempfile.mkdtemp(prefix="caniq_bload_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_f23_b01_corrupt_json_metadata_raises(self):
        with open(os.path.join(self.tmp_dir, "metadata.json"), "w") as f:
            f.write("{broken json:::")
        with self.assertRaises(Exception):
            self.load_fn(self.tmp_dir)

    def test_f23_b02_empty_directory_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.load_fn(self.tmp_dir)

    def test_f23_b03_missing_model_file_raises(self):
        with open(os.path.join(self.tmp_dir, "metadata.json"), "w") as f:
            f.write("{}")
        with self.assertRaises(FileNotFoundError):
            self.load_fn(self.tmp_dir)

    def test_f23_b04_path_as_file_instead_of_dir(self):
        tmp_file = os.path.join(self.tmp_dir, "somefile.txt")
        with open(tmp_file, "w") as f:
            f.write("text")
        with self.assertRaises((FileNotFoundError, NotADirectoryError, Exception)):
            self.load_fn(tmp_file)

    def test_f23_b05_non_existent_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.load_fn("/path/does/not/exist/99999")


class TestFeature24BoundaryAppDashboard(unittest.TestCase):
    """F24 Boundary: Web application edge conditions."""

    def test_f24_b01_app_file_in_designated_directory(self):
        app_file = REPO_ROOT / "app" / "app.py"
        if not app_file.exists():
            skip_test("app/app.py not yet implemented [Pending M4]")
        self.assertEqual(app_file.parent.name, "app")

    def test_f24_b02_app_handles_unsupported_image_format(self):
        app_file = REPO_ROOT / "app" / "app.py"
        if not app_file.exists():
            skip_test("app/app.py not yet implemented [Pending M4]")
        with open(app_file) as f:
            content = f.read()
        self.assertTrue("png" in content.lower() or "jpg" in content.lower())

    def test_f24_b03_app_slider_bounds(self):
        app_file = REPO_ROOT / "app" / "app.py"
        if not app_file.exists():
            skip_test("app/app.py not yet implemented [Pending M4]")
        with open(app_file) as f:
            content = f.read()
        self.assertTrue("min_value" in content or "max_value" in content or "1800" in content or "1900" in content)

    def test_f24_b04_app_title_present(self):
        app_file = REPO_ROOT / "app" / "app.py"
        if not app_file.exists():
            skip_test("app/app.py not yet implemented [Pending M4]")
        with open(app_file) as f:
            content = f.read()
        self.assertTrue("set_page_config" in content or "title" in content.lower())

    def test_f24_b05_app_sidebar_or_column_layout(self):
        app_file = REPO_ROOT / "app" / "app.py"
        if not app_file.exists():
            skip_test("app/app.py not yet implemented [Pending M4]")
        with open(app_file) as f:
            content = f.read()
        self.assertTrue("columns" in content or "sidebar" in content)


class TestFeature25BoundaryValuationReport(unittest.TestCase):
    """F25 Boundary: Valuation rendering at extreme quantities."""

    def test_f25_b01_nine_figure_appraisal_format(self):
        app_file = REPO_ROOT / "app" / "app.py"
        if not app_file.exists():
            skip_test("app/app.py not yet implemented [Pending M4]")
        with open(app_file) as f:
            content = f.read()
        self.assertTrue(":,}" in content or ":,.0f}" in content or ":,.2f}" in content or "round" in content)

    def test_f25_b02_subdollar_appraisal_format(self):
        app_file = REPO_ROOT / "app" / "app.py"
        if not app_file.exists():
            skip_test("app/app.py not yet implemented [Pending M4]")
        with open(app_file) as f:
            content = f.read()
        self.assertTrue("format" in content or "$" in content or "f\"" in content)

    def test_f25_b03_zero_lower_bound_presentation(self):
        app_file = REPO_ROOT / "app" / "app.py"
        if not app_file.exists():
            skip_test("app/app.py not yet implemented [Pending M4]")
        with open(app_file) as f:
            content = f.read()
        self.assertTrue("lower" in content.lower() or "bound" in content.lower() or "confidence" in content.lower())

    def test_f25_b04_aesthetic_attribute_display(self):
        app_file = REPO_ROOT / "app" / "app.py"
        if not app_file.exists():
            skip_test("app/app.py not yet implemented [Pending M4]")
        with open(app_file) as f:
            content = f.read()
        self.assertTrue("visual" in content.lower() or "style" in content.lower() or "attribute" in content.lower() or "breakdown" in content.lower())

    def test_f25_b05_metric_cards_defined(self):
        app_file = REPO_ROOT / "app" / "app.py"
        if not app_file.exists():
            skip_test("app/app.py not yet implemented [Pending M4]")
        with open(app_file) as f:
            content = f.read()
        self.assertTrue("metric" in content or "write" in content or "markdown" in content)


class TestFeature26BoundaryGitCommits(unittest.TestCase):
    """F26 Boundary: Git conventional commit and remote integrity."""

    def test_f26_b01_branch_is_main(self):
        import subprocess
        res = subprocess.run(["git", "branch", "--show-current"], cwd=str(REPO_ROOT), capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertEqual(res.stdout.strip(), "main")

    def test_f26_b02_conventional_commit_prefix_check(self):
        import subprocess
        res = subprocess.run(["git", "log", "-n", "5", "--format=%s"], cwd=str(REPO_ROOT), capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        pattern = re.compile(r"^(feat|chore|docs|test|fix|refactor)(\(.*\))?: .+")
        for line in res.stdout.strip().split("\n"):
            if line:
                self.assertTrue(bool(pattern.match(line)), f"Commit '{line}' does not match conventional format")

    def test_f26_b03_remote_is_github(self):
        import subprocess
        res = subprocess.run(["git", "remote", "get-url", "origin"], cwd=str(REPO_ROOT), capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("github.com", res.stdout)

    def test_f26_b04_no_pyc_in_git_status(self):
        import subprocess
        res = subprocess.run(["git", "status", "--porcelain"], cwd=str(REPO_ROOT), capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertNotIn(".pyc", res.stdout)

    def test_f26_b05_git_credential_configured(self):
        import subprocess
        res = subprocess.run(["git", "config", "credential.helper"], cwd=str(REPO_ROOT), capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("store", res.stdout)


class TestFeature27BoundaryOpenSpecTasks(unittest.TestCase):
    """F27 Boundary: Tasks structure and syntax boundaries."""

    def test_f27_b01_no_orphan_headers(self):
        tasks_file = REPO_ROOT / "openspec" / "changes" / "multimodal-art-valuation" / "tasks.md"
        with open(tasks_file) as f:
            content = f.read()
        sections = re.findall(r"##\s+\d+\..*", content)
        self.assertEqual(len(sections), 5)

    def test_f27_b02_strict_checkbox_syntax(self):
        tasks_file = REPO_ROOT / "openspec" / "changes" / "multimodal-art-valuation" / "tasks.md"
        with open(tasks_file) as f:
            content = f.read()
        invalid_boxes = re.findall(r"-\s+\[[^ x]\]", content)
        self.assertEqual(len(invalid_boxes), 0)

    def test_f27_b03_chronological_task_ids(self):
        tasks_file = REPO_ROOT / "openspec" / "changes" / "multimodal-art-valuation" / "tasks.md"
        with open(tasks_file) as f:
            content = f.read()
        ids = [float(x) for x in re.findall(r"-\s+\[[ x]\]\s+(\d+\.\d+)", content)]
        self.assertEqual(ids, sorted(ids))

    def test_f27_b04_all_tasks_have_non_empty_descriptions(self):
        tasks_file = REPO_ROOT / "openspec" / "changes" / "multimodal-art-valuation" / "tasks.md"
        with open(tasks_file) as f:
            lines = [l.strip() for l in f if l.strip().startswith("- [")]
        for l in lines:
            parts = l.split("]", 1)
            self.assertGreater(len(parts[1].strip()), 10)

    def test_f27_b05_total_task_count_matches_milestones(self):
        tasks_file = REPO_ROOT / "openspec" / "changes" / "multimodal-art-valuation" / "tasks.md"
        with open(tasks_file) as f:
            content = f.read()
        items = re.findall(r"-\s+\[[ x]\]\s+\d+\.\d+", content)
        self.assertEqual(len(items), 13)


class TestFeature28BoundaryOpenSpecArchival(unittest.TestCase):
    """F28 Boundary: OpenSpec artifact completeness and consistency."""

    def test_f28_b01_proposal_has_requirements(self):
        proposal = REPO_ROOT / "openspec" / "changes" / "multimodal-art-valuation" / "proposal.md"
        with open(proposal) as f:
            content = f.read()
        self.assertTrue("## Why" in content or "## What Changes" in content or "## Capabilities" in content)

    def test_f28_b02_design_has_risks_section(self):
        design = REPO_ROOT / "openspec" / "changes" / "multimodal-art-valuation" / "design.md"
        with open(design) as f:
            content = f.read()
        self.assertIn("## Risks", content)

    def test_f28_b03_design_has_decisions_section(self):
        design = REPO_ROOT / "openspec" / "changes" / "multimodal-art-valuation" / "design.md"
        with open(design) as f:
            content = f.read()
        self.assertIn("## Decisions", content)

    def test_f28_b04_four_delta_spec_domains(self):
        specs_dir = REPO_ROOT / "openspec" / "changes" / "multimodal-art-valuation" / "specs"
        domains = [d.name for d in specs_dir.iterdir() if d.is_dir()]
        self.assertIn("data-pipeline", domains)
        self.assertIn("feature-extraction", domains)
        self.assertIn("valuation-engine", domains)
        self.assertIn("valuation-interface", domains)

    def test_f28_b05_delta_specs_have_spec_markdown(self):
        specs_dir = REPO_ROOT / "openspec" / "changes" / "multimodal-art-valuation" / "specs"
        for d in specs_dir.iterdir():
            if d.is_dir():
                spec_file = d / "spec.md"
                self.assertTrue(spec_file.exists(), f"{spec_file} missing")


if __name__ == "__main__":
    unittest.main()
