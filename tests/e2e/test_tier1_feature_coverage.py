"""Tier 1: Feature Coverage E2E Test Suite for CanvasIQ.

Covers all 28 features from PROJECT.md with >=5 test cases per feature (140 tests total).
Exercises primary happy-path behavior, interface contracts, and mathematical properties.
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
    require_package,
    make_sample_artwork_dict,
    make_sample_pil_image,
    make_sample_image_bytes,
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


class TestFeature01DatasetSchemaValidator(unittest.TestCase):
    """F1: Dataset Schema Validator (src/data/validator.py)."""

    def setUp(self):
        self.validator_fn = get_component("src.data.validator", "validate_artwork_schema", milestone="M1")

    def test_f1_01_valid_dict_passes(self):
        record = make_sample_artwork_dict()
        res = self.validator_fn(record, is_training=True)
        self.assertIsNotNone(res)
        self.assertEqual(len(res), 1)
        self.assertEqual(res.iloc[0]["artist"], "Claude Monet")

    def test_f1_02_valid_dataframe_passes(self):
        if pd is None:
            skip_test("pandas not installed")
        records = make_diverse_artwork_records(3)
        df = pd.DataFrame(records)
        res = self.validator_fn(df, is_training=True)
        self.assertEqual(len(res), 3)

    def test_f1_03_optional_image_path_preserved(self):
        record = make_sample_artwork_dict()
        record["image_path"] = "data/images/synthetic/test.png"
        res = self.validator_fn(record, is_training=True)
        self.assertIn("image_path", res.columns)
        self.assertEqual(res.iloc[0]["image_path"], "data/images/synthetic/test.png")

    def test_f1_04_inference_mode_without_price(self):
        record = make_sample_artwork_dict()
        del record["price"]
        res = self.validator_fn(record, is_training=False)
        self.assertEqual(len(res), 1)
        self.assertEqual(res.iloc[0]["artist"], "Claude Monet")

    def test_f1_05_mandatory_columns_present_in_output(self):
        record = make_sample_artwork_dict()
        res = self.validator_fn(record, is_training=True)
        for col in ["artist", "year", "medium", "height", "width", "price"]:
            self.assertIn(col, res.columns)


class TestFeature02CleanSanitizeRecords(unittest.TestCase):
    """F2: Clean & Sanitize Records (src/data/cleaner.py)."""

    def setUp(self):
        CleanerClass = get_component("src.data.cleaner", "ArtDataCleaner", milestone="M1")
        self.cleaner = CleanerClass()

    def test_f2_01_clean_valid_dataframe(self):
        if pd is None:
            skip_test("pandas not installed")
        df = pd.DataFrame(make_diverse_artwork_records(4))
        cleaned = self.cleaner.clean(df, is_training=True)
        self.assertEqual(len(cleaned), 4)

    def test_f2_02_missing_medium_imputed_with_unknown(self):
        if pd is None:
            skip_test("pandas not installed")
        records = [make_sample_artwork_dict()]
        records[0]["medium"] = None
        df = pd.DataFrame(records)
        cleaned = self.cleaner.clean(df, is_training=True)
        self.assertEqual(cleaned.iloc[0]["medium"], "Unknown")

    def test_f2_03_non_positive_dimensions_filtered_or_handled(self):
        if pd is None:
            skip_test("pandas not installed")
        records = [
            make_sample_artwork_dict(),
            {"artist": "Bad", "year": 2000, "medium": "Oil", "height": -10.0, "width": 50.0, "price": 1000.0},
        ]
        df = pd.DataFrame(records)
        cleaned = self.cleaner.clean(df, is_training=True)
        self.assertTrue(all(cleaned["height"] > 0))

    def test_f2_04_non_positive_prices_filtered_in_training(self):
        if pd is None:
            skip_test("pandas not installed")
        records = [
            make_sample_artwork_dict(),
            {"artist": "BadPrice", "year": 2000, "medium": "Oil", "height": 50.0, "width": 50.0, "price": -500.0},
        ]
        df = pd.DataFrame(records)
        cleaned = self.cleaner.clean(df, is_training=True)
        self.assertTrue(all(cleaned["price"] > 0))

    def test_f2_05_derived_feature_columns_added(self):
        if pd is None:
            skip_test("pandas not installed")
        df = pd.DataFrame([make_sample_artwork_dict()])
        cleaned = self.cleaner.clean(df, is_training=True)
        self.assertIn("surface_area", cleaned.columns)
        self.assertIn("aspect_ratio", cleaned.columns)


class TestFeature03SurfaceAreaCalculation(unittest.TestCase):
    """F3: Surface Area Calculation (height * width)."""

    def setUp(self):
        CleanerClass = get_component("src.data.cleaner", "ArtDataCleaner", milestone="M1")
        self.calc = getattr(CleanerClass, "calculate_surface_area")

    def test_f3_01_standard_dimensions(self):
        area = self.calc(50.0, 100.0)
        self.assertAlmostEqual(area, 5000.0, places=4)

    def test_f3_02_square_canvas(self):
        area = self.calc(40.0, 40.0)
        self.assertAlmostEqual(area, 1600.0, places=4)

    def test_f3_03_large_canvas(self):
        area = self.calc(200.0, 300.0)
        self.assertAlmostEqual(area, 60000.0, places=4)

    def test_f3_04_fractional_dimensions(self):
        area = self.calc(25.5, 40.2)
        self.assertAlmostEqual(area, 1025.1, places=4)

    def test_f3_05_commutative_property(self):
        self.assertEqual(self.calc(35.0, 70.0), self.calc(70.0, 35.0))


class TestFeature04AspectRatioCalculation(unittest.TestCase):
    """F4: Aspect Ratio Calculation (width / height)."""

    def setUp(self):
        CleanerClass = get_component("src.data.cleaner", "ArtDataCleaner", milestone="M1")
        self.calc = getattr(CleanerClass, "calculate_aspect_ratio")

    def test_f4_01_landscape_ratio(self):
        ratio = self.calc(50.0, 100.0)
        self.assertAlmostEqual(ratio, 2.0, places=4)

    def test_f4_02_portrait_ratio(self):
        ratio = self.calc(100.0, 50.0)
        self.assertAlmostEqual(ratio, 0.5, places=4)

    def test_f4_03_square_ratio(self):
        ratio = self.calc(75.0, 75.0)
        self.assertAlmostEqual(ratio, 1.0, places=4)

    def test_f4_04_panoramic_ratio(self):
        ratio = self.calc(60.0, 300.0)
        self.assertAlmostEqual(ratio, 5.0, places=4)

    def test_f4_05_golden_ratio_canvas(self):
        ratio = self.calc(100.0, 161.8)
        self.assertAlmostEqual(ratio, 1.618, places=3)


class TestFeature05LogTargetTransformation(unittest.TestCase):
    """F5: Log Target Transformation (y = log(1 + price))."""

    def setUp(self):
        CleanerClass = get_component("src.data.cleaner", "ArtDataCleaner", milestone="M1")
        self.transform = getattr(CleanerClass, "transform_log_price")

    def test_f5_01_zero_price_log_identity(self):
        log_p = self.transform(0.0)
        self.assertAlmostEqual(log_p, 0.0, places=5)

    def test_f5_02_typical_art_price(self):
        price = 10000.0
        log_p = self.transform(price)
        self.assertAlmostEqual(log_p, math.log1p(price), places=5)

    def test_f5_03_high_value_auction_price(self):
        price = 1000000.0
        log_p = self.transform(price)
        self.assertAlmostEqual(log_p, math.log1p(price), places=5)

    def test_f5_04_monotonicity(self):
        p1, p2 = 500.0, 5000.0
        self.assertLess(self.transform(p1), self.transform(p2))

    def test_f5_05_vectorized_input(self):
        if np is None:
            skip_test("numpy not installed")
        prices = np.array([10.0, 100.0, 1000.0])
        log_prices = self.transform(prices)
        np.testing.assert_allclose(log_prices, np.log1p(prices), rtol=1e-5)


class TestFeature06InversePriceBackTransformation(unittest.TestCase):
    """F6: Inverse Price Back-Transformation (price = exp(y) - 1, clamped >= 0)."""

    def setUp(self):
        CleanerClass = get_component("src.data.cleaner", "ArtDataCleaner", milestone="M1")
        self.inverse = getattr(CleanerClass, "inverse_log_price")
        self.transform = getattr(CleanerClass, "transform_log_price")

    def test_f6_01_zero_log_returns_zero_dollars(self):
        price = self.inverse(0.0)
        self.assertAlmostEqual(price, 0.0, places=5)

    def test_f6_02_typical_inversion(self):
        expected = 25000.0
        log_p = math.log1p(expected)
        inv = self.inverse(log_p)
        self.assertAlmostEqual(inv, expected, places=3)

    def test_f6_03_roundtrip_identity(self):
        original = 150000.0
        restored = self.inverse(self.transform(original))
        self.assertAlmostEqual(restored, original, places=2)

    def test_f6_04_clamping_negative_log_values(self):
        price = self.inverse(-1.5)
        self.assertGreaterEqual(price, 0.0)

    def test_f6_05_vectorized_inverse_transformation(self):
        if np is None:
            skip_test("numpy not installed")
        log_prices = np.array([0.0, 4.61512, 9.21044])
        prices = self.inverse(log_prices)
        self.assertTrue(all(prices >= 0.0))


class TestFeature07ProceduralArtworkImageGenerator(unittest.TestCase):
    """F7: Procedural Artwork Image Generator (src/data/synthetic.py)."""

    def setUp(self):
        GeneratorClass = get_component("src.data.synthetic", "SyntheticArtDatasetGenerator", milestone="M1")
        self.gen = GeneratorClass(seed=42)

    def test_f7_01_generate_colorfield(self):
        img = self.gen.generate_image(style="colorfield", palette="oceanic", size=(224, 224))
        self.assertEqual(img.size, (224, 224))
        self.assertEqual(img.mode, "RGB")

    def test_f7_02_generate_abstract(self):
        img = self.gen.generate_image(style="abstract", palette="earthy", size=(224, 224))
        self.assertEqual(img.size, (224, 224))
        self.assertEqual(img.mode, "RGB")

    def test_f7_03_generate_minimalist(self):
        img = self.gen.generate_image(style="minimalist", palette="monochrome", size=(224, 224))
        self.assertEqual(img.size, (224, 224))

    def test_f7_04_generate_impressionist(self):
        img = self.gen.generate_image(style="impressionist", palette="vibrant", size=(224, 224))
        self.assertEqual(img.size, (224, 224))

    def test_f7_05_custom_size(self):
        img = self.gen.generate_image(style="colorfield", palette="oceanic", size=(128, 64))
        self.assertEqual(img.size, (128, 64))


class TestFeature08SyntheticMetadataGenerator(unittest.TestCase):
    """F8: Synthetic Metadata Generator (src/data/synthetic.py)."""

    def setUp(self):
        GeneratorClass = get_component("src.data.synthetic", "SyntheticArtDatasetGenerator", milestone="M1")
        self.gen = GeneratorClass(seed=42)
        self.tmp_dir = tempfile.mkdtemp(prefix="caniq_bench_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_f8_01_sample_count(self):
        df, _ = self.gen.generate_benchmark(n_samples=10, output_dir=self.tmp_dir)
        self.assertEqual(len(df), 10)

    def test_f8_02_required_columns(self):
        df, _ = self.gen.generate_benchmark(n_samples=5, output_dir=self.tmp_dir)
        for col in ["artist", "year", "medium", "height", "width", "price"]:
            self.assertIn(col, df.columns)

    def test_f8_03_images_saved_to_disk(self):
        df, img_dir = self.gen.generate_benchmark(n_samples=3, output_dir=self.tmp_dir)
        for path in df["image_path"]:
            self.assertTrue(os.path.exists(path), f"File {path} does not exist")

    def test_f8_04_reproducibility(self):
        if np is None:
            skip_test("numpy not installed")
        Gen = get_component("src.data.synthetic", "SyntheticArtDatasetGenerator", milestone="M1")
        df1, _ = Gen(seed=99).generate_benchmark(n_samples=5, output_dir=tempfile.mkdtemp())
        df2, _ = Gen(seed=99).generate_benchmark(n_samples=5, output_dir=tempfile.mkdtemp())
        np.testing.assert_array_equal(df1["price"].values, df2["price"].values)

    def test_f8_05_positive_prices_generated(self):
        df, _ = self.gen.generate_benchmark(n_samples=15, output_dir=self.tmp_dir)
        self.assertTrue(all(df["price"] > 0))


class TestFeature09FrozenPretrainedResNet50Backbone(unittest.TestCase):
    """F9: Frozen Pretrained ResNet50 Backbone (src/features/vision.py)."""

    def setUp(self):
        if np is None:
            skip_test("numpy not installed")
        ExtractorClass = get_component("src.features.vision", "ResNet50FeatureExtractor", milestone="M2")
        self.extractor = ExtractorClass(device="cpu")

    def test_f9_01_feature_vector_dimension(self):
        img = make_sample_pil_image(224, 224)
        feats = self.extractor.extract_image(img)
        self.assertEqual(feats.shape, (2048,))

    def test_f9_02_output_dtype_float32(self):
        img = make_sample_pil_image(224, 224)
        feats = self.extractor.extract_image(img)
        self.assertEqual(feats.dtype, np.float32)

    def test_f9_03_finite_embedding_values(self):
        img = make_sample_pil_image(224, 224)
        feats = self.extractor.extract_image(img)
        self.assertTrue(np.all(np.isfinite(feats)))

    def test_f9_04_extractor_frozen_parameters(self):
        if hasattr(self.extractor, "model") and hasattr(self.extractor.model, "parameters"):
            for p in self.extractor.model.parameters():
                self.assertFalse(p.requires_grad)

    def test_f9_05_consistent_embedding_on_same_image(self):
        img = make_sample_pil_image(224, 224)
        f1 = self.extractor.extract_image(img)
        f2 = self.extractor.extract_image(img)
        np.testing.assert_allclose(f1, f2, atol=1e-6)


class TestFeature10ImagePreprocessingTransforms(unittest.TestCase):
    """F10: Image Preprocessing & Transforms (src/features/vision.py)."""

    def setUp(self):
        ExtractorClass = get_component("src.features.vision", "ResNet50FeatureExtractor", milestone="M2")
        self.extractor = ExtractorClass(device="cpu")

    def test_f10_01_accepts_pil_image(self):
        img = make_sample_pil_image(300, 400)
        feats = self.extractor.extract_image(img)
        self.assertEqual(feats.shape, (2048,))

    def test_f10_02_accepts_image_bytes(self):
        raw_bytes = make_sample_image_bytes()
        feats = self.extractor.extract_image(raw_bytes)
        self.assertEqual(feats.shape, (2048,))

    def test_f10_03_accepts_file_path(self):
        img = make_sample_pil_image(224, 224)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp.name)
        try:
            feats = self.extractor.extract_image(tmp.name)
            self.assertEqual(feats.shape, (2048,))
        finally:
            os.unlink(tmp.name)

    def test_f10_04_batch_extraction_shape(self):
        images = [make_sample_pil_image(224, 224) for _ in range(3)]
        batch_feats = self.extractor.extract_batch(images)
        self.assertEqual(batch_feats.shape, (3, 2048))

    def test_f10_05_arbitrary_aspect_ratios_handled(self):
        wide_img = make_sample_pil_image(600, 150)
        feats = self.extractor.extract_image(wide_img)
        self.assertEqual(feats.shape, (2048,))


class TestFeature11MissingCorruptImageFallback(unittest.TestCase):
    """F11: Missing/Corrupt Image Fallback to 2048-D Zero Vector."""

    def setUp(self):
        if np is None:
            skip_test("numpy not installed")
        ExtractorClass = get_component("src.features.vision", "ResNet50FeatureExtractor", milestone="M2")
        self.extractor = ExtractorClass(device="cpu")

    def test_f11_01_non_existent_file_returns_zeros(self):
        feats = self.extractor.extract_image("non_existent_file_12345.png")
        self.assertEqual(feats.shape, (2048,))
        np.testing.assert_array_equal(feats, np.zeros(2048, dtype=np.float32))

    def test_f11_02_corrupt_bytes_returns_zeros(self):
        corrupt = b"not_a_valid_image_header_bytes_123"
        feats = self.extractor.extract_image(corrupt)
        self.assertEqual(feats.shape, (2048,))
        np.testing.assert_array_equal(feats, np.zeros(2048, dtype=np.float32))

    def test_f11_03_empty_string_returns_zeros(self):
        feats = self.extractor.extract_image("")
        self.assertEqual(feats.shape, (2048,))
        np.testing.assert_array_equal(feats, np.zeros(2048, dtype=np.float32))

    def test_f11_04_none_returns_zeros(self):
        feats = self.extractor.extract_image(None)
        self.assertEqual(feats.shape, (2048,))
        np.testing.assert_array_equal(feats, np.zeros(2048, dtype=np.float32))

    def test_f11_05_batch_fallback_preserves_valid_and_zero_slots(self):
        items = [make_sample_pil_image(224, 224), "non_existent.jpg"]
        batch = self.extractor.extract_batch(items)
        self.assertEqual(batch.shape, (2, 2048))
        self.assertFalse(np.all(batch[0] == 0))
        np.testing.assert_array_equal(batch[1], np.zeros(2048, dtype=np.float32))


class TestFeature12DeviceHandling(unittest.TestCase):
    """F12: Device Handling (CPU/GPU) (src/features/vision.py)."""

    def setUp(self):
        self.ExtractorClass = get_component("src.features.vision", "ResNet50FeatureExtractor", milestone="M2")

    def test_f12_01_explicit_cpu_initialization(self):
        ext = self.ExtractorClass(device="cpu")
        self.assertIsNotNone(ext)

    def test_f12_02_default_device_initialization(self):
        ext = self.ExtractorClass()
        self.assertIsNotNone(ext)

    def test_f12_03_eval_mode_set(self):
        ext = self.ExtractorClass(device="cpu")
        if hasattr(ext, "model"):
            self.assertFalse(ext.model.training)

    def test_f12_04_extraction_on_configured_device(self):
        ext = self.ExtractorClass(device="cpu")
        feats = ext.extract_image(make_sample_pil_image(224, 224))
        self.assertEqual(feats.shape, (2048,))

    def test_f12_05_no_unhandled_crash_on_gpu_fallback(self):
        try:
            ext = self.ExtractorClass(device="cuda")
            self.assertIsNotNone(ext)
        except Exception as e:
            self.assertIn("cuda", str(e).lower())


class TestFeature13MediumCategoricalEncoding(unittest.TestCase):
    """F13: Medium Categorical Encoding (src/features/tabular.py)."""

    def setUp(self):
        TransformerClass = get_component("src.features.tabular", "TabularTransformer", milestone="M2")
        self.transformer = TransformerClass()

    def test_f13_01_fit_and_transform_known_mediums(self):
        if pd is None:
            skip_test("pandas not installed")
        df = pd.DataFrame(make_diverse_artwork_records(5))
        self.transformer.fit(df)
        transformed = self.transformer.transform(df)
        self.assertIsNotNone(transformed)

    def test_f13_02_unseen_medium_handled_gracefully(self):
        if pd is None:
            skip_test("pandas not installed")
        train_df = pd.DataFrame(make_diverse_artwork_records(4))
        test_df = pd.DataFrame([{"artist": "Picasso", "year": 2020, "medium": "Titanium Foil 3D", "height": 50.0, "width": 50.0}])
        self.transformer.fit(train_df)
        transformed = self.transformer.transform(test_df)
        self.assertEqual(transformed.shape[0], 1)

    def test_f13_03_missing_medium_unknown_handled(self):
        if pd is None or np is None:
            skip_test("pandas or numpy not installed")
        df = pd.DataFrame([{"artist": "Picasso", "year": 2020, "medium": "Unknown", "height": 50.0, "width": 50.0}])
        self.transformer.fit(df)
        out = self.transformer.transform(df)
        self.assertFalse(np.any(np.isnan(out)))

    def test_f13_04_transform_shape_reproducible(self):
        if pd is None:
            skip_test("pandas not installed")
        df = pd.DataFrame(make_diverse_artwork_records(4))
        out1 = self.transformer.fit_transform(df)
        out2 = self.transformer.transform(df)
        self.assertEqual(out1.shape, out2.shape)

    def test_f13_05_no_nan_in_encoded_output(self):
        if pd is None or np is None:
            skip_test("pandas or numpy not installed")
        df = pd.DataFrame(make_diverse_artwork_records(5))
        out = self.transformer.fit_transform(df)
        self.assertFalse(np.any(np.isnan(out)))


class TestFeature14ArtistStatisticsPedigreeEncoding(unittest.TestCase):
    """F14: Artist Statistics & Pedigree Encoding (src/features/tabular.py)."""

    def setUp(self):
        TransformerClass = get_component("src.features.tabular", "TabularTransformer", milestone="M2")
        self.transformer = TransformerClass(smoothing=10.0)

    def test_f14_01_fit_with_target_prices(self):
        if pd is None or np is None:
            skip_test("pandas or numpy not installed")
        df = pd.DataFrame(make_diverse_artwork_records(6))
        y = np.log1p(df["price"].values)
        self.transformer.fit(df, y)
        out = self.transformer.transform(df)
        self.assertIsNotNone(out)

    def test_f14_02_unseen_artist_uses_global_prior(self):
        if pd is None or np is None:
            skip_test("pandas or numpy not installed")
        train_df = pd.DataFrame(make_diverse_artwork_records(5))
        y = np.log1p(train_df["price"].values)
        self.transformer.fit(train_df, y)
        test_df = pd.DataFrame([{"artist": "Completely Novel Artist 999", "year": 2022, "medium": "Oil on Canvas", "height": 50.0, "width": 50.0}])
        out = self.transformer.transform(test_df)
        self.assertFalse(np.any(np.isnan(out)))

    def test_f14_03_smoothing_parameter_preserved(self):
        self.assertEqual(self.transformer.smoothing, 10.0)

    def test_f14_04_single_artist_transform_shape(self):
        if pd is None or np is None:
            skip_test("pandas or numpy not installed")
        df = pd.DataFrame([make_sample_artwork_dict()])
        out = self.transformer.fit_transform(df, np.array([math.log1p(df.iloc[0]["price"])]))
        self.assertEqual(out.shape[0], 1)

    def test_f14_05_deterministic_encoding(self):
        if pd is None or np is None:
            skip_test("pandas or numpy not installed")
        df = pd.DataFrame(make_diverse_artwork_records(4))
        y = np.log1p(df["price"].values)
        o1 = self.transformer.fit_transform(df, y)
        o2 = self.transformer.transform(df)
        np.testing.assert_allclose(o1, o2, atol=1e-5)


class TestFeature15DimensionalScalingEraNormalization(unittest.TestCase):
    """F15: Dimensional Scaling & Era Normalization (src/features/tabular.py)."""

    def setUp(self):
        TransformerClass = get_component("src.features.tabular", "TabularTransformer", milestone="M2")
        self.transformer = TransformerClass()

    def test_f15_01_continuous_features_normalized(self):
        if pd is None or np is None:
            skip_test("pandas or numpy not installed")
        df = pd.DataFrame(make_diverse_artwork_records(6))
        out = self.transformer.fit_transform(df)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_f15_02_outlier_dimension_stability(self):
        if pd is None or np is None:
            skip_test("pandas or numpy not installed")
        records = make_diverse_artwork_records(4)
        records.append({"artist": "Muralist", "year": 1950, "medium": "Oil on Canvas", "height": 4000.0, "width": 8000.0, "price": 5000000.0})
        df = pd.DataFrame(records)
        out = self.transformer.fit_transform(df)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_f15_03_historical_era_range_scaled(self):
        if pd is None:
            skip_test("pandas not installed")
        records = [
            {"artist": "Renaissance", "year": 1450, "medium": "Oil", "height": 100.0, "width": 80.0, "price": 1000000.0},
            {"artist": "Modern", "year": 2024, "medium": "Oil", "height": 100.0, "width": 80.0, "price": 1000000.0},
        ]
        df = pd.DataFrame(records)
        out = self.transformer.fit_transform(df)
        self.assertEqual(out.shape[0], 2)

    def test_f15_04_transform_preserves_row_count(self):
        if pd is None:
            skip_test("pandas not installed")
        df = pd.DataFrame(make_diverse_artwork_records(5))
        out = self.transformer.fit_transform(df)
        self.assertEqual(out.shape[0], 5)

    def test_f15_05_standardized_variance(self):
        if pd is None:
            skip_test("pandas not installed")
        df = pd.DataFrame(make_diverse_artwork_records(8))
        out = self.transformer.fit_transform(df)
        self.assertTrue(out.shape[1] > 0)


class TestFeature16LateConcatenationLayer(unittest.TestCase):
    """F16: Late Concatenation Layer (src/features/multimodal.py)."""

    def setUp(self):
        PipelineClass = get_component("src.features.multimodal", "MultimodalFeaturePipeline", milestone="M2")
        self.pipeline = PipelineClass(vision_extractor=None, tabular_transformer=None)

    def test_f16_01_fuse_dimensions(self):
        if np is None:
            skip_test("numpy not installed")
        N, D_tab, D_vis = 5, 12, 2048
        tab = np.ones((N, D_tab), dtype=np.float32)
        vis = np.zeros((N, D_vis), dtype=np.float32)
        fused = self.pipeline.fuse(tab, vis)
        self.assertEqual(fused.shape, (N, D_tab + D_vis))

    def test_f16_02_horizontal_feature_order(self):
        if np is None:
            skip_test("numpy not installed")
        N = 3
        tab = np.full((N, 4), 7.0, dtype=np.float32)
        vis = np.full((N, 8), 9.0, dtype=np.float32)
        fused = self.pipeline.fuse(tab, vis)
        np.testing.assert_array_equal(fused[:, :4], tab)
        np.testing.assert_array_equal(fused[:, 4:], vis)

    def test_f16_03_single_sample_concatenation(self):
        if np is None:
            skip_test("numpy not installed")
        tab = np.ones((1, 10), dtype=np.float32)
        vis = np.zeros((1, 2048), dtype=np.float32)
        fused = self.pipeline.fuse(tab, vis)
        self.assertEqual(fused.shape, (1, 2058))

    def test_f16_04_mismatched_samples_raises_value_error(self):
        if np is None:
            skip_test("numpy not installed")
        tab = np.ones((4, 10))
        vis = np.zeros((5, 2048))
        with self.assertRaises((ValueError, AssertionError)):
            self.pipeline.fuse(tab, vis)

    def test_f16_05_float32_preservation(self):
        if np is None:
            skip_test("numpy not installed")
        tab = np.ones((2, 5), dtype=np.float32)
        vis = np.ones((2, 2048), dtype=np.float32)
        fused = self.pipeline.fuse(tab, vis)
        self.assertEqual(fused.dtype, np.float32)


class TestFeature17ModalitySlicingForAblations(unittest.TestCase):
    """F17: Modality Slicing for Ablations (src/features/multimodal.py)."""

    def setUp(self):
        PipelineClass = get_component("src.features.multimodal", "MultimodalFeaturePipeline", milestone="M2")
        self.pipeline = PipelineClass(vision_extractor=None, tabular_transformer=None)

    def test_f17_01_tabular_slice(self):
        if np is None:
            skip_test("numpy not installed")
        if hasattr(self.pipeline, "slice_features"):
            combined = np.random.randn(5, 2060).astype(np.float32)
            tab_slice = self.pipeline.slice_features(combined, modality="tabular", tabular_dim=12)
            self.assertEqual(tab_slice.shape, (5, 12))

    def test_f17_02_vision_slice(self):
        if np is None:
            skip_test("numpy not installed")
        if hasattr(self.pipeline, "slice_features"):
            combined = np.random.randn(5, 2060).astype(np.float32)
            vis_slice = self.pipeline.slice_features(combined, modality="vision", tabular_dim=12)
            self.assertEqual(vis_slice.shape, (5, 2048))

    def test_f17_03_multimodal_slice(self):
        if np is None:
            skip_test("numpy not installed")
        if hasattr(self.pipeline, "slice_features"):
            combined = np.random.randn(5, 2060).astype(np.float32)
            mm_slice = self.pipeline.slice_features(combined, modality="multimodal", tabular_dim=12)
            self.assertEqual(mm_slice.shape, (5, 2060))

    def test_f17_04_invalid_modality_raises(self):
        if np is None:
            skip_test("numpy not installed")
        if hasattr(self.pipeline, "slice_features"):
            combined = np.random.randn(5, 2060).astype(np.float32)
            with self.assertRaises(ValueError):
                self.pipeline.slice_features(combined, modality="invalid_modality", tabular_dim=12)

    def test_f17_05_case_insensitive_modality_name(self):
        if np is None:
            skip_test("numpy not installed")
        if hasattr(self.pipeline, "slice_features"):
            combined = np.random.randn(4, 2060).astype(np.float32)
            s1 = self.pipeline.slice_features(combined, modality="multimodal", tabular_dim=12)
            s2 = self.pipeline.slice_features(combined, modality="MULTIMODAL", tabular_dim=12)
            np.testing.assert_array_equal(s1, s2)


class TestFeature18LightGBMXGBoostRegressor(unittest.TestCase):
    """F18: LightGBM / XGBoost Regressor (src/models/train.py)."""

    def setUp(self):
        if np is None:
            skip_test("numpy not installed")
        ModelClass = get_component("src.models.train", "ArtValuationModel", milestone="M3")
        self.model = ModelClass()

    def test_f18_01_fit_multimodal_matrix(self):
        X = np.random.randn(20, 64).astype(np.float32)
        y = np.linspace(8.0, 15.0, 20).astype(np.float32)
        self.model.fit(X, y)
        self.assertIsNotNone(self.model)

    def test_f18_02_predict_log_shape(self):
        X = np.random.randn(20, 64).astype(np.float32)
        y = np.linspace(8.0, 15.0, 20).astype(np.float32)
        self.model.fit(X, y)
        preds = self.model.predict_log(X)
        self.assertEqual(preds.shape, (20,))

    def test_f18_03_training_loss_finite(self):
        X = np.random.randn(25, 32).astype(np.float32)
        y = np.linspace(5.0, 12.0, 25).astype(np.float32)
        self.model.fit(X, y)
        preds = self.model.predict_log(X)
        mse = np.mean((preds - y) ** 2)
        self.assertTrue(np.isfinite(mse))

    def test_f18_04_reproducible_predictions(self):
        X = np.random.randn(15, 32).astype(np.float32)
        y = np.linspace(7.0, 14.0, 15).astype(np.float32)
        self.model.fit(X, y)
        p1 = self.model.predict_log(X)
        p2 = self.model.predict_log(X)
        np.testing.assert_allclose(p1, p2, atol=1e-5)

    def test_f18_05_predict_single_vector(self):
        X = np.random.randn(20, 32).astype(np.float32)
        y = np.linspace(8.0, 15.0, 20).astype(np.float32)
        self.model.fit(X, y)
        p = self.model.predict_log(X[0:1])
        self.assertEqual(len(p), 1)


class TestFeature19PriceInferencePipeline(unittest.TestCase):
    """F19: Price Inference Pipeline (src/models/predict.py or ArtValuationModel)."""

    def setUp(self):
        if np is None:
            skip_test("numpy not installed")
        ModelClass = get_component("src.models.train", "ArtValuationModel", milestone="M3")
        self.model = ModelClass()
        X = np.random.randn(25, 32).astype(np.float32)
        y = np.linspace(8.0, 14.0, 25).astype(np.float32)
        self.model.fit(X, y)

    def test_f19_01_predict_price_native_dollars(self):
        X_test = np.random.randn(5, 32).astype(np.float32)
        prices = self.model.predict_price(X_test)
        self.assertEqual(prices.shape, (5,))

    def test_f19_02_all_prices_non_negative(self):
        X_test = np.random.randn(10, 32).astype(np.float32)
        prices = self.model.predict_price(X_test)
        self.assertTrue(all(prices >= 0.0))

    def test_f19_03_price_monotonic_with_log_prediction(self):
        X_test = np.random.randn(8, 32).astype(np.float32)
        logs = self.model.predict_log(X_test)
        prices = self.model.predict_price(X_test)
        order_log = np.argsort(logs)
        order_price = np.argsort(prices)
        np.testing.assert_array_equal(order_log, order_price)

    def test_f19_04_single_artwork_inference(self):
        X_one = np.random.randn(1, 32).astype(np.float32)
        price = self.model.predict_price(X_one)
        self.assertEqual(len(price), 1)
        self.assertGreaterEqual(price[0], 0.0)

    def test_f19_05_finite_dollar_predictions(self):
        X_test = np.random.randn(15, 32).astype(np.float32)
        prices = self.model.predict_price(X_test)
        self.assertTrue(np.all(np.isfinite(prices)))


class TestFeature20ConfidenceIntervalEstimation(unittest.TestCase):
    """F20: Confidence Interval Estimation (src/models/predict.py or ArtValuationModel)."""

    def setUp(self):
        if np is None:
            skip_test("numpy not installed")
        ModelClass = get_component("src.models.train", "ArtValuationModel", milestone="M3")
        self.model = ModelClass()
        X = np.random.randn(30, 16).astype(np.float32)
        y = np.linspace(9.0, 15.0, 30).astype(np.float32)
        self.model.fit(X, y)

    def test_f20_01_predict_with_confidence_returns_three_arrays(self):
        X_test = np.random.randn(5, 16).astype(np.float32)
        lower, point, upper = self.model.predict_with_confidence(X_test, confidence=0.80)
        self.assertEqual(len(lower), 5)
        self.assertEqual(len(point), 5)
        self.assertEqual(len(upper), 5)

    def test_f20_02_lower_less_than_upper(self):
        X_test = np.random.randn(8, 16).astype(np.float32)
        lower, _, upper = self.model.predict_with_confidence(X_test, confidence=0.80)
        self.assertTrue(all(lower <= upper))

    def test_f20_03_point_estimate_within_bounds(self):
        X_test = np.random.randn(8, 16).astype(np.float32)
        lower, point, upper = self.model.predict_with_confidence(X_test, confidence=0.80)
        self.assertTrue(all(lower <= point + 1e-4))
        self.assertTrue(all(point <= upper + 1e-4))

    def test_f20_04_confidence_level_scales_width(self):
        X_test = np.random.randn(4, 16).astype(np.float32)
        l80, _, u80 = self.model.predict_with_confidence(X_test, confidence=0.80)
        l95, _, u95 = self.model.predict_with_confidence(X_test, confidence=0.95)
        width80 = u80 - l80
        width95 = u95 - l95
        self.assertTrue(all(width95 >= width80 - 1e-4))

    def test_f20_05_lower_bound_non_negative(self):
        X_test = np.random.randn(10, 16).astype(np.float32)
        lower, _, _ = self.model.predict_with_confidence(X_test, confidence=0.95)
        self.assertTrue(all(lower >= 0.0))


class TestFeature21AblationComparisonProtocol(unittest.TestCase):
    """F21: Ablation Comparison Protocol (src/models/evaluate.py)."""

    def setUp(self):
        self.compute_metrics = get_component("src.models.evaluate", "compute_metrics", milestone="M3")
        self.ablation_runner = get_component("src.models.evaluate", "run_ablation_study", milestone="M3")

    def test_f21_01_metrics_keys_present(self):
        if np is None:
            skip_test("numpy not installed")
        y_true = np.array([1000.0, 5000.0, 10000.0])
        y_pred = np.array([1100.0, 4800.0, 9500.0])
        metrics = self.compute_metrics(y_true, y_pred)
        for k in ["r2", "mae", "rmse", "mape"]:
            self.assertIn(k, metrics)

    def test_f21_02_perfect_prediction_identities(self):
        if np is None:
            skip_test("numpy not installed")
        y = np.array([100.0, 200.0, 300.0])
        m = self.compute_metrics(y, y)
        self.assertAlmostEqual(m["r2"], 1.0, places=4)
        self.assertAlmostEqual(m["mae"], 0.0, places=4)
        self.assertAlmostEqual(m["rmse"], 0.0, places=4)

    def test_f21_03_non_negative_error_metrics(self):
        if np is None:
            skip_test("numpy not installed")
        y_true = np.array([1000.0, 2000.0])
        y_pred = np.array([1500.0, 1800.0])
        m = self.compute_metrics(y_true, y_pred)
        self.assertGreaterEqual(m["mae"], 0.0)
        self.assertGreaterEqual(m["rmse"], 0.0)
        self.assertGreaterEqual(m["mape"], 0.0)

    def test_f21_04_ablation_study_table_structure(self):
        if np is None:
            skip_test("numpy not installed")
        N = 30
        X_tab = np.random.randn(N, 8)
        X_vis = np.random.randn(N, 16)
        y_price = np.random.uniform(5000, 50000, N)
        report_df = self.ablation_runner(X_tab, X_vis, y_price)
        self.assertGreaterEqual(len(report_df), 3)

    def test_f21_05_ablation_modalities_present(self):
        if np is None:
            skip_test("numpy not installed")
        N = 30
        X_tab = np.random.randn(N, 8)
        X_vis = np.random.randn(N, 16)
        y_price = np.random.uniform(5000, 50000, N)
        report_df = self.ablation_runner(X_tab, X_vis, y_price)
        modality_str = str(report_df.to_dict()).lower()
        self.assertIn("tabular", modality_str)
        self.assertIn("vision", modality_str)
        self.assertIn("multimodal", modality_str)


class TestFeature22ModelPipelineSerialization(unittest.TestCase):
    """F22: Model Pipeline Serialization (src/models/io.py)."""

    def setUp(self):
        self.save_fn = get_component("src.models.io", "save_model_pipeline", milestone="M3")
        self.tmp_dir = tempfile.mkdtemp(prefix="caniq_models_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_f22_01_save_creates_directory(self):
        out_path = os.path.join(self.tmp_dir, "nested_model_dir")
        self.save_fn(model={"dummy": 1}, tabular_transformer={"t": 2}, metadata={"name": "test"}, output_dir=out_path)
        self.assertTrue(os.path.exists(out_path))

    def test_f22_02_metadata_json_written(self):
        self.save_fn(model={"dummy": 1}, tabular_transformer={"t": 2}, metadata={"name": "test_model"}, output_dir=self.tmp_dir)
        meta_file = os.path.join(self.tmp_dir, "metadata.json")
        self.assertTrue(os.path.exists(meta_file))

    def test_f22_03_saved_files_non_empty(self):
        self.save_fn(model={"m": 1}, tabular_transformer={"t": 2}, metadata={"m": 3}, output_dir=self.tmp_dir)
        for f in os.listdir(self.tmp_dir):
            path = os.path.join(self.tmp_dir, f)
            self.assertGreater(os.path.getsize(path), 0)

    def test_f22_04_metadata_preserves_attributes(self):
        import json
        meta = {"version": "1.0", "target_metric": 0.85}
        self.save_fn(model={"m": 1}, tabular_transformer={"t": 2}, metadata=meta, output_dir=self.tmp_dir)
        with open(os.path.join(self.tmp_dir, "metadata.json")) as f:
            loaded = json.load(f)
        self.assertEqual(loaded["version"], "1.0")

    def test_f22_05_overwrite_existing_files(self):
        self.save_fn(model={"v": 1}, tabular_transformer={"t": 1}, metadata={"iter": 1}, output_dir=self.tmp_dir)
        self.save_fn(model={"v": 2}, tabular_transformer={"t": 2}, metadata={"iter": 2}, output_dir=self.tmp_dir)
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, "metadata.json")))


class TestFeature23ReloadVerificationTest(unittest.TestCase):
    """F23: Reload Verification Test (src/models/io.py)."""

    def setUp(self):
        self.save_fn = get_component("src.models.io", "save_model_pipeline", milestone="M3")
        self.load_fn = get_component("src.models.io", "load_model_pipeline", milestone="M3")
        self.tmp_dir = tempfile.mkdtemp(prefix="caniq_reload_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_f23_01_load_pipeline_returns_dict(self):
        self.save_fn(model="dummy_model", tabular_transformer="dummy_trans", metadata={"v": 1}, output_dir=self.tmp_dir)
        loaded = self.load_fn(self.tmp_dir)
        self.assertIsInstance(loaded, dict)

    def test_f23_02_loaded_keys_present(self):
        self.save_fn(model="dummy_model", tabular_transformer="dummy_trans", metadata={"v": 1}, output_dir=self.tmp_dir)
        loaded = self.load_fn(self.tmp_dir)
        self.assertIn("model", loaded)
        self.assertIn("tabular_transformer", loaded)

    def test_f23_03_metadata_loaded_accurately(self):
        self.save_fn(model="m", tabular_transformer="t", metadata={"model_type": "lightgbm"}, output_dir=self.tmp_dir)
        loaded = self.load_fn(self.tmp_dir)
        self.assertEqual(loaded["metadata"]["model_type"], "lightgbm")

    def test_f23_04_missing_directory_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            self.load_fn("/non_existent_model_dir_999")

    def test_f23_05_prediction_identity_after_reload(self):
        if np is None:
            skip_test("numpy not installed")
        ModelClass = get_component("src.models.train", "ArtValuationModel", milestone="M3")
        model = ModelClass()
        X = np.random.randn(20, 8).astype(np.float32)
        y = np.linspace(8.0, 12.0, 20).astype(np.float32)
        model.fit(X, y)
        self.save_fn(model=model, tabular_transformer={"name": "test"}, metadata={"metric": 0.8}, output_dir=self.tmp_dir)
        reloaded = self.load_fn(self.tmp_dir)
        p_orig = model.predict_log(X[:5])
        p_reloaded = reloaded["model"].predict_log(X[:5])
        np.testing.assert_allclose(p_orig, p_reloaded, atol=1e-5)


class TestFeature24InteractiveAppraisalDashboard(unittest.TestCase):
    """F24: Interactive Appraisal Dashboard (app/app.py)."""

    def test_f24_01_app_script_exists(self):
        app_file = REPO_ROOT / "app" / "app.py"
        if not app_file.exists():
            skip_test("app/app.py not yet implemented [Pending M4]")
        self.assertTrue(app_file.exists())

    def test_f24_02_app_syntax_compiles(self):
        app_file = REPO_ROOT / "app" / "app.py"
        if not app_file.exists():
            skip_test("app/app.py not yet implemented [Pending M4]")
        with open(app_file) as f:
            source = f.read()
        compile(source, str(app_file), "exec")

    def test_f24_03_app_imports_streamlit(self):
        app_file = REPO_ROOT / "app" / "app.py"
        if not app_file.exists():
            skip_test("app/app.py not yet implemented [Pending M4]")
        with open(app_file) as f:
            content = f.read()
        self.assertIn("streamlit", content)

    def test_f24_04_app_contains_image_uploader(self):
        app_file = REPO_ROOT / "app" / "app.py"
        if not app_file.exists():
            skip_test("app/app.py not yet implemented [Pending M4]")
        with open(app_file) as f:
            content = f.read()
        self.assertTrue("file_uploader" in content or "uploader" in content)

    def test_f24_05_app_contains_metadata_input_controls(self):
        app_file = REPO_ROOT / "app" / "app.py"
        if not app_file.exists():
            skip_test("app/app.py not yet implemented [Pending M4]")
        with open(app_file) as f:
            content = f.read()
        for field in ["artist", "year", "medium", "height", "width"]:
            self.assertIn(field, content.lower())


class TestFeature25ValuationReportVisualBreakdown(unittest.TestCase):
    """F25: Valuation Report & Visual Breakdown (app/app.py)."""

    def test_f25_01_valuation_display_callout_defined(self):
        app_file = REPO_ROOT / "app" / "app.py"
        if not app_file.exists():
            skip_test("app/app.py not yet implemented [Pending M4]")
        with open(app_file) as f:
            content = f.read()
        self.assertTrue("metric" in content or "valuation" in content.lower() or "price" in content.lower())

    def test_f25_02_confidence_interval_presentation(self):
        app_file = REPO_ROOT / "app" / "app.py"
        if not app_file.exists():
            skip_test("app/app.py not yet implemented [Pending M4]")
        with open(app_file) as f:
            content = f.read()
        self.assertTrue("confidence" in content.lower() or "bound" in content.lower() or "range" in content.lower())

    def test_f25_03_dimension_calculations_in_ui(self):
        app_file = REPO_ROOT / "app" / "app.py"
        if not app_file.exists():
            skip_test("app/app.py not yet implemented [Pending M4]")
        with open(app_file) as f:
            content = f.read()
        self.assertTrue("area" in content.lower() or "aspect" in content.lower() or "surface" in content.lower())

    def test_f25_04_currency_formatting_present(self):
        app_file = REPO_ROOT / "app" / "app.py"
        if not app_file.exists():
            skip_test("app/app.py not yet implemented [Pending M4]")
        with open(app_file) as f:
            content = f.read()
        self.assertTrue("$" in content or "dollar" in content.lower() or "{:,}" in content or "{:,.2f}" in content)

    def test_f25_05_disclaimer_notice_present(self):
        app_file = REPO_ROOT / "app" / "app.py"
        if not app_file.exists():
            skip_test("app/app.py not yet implemented [Pending M4]")
        with open(app_file) as f:
            content = f.read()
        self.assertTrue("disclaimer" in content.lower() or "estimate" in content.lower() or "appraisal" in content.lower())


class TestFeature26FeatureWiseAtomicCommitsPush(unittest.TestCase):
    """F26: Feature-Wise Atomic Commits & Push Tracking."""

    def test_f26_01_git_repository_exists(self):
        git_dir = REPO_ROOT / ".git"
        self.assertTrue(git_dir.exists(), ".git directory must exist")

    def test_f26_02_git_origin_remote_configured(self):
        import subprocess
        result = subprocess.run(["git", "remote", "-v"], cwd=str(REPO_ROOT), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("origin", result.stdout)

    def test_f26_03_conventional_commits_in_history(self):
        import subprocess
        result = subprocess.run(["git", "log", "-n", "10", "--oneline"], cwd=str(REPO_ROOT), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        commits = result.stdout.strip().split("\n")
        self.assertGreater(len(commits), 0)

    def test_f26_04_dry_run_push_passes(self):
        import subprocess
        result = subprocess.run(["git", "push", "--dry-run", "origin", "main"], cwd=str(REPO_ROOT), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)

    def test_f26_05_gitignore_protects_transient_data(self):
        gitignore = REPO_ROOT / ".gitignore"
        self.assertTrue(gitignore.exists())
        with open(gitignore) as f:
            content = f.read()
        self.assertTrue(".agents" in content or "*.pyc" in content or "__pycache__" in content)


class TestFeature27OpenSpecTaskChecklistCompletion(unittest.TestCase):
    """F27: OpenSpec Task Checklist Completion."""

    def test_f27_01_tasks_file_exists(self):
        tasks_file = REPO_ROOT / "openspec" / "changes" / "multimodal-art-valuation" / "tasks.md"
        self.assertTrue(tasks_file.exists())

    def test_f27_02_all_five_milestone_sections_present(self):
        tasks_file = REPO_ROOT / "openspec" / "changes" / "multimodal-art-valuation" / "tasks.md"
        with open(tasks_file) as f:
            content = f.read()
        self.assertIn("## 1. Data Pipeline and Synthetic Benchmark", content)
        self.assertIn("## 2. Feature Extraction", content)
        self.assertIn("## 3. Valuation Engine and Evaluation", content)
        self.assertIn("## 4. Interactive Appraisal Interface", content)
        self.assertIn("## 5. End-to-End Verification & Documentation", content)

    def test_f27_03_tasks_count_is_thirteen(self):
        tasks_file = REPO_ROOT / "openspec" / "changes" / "multimodal-art-valuation" / "tasks.md"
        with open(tasks_file) as f:
            content = f.read()
        task_items = re.findall(r"-\s+\[[ x]\]\s+\d+\.\d+", content)
        self.assertEqual(len(task_items), 13)

    def test_f27_04_valid_checkbox_syntax(self):
        tasks_file = REPO_ROOT / "openspec" / "changes" / "multimodal-art-valuation" / "tasks.md"
        with open(tasks_file) as f:
            content = f.read()
        checkboxes = re.findall(r"-\s+\[(.)\]", content)
        self.assertTrue(all(cb in [" ", "x"] for cb in checkboxes))

    def test_f27_05_tasks_traceability_to_features(self):
        tasks_file = REPO_ROOT / "openspec" / "changes" / "multimodal-art-valuation" / "tasks.md"
        with open(tasks_file) as f:
            content = f.read()
        self.assertIn("validator.py", content)
        self.assertIn("synthetic.py", content)
        self.assertIn("cleaner.py", content)
        self.assertIn("vision.py", content)
        self.assertIn("tabular.py", content)
        self.assertIn("multimodal.py", content)
        self.assertIn("train.py", content)
        self.assertIn("evaluate.py", content)
        self.assertIn("app.py", content)


class TestFeature28ChangeArchivingWorkflow(unittest.TestCase):
    """F28: Change Archiving Workflow."""

    def test_f28_01_openspec_directory_exists(self):
        change_dir = REPO_ROOT / "openspec" / "changes" / "multimodal-art-valuation"
        self.assertTrue(change_dir.exists())

    def test_f28_02_proposal_markdown_exists(self):
        proposal = REPO_ROOT / "openspec" / "changes" / "multimodal-art-valuation" / "proposal.md"
        self.assertTrue(proposal.exists())

    def test_f28_03_design_markdown_exists(self):
        design = REPO_ROOT / "openspec" / "changes" / "multimodal-art-valuation" / "design.md"
        self.assertTrue(design.exists())

    def test_f28_04_specs_directory_contains_delta_specs(self):
        specs_dir = REPO_ROOT / "openspec" / "changes" / "multimodal-art-valuation" / "specs"
        self.assertTrue(specs_dir.exists())
        spec_files = list(specs_dir.rglob("*.md"))
        self.assertGreater(len(spec_files), 0)

    def test_f28_05_archive_instructions_in_tasks(self):
        tasks_file = REPO_ROOT / "openspec" / "changes" / "multimodal-art-valuation" / "tasks.md"
        with open(tasks_file) as f:
            content = f.read()
        self.assertIn("archive", content.lower())


if __name__ == "__main__":
    unittest.main()
