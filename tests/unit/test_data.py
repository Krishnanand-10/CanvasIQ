"""Comprehensive unit tests for CanvasIQ data pipeline (Milestone 1).

Tests:
1. src/data/validator.py (Dataset Schema Validation & Exception Hierarchy)
2. src/data/synthetic.py (Procedural Synthetic Art Benchmark Generator)
3. src/data/cleaner.py (Data Cleaning, Feature Engineering & Transformations)
4. Integration flow (Synthetic -> Validator -> Cleaner)
"""

import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src.data.cleaner import (
    ArtDataCleaner,
    calculate_age,
    calculate_aspect_ratio,
    calculate_surface_area,
    inverse_log_price,
    transform_log_price,
)
from src.data.synthetic import SyntheticArtDatasetGenerator
from src.data.validator import (
    CanvasIQError,
    DataValidationError,
    InvalidImageError,
    InvalidRangeError,
    MissingColumnError,
    SchemaError,
    validate_artwork_record,
    validate_artwork_schema,
    validate_image_file,
)


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture
def valid_artwork_dict():
    """A valid artwork dictionary conforming to training schema."""
    return {
        "artist": "Claude Monet",
        "year": 1899,
        "medium": "Oil on Canvas",
        "height": 65.0,
        "width": 81.0,
        "price": 25000000.0,
        "image_path": "images/synthetic/art_00001.jpg",
    }


@pytest.fixture
def valid_artwork_df():
    """A valid DataFrame with multiple artwork records."""
    return pd.DataFrame([
        {
            "artist": "Vincent van Gogh",
            "year": 1889,
            "medium": "Oil on Canvas",
            "height": 73.7,
            "width": 92.1,
            "price": 82500000.0,
            "image_path": "images/synthetic/art_00001.jpg",
        },
        {
            "artist": "Georgia O'Keeffe",
            "year": 1932,
            "medium": "Oil on Canvas",
            "height": 91.4,
            "width": 121.9,
            "price": 44400000.0,
            "image_path": "images/synthetic/art_00002.jpg",
        },
        {
            "artist": "Mark Rothko",
            "year": 1958,
            "medium": "Acrylic on Canvas",
            "height": 233.0,
            "width": 175.0,
            "price": 86800000.0,
            "image_path": "images/synthetic/art_00003.jpg",
        },
    ])


@pytest.fixture
def dirty_artwork_df():
    """A DataFrame containing missing values, zero dimensions, and negative prices."""
    return pd.DataFrame([
        # Row 0: Valid record
        {
            "artist": "Pablo Picasso",
            "year": 1907,
            "medium": "Oil on Canvas",
            "height": 243.9,
            "width": 233.7,
            "price": 179365000.0,
        },
        # Row 1: Zero height (invalid dimension)
        {
            "artist": "Henri Matisse",
            "year": 1905,
            "medium": "Oil on Canvas",
            "height": 0.0,
            "width": 60.0,
            "price": 12000000.0,
        },
        # Row 2: Negative width (invalid dimension)
        {
            "artist": "Joan Miro",
            "year": 1925,
            "medium": "Oil on Canvas",
            "height": 80.0,
            "width": -50.0,
            "price": 15000000.0,
        },
        # Row 3: Negative price (invalid training target)
        {
            "artist": "Salvador Dali",
            "year": 1931,
            "medium": "Oil on Canvas",
            "height": 24.1,
            "width": 33.0,
            "price": -500.0,
        },
        # Row 4: Missing medium (should be imputed with 'Unknown')
        {
            "artist": "Jackson Pollock",
            "year": 1948,
            "medium": None,
            "height": 172.7,
            "width": 264.2,
            "price": 140000000.0,
        },
        # Row 5: Missing price (NaN)
        {
            "artist": "Andy Warhol",
            "year": 1962,
            "medium": "Silkscreen ink",
            "height": 208.0,
            "width": 145.0,
            "price": np.nan,
        },
    ])


@pytest.fixture
def synthetic_generator():
    """A seeded synthetic art dataset generator."""
    return SyntheticArtDatasetGenerator(seed=42)


# ============================================================================
# 1. Exception Hierarchy Tests
# ============================================================================

class TestExceptionHierarchy:
    """Test custom exception relationships and standard exception inheritance."""

    def test_schema_error_inherits_value_error(self):
        assert issubclass(SchemaError, ValueError)
        assert issubclass(SchemaError, CanvasIQError)

    def test_data_validation_error_hierarchy(self):
        assert issubclass(DataValidationError, SchemaError)
        assert issubclass(DataValidationError, ValueError)

    def test_missing_column_error_hierarchy(self):
        assert issubclass(MissingColumnError, DataValidationError)
        assert issubclass(MissingColumnError, KeyError)

    def test_invalid_range_error_hierarchy(self):
        assert issubclass(InvalidRangeError, DataValidationError)
        assert issubclass(InvalidRangeError, ValueError)

    def test_invalid_image_error_hierarchy(self):
        assert issubclass(InvalidImageError, CanvasIQError)
        assert issubclass(InvalidImageError, ValueError)


# ============================================================================
# 2. Dataset Schema Validator Tests
# ============================================================================

class TestArtworkSchemaValidator:
    """Unit tests for src/data/validator.py."""

    def test_validate_valid_dict_training(self, valid_artwork_dict):
        """Verify that a valid single dictionary passes training validation."""
        result = validate_artwork_schema(valid_artwork_dict, is_training=True)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.loc[0, "artist"] == "Claude Monet"
        assert result.loc[0, "price"] == 25000000.0

    def test_validate_valid_dataframe_training(self, valid_artwork_df):
        """Verify that a valid DataFrame passes training validation."""
        result = validate_artwork_schema(valid_artwork_df, is_training=True)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_validate_inference_without_price(self, valid_artwork_dict):
        """Verify that price is optional when is_training=False."""
        dict_no_price = valid_artwork_dict.copy()
        del dict_no_price["price"]
        result = validate_artwork_schema(dict_no_price, is_training=False)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert "price" not in result.columns or pd.isna(result.loc[0, "price"])

    def test_validate_inference_with_optional_price(self, valid_artwork_dict):
        """Verify that price is preserved if present in inference mode."""
        result = validate_artwork_schema(valid_artwork_dict, is_training=False)
        assert isinstance(result, pd.DataFrame)
        assert result.loc[0, "price"] == 25000000.0

    def test_numeric_string_coercion(self, valid_artwork_dict):
        """Verify that numeric strings in year, height, width, and price are coerced."""
        data = valid_artwork_dict.copy()
        data["year"] = "1920"
        data["height"] = "50.5"
        data["width"] = "75.0"
        data["price"] = "100000.0"
        df = validate_artwork_schema(data, is_training=True)
        assert df["year"].iloc[0] == 1920
        assert isinstance(df["year"].iloc[0], (int, np.integer))
        assert df["height"].iloc[0] == 50.5
        assert df["width"].iloc[0] == 75.0
        assert df["price"].iloc[0] == 100000.0

    def test_column_alias_normalization(self):
        """Verify automatic renaming of common column aliases."""
        aliased_dict = {
            "artist": "Pablo Picasso",
            "creation_year": 1907,
            "medium": "Oil on Canvas",
            "height_cm": 243.9,
            "width_cm": 233.7,
            "hammer_price": 179365000.0,
        }
        df = validate_artwork_schema(aliased_dict, is_training=True)
        assert "year" in df.columns
        assert "height" in df.columns
        assert "width" in df.columns
        assert "price" in df.columns
        assert df["year"].iloc[0] == 1907
        assert df["height"].iloc[0] == pytest.approx(243.9)

    def test_extra_columns_allowed_by_default(self, valid_artwork_dict):
        """Verify that extra non-mandatory columns are preserved by default."""
        data = valid_artwork_dict.copy()
        data["genre"] = "Post-Impressionism"
        data["auction_house"] = "Christie's"
        df = validate_artwork_schema(data, is_training=True, allow_extra_columns=True)
        assert "genre" in df.columns
        assert df["genre"].iloc[0] == "Post-Impressionism"

    def test_extra_columns_rejected_when_strict(self, valid_artwork_dict):
        """Verify that extra columns raise DataValidationError when allow_extra_columns=False."""
        data = valid_artwork_dict.copy()
        data["unrecognized_field"] = 12345
        with pytest.raises(DataValidationError, match="Unrecognized columns detected"):
            validate_artwork_schema(data, is_training=True, allow_extra_columns=False)

    def test_validate_artwork_record_helper(self, valid_artwork_dict):
        """Verify validate_artwork_record returns a clean dictionary."""
        record = validate_artwork_record(valid_artwork_dict, is_training=True)
        assert isinstance(record, dict)
        assert record["artist"] == "Claude Monet"
        assert record["year"] == 1899

    def test_none_input_raises_validation_error(self):
        with pytest.raises(DataValidationError, match="cannot be None"):
            validate_artwork_schema(None)

    def test_empty_dict_raises_validation_error(self):
        with pytest.raises(DataValidationError, match="cannot be empty"):
            validate_artwork_schema({})

    def test_empty_dataframe_raises_validation_error(self):
        with pytest.raises(DataValidationError, match="cannot be empty"):
            validate_artwork_schema(pd.DataFrame())

    @pytest.mark.parametrize("missing_col", ["artist", "year", "medium", "height", "width"])
    def test_missing_mandatory_columns_raises_error(self, valid_artwork_dict, missing_col):
        invalid_dict = valid_artwork_dict.copy()
        del invalid_dict[missing_col]
        with pytest.raises(MissingColumnError) as exc_info:
            validate_artwork_schema(invalid_dict, is_training=True)
        assert missing_col in exc_info.value.missing_columns

    def test_missing_price_in_training_raises_error(self, valid_artwork_dict):
        invalid_dict = valid_artwork_dict.copy()
        del invalid_dict["price"]
        with pytest.raises(MissingColumnError) as exc_info:
            validate_artwork_schema(invalid_dict, is_training=True)
        assert "price" in exc_info.value.missing_columns

    def test_nan_in_mandatory_column_raises(self, valid_artwork_dict):
        data = valid_artwork_dict.copy()
        data["height"] = np.nan
        with pytest.raises(DataValidationError, match="contains .* missing/NaN"):
            validate_artwork_schema(data, is_training=True)

    def test_blank_artist_string_raises(self, valid_artwork_dict):
        data = valid_artwork_dict.copy()
        data["artist"] = "   "
        with pytest.raises(DataValidationError, match="must contain non-empty, non-blank strings"):
            validate_artwork_schema(data, is_training=True)

    def test_non_numeric_dimension_raises(self, valid_artwork_dict):
        data = valid_artwork_dict.copy()
        data["height"] = "sixty-five"
        with pytest.raises(DataValidationError, match="contains non-numeric values"):
            validate_artwork_schema(data, is_training=True)

    def test_zero_or_negative_dimensions_raise_error(self, valid_artwork_dict):
        zero_height = valid_artwork_dict.copy()
        zero_height["height"] = 0.0
        with pytest.raises(InvalidRangeError, match="must be strictly positive"):
            validate_artwork_schema(zero_height, is_training=True)

        negative_width = valid_artwork_dict.copy()
        negative_width["width"] = -10.0
        with pytest.raises(InvalidRangeError, match="must be strictly positive"):
            validate_artwork_schema(negative_width, is_training=True)

    def test_exorbitant_dimension_raises_invalid_range(self, valid_artwork_dict):
        data = valid_artwork_dict.copy()
        data["height"] = 50000.0
        with pytest.raises(InvalidRangeError, match="exceeds maximum allowable dimension"):
            validate_artwork_schema(data, is_training=True)

    def test_negative_or_zero_price_in_training_raises_error(self, valid_artwork_dict):
        zero_price = valid_artwork_dict.copy()
        zero_price["price"] = 0.0
        with pytest.raises(InvalidRangeError, match="must be strictly positive"):
            validate_artwork_schema(zero_price, is_training=True)

        neg_price = valid_artwork_dict.copy()
        neg_price["price"] = -500.0
        with pytest.raises(InvalidRangeError, match="must be strictly positive"):
            validate_artwork_schema(neg_price, is_training=True)

    def test_exorbitant_price_raises_invalid_range(self, valid_artwork_dict):
        data = valid_artwork_dict.copy()
        data["price"] = 5_000_000_000.0
        with pytest.raises(InvalidRangeError, match="exceeds maximum allowable valuation"):
            validate_artwork_schema(data, is_training=True)

    def test_future_creation_year_raises(self, valid_artwork_dict):
        data = valid_artwork_dict.copy()
        data["year"] = 2050
        with pytest.raises(InvalidRangeError, match="outside allowable historical range"):
            validate_artwork_schema(data, is_training=True)

    def test_ancient_creation_year_raises(self, valid_artwork_dict):
        data = valid_artwork_dict.copy()
        data["year"] = 500
        with pytest.raises(InvalidRangeError, match="outside allowable historical range"):
            validate_artwork_schema(data, is_training=True)


# ============================================================================
# 3. Image File Validation Tests
# ============================================================================

class TestValidateImageFile:
    """Test image file path and header verification."""

    def test_valid_image_file_passes(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img = Image.new("RGB", (100, 100), color="red")
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            path_obj = validate_image_file(tmp_path, check_readable=True)
            assert isinstance(path_obj, Path)
            assert path_obj.exists()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_nonexistent_image_raises_invalid_image_error(self):
        with pytest.raises(InvalidImageError, match="does not exist"):
            validate_image_file("nonexistent_directory/missing_artwork.png")

    def test_unsupported_image_extension_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"Not an image")
            tmp_path = tmp.name

        try:
            with pytest.raises(InvalidImageError, match="Unsupported image file extension"):
                validate_image_file(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ============================================================================
# 4. Synthetic Art Dataset Generator Tests
# ============================================================================

class TestSyntheticArtDatasetGenerator:
    """Unit tests for src/data/synthetic.py."""

    def test_initialization(self, synthetic_generator):
        assert synthetic_generator.seed == 42
        assert hasattr(synthetic_generator, "py_rng")
        assert hasattr(synthetic_generator, "np_rng")

    def test_generator_deterministic_reproducibility(self):
        """Verify that two generators with identical seeds produce identical images and data."""
        gen1 = SyntheticArtDatasetGenerator(seed=123)
        gen2 = SyntheticArtDatasetGenerator(seed=123)

        img1 = gen1.generate_image(style="color_field", palette="oceanic_cool")
        img2 = gen2.generate_image(style="color_field", palette="oceanic_cool")

        arr1 = np.array(img1)
        arr2 = np.array(img2)
        assert np.array_equal(arr1, arr2)

    @pytest.mark.parametrize("size", [(224, 224), (128, 128), (64, 64)])
    def test_generate_image_dimensions_and_mode(self, synthetic_generator, size):
        img = synthetic_generator.generate_image(size=size)
        assert isinstance(img, Image.Image)
        assert img.size == size
        assert img.mode == "RGB"

    @pytest.mark.parametrize("style", ["color_field", "abstract_splatter", "geometric", "impressionist"])
    @pytest.mark.parametrize("palette", [
        "earthy_ochre", "oceanic_cool", "monochrome", "neon_vibrant", "pastel_wash"
    ])
    def test_all_styles_and_palettes(self, synthetic_generator, style, palette):
        img = synthetic_generator.generate_image(style=style, palette=palette, size=(64, 64))
        assert isinstance(img, Image.Image)
        assert img.size == (64, 64)
        assert img.mode == "RGB"
        arr = np.array(img)
        assert arr.std() > 0.5, f"Image for {style}/{palette} should have texture"

    def test_generate_image_aliases(self, synthetic_generator):
        img1 = synthetic_generator.generate_image(style="rothko", palette="earthy_ochre", size=(32, 32))
        img2 = synthetic_generator.generate_image(style="pollock", palette="monochrome", size=(32, 32))
        img3 = synthetic_generator.generate_image(style="minimalist", palette="neon_vibrant", size=(32, 32))
        img4 = synthetic_generator.generate_image(style="pointillist", palette="pastel_wash", size=(32, 32))
        assert img1.size == (32, 32)
        assert img2.size == (32, 32)
        assert img3.size == (32, 32)
        assert img4.size == (32, 32)

    def test_generate_image_invalid_inputs(self, synthetic_generator):
        with pytest.raises(ValueError, match="Unknown style"):
            synthetic_generator.generate_image(style="cubist_cyberpunk")
        with pytest.raises(ValueError, match="Unknown palette"):
            synthetic_generator.generate_image(palette="rainbow_glitter")

    def test_generate_benchmark_creates_files(self, synthetic_generator, tmp_path):
        """Verify that generate_benchmark persists image files and CSV metadata."""
        n_samples = 5
        df, csv_path = synthetic_generator.generate_benchmark(n_samples=n_samples, output_dir=tmp_path)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == n_samples
        assert os.path.exists(csv_path)

        for col in ["artist", "year", "medium", "height", "width", "price"]:
            assert col in df.columns

        assert (df["price"] > 0).all()
        assert (df["height"] > 0).all()
        assert (df["width"] > 0).all()
        assert ((df["year"] >= 1850) & (df["year"] <= 2026)).all()

        for p in df["image_path"]:
            full_path = Path(p) if Path(p).is_absolute() else (tmp_path / p)
            assert full_path.exists(), f"Image file {full_path} not found"

    def test_deterministic_benchmark_reproducibility(self, tmp_path):
        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"

        gen1 = SyntheticArtDatasetGenerator(seed=999)
        df1, _ = gen1.generate_benchmark(n_samples=3, output_dir=dir1)

        gen2 = SyntheticArtDatasetGenerator(seed=999)
        df2, _ = gen2.generate_benchmark(n_samples=3, output_dir=dir2)

        pd.testing.assert_frame_equal(
            df1.drop(columns=["image_path"]), df2.drop(columns=["image_path"])
        )
        assert [Path(p).name for p in df1["image_path"]] == [
            Path(p).name for p in df2["image_path"]
        ]

        for i in range(1, 4):
            img_path1 = dir1 / f"images/synthetic/art_{i:05d}.jpg"
            img_path2 = dir2 / f"images/synthetic/art_{i:05d}.jpg"
            assert img_path1.read_bytes() == img_path2.read_bytes()

    def test_hedonic_pricing_tier_monotonicity(self, tmp_path):
        gen = SyntheticArtDatasetGenerator(seed=42)
        df, _ = gen.generate_benchmark(n_samples=60, output_dir=tmp_path)

        tier1_median = df[df["artist_tier"] == "Tier 1 - Master"]["price"].median()
        tier2_median = df[df["artist_tier"] == "Tier 2 - Mid-Career"]["price"].median()
        tier3_median = df[df["artist_tier"] == "Tier 3 - Emerging"]["price"].median()

        assert tier1_median > tier2_median, f"Tier 1 (${tier1_median}) should exceed Tier 2 (${tier2_median})"
        assert tier2_median > tier3_median, f"Tier 2 (${tier2_median}) should exceed Tier 3 (${tier3_median})"


# ============================================================================
# 5. Art Data Cleaner & Feature Transformation Tests
# ============================================================================

class TestArtDataCleaner:
    """Unit tests for src/data/cleaner.py."""

    def test_calculate_surface_area_scalar(self):
        area = calculate_surface_area(50.0, 80.0)
        assert area == 4000.0
        assert ArtDataCleaner.calculate_surface_area(10.0, 10.0) == 100.0

    def test_calculate_surface_area_negative_raises(self):
        with pytest.raises(ValueError, match="cannot be negative"):
            calculate_surface_area(-10.0, 50.0)
        with pytest.raises(ValueError, match="cannot be negative"):
            calculate_surface_area(50.0, -10.0)
        assert calculate_surface_area(0.0, 50.0) == 0.0

    def test_calculate_aspect_ratio_scalar(self):
        assert calculate_aspect_ratio(height=50.0, width=100.0) == 2.0
        assert calculate_aspect_ratio(height=100.0, width=50.0) == 0.5
        assert calculate_aspect_ratio(height=60.0, width=60.0) == 1.0

    def test_calculate_aspect_ratio_zero_height_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            calculate_aspect_ratio(height=0.0, width=50.0)
        with pytest.raises(ValueError, match="strictly positive"):
            calculate_aspect_ratio(height=-20.0, width=50.0)

    def test_transform_log_price_scalar(self):
        assert transform_log_price(0.0) == 0.0
        expected = float(np.log1p(10000.0))
        assert transform_log_price(10000.0) == pytest.approx(expected)

    def test_transform_log_price_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            transform_log_price(-50.0)

    def test_inverse_log_price_scalar(self):
        assert inverse_log_price(0.0) == 0.0
        val = float(np.log1p(50000.0))
        assert inverse_log_price(val) == pytest.approx(50000.0, rel=1e-5)

    def test_inverse_log_price_clamped_at_zero(self):
        assert inverse_log_price(-2.5) == 0.0
        assert inverse_log_price(-100.0) == 0.0

    @pytest.mark.parametrize("price", [1.0, 10.0, 100.0, 5000.0, 75000.0, 1200000.0, 45000000.0])
    def test_log_price_roundtrip_identity(self, price):
        log_val = transform_log_price(price)
        recovered_price = inverse_log_price(log_val)
        assert recovered_price == pytest.approx(price, rel=1e-5)

    def test_log_price_roundtrip_array(self):
        prices = np.array([100.0, 2500.0, 18000.0, 950000.0, 35000000.0])
        log_prices = transform_log_price(prices)
        recovered = inverse_log_price(log_prices)
        assert np.allclose(recovered, prices, rtol=1e-5)

    def test_calculate_age(self):
        assert calculate_age(year=2000, reference_year=2026) == 26
        assert calculate_age(year=1850, reference_year=2026) == 176
        assert calculate_age(year=2030, reference_year=2026) == 0

    def test_clean_training_dataframe_drops_invalids(self, dirty_artwork_df):
        cleaner = ArtDataCleaner(reference_year=2026)
        cleaned = cleaner.clean(dirty_artwork_df, is_training=True)

        assert len(cleaned) == 2
        assert (cleaned["height"] > 0).all()
        assert (cleaned["width"] > 0).all()
        assert (cleaned["price"] > 0).all()
        assert (cleaned["medium"] != "").all()
        assert (cleaned["medium"].notna()).all()
        assert "Unknown" in cleaned["medium"].values

    def test_clean_derived_columns_exist(self, valid_artwork_df):
        cleaner = ArtDataCleaner(reference_year=2026)
        cleaned = cleaner.clean(valid_artwork_df, is_training=True)

        for col in ["surface_area", "aspect_ratio", "age", "log_price"]:
            assert col in cleaned.columns

        row0 = cleaned.iloc[0]
        assert row0["surface_area"] == pytest.approx(73.7 * 92.1)
        assert row0["aspect_ratio"] == pytest.approx(92.1 / 73.7)
        assert row0["age"] == 2026 - 1889
        assert row0["log_price"] == pytest.approx(np.log1p(82500000.0))

    def test_clean_inference_mode_without_price(self, valid_artwork_df):
        unpriced_df = valid_artwork_df.drop(columns=["price"])
        cleaner = ArtDataCleaner(reference_year=2026)
        cleaned = cleaner.clean(unpriced_df, is_training=False)

        assert len(cleaned) == len(unpriced_df)
        assert "surface_area" in cleaned.columns
        assert "aspect_ratio" in cleaned.columns
        assert "age" in cleaned.columns

    def test_clean_returns_copy_not_view(self, valid_artwork_df):
        original_copy = valid_artwork_df.copy()
        cleaner = ArtDataCleaner()
        cleaned = cleaner.clean(valid_artwork_df, is_training=True)

        cleaned.loc[0, "artist"] = "MUTATED"
        assert valid_artwork_df.loc[0, "artist"] == original_copy.loc[0, "artist"]


# ============================================================================
# 6. End-to-End Milestone 1 Data Pipeline Integration Tests
# ============================================================================

class TestDataPipelineIntegration:
    """Integration tests combining Synthetic Generator -> Validator -> Cleaner."""

    def test_synthetic_to_validator_to_cleaner_flow(self, tmp_path):
        """Generate synthetic records, validate schema, clean data, and verify outputs."""
        # Step 1: Synthesize benchmark records
        generator = SyntheticArtDatasetGenerator(seed=999)
        raw_df, _ = generator.generate_benchmark(n_samples=25, output_dir=tmp_path)
        assert len(raw_df) == 25

        # Step 2: Validate Schema
        validated_df = validate_artwork_schema(raw_df, is_training=True)
        assert len(validated_df) == 25

        # Step 3: Clean and Feature Engineer
        cleaner = ArtDataCleaner(reference_year=2026)
        cleaned_df = cleaner.clean(validated_df, is_training=True)
        assert len(cleaned_df) == 25

        # Step 4: Verify complete output integrity
        expected_cols = [
            "artist", "year", "medium", "height", "width", "price",
            "surface_area", "aspect_ratio", "age", "log_price",
        ]
        for col in expected_cols:
            assert col in cleaned_df.columns

        assert cleaned_df[expected_cols].isna().sum().sum() == 0
        assert (cleaned_df["surface_area"] > 0).all()
        assert (cleaned_df["aspect_ratio"] > 0).all()
        assert (cleaned_df["age"] >= 0).all()
        assert (cleaned_df["log_price"] > 0).all()
