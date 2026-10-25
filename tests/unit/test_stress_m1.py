"""Empirical Stress Tests and Challenge Harness for Milestone 1 (CanvasIQ).

Attacks schema validation and cleaner transformations:
1. Numerical precision and roundtrip fidelity ($0.01 to $1,000,000,000, subnormals, infinities).
2. Extreme values, zero dimensions, negative prices, NaN/Inf combinations.
3. String encoding corner cases (emojis, unicode, non-Latin scripts, whitespace variants).
4. High-volume DataFrame throughput (10,000+ rows).
"""

import time
import numpy as np
import pandas as pd
import pytest

from src.data.validator import (
    validate_artwork_schema,
    validate_artwork_record,
    DataValidationError,
    InvalidRangeError,
    MissingColumnError,
    MIN_DIMENSION,
    MAX_DIMENSION,
    MIN_PRICE,
    MAX_PRICE,
    MIN_YEAR,
    MAX_YEAR,
)
from src.data.cleaner import (
    ArtDataCleaner,
    transform_log_price,
    inverse_log_price,
    calculate_surface_area,
    calculate_aspect_ratio,
    calculate_age,
)


class TestNumericalFidelity:
    """Stress tests on mathematical precision and roundtrip fidelity of price transformations."""

    def test_log_price_roundtrip_broad_range(self):
        """Verify roundtrip fidelity inverse(transform(p)) == p across $0.01 to $1,000,000,000."""
        prices = np.logspace(-2, 9, num=10000)
        log_prices = transform_log_price(prices)
        recovered_prices = inverse_log_price(log_prices)

        rel_errors = np.abs(recovered_prices - prices) / prices
        max_rel_error = np.max(rel_errors)

        # 64-bit IEEE-754 precision should guarantee relative error < 1e-12
        assert max_rel_error < 1e-12, f"Max relative error {max_rel_error} exceeded 1e-12 tolerance"

    def test_log_price_roundtrip_scalars(self):
        """Verify scalar roundtrip precision at distinct benchmark points."""
        test_points = [0.01, 0.50, 1.0, 10.0, 1000.0, 1e6, 1e8, 1e9]
        for p in test_points:
            rec = inverse_log_price(transform_log_price(p))
            assert abs(rec - p) / p < 1e-12, f"Failed precision for price {p}: recovered {rec}"

    def test_log_price_monotonicity(self):
        """Strict monotonicity: p1 < p2 must guarantee log(1+p1) < log(1+p2)."""
        prices = np.linspace(0.01, 100000.0, num=5000)
        log_prices = transform_log_price(prices)
        diffs = np.diff(log_prices)
        assert np.all(diffs > 0), "Monotonicity violated in transform_log_price"

    def test_log_price_zero_and_subnormals(self):
        """Verify behavior at exact zero and subnormal floating point values."""
        assert transform_log_price(0.0) == 0.0
        assert inverse_log_price(0.0) == 0.0

        # Subnormal float
        subnormal = 1e-320
        lp = transform_log_price(subnormal)
        rec = inverse_log_price(lp)
        assert rec == subnormal, f"Subnormal {subnormal} was not preserved exactly"

    def test_log_price_extreme_large(self):
        """Verify behavior near IEEE 754 float max (1.79e308)."""
        large_p = 1e308
        lp = transform_log_price(large_p)
        assert np.isfinite(lp)
        rec = inverse_log_price(lp)
        assert abs(rec - large_p) / large_p < 1e-12

    def test_inverse_log_price_clamping_and_nans(self):
        """Verify clamping of negative log-prices to 0.0 and propagation of NaNs."""
        assert inverse_log_price(-10.0) == 0.0
        assert inverse_log_price(-0.01) == 0.0
        assert np.isnan(inverse_log_price(float("nan")))

        arr = np.array([-5.0, -0.1, 0.0, 10.0, np.nan])
        res = inverse_log_price(arr)
        assert np.array_equal(res[:3], [0.0, 0.0, 0.0])
        assert np.isnan(res[4])

    def test_transform_log_price_negative_scalar_raises(self):
        """Scalar negative price must raise ValueError."""
        with pytest.raises(ValueError):
            transform_log_price(-1.0)
        with pytest.raises(ValueError):
            transform_log_price(-1e-15)


class TestExtremeValuesAndInfinities:
    """Stress tests on extreme numerical inputs, infinities, subnormals, and boundary rules."""

    def test_validator_rejects_infinities(self):
        """validate_artwork_schema must reject +inf and -inf in all numerical fields."""
        base = {
            "artist": "Test Artist",
            "year": 2000,
            "medium": "Oil on Canvas",
            "height": 50.0,
            "width": 50.0,
            "price": 1000.0,
        }

        # Year inf
        d = base.copy()
        d["year"] = float("inf")
        with pytest.raises((InvalidRangeError, DataValidationError)):
            validate_artwork_schema(d)

        # Height inf
        d = base.copy()
        d["height"] = float("inf")
        with pytest.raises((InvalidRangeError, DataValidationError)):
            validate_artwork_schema(d)

        # Width inf
        d = base.copy()
        d["width"] = float("inf")
        with pytest.raises((InvalidRangeError, DataValidationError)):
            validate_artwork_schema(d)

        # Price inf
        d = base.copy()
        d["price"] = float("inf")
        with pytest.raises((InvalidRangeError, DataValidationError)):
            validate_artwork_schema(d)

        # Negative infinities
        for field in ["year", "height", "width", "price"]:
            d = base.copy()
            d[field] = float("-inf")
            with pytest.raises((InvalidRangeError, DataValidationError)):
                validate_artwork_schema(d)

    def test_cleaner_crash_on_infinite_year(self):
        """
        VERIFY FIX:
        ArtDataCleaner.clean() drops non-finite years without raising IntCastingNaNError.
        """
        df = pd.DataFrame([{
            "artist": "Test Artist",
            "year": float("inf"),
            "medium": "Oil on Canvas",
            "height": 50.0,
            "width": 50.0,
            "price": 1000.0,
        }])
        cleaner = ArtDataCleaner()
        cleaned = cleaner.clean(df)
        assert len(cleaned) == 0

    def test_cleaner_calculate_age_infinite_scalar_overflow(self):
        """
        VERIFY FIX:
        calculate_age(float('inf')) handles non-finite inputs gracefully and returns 0.
        """
        assert calculate_age(float("inf")) == 0
        assert calculate_age(float("-inf")) == 0
        assert calculate_age(float("nan")) == 0

    def test_min_dimension_constant_not_enforced_in_validator(self):
        """
        VERIFY FIX:
        MIN_DIMENSION (0.001) is strictly enforced in validator.py, raising InvalidRangeError.
        """
        df = pd.DataFrame([{
            "artist": "Test Artist",
            "year": 2000,
            "medium": "Oil",
            "height": 0.0001,  # Below MIN_DIMENSION (0.001)
            "width": 50.0,
            "price": 1000.0,
        }])
        with pytest.raises(InvalidRangeError):
            validate_artwork_schema(df)

    def test_min_price_constant_not_enforced_in_validator(self):
        """
        VERIFY FIX:
        MIN_PRICE (0.001) is strictly enforced in validator.py in training mode, raising InvalidRangeError.
        """
        df = pd.DataFrame([{
            "artist": "Test Artist",
            "year": 2000,
            "medium": "Oil",
            "height": 50.0,
            "width": 50.0,
            "price": 0.00001,  # Below MIN_PRICE (0.001)
        }])
        with pytest.raises(InvalidRangeError):
            validate_artwork_schema(df)

    def test_calculate_surface_area_zero_dimension_discrepancy(self):
        """
        VERIFY FIX:
        calculate_surface_area raises ValueError for non-positive scalar dimensions,
        matching docstrings and calculate_aspect_ratio.
        """
        with pytest.raises(ValueError):
            calculate_surface_area(0.0, 50.0)
        with pytest.raises(ValueError):
            calculate_surface_area(50.0, 0.0)

        with pytest.raises(ValueError):
            calculate_aspect_ratio(0.0, 50.0)

    def test_cleaner_log_price_with_zero_price_in_inference(self):
        """
        VERIFY FIX:
        In inference mode (is_training=False), if price is 0.0,
        ArtDataCleaner computes log_price as log1p(0.0) == 0.0 instead of NaN.
        """
        df = pd.DataFrame([{
            "artist": "Artist",
            "year": 2000,
            "medium": "Oil",
            "height": 50.0,
            "width": 50.0,
            "price": 0.0,
        }])
        cleaner = ArtDataCleaner()
        res = cleaner.clean(df, is_training=False)
        assert res["log_price"].iloc[0] == 0.0


class TestStringEncodingCornerCases:
    """Stress tests on Unicode, emojis, diverse scripts, and whitespace variants."""

    def test_unicode_and_emojis_in_metadata(self):
        """Ensure full Unicode coverage across non-Latin scripts, accents, and emojis."""
        test_records = [
            {"artist": "🎨 Leonardo da Vinci 🖌️", "medium": "🖼️ Oil on Poplar ✨"},
            {"artist": "Пабло Пикассо", "medium": "Масло на холсте"},
            {"artist": "葛飾北斎", "medium": "木版画"},
            {"artist": "غوستاف كليمت", "medium": "زيت على قماش"},
            {"artist": "René François Ghislain Magritte", "medium": "Huile sur toile"},
            {"artist": "Søren Kierkegaard & Edvard Munch", "medium": "Tempera og olje"},
            {"artist": "राजा रवि वर्मा", "medium": "कैनवास पर तैल चित्र"},
        ]

        df = pd.DataFrame([
            {**r, "year": 1900, "height": 60.0, "width": 80.0, "price": 500000.0}
            for r in test_records
        ])

        # Validate schema
        validated_df = validate_artwork_schema(df)
        assert len(validated_df) == len(test_records)
        for i, r in enumerate(test_records):
            assert validated_df["artist"].iloc[i] == r["artist"]
            assert validated_df["medium"].iloc[i] == r["medium"]

        # Clean metadata
        cleaner = ArtDataCleaner()
        cleaned_df = cleaner.clean(validated_df)
        assert len(cleaned_df) == len(test_records)
        assert "surface_area" in cleaned_df.columns

    def test_whitespace_and_empty_strings(self):
        """Check rejection of blank strings, tabs, newlines, and NBSP in validator."""
        blank_cases = ["   ", "\t\t", "\n\r\n", " \t \n ", " \xa0 "]
        for blank in blank_cases:
            d = {
                "artist": blank,
                "year": 2000,
                "medium": "Oil",
                "height": 50.0,
                "width": 50.0,
                "price": 1000.0,
            }
            with pytest.raises(DataValidationError):
                validate_artwork_schema(d)

    def test_null_representation_strings(self):
        """Strings 'nan', 'none', 'null' in artist/medium should be rejected by validator."""
        pseudo_nulls = ["nan", "NaN", "NAN", "none", "None", "null", "NULL"]
        for val in pseudo_nulls:
            d = {
                "artist": val,
                "year": 2000,
                "medium": "Oil",
                "height": 50.0,
                "width": 50.0,
                "price": 1000.0,
            }
            with pytest.raises(DataValidationError):
                validate_artwork_schema(d)


class TestHighVolumeThroughput:
    """Throughput benchmarks on high-volume DataFrames."""

    def test_validator_and_cleaner_throughput_10k_rows(self):
        """Benchmark validation and cleaning on 10,000 rows."""
        n_rows = 10000
        np.random.seed(123)

        df = pd.DataFrame({
            "artist": np.random.choice(["Picasso", "Monet", "Warhol", "Rothko", "Klimt"], size=n_rows),
            "year": np.random.randint(1880, 2025, size=n_rows),
            "medium": np.random.choice(["Oil on Canvas", "Acrylic", "Lithograph", "Watercolor"], size=n_rows),
            "height": np.random.uniform(20.0, 250.0, size=n_rows),
            "width": np.random.uniform(20.0, 250.0, size=n_rows),
            "price": np.random.lognormal(mean=11.0, sigma=1.2, size=n_rows),
        })

        # Measure validator latency
        t0 = time.perf_counter()
        val_df = validate_artwork_schema(df)
        val_time = time.perf_counter() - t0
        val_rate = n_rows / val_time

        assert len(val_df) == n_rows
        assert val_rate > 100000, f"Validator throughput {val_rate:.0f} rows/s below 100k target"

        # Measure cleaner latency
        cleaner = ArtDataCleaner()
        t1 = time.perf_counter()
        clean_df = cleaner.clean(val_df)
        clean_time = time.perf_counter() - t1
        clean_rate = n_rows / clean_time

        assert len(clean_df) == n_rows
        assert clean_rate > 50000, f"Cleaner throughput {clean_rate:.0f} rows/s below 50k target"
