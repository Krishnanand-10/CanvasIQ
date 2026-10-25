"""Data cleaning, transformation, and feature engineering for artwork valuation records.

This module provides the ArtDataCleaner class and utility functions for calculating
physical metrics (surface area, aspect ratio), artwork age, logarithmic price target
scaling, and inverse price transformations.
"""

from typing import Optional, Union
import numpy as np
import pandas as pd


class ArtDataCleaner:
    """Cleans artwork metadata, imputes missing attributes, and derives geometric/target features."""

    def __init__(
        self,
        reference_year: int = 2026,
        unknown_medium: str = "Unknown",
        unknown_artist: str = "Unknown",
    ) -> None:
        """Initialize ArtDataCleaner.

        Args:
            reference_year: Base year for calculating artwork age. Defaults to 2026.
            unknown_medium: Replacement string for missing or empty medium fields.
            unknown_artist: Replacement string for missing or empty artist fields.
        """
        self.reference_year = reference_year
        self.unknown_medium = unknown_medium
        self.unknown_artist = unknown_artist

    @staticmethod
    def calculate_surface_area(
        height: Union[float, int, np.ndarray, pd.Series],
        width: Union[float, int, np.ndarray, pd.Series],
    ) -> Union[float, np.ndarray, pd.Series]:
        """Calculate physical surface area in cm² (height * width).

        Args:
            height: Artwork height in centimeters (> 0).
            width: Artwork width in centimeters (> 0).

        Returns:
            Surface area as a float or matching array/Series.

        Raises:
            ValueError: If scalar height or width is non-positive or NaN.
        """
        if np.isscalar(height) and np.isscalar(width):
            if np.isnan(height) or np.isnan(width):
                raise ValueError("Height and width cannot be NaN")
            if height <= 0 or width <= 0:
                raise ValueError(
                    f"Dimensions cannot be negative or zero (must be strictly positive): height={height}, width={width}"
                )
            return float(height * width)

        h = np.asarray(height)
        w = np.asarray(width)
        res = h * w
        if isinstance(height, pd.Series):
            return pd.Series(res, index=height.index)
        if isinstance(width, pd.Series):
            return pd.Series(res, index=width.index)
        return res

    @staticmethod
    def calculate_aspect_ratio(
        height: Union[float, int, np.ndarray, pd.Series],
        width: Union[float, int, np.ndarray, pd.Series],
    ) -> Union[float, np.ndarray, pd.Series]:
        """Calculate canvas aspect ratio (width / height).

        Args:
            height: Artwork height in centimeters (> 0).
            width: Artwork width in centimeters (> 0).

        Returns:
            Aspect ratio as a float or matching array/Series.

        Raises:
            ValueError: If scalar height <= 0 or width <= 0, or if NaN.
        """
        if np.isscalar(height) and np.isscalar(width):
            if np.isnan(height) or np.isnan(width):
                raise ValueError("Height and width cannot be NaN")
            if height <= 0:
                raise ValueError(f"Height must be strictly positive and non-zero: height={height}")
            if width <= 0:
                raise ValueError(f"Width must be strictly positive: width={width}")
            return float(width / height)

        h = np.asarray(height, dtype=float)
        w = np.asarray(width, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            res = w / h
        if isinstance(width, pd.Series):
            return pd.Series(res, index=width.index)
        if isinstance(height, pd.Series):
            return pd.Series(res, index=height.index)
        return res

    @staticmethod
    def transform_log_price(
        price: Union[float, int, np.ndarray, pd.Series],
    ) -> Union[float, np.ndarray, pd.Series]:
        """Transform raw monetary price to logarithmic scale via log(1 + price).

        Args:
            price: Raw hammer auction price in dollars (>= 0).

        Returns:
            Log-transformed price as float or matching array/Series.

        Raises:
            ValueError: If scalar price is negative or NaN.
        """
        if np.isscalar(price):
            if np.isnan(price):
                raise ValueError("Price cannot be NaN")
            if price < 0:
                raise ValueError(f"Price must be non-negative for log transformation: price={price}")
            return float(np.log1p(price))

        p = np.asarray(price, dtype=float)
        res = np.log1p(p)
        if isinstance(price, pd.Series):
            return pd.Series(res, index=price.index)
        return res

    @staticmethod
    def inverse_log_price(
        log_price: Union[float, int, np.ndarray, pd.Series],
    ) -> Union[float, np.ndarray, pd.Series]:
        """Invert log-scale price back to native US dollars via exp(y) - 1, clamped >= 0.

        Args:
            log_price: Predicted log-transformed price.

        Returns:
            Estimated dollar price (>= 0.0) as float or matching array/Series.
        """
        if np.isscalar(log_price):
            if np.isnan(log_price):
                return float("nan")
            val = float(np.expm1(log_price))
            return max(0.0, val)

        lp = np.asarray(log_price, dtype=float)
        res = np.maximum(0.0, np.expm1(lp))
        if isinstance(log_price, pd.Series):
            return pd.Series(res, index=log_price.index)
        return res

    @staticmethod
    def calculate_age(
        year: Union[int, float, np.ndarray, pd.Series],
        reference_year: int = 2026,
    ) -> Union[int, np.ndarray, pd.Series]:
        """Calculate artwork age in years relative to a reference year.

        Args:
            year: Creation year of the artwork.
            reference_year: Reference year. Defaults to 2026.

        Returns:
            Age in years (clamped >= 0).
        """
        if np.isscalar(year):
            if year is None:
                return 0
            try:
                y_float = float(year)
            except (ValueError, TypeError):
                return 0
            if not np.isfinite(y_float):
                return 0
            try:
                return max(0, int(reference_year - y_float))
            except (OverflowError, ValueError):
                return 0

        y = np.asarray(year, dtype=float)
        finite_mask = np.isfinite(y)
        safe_y = np.where(finite_mask, y, reference_year)
        age = np.maximum(0.0, reference_year - safe_y)
        age = np.clip(age, 0, np.iinfo(int).max)
        if isinstance(year, pd.Series):
            return pd.Series(age.astype(int), index=year.index)
        return age.astype(int)

    def clean(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Clean raw artwork DataFrame and append derived feature columns.

        Performs:
            1. Copying to prevent mutations on source data.
            2. Missing value imputation: medium imputed with self.unknown_medium.
            3. Numeric type conversion and filtering of non-positive dimensions (height, width).
            4. If is_training=True: filtering non-positive or missing prices.
            5. Derived columns calculation: surface_area, aspect_ratio, age, and log_price.

        Args:
            df: Input DataFrame containing artwork records.
            is_training: If True, mandates valid positive prices. If False, allows unpriced inputs.

        Returns:
            Sanitized pd.DataFrame with derived columns and reset index.

        Raises:
            TypeError: If input is not a pandas DataFrame.
            ValueError: If input DataFrame is empty or missing mandatory columns.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

        if df.empty:
            return pd.DataFrame()

        clean_df = df.copy()

        # Normalize column names: strip whitespace
        clean_df.columns = [c.strip() if isinstance(c, str) else c for c in clean_df.columns]

        # Impute medium if present
        if "medium" in clean_df.columns:
            clean_df["medium"] = (
                clean_df["medium"]
                .fillna(self.unknown_medium)
                .astype(str)
                .replace(r"^\s*$", self.unknown_medium, regex=True)
            )
        else:
            clean_df["medium"] = self.unknown_medium

        # Impute artist if present
        if "artist" in clean_df.columns:
            clean_df["artist"] = (
                clean_df["artist"]
                .fillna(self.unknown_artist)
                .astype(str)
                .replace(r"^\s*$", self.unknown_artist, regex=True)
            )

        # Ensure numeric dimensions
        for col in ["height", "width", "year"]:
            if col in clean_df.columns:
                clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

        # Drop rows with NaN, non-finite, or non-positive dimensions
        if "height" in clean_df.columns and "width" in clean_df.columns:
            valid_dims = (
                clean_df["height"].notna()
                & clean_df["width"].notna()
                & np.isfinite(clean_df["height"])
                & np.isfinite(clean_df["width"])
                & (clean_df["height"] > 0)
                & (clean_df["width"] > 0)
            )
            clean_df = clean_df[valid_dims]

        # Drop rows with NaN or non-finite year
        if "year" in clean_df.columns:
            valid_year = clean_df["year"].notna() & np.isfinite(clean_df["year"])
            clean_df = clean_df[valid_year]

        # Handle price filtering
        if "price" in clean_df.columns:
            clean_df["price"] = pd.to_numeric(clean_df["price"], errors="coerce")
            if is_training:
                valid_price = (
                    clean_df["price"].notna()
                    & np.isfinite(clean_df["price"])
                    & (clean_df["price"] > 0)
                )
                clean_df = clean_df[valid_price]
        elif is_training:
            raise ValueError("Training dataset must contain a 'price' column")

        if clean_df.empty:
            return pd.DataFrame()

        # Compute derived features
        clean_df["surface_area"] = clean_df["height"] * clean_df["width"]
        clean_df["aspect_ratio"] = clean_df["width"] / clean_df["height"]
        clean_df["age"] = np.maximum(0, self.reference_year - clean_df["year"].astype(int))

        # Compute log_price
        if "price" in clean_df.columns:
            if is_training:
                clean_df["log_price"] = np.log1p(clean_df["price"].astype(float))
            else:
                clean_df["log_price"] = np.where(
                    clean_df["price"].notna() & (clean_df["price"] >= 0),
                    np.log1p(clean_df["price"].astype(float)),
                    np.nan,
                )

        return clean_df.reset_index(drop=True)


# Top-level functional aliases for direct module imports
calculate_surface_area = ArtDataCleaner.calculate_surface_area
calculate_aspect_ratio = ArtDataCleaner.calculate_aspect_ratio
transform_log_price = ArtDataCleaner.transform_log_price
inverse_log_price = ArtDataCleaner.inverse_log_price
calculate_age = ArtDataCleaner.calculate_age
