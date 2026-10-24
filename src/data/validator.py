"""
CanvasIQ Dataset Schema Validation Module.

Enforces schema contracts, type integrity, and physical/economic range boundaries
on single artwork records and batch pandas DataFrames for both training and appraisal inference.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import numpy as np
import pandas as pd
from PIL import Image


# ==============================================================================
# Custom Domain Exception Hierarchy
# ==============================================================================

class CanvasIQError(Exception):
    """Base exception class for all CanvasIQ domain exceptions."""
    pass


class SchemaError(CanvasIQError, ValueError):
    """
    Base exception for schema, format, and structure violations.
    Inherits from ValueError for broad standard library compatibility.
    """
    pass


class DataValidationError(SchemaError):
    """Raised when artwork tabular data violates schema, data types, or integrity rules."""
    def __init__(
        self,
        message: str,
        invalid_records: Optional[List[Any]] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.invalid_records = invalid_records or []
        self.details = details or {}


class MissingColumnError(DataValidationError, KeyError):
    """
    Raised when mandatory schema columns are missing.
    Inherits from KeyError for dictionary-like semantics.
    """
    def __init__(self, message: str, missing_columns: Optional[List[str]] = None):
        super().__init__(message)
        self.missing_columns = missing_columns or []


class InvalidRangeError(DataValidationError):
    """Raised when numeric attributes fall outside allowable physical or economic ranges."""
    def __init__(
        self,
        message: str,
        column: Optional[str] = None,
        invalid_values: Optional[List[Any]] = None,
    ):
        super().__init__(message)
        self.column = column
        self.invalid_values = invalid_values or []


class InvalidImageError(CanvasIQError, ValueError):
    """Raised when an artwork image is missing, corrupt, or invalid format."""
    def __init__(self, message: str, image_path: Optional[str] = None):
        super().__init__(message)
        self.image_path = image_path


# ==============================================================================
# Schema Definitions & Constraints
# ==============================================================================

MANDATORY_METADATA_COLUMNS: List[str] = [
    "artist",
    "year",
    "medium",
    "height",
    "width",
]

MANDATORY_TRAINING_COLUMNS: List[str] = MANDATORY_METADATA_COLUMNS + ["price"]
MANDATORY_INFERENCE_COLUMNS: List[str] = MANDATORY_METADATA_COLUMNS

OPTIONAL_COLUMNS: List[str] = [
    "image_path",
    "image_filename",
    "artwork_id",
    "title",
]

COLUMN_ALIASES: Dict[str, str] = {
    "creation_year": "year",
    "height_cm": "height",
    "width_cm": "width",
    "id": "artwork_id",
    "artwork_price": "price",
    "hammer_price": "price",
}

ALLOWED_IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp"}

MIN_YEAR: int = 1000
MAX_YEAR: int = 2026
MIN_DIMENSION: float = 0.001
MAX_DIMENSION: float = 10000.0  # 100 meters
MIN_PRICE: float = 0.001
MAX_PRICE: float = 1_000_000_000.0  # $1 Billion USD


# ==============================================================================
# Validation Functions
# ==============================================================================

def validate_artwork_schema(
    data: Union[Dict[str, Any], pd.DataFrame, pd.Series, List[Dict[str, Any]]],
    is_training: bool = True,
    allow_extra_columns: bool = True,
) -> pd.DataFrame:
    """
    Validate artwork data against schema rules, data types, and domain ranges.

    Parameters:
        data: Artwork data supplied as dict, DataFrame, Series, or list of dicts.
        is_training: If True, strictly enforces that 'price' exists and is positive (> 0).
                     If False (appraisal inference), 'price' is optional.
        allow_extra_columns: If True, preserves additional columns in returned DataFrame.
                             If False, raises DataValidationError if extra columns exist.

    Returns:
        pd.DataFrame: Validated copy of the dataset with standardized column names,
                      enforced dtypes, and reset index.

    Raises:
        DataValidationError: If input is None, empty, has nulls in mandatory columns,
                             contains non-numeric values in numeric columns, or blank strings.
        MissingColumnError: If any mandatory column is absent.
        InvalidRangeError: If numeric values (dimensions, price, year) violate allowable bounds.
    """
    # 1. Null / None Check
    if data is None:
        raise DataValidationError("Input data cannot be None.")

    # 2. Format Ingestion & Coercion to DataFrame
    if isinstance(data, dict):
        if len(data) == 0:
            raise DataValidationError("Artwork record dictionary cannot be empty.")
        df = pd.DataFrame([data])
    elif isinstance(data, pd.DataFrame):
        if data.empty or len(data) == 0:
            raise DataValidationError("Dataset DataFrame cannot be empty (0 rows found).")
        df = data.copy()
    elif isinstance(data, pd.Series):
        if data.empty:
            raise DataValidationError("Input Series cannot be empty.")
        df = data.to_frame().T
    elif isinstance(data, list):
        if len(data) == 0:
            raise DataValidationError("List of artwork records cannot be empty.")
        df = pd.DataFrame(data)
    else:
        raise DataValidationError(
            f"Unsupported data type: '{type(data).__name__}'. "
            "Expected dict, pd.DataFrame, pd.Series, or list of dicts."
        )

    # 3. Column Alias Normalization
    rename_mapping: Dict[str, str] = {}
    for alias, canonical in COLUMN_ALIASES.items():
        if alias in df.columns and canonical not in df.columns:
            rename_mapping[alias] = canonical
    if rename_mapping:
        df = df.rename(columns=rename_mapping)

    # 4. Mandatory Columns Presence Check
    required_cols = MANDATORY_TRAINING_COLUMNS if is_training else MANDATORY_INFERENCE_COLUMNS
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        mode_str = "training" if is_training else "appraisal inference"
        raise MissingColumnError(
            f"Missing mandatory column(s) for {mode_str}: {missing_cols}. "
            f"Required columns: {required_cols}.",
            missing_columns=missing_cols,
        )

    # 5. Unrecognized Columns Check (Strict Mode)
    if not allow_extra_columns:
        allowed_columns = set(required_cols).union(OPTIONAL_COLUMNS)
        extra_cols = [col for col in df.columns if col not in allowed_columns]
        if extra_cols:
            raise DataValidationError(
                f"Unrecognized columns detected: {extra_cols}. "
                "Extra columns are not permitted when allow_extra_columns=False.",
                details={"extra_columns": extra_cols},
            )

    # 6. Null / NaN Values Check in Mandatory Columns
    for col in required_cols:
        null_mask = df[col].isna()
        if null_mask.any():
            bad_indices = df.index[null_mask].tolist()
            raise DataValidationError(
                f"Mandatory column '{col}' contains {len(bad_indices)} missing/NaN value(s) "
                f"at row index(es): {bad_indices[:10]}.",
                details={"column": col, "null_count": len(bad_indices), "indices": bad_indices},
            )

    # 7. Categorical String Columns Validation ('artist', 'medium')
    for str_col in ["artist", "medium"]:
        str_series = df[str_col].astype(str).str.strip()
        empty_mask = (
            (str_series == "")
            | (str_series.str.lower() == "nan")
            | (str_series.str.lower() == "none")
            | (str_series.str.lower() == "null")
        )
        if empty_mask.any():
            bad_indices = df.index[empty_mask].tolist()
            raise DataValidationError(
                f"Column '{str_col}' must contain non-empty, non-blank strings. "
                f"Empty/blank values found at row index(es): {bad_indices[:10]}.",
                details={"column": str_col, "indices": bad_indices},
            )
        df[str_col] = str_series

    # 8. Temporal Validation ('year')
    try:
        numeric_year = pd.to_numeric(df["year"], errors="raise")
    except Exception as exc:
        raise DataValidationError(
            f"Column 'year' contains non-numeric values: {exc}."
        ) from exc

    out_of_range_year = (numeric_year < MIN_YEAR) | (numeric_year > MAX_YEAR)
    if out_of_range_year.any():
        bad_years = numeric_year[out_of_range_year].tolist()
        raise InvalidRangeError(
            f"Column 'year' contains values outside allowable historical range "
            f"[{MIN_YEAR}, {MAX_YEAR}]. Invalid year values found: {bad_years[:10]}.",
            column="year",
            invalid_values=bad_years,
        )
    df["year"] = numeric_year.astype(int)

    # 9. Physical Dimensions Validation ('height', 'width')
    for dim_col in ["height", "width"]:
        try:
            dim_series = pd.to_numeric(df[dim_col], errors="raise")
        except Exception as exc:
            raise DataValidationError(
                f"Column '{dim_col}' contains non-numeric values: {exc}."
            ) from exc

        non_positive_dim = dim_series <= 0
        if non_positive_dim.any():
            bad_dims = dim_series[non_positive_dim].tolist()
            raise InvalidRangeError(
                f"Column '{dim_col}' must be strictly positive (> 0). "
                f"Invalid non-positive values found: {bad_dims[:10]}.",
                column=dim_col,
                invalid_values=bad_dims,
            )

        too_large_dim = dim_series > MAX_DIMENSION
        if too_large_dim.any():
            bad_dims = dim_series[too_large_dim].tolist()
            raise InvalidRangeError(
                f"Column '{dim_col}' exceeds maximum allowable dimension ({MAX_DIMENSION} cm). "
                f"Invalid values: {bad_dims[:10]}.",
                column=dim_col,
                invalid_values=bad_dims,
            )
        df[dim_col] = dim_series.astype(float)

    # 10. Economic Target Validation ('price')
    if "price" in df.columns:
        price_series = df["price"]
        non_null_price = price_series[price_series.notna()]
        if len(non_null_price) > 0:
            try:
                numeric_price = pd.to_numeric(non_null_price, errors="raise")
            except Exception as exc:
                raise DataValidationError(
                    f"Column 'price' contains non-numeric values: {exc}."
                ) from exc

            non_positive_price = numeric_price <= 0
            if non_positive_price.any():
                bad_prices = numeric_price[non_positive_price].tolist()
                raise InvalidRangeError(
                    f"Column 'price' must be strictly positive (> 0). "
                    f"Invalid non-positive values found: {bad_prices[:10]}.",
                    column="price",
                    invalid_values=bad_prices,
                )

            too_expensive = numeric_price > MAX_PRICE
            if too_expensive.any():
                bad_prices = numeric_price[too_expensive].tolist()
                raise InvalidRangeError(
                    f"Column 'price' exceeds maximum allowable valuation ({MAX_PRICE}). "
                    f"Invalid values: {bad_prices[:10]}.",
                    column="price",
                    invalid_values=bad_prices,
                )
            if is_training:
                df["price"] = numeric_price.astype(float)
            else:
                df["price"] = pd.to_numeric(df["price"], errors="coerce").astype(float)

    # 11. Optional Image Columns Sanitization
    for img_col in ["image_path", "image_filename"]:
        if img_col in df.columns:
            df[img_col] = df[img_col].where(df[img_col].isna(), df[img_col].astype(str))

    # 12. Return Validated Copy with Clean Index
    return df.reset_index(drop=True)


def validate_artwork_record(
    record: Dict[str, Any],
    is_training: bool = True,
    allow_extra_columns: bool = True,
) -> Dict[str, Any]:
    """
    Validate a single artwork record dictionary against schema and range rules.

    Parameters:
        record: Dictionary containing artwork attributes.
        is_training: If True, mandates presence and positive value of 'price'.
        allow_extra_columns: If True, preserves additional keys in the dictionary.

    Returns:
        dict: Validated and sanitized record dictionary with coerced types.
    """
    validated_df = validate_artwork_schema(
        data=record,
        is_training=is_training,
        allow_extra_columns=allow_extra_columns,
    )
    return validated_df.iloc[0].to_dict()


def validate_image_file(
    image_path: Union[str, Path],
    check_readable: bool = True,
) -> Path:
    """
    Validate that an artwork image file exists, has a permitted extension, and is readable.

    Parameters:
        image_path: String or Path to image file.
        check_readable: If True, attempts to open image header via PIL.Image.open to verify validity.

    Returns:
        Path: Resolved pathlib.Path object pointing to the valid image.

    Raises:
        InvalidImageError: If file does not exist, has bad extension, or fails PIL verification.
    """
    if image_path is None:
        raise InvalidImageError("Image path cannot be None.")

    path_obj = Path(image_path).resolve()

    if not path_obj.exists():
        raise InvalidImageError(
            f"Artwork image file does not exist: {path_obj}",
            image_path=str(path_obj),
        )

    if not path_obj.is_file():
        raise InvalidImageError(
            f"Artwork image path is not a regular file: {path_obj}",
            image_path=str(path_obj),
        )

    if path_obj.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        raise InvalidImageError(
            f"Unsupported image file extension: '{path_obj.suffix}'. "
            f"Allowed extensions: {sorted(ALLOWED_IMAGE_EXTENSIONS)}",
            image_path=str(path_obj),
        )

    if check_readable:
        try:
            with Image.open(path_obj) as img:
                img.verify()
        except Exception as exc:
            raise InvalidImageError(
                f"Artwork image is corrupt or unreadable: {exc}",
                image_path=str(path_obj),
            ) from exc

    return path_obj
