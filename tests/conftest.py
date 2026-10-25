"""Shared test fixtures and helpers for CanvasIQ test suite."""

import os
import sys
import io
import shutil
import tempfile
import importlib
import unittest
from pathlib import Path
from typing import Any, Optional, Dict, List

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pytest
except ImportError:
    pytest = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None


def skip_test(reason: str):
    """Raise appropriate skip exception depending on test framework in use."""
    if pytest is not None:
        pytest.skip(reason)
    else:
        raise unittest.SkipTest(reason)


def require_package(pkg_name: str) -> Any:
    """Ensure a third-party package is installed, otherwise skip test cleanly."""
    try:
        return importlib.import_module(pkg_name)
    except (ImportError, ModuleNotFoundError) as err:
        skip_test(f"Required package '{pkg_name}' not installed in environment: {err}")


def get_component(module_name: str, symbol_name: Optional[str] = None, milestone: str = "M1") -> Any:
    """Safely import a module and optional symbol with progressive testability skip.
    
    If the module or symbol is not yet implemented (e.g. pending a future milestone),
    the test will be cleanly skipped instead of failing with an unhandled import error.
    """
    try:
        mod = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as err:
        skip_test(f"Module '{module_name}' not yet available [Pending {milestone}]: {err}")
    
    if symbol_name is not None:
        if not hasattr(mod, symbol_name):
            skip_test(f"Symbol '{symbol_name}' not yet defined in '{module_name}' [Pending {milestone}]")
        return getattr(mod, symbol_name)
    
    return mod


# Fixture factories for both pytest and standard unittest

def make_sample_artwork_dict() -> Dict[str, Any]:
    """Generates a canonical valid artwork dictionary."""
    return {
        "artist": "Claude Monet",
        "year": 1899,
        "medium": "Oil on Canvas",
        "height": 65.0,
        "width": 81.0,
        "price": 12500000.0,
        "image_path": "data/images/synthetic/monet_waterlilies.png"
    }


def make_sample_pil_image(width: int = 224, height: int = 224, color: str = "blue") -> Any:
    """Generates a test PIL Image with a simple colored rectangle."""
    if Image is None:
        skip_test("Pillow is not available in environment")
    img = Image.new("RGB", (width, height), color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([width // 4, height // 4, 3 * width // 4, 3 * height // 4], fill="yellow")
    return img


def make_sample_image_bytes(format: str = "PNG") -> bytes:
    """Generates valid image bytes in memory."""
    img = make_sample_pil_image(128, 128, "red")
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return buffer.getvalue()


def make_diverse_artwork_records(n: int = 5) -> List[Dict[str, Any]]:
    """Generates a list of diverse artwork records for batch testing."""
    records = [
        {"artist": "Pablo Picasso", "year": 1937, "medium": "Oil on Canvas", "height": 349.3, "width": 776.6, "price": 45000000.0},
        {"artist": "Vincent van Gogh", "year": 1889, "medium": "Oil on Canvas", "height": 73.7, "width": 92.1, "price": 82500000.0},
        {"artist": "Georgia O'Keeffe", "year": 1932, "medium": "Oil on Canvas", "height": 101.6, "width": 76.2, "price": 44400000.0},
        {"artist": "Andy Warhol", "year": 1962, "medium": "Silkscreen Ink", "height": 208.0, "width": 145.0, "price": 195000000.0},
        {"artist": "Jean-Michel Basquiat", "year": 1982, "medium": "Acrylic and Oilstick", "height": 183.0, "width": 173.0, "price": 110500000.0},
        {"artist": "Yayoi Kusama", "year": 2014, "medium": "Acrylic on Canvas", "height": 130.3, "width": 162.0, "price": 8500000.0},
        {"artist": "Gerhard Richter", "year": 1986, "medium": "Oil on Canvas", "height": 260.0, "width": 200.0, "price": 34000000.0},
        {"artist": "Joan Mitchell", "year": 1959, "medium": "Oil on Canvas", "height": 240.0, "width": 195.0, "price": 16600000.0},
    ]
    return records[:n]


# Pytest fixtures if pytest is installed
if pytest is not None:
    @pytest.fixture
    def sample_artwork():
        return make_sample_artwork_dict()

    @pytest.fixture
    def sample_image():
        return make_sample_pil_image()

    @pytest.fixture
    def sample_image_bytes_fixture():
        return make_sample_image_bytes()

    @pytest.fixture
    def temp_workspace():
        tmp_dir = tempfile.mkdtemp(prefix="caniq_test_")
        yield Path(tmp_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)
