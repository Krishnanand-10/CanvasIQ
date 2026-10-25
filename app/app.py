"""
CanvasIQ: Multimodal Art Piece Valuation Engine.

Interactive Streamlit Web Application for fine art auction appraisals.
Fuses high-dimensional visual aesthetic embeddings (ResNet50 backbone)
with historical tabular catalog metadata (artist reputation, medium,
dimensions, age) to predict real-time fair market value with confidence intervals.
"""

from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Configure repository root on python search path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from app.ui_theme import (
        disclaimer,
        empty_state,
        estimate_card,
        inject_theme,
        page_header,
        sidebar_stats,
    )
except ImportError:
    from ui_theme import (
        disclaimer,
        empty_state,
        estimate_card,
        inject_theme,
        page_header,
        sidebar_stats,
    )

from src.data.cleaner import ArtDataCleaner
from src.data.synthetic import SyntheticArtDatasetGenerator
from src.data.validator import validate_artwork_schema
from src.features.multimodal import MultimodalFeaturePipeline
from src.features.tabular import TabularTransformer
from src.features.vision import ResNet50FeatureExtractor
from src.models.io import load_model_pipeline, save_model_pipeline
from src.models.predict import predict_valuation
from src.models.train import ArtValuationModel, train_valuation_model

logger = logging.getLogger("canvas_iq_app")

# Page Configuration & Metadata
st.set_page_config(
    page_title="CanvasIQ - Art Valuation Engine",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = REPO_ROOT / "models"

COMMON_ARTISTS = [
    "Claude Monet",
    "Pablo Picasso",
    "Vincent van Gogh",
    "Andy Warhol",
    "Georgia O'Keeffe",
    "Rembrandt van Rijn",
    "Jackson Pollock",
    "Mark Rothko",
    "Jean-Michel Basquiat",
    "Nova Synthetics",
    "Custom / Other...",
]

STANDARD_MEDIA = [
    "Oil on Canvas",
    "Acrylic on Canvas",
    "Watercolor on Paper",
    "Bronze Sculpture",
    "Marble Sculpture",
    "Mixed Media",
    "Lithograph / Screenprint",
    "Algorithmic Lightbox",
    "Unknown",
]

PRESET_ARTWORKS = {
    "Claude Monet — Water Lilies (1899)": {
        "artist": "Claude Monet",
        "year": 1899,
        "medium": "Oil on Canvas",
        "height": 65.0,
        "width": 81.0,
    },
    "Pablo Picasso — Cubist Still Life (1912)": {
        "artist": "Pablo Picasso",
        "year": 1912,
        "medium": "Oil on Canvas",
        "height": 46.0,
        "width": 38.0,
    },
    "Andy Warhol — Pop Screenprint (1964)": {
        "artist": "Andy Warhol",
        "year": 1964,
        "medium": "Lithograph / Screenprint",
        "height": 40.0,
        "width": 40.0,
    },
    "Contemporary Masterpiece (2024)": {
        "artist": "Nova Synthetics",
        "year": 2024,
        "medium": "Acrylic on Canvas",
        "height": 72.0,
        "width": 96.0,
    },
}


def format_currency(amount: float) -> str:
    """Formats numeric price into standard US dollar currency string with comma separators."""
    amount_clean = max(0.0, float(amount))
    return f"${amount_clean:,.2f}"


def calculate_dimensions(height: float, width: float) -> Dict[str, float]:
    """Calculates surface area (H * W) and aspect ratio (W / H) with boundary resilience."""
    h = float(height)
    w = float(width)
    if h <= 0.0 or w <= 0.0:
        return {
            "surface_area": max(0.0, h * w),
            "aspect_ratio": 0.0,
        }
    return {
        "surface_area": float(ArtDataCleaner.calculate_surface_area(h, w)),
        "aspect_ratio": float(ArtDataCleaner.calculate_aspect_ratio(h, w)),
    }


def calculate_artwork_dimensions(height: float, width: float) -> Tuple[float, float]:
    """Calculates surface area and aspect ratio as a tuple."""
    dims = calculate_dimensions(height, width)
    return dims["surface_area"], dims["aspect_ratio"]


def format_confidence_interval(lower: float, upper: float) -> Tuple[str, str]:
    """Returns formatted currency strings for lower and upper valuation bounds, ensuring lower >= 0.0."""
    lower_clamped = max(0.0, float(lower))
    upper_clamped = max(lower_clamped, float(upper))
    return format_currency(lower_clamped), format_currency(upper_clamped)


def validate_inputs(metadata: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validates user metadata input dictionary (artist, year, medium, height, width)."""
    required = ["artist", "year", "medium", "height", "width"]
    for field in required:
        if field not in metadata or metadata[field] is None:
            return False, f"Missing required field: {field}"
        if isinstance(metadata[field], str) and not metadata[field].strip():
            return False, f"Field '{field}' cannot be empty."

    try:
        year = int(metadata["year"])
        if not (1000 <= year <= 2026):
            return False, f"Creation year must be between 1000 and 2026, got {year}"
    except (ValueError, TypeError):
        return False, "Creation year must be an integer"

    try:
        h = float(metadata["height"])
        w = float(metadata["width"])
        if h <= 0.0 or w <= 0.0:
            return False, "Artwork dimensions must be strictly positive (> 0)"
    except (ValueError, TypeError):
        return False, "Artwork dimensions must be numeric"

    return True, None


@st.cache_resource(show_spinner=False)
def get_vision_extractor(device: Optional[str] = "cpu") -> ResNet50FeatureExtractor:
    """Caches deep visual feature extractor instance (2048-D ResNet50 backbone)."""
    return ResNet50FeatureExtractor(device=device)


def bootstrap_model_pipeline(
    output_dir: Union[str, Path] = MODEL_DIR,
    n_samples: int = 60,
) -> Dict[str, Any]:
    """Bootstraps and persists a default multimodal valuation pipeline using procedural synthetic benchmark data."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    gen = SyntheticArtDatasetGenerator(seed=42)
    bench_dir = REPO_ROOT / "data"
    bench_df, _ = gen.generate_benchmark(n_samples=n_samples, output_dir=str(bench_dir))

    cleaner = ArtDataCleaner()
    bench_clean = cleaner.clean(bench_df, is_training=True)

    tab_transformer = TabularTransformer(smoothing=10.0)
    y_log = bench_clean["log_price"].values
    X_tab = tab_transformer.fit_transform(bench_clean, y_log)

    extractor = get_vision_extractor(device="cpu")
    img_paths = bench_clean["image_path"].tolist()
    X_vis = extractor.extract_batch(img_paths, batch_size=32)

    fused_pipeline = MultimodalFeaturePipeline(vision_extractor=extractor, tabular_transformer=tab_transformer)
    X_fused = fused_pipeline.fuse(X_tab, X_vis)

    model, metrics = train_valuation_model(
        X_fused,
        y_log,
        model_type="lightgbm",
        n_estimators=50,
        max_depth=5,
        is_log_target=True,
    )

    metadata = {
        "model_type": getattr(model, "model_type", "lightgbm"),
        "n_samples": len(bench_clean),
        "tabular_dim": int(X_tab.shape[1]),
        "vision_dim": int(X_vis.shape[1]),
        "total_dim": int(X_fused.shape[1]),
        "residual_std": float(model.residual_std_),
        "metrics": metrics,
        "status": "Bootstrapped Synthetic Benchmark Model",
    }

    save_model_pipeline(
        model=model,
        tabular_transformer=tab_transformer,
        metadata=metadata,
        output_dir=str(target_dir),
    )

    return {
        "model": model,
        "tabular_transformer": tab_transformer,
        "metadata": metadata,
    }


@st.cache_resource(show_spinner=False)
def load_or_bootstrap_pipeline(models_dir: Union[str, Path] = MODEL_DIR) -> Dict[str, Any]:
    """Loads saved model pipeline from disk, or automatically bootstraps one from synthetic data."""
    path = Path(models_dir)
    model_file = path / "valuation_model.joblib"
    transformer_file = path / "tabular_transformer.joblib"
    metadata_file = path / "metadata.json"

    if model_file.exists() and transformer_file.exists() and metadata_file.exists():
        try:
            return load_model_pipeline(path)
        except Exception as e:
            logger.warning("Failed to load existing pipeline from %s: %s. Re-bootstrapping.", path, e)

    return bootstrap_model_pipeline(output_dir=path, n_samples=60)


def load_pipeline(model_dir: Union[str, Path] = MODEL_DIR) -> Dict[str, Any]:
    """Convenience alias for load_or_bootstrap_pipeline."""
    return load_or_bootstrap_pipeline(model_dir)


def appraise_artwork(
    image_input: Optional[Union[bytes, Image.Image, str, Path]] = None,
    artist: str = "Claude Monet",
    year: int = 1899,
    medium: str = "Oil on Canvas",
    height: float = 65.0,
    width: float = 81.0,
    pipeline: Optional[Dict[str, Any]] = None,
    confidence: float = 0.80,
) -> Dict[str, Any]:
    """Executes end-to-end multimodal art appraisal with statistical confidence intervals."""
    if pipeline is None:
        pipeline = load_or_bootstrap_pipeline()

    # 1. Input validation
    raw_record = {
        "artist": str(artist).strip() or "Unknown",
        "year": int(year),
        "medium": str(medium).strip() or "Unknown",
        "height": float(height),
        "width": float(width),
    }
    is_valid, err_msg = validate_inputs(raw_record)
    if not is_valid:
        raise ValueError(f"Input validation error: {err_msg}")

    val_df = validate_artwork_schema(raw_record, is_training=False)
    cleaner = ArtDataCleaner()
    df_clean = cleaner.clean(val_df, is_training=False)
    if df_clean.empty:
        raise ValueError("Artwork dimensions must be strictly positive (> 0).")

    # 2. Tabular feature transform
    transformer: TabularTransformer = pipeline["tabular_transformer"]
    X_tab = transformer.transform(df_clean)

    # 3. Vision feature extraction (falls back to zeros on None or invalid)
    extractor: ResNet50FeatureExtractor = pipeline.get("vision_extractor") or get_vision_extractor()
    if image_input is not None:
        X_vis = extractor.extract_image(image_input)
    else:
        X_vis = np.zeros(2048, dtype=np.float32)

    X_vis_2d = X_vis.reshape(1, -1)

    # 4. Multimodal feature late fusion
    fused_pipe = MultimodalFeaturePipeline(vision_extractor=extractor, tabular_transformer=transformer)
    X_mm = fused_pipe.fuse(X_tab, X_vis_2d)

    # 5. Predict with confidence interval
    model: ArtValuationModel = pipeline["model"]
    conf = float(confidence)
    if conf > 1.0:
        conf = conf / 100.0
    conf = min(0.99, max(0.50, conf))

    lower_arr, point_arr, upper_arr = model.predict_with_confidence(X_mm, confidence=conf)

    # 6. Sanitize and clamp bounds: 0.0 <= lower <= point <= upper
    lower_price = max(0.0, float(lower_arr[0]))
    point_price = max(0.0, float(point_arr[0]))
    upper_price = max(point_price, float(upper_arr[0]))
    lower_price = min(lower_price, point_price)

    # 7. Physical geometry
    surface_area = float(df_clean.iloc[0]["surface_area"])
    aspect_ratio = float(df_clean.iloc[0]["aspect_ratio"])
    artwork_age = int(df_clean.iloc[0]["age"])

    # 8. Visual representation metrics
    vis_norm = float(np.linalg.norm(X_vis))
    vis_sparsity = float(np.mean(X_vis == 0.0) * 100.0)
    has_image = bool(vis_norm > 1e-6)

    formatted_point = format_currency(point_price)
    formatted_lower = format_currency(lower_price)
    formatted_upper = format_currency(upper_price)

    return {
        "point_estimate": point_price,
        "lower_bound": lower_price,
        "upper_bound": upper_price,
        "formatted_point": formatted_point,
        "formatted_lower": formatted_lower,
        "formatted_upper": formatted_upper,
        "confidence_level": conf,
        "confidence_pct": int(round(conf * 100)),
        "surface_area": surface_area,
        "aspect_ratio": aspect_ratio,
        "artwork_age": artwork_age,
        "has_image": has_image,
        "visual_embedding_norm": vis_norm,
        "visual_sparsity": vis_sparsity,
        "visual_vector": X_vis,
        "tabular_features": X_tab,
        "fused_features": X_mm,
    }


def main() -> None:
    """Main execution entrypoint for CanvasIQ Streamlit appraisal web application."""
    # Inject auction-house catalogue typography & palette
    inject_theme()

    # Load or bootstrap cached valuation pipeline and vision extractor
    pipeline = load_or_bootstrap_pipeline()
    vision_extractor = get_vision_extractor(device="cpu")

    # Sidebar: Presets at top, compact diagnostics in middle, Retrain at bottom under divider
    st.sidebar.markdown(
        '<div style="font-family: \'Newsreader\', Georgia, serif; font-size: 1.35rem; color: #F5F6F4; '
        'font-weight: 500; margin-bottom: 0.25rem;">CanvasIQ</div>'
        '<div style="font-size: 0.8rem; color: #A0A4AB; margin-bottom: 1rem;">Auction valuation engine</div>',
        unsafe_allow_html=True,
    )

    preset_name = st.sidebar.selectbox(
        "Catalogue presets",
        options=["Select a preset..."] + list(PRESET_ARTWORKS.keys()),
        index=0,
        help="Select a curated artwork to populate details.",
    )
    selected_preset = PRESET_ARTWORKS.get(preset_name)

    st.sidebar.divider()

    # Model diagnostics as compact label/value rows
    sidebar_stats(pipeline.get("metadata", {}), pipeline.get("model"))

    st.sidebar.divider()
    if st.sidebar.button("Retrain / Re-bootstrap", use_container_width=True):
        with st.spinner("Retraining baseline model..."):
            pipeline = bootstrap_model_pipeline(MODEL_DIR, n_samples=60)
            st.cache_resource.clear()
            st.sidebar.success("Model pipeline retrained successfully.")

    # Page Header: serif title + single subtitle line
    page_header()

    # Two columns (wide layout, 1:1)
    col_details, col_appraisal = st.columns([1, 1], gap="large")

    # Column 1: Artwork details
    with col_details:
        st.subheader("Artwork details")

        uploaded_file = st.file_uploader(
            "Upload artwork image",
            type=["png", "jpg", "jpeg"],
            help="High-resolution image of the painting or artwork.",
        )

        pil_image: Optional[Image.Image] = None
        if uploaded_file is not None:
            try:
                pil_image = Image.open(uploaded_file).convert("RGB")
                st.image(
                    pil_image,
                    caption=f"{pil_image.width} × {pil_image.height} px",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Error decoding image: {e}")
                pil_image = None
        else:
            st.caption("No image added. The estimate will use catalogue details only.")

        # Artist name
        preset_artist = selected_preset["artist"] if selected_preset else "Claude Monet"
        artist_idx = COMMON_ARTISTS.index(preset_artist) if preset_artist in COMMON_ARTISTS else (len(COMMON_ARTISTS) - 1)
        artist_select = st.selectbox("Artist name", options=COMMON_ARTISTS, index=artist_idx)

        if artist_select == "Custom / Other...":
            artist = st.text_input("Custom artist name", value="Contemporary Artist", placeholder="e.g. Gerhard Richter")
        else:
            artist = artist_select

        # Creation year
        default_year = int(selected_preset["year"]) if selected_preset else 1920
        year = st.slider(
            "Creation year",
            min_value=1800,
            max_value=2026,
            value=default_year,
            step=1,
            help="Year the artwork was created.",
        )

        # Medium
        preset_medium = selected_preset["medium"] if selected_preset else "Oil on Canvas"
        medium_idx = STANDARD_MEDIA.index(preset_medium) if preset_medium in STANDARD_MEDIA else 0
        medium = st.selectbox("Medium", options=STANDARD_MEDIA, index=medium_idx)

        # Dimensions: Height & Width
        col_h, col_w = st.columns(2)
        with col_h:
            default_h = float(selected_preset["height"]) if selected_preset else 40.0
            height = st.number_input(
                "Height (inches)",
                min_value=1.0,
                max_value=1000.0,
                value=default_h,
                step=0.5,
            )
        with col_w:
            default_w = float(selected_preset["width"]) if selected_preset else 50.0
            width = st.number_input(
                "Width (inches)",
                min_value=1.0,
                max_value=1000.0,
                value=default_w,
                step=0.5,
            )

        # Advanced expander: statistical confidence slider
        with st.expander("Advanced"):
            confidence_pct = st.slider(
                "Confidence level (%)",
                min_value=50,
                max_value=99,
                value=80,
                step=1,
                help="Statistical confidence interval for valuation bounds.",
            )

        # Primary action button (full width, active-voice, madder red accent)
        appraise_clicked = st.button("Appraise artwork", type="primary", use_container_width=True)

    # Column 2: Appraisal
    with col_appraisal:
        st.subheader("Appraisal")

        should_appraise = appraise_clicked or (selected_preset is not None)

        if should_appraise:
            with st.spinner("Valuating artwork..."):
                try:
                    result = appraise_artwork(
                        image_input=pil_image,
                        artist=artist,
                        year=year,
                        medium=medium,
                        height=height,
                        width=width,
                        pipeline=pipeline,
                        confidence=confidence_pct / 100.0,
                    )
                except Exception as exc:
                    st.error(f"Appraisal calculation failed: {exc}")
                    result = None

            if result is not None:
                estimate_card(
                    result=result,
                    confidence_pct=confidence_pct,
                    artist=artist,
                    year=year,
                    medium=medium,
                    height=height,
                    width=width,
                )
        else:
            empty_state()

    # Small muted footnote disclaimer at the page bottom
    disclaimer()


if __name__ == "__main__":
    main()
