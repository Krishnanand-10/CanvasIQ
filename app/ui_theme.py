"""
UI theme, typography, and styling components for CanvasIQ.

Design direction: Auction-house catalogue / gallery wall.
- Palette: Gallery white (#F5F6F4) main background, Ink (#16181D) sidebar,
  Madder red (#A3342B) single accent for primary actions and range bar.
- Typography: Newsreader (serif) for headings and valuation figures;
  Public Sans for body copy and metadata.
- Geometry: 4px corners, 1px #DADCD8 borders, no gradients, no emoji in headings.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import streamlit as st


THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400..700;1,6..72,400..700&family=Public+Sans:wght@300;400;500;600;700&display=swap');

/* Main application container */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    font-family: 'Public Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    background-color: #F5F6F4 !important;
    color: #16181D !important;
}

/* Hide default menu, header decoration, and footer */
#MainMenu {visibility: hidden; display: none !important;}
footer {visibility: hidden; display: none !important;}
header[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* Headings in Newsreader serif */
h1, h2, h3, h4, h5, h6,
[data-testid="stTitle"],
[data-testid="stSubheader"] {
    font-family: 'Newsreader', Georgia, 'Times New Roman', serif !important;
    color: #16181D !important;
    font-weight: 500 !important;
    letter-spacing: -0.015em !important;
}

/* Sidebar styling: Ink (#16181D) with light text */
section[data-testid="stSidebar"], [data-testid="stSidebar"] {
    background-color: #16181D !important;
    color: #F5F6F4 !important;
    border-right: 1px solid #232731 !important;
}

section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: #F5F6F4 !important;
}

section[data-testid="stSidebar"] hr {
    border-color: #2B2F38 !important;
    margin: 1.25rem 0 !important;
}

/* Sidebar form inputs */
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #1F232B !important;
    border: 1px solid #383D48 !important;
    border-radius: 4px !important;
    color: #F5F6F4 !important;
}

section[data-testid="stSidebar"] [data-baseweb="select"] span {
    color: #F5F6F4 !important;
}

/* Sidebar secondary button */
section[data-testid="stSidebar"] button {
    background-color: #222630 !important;
    color: #F5F6F4 !important;
    border: 1px solid #383D48 !important;
    border-radius: 4px !important;
    font-family: 'Public Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
    box-shadow: none !important;
}

section[data-testid="stSidebar"] button:hover {
    background-color: #2C3240 !important;
    border-color: #4E5566 !important;
    color: #FFFFFF !important;
}

/* Primary Action Button: Madder red (#A3342B) */
button[kind="primary"], .stButton > button[kind="primary"] {
    background-color: #A3342B !important;
    color: #FFFFFF !important;
    border: 1px solid #8C2B23 !important;
    border-radius: 4px !important;
    font-family: 'Public Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    box-shadow: none !important;
    padding: 0.6rem 1.25rem !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
}

button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover {
    background-color: #8C2B23 !important;
    border-color: #75221B !important;
    color: #FFFFFF !important;
}

button[kind="primary"]:active, .stButton > button[kind="primary"]:active {
    background-color: #75221B !important;
}

/* Keyboard focus states: Madder red (#A3342B) */
*:focus-visible {
    outline: 2px solid #A3342B !important;
    outline-offset: 2px !important;
}

/* Input fields, selectboxes, number inputs in main content area */
[data-testid="stAppViewContainer"] [data-baseweb="input"],
[data-testid="stAppViewContainer"] [data-baseweb="base-input"] {
    background-color: #FFFFFF !important;
    border: 1px solid #DADCD8 !important;
    border-radius: 4px !important;
    color: #16181D !important;
}

[data-testid="stAppViewContainer"] [data-baseweb="input"]:focus-within {
    border-color: #A3342B !important;
    box-shadow: 0 0 0 1px #A3342B !important;
}

[data-testid="stAppViewContainer"] [data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #DADCD8 !important;
    border-radius: 4px !important;
}

[data-testid="stAppViewContainer"] [data-baseweb="select"] > div:focus-within {
    border-color: #A3342B !important;
    box-shadow: 0 0 0 1px #A3342B !important;
}

/* File Uploader */
[data-testid="stFileUploader"] section {
    background-color: #FFFFFF !important;
    border: 1px dashed #DADCD8 !important;
    border-radius: 4px !important;
    padding: 20px 14px !important;
    transition: border-color 0.15s ease !important;
}

[data-testid="stFileUploader"] section:hover {
    border-color: #A3342B !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #DADCD8 !important;
    border-radius: 4px !important;
    box-shadow: none !important;
}

[data-testid="stExpander"] details summary {
    font-size: 0.88rem !important;
    color: #555A63 !important;
}

/* Sliders: Madder red thumb and active track */
div[data-testid="stSlider"] div[role="slider"] {
    background-color: #A3342B !important;
    border: 2px solid #FFFFFF !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.2) !important;
}

div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div:nth-child(1) {
    background-color: #A3342B !important;
}

/* Artwork frame: white mat with soft shadow */
[data-testid="stImage"] {
    background-color: #FFFFFF !important;
    padding: 12px !important;
    border: 1px solid #DADCD8 !important;
    border-radius: 4px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06) !important;
    margin-bottom: 0.75rem !important;
}

[data-testid="stImage"] img {
    border: 1px solid #ECEEEA !important;
    border-radius: 2px !important;
}

/* Metric styling: Serif type for valuation */
[data-testid="stMetric"] {
    background-color: transparent !important;
    border: none !important;
    padding: 0 !important;
}

[data-testid="stMetricLabel"] p {
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #717680 !important;
    font-family: 'Public Sans', sans-serif !important;
}

[data-testid="stMetricValue"] div {
    font-family: 'Newsreader', Georgia, serif !important;
    font-size: 2.8rem !important;
    color: #16181D !important;
    font-weight: 500 !important;
    letter-spacing: -0.02em !important;
    line-height: 1.1 !important;
}

/* Custom Catalogue Components */
.page-header-container {
    padding-bottom: 1.25rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid #DADCD8;
}

.page-header-title {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 2.6rem;
    font-weight: 500;
    color: #16181D;
    line-height: 1.15;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.02em;
}

.page-header-subtitle {
    font-size: 1.05rem;
    color: #555A63;
    margin: 0;
    line-height: 1.45;
}

.empty-appraisal-card {
    background-color: #FFFFFF;
    border: 1px dashed #DADCD8;
    border-radius: 4px;
    padding: 72px 24px;
    text-align: center;
    min-height: 360px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.empty-appraisal-title {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.4rem;
    color: #16181D;
    margin-bottom: 0.4rem;
    font-weight: 500;
}

.empty-appraisal-caption {
    font-size: 0.9rem;
    color: #717680;
    max-width: 280px;
    line-height: 1.45;
}

.lot-card {
    background-color: #FFFFFF;
    border: 1px solid #DADCD8;
    border-radius: 4px;
    padding: 24px;
    margin-bottom: 1.25rem;
}

.lot-card-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #F0F1EE;
}

.lot-card-tag {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #717680;
    font-weight: 600;
}

.lot-card-meta {
    font-size: 0.85rem;
    color: #555A63;
}

.lot-price-display {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 3rem;
    font-weight: 500;
    color: #16181D;
    letter-spacing: -0.02em;
    line-height: 1.05;
    margin: 0.5rem 0 1.25rem 0;
}

/* Range bar with tick at midpoint */
.range-card {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid #F0F1EE;
}

.range-labels {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.85rem;
    color: #555A63;
    margin-bottom: 0.6rem;
}

.range-bound-value {
    font-weight: 600;
    color: #16181D;
}

.range-confidence-badge {
    font-size: 0.75rem;
    color: #717680;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.range-bar-track {
    position: relative;
    width: 100%;
    height: 6px;
    background-color: #E6E8E4;
    border-radius: 3px;
    margin: 8px 0;
}

.range-bar-fill {
    position: absolute;
    left: 0;
    right: 0;
    top: 0;
    bottom: 0;
    background-color: #A3342B;
    border-radius: 3px;
}

.range-bar-tick {
    position: absolute;
    left: 50%;
    top: -5px;
    width: 2px;
    height: 16px;
    background-color: #A3342B;
    transform: translateX(-50%);
    border-radius: 1px;
    box-shadow: 0 0 0 2px #FFFFFF;
}

.lot-details-grid {
    margin-top: 1.25rem;
    padding-top: 1rem;
    border-top: 1px solid #F0F1EE;
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.75rem 1rem;
    font-size: 0.85rem;
}

.lot-detail-item {
    display: flex;
    flex-direction: column;
}

.lot-detail-label {
    color: #717680;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.15rem;
}

.lot-detail-val {
    color: #16181D;
    font-weight: 500;
}

/* Sidebar diagnostics table */
.sidebar-diag-header {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.15rem;
    color: #F5F6F4;
    font-weight: 500;
    margin-bottom: 0.75rem;
}

.sidebar-diag-table {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    margin-bottom: 1rem;
}

.sidebar-diag-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0.35rem 0;
    border-bottom: 1px solid #232731;
    font-size: 0.82rem;
}

.sidebar-diag-row:last-child {
    border-bottom: none;
}

.sidebar-diag-label {
    color: #A0A4AB;
}

.sidebar-diag-value {
    color: #F5F6F4;
    font-weight: 500;
    font-variant-numeric: tabular-nums;
}

/* Footnote disclaimer */
.footnote-disclaimer {
    margin-top: 3.5rem;
    padding-top: 1.25rem;
    border-top: 1px solid #DADCD8;
    font-size: 0.78rem;
    color: #717680;
    line-height: 1.5;
    text-align: center;
}

@media (max-width: 768px) {
    .page-header-title {
        font-size: 2.1rem;
    }
    .lot-price-display {
        font-size: 2.4rem;
    }
    .lot-details-grid {
        grid-template-columns: 1fr;
    }
}
</style>
"""


def inject_theme() -> None:
    """Injects auction-house catalogue CSS styles into the Streamlit document."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def page_header() -> None:
    """Renders the top catalogue header."""
    st.title("CanvasIQ")
    st.markdown(
        '<p class="page-header-subtitle">Estimate what a painting could fetch at auction from its image and catalogue details.</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="height: 1.25rem;"></div>', unsafe_allow_html=True)


def empty_state() -> None:
    """Renders the un-appraised empty placeholder state."""
    st.markdown(
        """
        <div class="empty-appraisal-card">
            <div class="empty-appraisal-title">No appraisal yet</div>
            <div class="empty-appraisal-caption">
                Enter artwork details on the left and select <strong>Appraise artwork</strong> to generate an auction valuation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def estimate_card(
    result: Dict[str, Any],
    confidence_pct: int,
    artist: str,
    year: int,
    medium: str,
    height: float,
    width: float,
) -> None:
    """
    Renders the auction lot card with large serif estimate, range bar with midpoint tick,
    and catalogue metadata breakdown.
    """
    surface_area = result.get("surface_area", height * width)
    aspect_ratio = result.get("aspect_ratio", (width / height) if height > 0 else 1.0)
    has_image = result.get("has_image", False)
    image_status = "Image embeddings included" if has_image else "Catalogue details only"

    # Render primary metric callout for Streamlit testing compatibility and clean display
    st.metric(
        label="Estimated auction value",
        value=result["formatted_point"],
    )

    # Render lot card details with range bar
    st.markdown(
        f"""
        <div class="lot-card">
            <div class="lot-card-header">
                <span class="lot-card-tag">Lot Appraisal</span>
                <span class="lot-card-meta">{artist} ({year})</span>
            </div>
            <div class="range-card">
                <div class="range-labels">
                    <div>
                        <span style="font-size: 0.72rem; text-transform: uppercase; color: #717680; display: block;">Low estimate</span>
                        <span class="range-bound-value">{result['formatted_lower']}</span>
                    </div>
                    <span class="range-confidence-badge">{confidence_pct}% confidence interval</span>
                    <div style="text-align: right;">
                        <span style="font-size: 0.72rem; text-transform: uppercase; color: #717680; display: block;">High estimate</span>
                        <span class="range-bound-value">{result['formatted_upper']}</span>
                    </div>
                </div>
                <div class="range-bar-track">
                    <div class="range-bar-fill"></div>
                    <div class="range-bar-tick" title="Midpoint estimate"></div>
                </div>
            </div>
            <div class="lot-details-grid">
                <div class="lot-detail-item">
                    <span class="lot-detail-label">Artist</span>
                    <span class="lot-detail-val">{artist}</span>
                </div>
                <div class="lot-detail-item">
                    <span class="lot-detail-label">Creation year</span>
                    <span class="lot-detail-val">{year}</span>
                </div>
                <div class="lot-detail-item">
                    <span class="lot-detail-label">Medium</span>
                    <span class="lot-detail-val">{medium}</span>
                </div>
                <div class="lot-detail-item">
                    <span class="lot-detail-label">Dimensions</span>
                    <span class="lot-detail-val">{height:,.1f} × {width:,.1f} in ({surface_area:,.1f} sq in)</span>
                </div>
                <div class="lot-detail-item">
                    <span class="lot-detail-label">Aspect ratio</span>
                    <span class="lot-detail-val">{aspect_ratio:.2f}</span>
                </div>
                <div class="lot-detail-item">
                    <span class="lot-detail-label">Analysis modality</span>
                    <span class="lot-detail-val">{image_status}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_stats(metadata: Dict[str, Any], model_obj: Any) -> None:
    """Renders the compact label/value model diagnostics table in the sidebar."""
    model_type = str(metadata.get("model_type", getattr(model_obj, "model_type", "LightGBM"))).title()
    tab_dim = metadata.get("tabular_dim", 11)
    vis_dim = metadata.get("vision_dim", 2048)
    n_samples = metadata.get("n_samples", 60)
    res_std = getattr(model_obj, "residual_std_", metadata.get("residual_std", 0.20))

    st.sidebar.markdown(
        f"""
        <div class="sidebar-diag-header">Model diagnostics</div>
        <div class="sidebar-diag-table">
            <div class="sidebar-diag-row">
                <span class="sidebar-diag-label">Model architecture</span>
                <span class="sidebar-diag-value">{model_type}</span>
            </div>
            <div class="sidebar-diag-row">
                <span class="sidebar-diag-label">Tabular features</span>
                <span class="sidebar-diag-value">{tab_dim}</span>
            </div>
            <div class="sidebar-diag-row">
                <span class="sidebar-diag-label">Vision features</span>
                <span class="sidebar-diag-value">{vis_dim:,}</span>
            </div>
            <div class="sidebar-diag-row">
                <span class="sidebar-diag-label">Training records</span>
                <span class="sidebar-diag-value">{n_samples}</span>
            </div>
            <div class="sidebar-diag-row">
                <span class="sidebar-diag-label">Residual std (log)</span>
                <span class="sidebar-diag-value">{res_std:.4f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def disclaimer() -> None:
    """Renders the small muted footnote disclaimer at the page bottom."""
    st.markdown(
        """
        <div class="footnote-disclaimer">
            Valuation estimates and ranges are generated by multimodal machine learning models for informational and research purposes only. Actual auction realization prices depend on provenance, physical condition, market liquidity, and specialist examination.
        </div>
        """,
        unsafe_allow_html=True,
    )
