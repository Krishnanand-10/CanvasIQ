# Project: CanvasIQ

CanvasIQ is an end-to-end multimodal machine learning system that estimates fair market auction valuations from artwork images and historical metadata, complete with a self-contained benchmark generator, evaluation pipeline, and an interactive Streamlit appraisal interface.

## Architecture

CanvasIQ is structured into clean, decoupled layers separating data ingestion, feature extraction, gradient-boosted modeling, artifact persistence, and presentation:

```
[Raw Metadata / Synthetic Generator] ──> [Data Validator & Cleaner]
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
             [Visual Feature Extractor]                        [Tabular Transformer]
             (ResNet50 Penultimate Pool: 2048-D)               (Encoders & Scalers)
                        │                                                 │
                        └────────────────────────┬────────────────────────┘
                                                 ▼
                                     [Multimodal Concatenator]
                                      ([Tabular || Vision])
                                                 │
                                                 ▼
                                    [Valuation Regressor Engine]
                                    (LightGBM / XGBoost on log1p)
                                                 │
                        ┌────────────────────────┼────────────────────────┐
                        ▼                        ▼                        ▼
                 [Ablation Runner]        [Model Artifacts]       [Streamlit Web App]
             (Tabular vs Vision vs MM)     (models/*.joblib)         (app/app.py)
```

### Module Boundaries
- `src/data/`: Schema validation (`validator.py`), procedural synthetic benchmark generation (`synthetic.py`), and data cleaning/transformations (`cleaner.py`).
- `src/features/`: Deep vision embeddings via frozen ResNet50 (`vision.py`), tabular transformation & target encoding (`tabular.py`), and multimodal late fusion (`multimodal.py`).
- `src/models/`: Regressor training with log-price modeling (`train.py`), price back-transformation & confidence interval prediction (`predict.py`), ablation study runner (`evaluate.py`), and atomic pipeline persistence (`io.py`).
- `src/utils/`: Shared logging, environment configurations, and metrics utilities.
- `app/`: Streamlit interactive appraisal web application (`app/app.py`).
- `tests/`: Isolated unit tests in `tests/unit/` and requirement-driven opaque-box 4-tier E2E tests in `tests/e2e/`.

---

## Feature Inventory

Every feature from the Survey phase is mapped to a concrete milestone below:

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Dataset Schema Validator | Validates presence and types of mandatory fields (`artist`, `year`, `medium`, `height`, `width`, `price`). | M1 | Survey (Task 1.1) |
| 2 | Clean & Sanitize Records | Cleans missing values, filters non-positive dimensions/prices, imputes missing categorical values with `"Unknown"`. | M1 | Survey (Task 1.3) |
| 3 | Surface Area Calculation | Computes canvas surface area: $\text{height} \times \text{width}$. | M1 | Survey (Task 1.3) |
| 4 | Aspect Ratio Calculation | Computes canvas aspect ratio: $\text{width} / \text{height}$. | M1 | Survey (Task 1.3) |
| 5 | Log Target Transformation | Transforms raw positive hammer prices using $y = \log(1 + \text{price})$. | M1 | Survey (Task 1.3) |
| 6 | Inverse Price Back-Transformation | Inverts log-scale predictions back into native dollars using $\hat{\text{price}} = \exp(\hat{y}) - 1$, clamped to $\ge 0$. | M1 | Survey (Task 1.3, 3.1) |
| 7 | Procedural Artwork Image Generator | Synthesizes procedural canvas images (colorfield, abstract, minimalist, impressionist) with distinct color palettes. | M1 | Survey (Task 1.2) |
| 8 | Synthetic Metadata Generator | Generates realistic paired artwork metadata matching empirical distributions of artist tiers, media, dimensions, and power-law prices. | M1 | Survey (Task 1.2) |
| 9 | Frozen Pretrained ResNet50 Backbone | Uses ImageNet-pretrained ResNet50 truncated at penultimate `avgpool` layer producing 2048-D aesthetic embeddings without backprop. | M2 | Survey (Task 2.1) |
| 10 | Image Preprocessing & Transforms | Resizes to 256, center-crops to 224x224, converts to tensor, and normalizes using ImageNet statistics. | M2 | Survey (Task 2.1) |
| 11 | Missing/Corrupt Image Fallback | Automatically falls back to 2048-dimensional zero-vector when an image is corrupt, unreadable, or missing. | M2 | Survey (Task 2.1) |
| 12 | Device Handling (CPU/GPU) | Transparently selects CUDA GPU if available, with robust CPU fallback without throwing device errors. | M2 | Survey (Task 2.1) |
| 13 | Medium Categorical Encoding | One-hot encodes or categorical encodes artwork medium types, gracefully handling unseen media. | M2 | Survey (Task 2.2) |
| 14 | Artist Statistics & Pedigree Encoding | Encodes artist tiers, historical median values, or smoothed target encoding, with fallback to global median for unseen artists. | M2 | Survey (Task 2.2) |
| 15 | Dimensional Scaling & Era Normalization | Scales physical metrics (`surface_area`, `aspect_ratio`, `height`, `width`) and creation age using `StandardScaler`/`RobustScaler`. | M2 | Survey (Task 2.2) |
| 16 | Late Concatenation Layer | Horizontally concatenates tabular feature array $T$ ($D_t$) and vision embedding array $V$ ($D_v$) into unified $[T \parallel V]$. | M2 | Survey (Task 2.3) |
| 17 | Modality Slicing for Ablations | Provides functional interface to extract Tabular-only, Vision-only, or Combined feature subsets from the unified pipeline. | M2 | Survey (Task 2.3, 3.2) |
| 18 | LightGBM / XGBoost Regressor | Fits gradient boosted decision trees on multimodal features to predict log target prices. | M3 | Survey (Task 3.1) |
| 19 | Price Inference Pipeline | Accepts raw artwork record (image + metadata), applies preprocessing and feature extraction, and computes estimated dollar valuation. | M3 | Survey (Task 3.1) |
| 20 | Confidence Interval Estimation | Calculates lower and upper valuation bounds based on validation residual variance ($\hat{y} \pm z \cdot \sigma_\epsilon$) back-transformed to dollars. | M3 | Survey (Task 3.1, R4) |
| 21 | Ablation Comparison Protocol | Trains and evaluates Tabular-only, Vision-only, and Multimodal models on identical test data, computing $R^2$, MAE, RMSE, and MAPE. | M3 | Survey (Task 3.2) |
| 22 | Model Pipeline Serialization | Serializes trained regressor, tabular transformer, scaler, and metadata JSON into `models/` directory. | M3 | Survey (Task 3.3) |
| 23 | Reload Verification Test | Loads saved pipeline artifacts from disk and verifies prediction outputs on identical sample inputs match within float precision. | M3 | Survey (Task 3.3) |
| 24 | Interactive Appraisal Dashboard | Web user interface providing image file uploader, artist, creation year, medium, and dimension input controls. | M4 | Survey (Task 4.1) |
| 25 | Valuation Report & Visual Breakdown | Displays predicted fair market value in large callout, confidence range, calculated dimensions, and aesthetic/metadata breakdown. | M4 | Survey (Task 4.2) |
| 26 | Feature-Wise Atomic Commits & Push | Groups code into conventional commits (`feat(data)`, `feat(features)`, `feat(models)`, `feat(app)`, `docs(openspec)`) pushed to `origin/main`. | M1-M5 | Survey (R5) |
| 27 | OpenSpec Task Checklist Completion | Tracks implementation progress by updating `[ ]` to `[x]` for all 13 tasks in `tasks.md`. | M1-M5 | Survey (Tasks 1.1-5.2) |
| 28 | Change Archiving Workflow | Archives completed change via `openspec archive multimodal-art-valuation --yes`, promoting delta specs to main specs. | M5 | Survey (Workflow follow-up) |

---

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| E2E | E2E Testing Track | Requirement-driven opaque-box test harness in `tests/e2e/` covering Tiers 1-4, publishing `TEST_READY.md`. | None | DONE |
| M1 | Data Pipeline & Synthetic Benchmark | `src/data/validator.py`, `src/data/synthetic.py`, `src/data/cleaner.py`, and unit tests in `tests/unit/test_data.py`. | None | DONE |
| M2 | Multimodal Feature Extraction | `src/features/vision.py`, `src/features/tabular.py`, `src/features/multimodal.py`, and unit tests in `tests/unit/test_features.py`. | M1 | PLANNED |
| M3 | Valuation Engine, Ablation & Persistence | `src/models/train.py`, `src/models/predict.py`, `src/models/evaluate.py`, `src/models/io.py`, and unit tests in `tests/unit/test_models.py`. | M2 | PLANNED |
| M4 | Interactive Appraisal Web Application | `app/app.py` Streamlit dashboard, valuation presentation, confidence bounds, and smoke/integration test. | M3 | PLANNED |
| M5 | Final Milestone: 100% E2E Pass & Adversarial Hardening | Pass 100% of E2E suite, execute Tier 5 adversarial tests, update README documentation, mark all OpenSpec tasks `[x]`, archive change, push to `origin/main`. | M4, E2E | PLANNED |

---

## Interface Contracts

### `src/data/validator.py`
```python
def validate_artwork_schema(data: Union[dict, pd.DataFrame], is_training: bool = True) -> pd.DataFrame:
    """Validates presence and types of mandatory fields. Raises SchemaError or ValueError on invalid inputs."""
```

### `src/data/synthetic.py`
```python
class SyntheticArtDatasetGenerator:
    def __init__(self, seed: int = 42): ...
    def generate_image(self, style: str, palette: str, size: Tuple[int, int] = (224, 224)) -> Image.Image: ...
    def generate_benchmark(self, n_samples: int = 200, output_dir: str = "data") -> Tuple[pd.DataFrame, str]: ...
```

### `src/data/cleaner.py`
```python
class ArtDataCleaner:
    def clean(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame: ...
    @staticmethod
    def calculate_surface_area(height: float, width: float) -> float: ...
    @staticmethod
    def calculate_aspect_ratio(height: float, width: float) -> float: ...
    @staticmethod
    def transform_log_price(price: Union[float, np.ndarray]) -> Union[float, np.ndarray]: ...
    @staticmethod
    def inverse_log_price(log_price: Union[float, np.ndarray]) -> Union[float, np.ndarray]: ...
```

### `src/features/vision.py`
```python
class ResNet50FeatureExtractor:
    def __init__(self, device: Optional[str] = None): ...
    def extract_image(self, image_input: Union[str, Path, Image.Image, bytes]) -> np.ndarray:
        """Returns 2048-D float32 vector; returns np.zeros(2048, dtype=np.float32) on missing/corrupt image."""
    def extract_batch(self, image_inputs: List[Union[str, Path, Image.Image]]) -> np.ndarray: ...
```

### `src/features/tabular.py`
```python
class TabularTransformer:
    def __init__(self, smoothing: float = 10.0): ...
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "TabularTransformer": ...
    def transform(self, X: pd.DataFrame) -> np.ndarray: ...
    def fit_transform(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> np.ndarray: ...
```

### `src/features/multimodal.py`
```python
class MultimodalFeaturePipeline:
    def __init__(self, vision_extractor: ResNet50FeatureExtractor, tabular_transformer: TabularTransformer): ...
    def fuse(self, tabular_features: np.ndarray, visual_features: np.ndarray) -> np.ndarray: ...
    def get_features(self, df: pd.DataFrame, image_inputs: List[Any], modality: str = "multimodal") -> np.ndarray: ...
```

### `src/models/train.py` & `src/models/predict.py`
```python
class ArtValuationModel:
    def fit(self, X: np.ndarray, y_log: np.ndarray) -> "ArtValuationModel": ...
    def predict_log(self, X: np.ndarray) -> np.ndarray: ...
    def predict_price(self, X: np.ndarray) -> np.ndarray: ...
    def predict_with_confidence(self, X: np.ndarray, confidence: float = 0.80) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
```

### `src/models/evaluate.py`
```python
def compute_metrics(y_true_price: np.ndarray, y_pred_price: np.ndarray) -> Dict[str, float]:
    """Returns {'r2': float, 'mae': float, 'rmse': float, 'mape': float}."""
def run_ablation_study(X_tab: np.ndarray, X_vis: np.ndarray, y_price: np.ndarray) -> pd.DataFrame:
    """Compares Tabular-only, Vision-only, and Multimodal models."""
```

### `src/models/io.py`
```python
def save_model_pipeline(model: Any, tabular_transformer: Any, metadata: dict, output_dir: str = "models") -> None: ...
def load_model_pipeline(model_dir: str = "models") -> Dict[str, Any]: ...
```

---

## Code Layout

```
/home/krishna/Documents/CanvasIQ/
├── app/
│   └── app.py
├── data/
│   ├── images/
│   ├── processed/
│   └── raw/
├── models/
│   ├── metadata.json
│   ├── tabular_transformer.joblib
│   └── valuation_model.joblib
├── openspec/
│   └── changes/multimodal-art-valuation/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── cleaner.py
│   │   ├── synthetic.py
│   │   └── validator.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── multimodal.py
│   │   ├── tabular.py
│   │   └── vision.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── evaluate.py
│   │   ├── io.py
│   │   ├── predict.py
│   │   └── train.py
│   └── utils/
│       ├── __init__.py
│       └── logger.py
└── tests/
    ├── e2e/
    │   ├── test_tier1_feature_coverage.py
    │   ├── test_tier2_boundaries.py
    │   ├── test_tier3_interactions.py
    │   ├── test_tier4_real_world.py
    │   └── test_tier5_adversarial.py
    └── unit/
        ├── test_data.py
        ├── test_features.py
        └── test_models.py
```
