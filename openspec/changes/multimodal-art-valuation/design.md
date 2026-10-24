# Design

## Context

See `proposal.md` for system motivation and problem context. CanvasIQ operates on heterogeneous data: unstructured 2D canvas images and structured tabular metadata (artist, dimensions, medium, creation year). The system requires a low-latency, modular architecture enabling reproducible training, offline benchmarking, and interactive appraisal.

## Goals / Non-Goals

**Goals:**
- Provide a decoupled, two-stage multimodal pipeline: visual feature extraction followed by tabular-visual late fusion.
- Enable zero-shot aesthetic feature extraction using a standard pretrained vision model (ResNet50 / EfficientNet) without requiring prohibitive GPU fine-tuning.
- Support both synthetic art benchmarking (for immediate local testing and reproducible validation) and real auction dataset ingestion.
- Implement robust evaluation comparing Tabular-only, Visual-only, and Multimodal Fusion models.
- Deliver an interactive Streamlit web dashboard for end users.

**Non-Goals:**
- End-to-end joint gradient backpropagation into the vision backbone during gradient boosting training (late fusion cleanly separates feature extraction from tree ensembles).
- Real-time web scraping during live user valuation requests (all training data is ingested offline).

## Decisions

### 1. Visual Feature Extraction: Frozen Pretrained CNN (ResNet50)
- **Decision:** Use a frozen pretrained ResNet50 backbone (weights trained on ImageNet) truncated at the penultimate average pooling layer, producing a 2048-dimensional aesthetic/composition vector.
- **Rationale:** Fine art style, palette, and structural composition are well captured by deep convolutional filter activations. Freezing weights prevents overfitting on modest art dataset sizes and executes efficiently on standard CPU or single GPU.
- **Alternatives Considered:**
  - *Train CNN from scratch:* High risk of severe overfitting due to limited art transaction sample sizes.
  - *CLIP ViT embeddings:* Effective, but ResNet50 provides faster inference and lower dependency overhead on standard Linux environments.

### 2. Fusion Architecture: Late Fusion via Gradient Boosted Trees (LightGBM / XGBoost)
- **Decision:** Concatenate tabular feature vectors (artist encoding, dimensions, medium, age) with visual embeddings and train LightGBM / XGBoost regressor.
- **Rationale:** Gradient boosted decision trees excel on heterogeneous tabular data with mixed scales and dense visual embeddings, training significantly faster and with greater stability than deep MLPs on smaller sample sizes.
- **Alternatives Considered:**
  - *Deep Multimodal MLP:* Prone to unstable convergence on small tabular feature sets and requires extensive hyperparameter tuning.

### 3. Target Modeling: Logarithmic Price Transformation
- **Decision:** Transform target valuation to $y = \log(1 + \text{price})$ for training, and invert via $\hat{\text{price}} = \exp(\hat{y}) - 1$ at inference.
- **Rationale:** Fine art auction prices follow a power-law distribution. Fitting standard squared loss directly on raw prices causes the model to optimize exclusively for multi-million-dollar outliers while failing on standard works.

### 4. Synthetic Art Benchmark Suite
- **Decision:** Provide a built-in synthetic benchmark generator capable of generating procedural canvas images (abstract, colorfield, impressionistic patterns) and realistic metadata distributions.
- **Rationale:** Guarantees immediate out-of-the-box pipeline testability and unit test execution without requiring immediate multi-gigabyte external auction image downloads.

## Risks / Trade-offs

- **[Risk: Missing or Corrupt User Image]** → *Mitigation:* Fallback to a zero-vector embedding and toggle the valuation model to tabular-only inference mode with a warning in the UI.
- **[Risk: Unseen or Rare Artists at Inference]** → *Mitigation:* Apply target encoding with smoothing and handle unknown artists via overall market median artist tiers.
- **[Risk: High Dimensionality of Vision Vectors Overwhelming Tabular Features]** → *Mitigation:* Apply optional PCA dimensionality reduction (e.g. 32 to 64 components) or tree feature fraction subsampling in LightGBM/XGBoost.
