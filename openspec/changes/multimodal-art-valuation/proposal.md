# Proposal

## Why

Determining the fair market value of fine art is traditionally subjective, opaque, and dependent on human appraisal, making valuation inaccessible to casual collectors, auction participants, and smaller galleries. Fine art pricing is uniquely driven by both historical market context (artist reputation, age, medium, dimensions) and visual aesthetics (palette, composition, textures, artistic style). CanvasIQ solves this by building an automated, multimodal machine learning valuation system that unifies deep visual representations and tabular auction metadata into calibrated price predictions.

## What Changes

- Implement a robust data pipeline supporting both curated art auction datasets and a high-fidelity synthetic benchmark dataset for offline reproducible experimentation.
- Implement a deep visual feature extractor using transfer learning (pretrained CNN backbone such as ResNet50) to extract aesthetic embeddings from artwork images.
- Implement tabular feature engineering pipelines handling artist market metrics, dimensional attributes (surface area, aspect ratio), creation era/age, and categorical medium encodings.
- Build a multimodal late-fusion regression model (Gradient Boosted Decision Trees / XGBoost / LightGBM) trained on target log-transformed hammer prices.
- Build an interactive, production-ready Streamlit web application enabling users to upload artwork images, input historical metadata, and receive fair market valuations with confidence ranges and explainable attribute breakdowns.

## Capabilities

### New Capabilities
- `data-pipeline`: Ingestion, schema validation, data preprocessing, and realistic benchmark dataset generation for art sales and images.
- `feature-extraction`: Visual embedding extraction via pretrained vision backbones and tabular feature engineering for artwork metadata.
- `valuation-engine`: Multimodal regression modeling, model serialization, evaluation metrics (R², MAE, RMSE, MAPE), and price inference on log scale.
- `valuation-interface`: Interactive web-based user interface allowing image upload, metadata input, real-time prediction, and visual appraisal reports.

### Modified Capabilities
<!-- No existing capabilities to modify in initial greenfield project -->

## Impact

- Codebase additions: `src/data/`, `src/features/`, `src/models/`, `app/` and test suites.
- Dependencies: PyTorch, Torchvision, Scikit-learn, XGBoost/LightGBM, Streamlit, PIL, Pandas, NumPy.
- Operational: End-to-end local training and inference without external API lock-in.
