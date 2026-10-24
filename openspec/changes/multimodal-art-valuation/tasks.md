# Tasks

## 1. Data Pipeline and Synthetic Benchmark

- [x] 1.1 Implement dataset schema validator in `src/data/validator.py` and verify with unit tests on valid and invalid metadata dictionaries
- [x] 1.2 Implement synthetic benchmark generator in `src/data/synthetic.py` generating procedural artwork images and paired metadata, verifying generated image files and CSV output
- [x] 1.3 Implement data cleaning and transformation in `src/data/cleaner.py` (aspect ratio, surface area, `log1p(price)` target transformation) and verify with tests

## 2. Feature Extraction (Visual and Tabular)

- [ ] 2.1 Implement deep vision feature extractor in `src/features/vision.py` using pretrained ResNet50 with pooling and fallback handling, verifying feature dimensions on sample images
- [ ] 2.2 Implement tabular feature transformer in `src/features/tabular.py` for medium categorical encoding, artist statistics, and dimensional scaling, verifying output array shapes
- [ ] 2.3 Implement multimodal feature concatenator in `src/features/multimodal.py` combining vision embeddings and tabular features, verifying end-to-end vector fusion

## 3. Valuation Engine and Evaluation

- [ ] 3.1 Implement regression model training and inference in `src/models/train.py` and `src/models/predict.py` supporting LightGBM / XGBoost with log-price back-transformation, verifying training execution
- [ ] 3.2 Implement model evaluation and ablation runner in `src/models/evaluate.py` reporting Tabular-only, Vision-only, and Multimodal metrics (R², MAE, RMSE, MAPE), verifying comparison table output
- [ ] 3.3 Implement model persistence and artifact saving in `src/models/io.py` verifying model reload and identical prediction outputs

## 4. Interactive Appraisal Interface

- [ ] 4.1 Implement Streamlit web app in `app/app.py` with artwork image upload, metadata input controls (artist, creation year, medium, dimensions), and responsive appraisal layout
- [ ] 4.2 Connect frontend form to the multimodal valuation inference engine, displaying estimated price, confidence interval, and visual breakdown, verifying interactive test flow

## 5. End-to-End Verification & Documentation

- [ ] 5.1 Execute end-to-end benchmark dataset generation, model training, and automated test suite, verifying test pass status
- [ ] 5.2 Document quickstart execution instructions in `README.md` and commit all working artifacts to Git repository

## Workflow follow-up

- Archive the change via OpenSpec once review and task implementation are verified.
