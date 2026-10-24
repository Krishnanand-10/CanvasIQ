# Spec Delta

## Purpose

Trains regression models on multimodal art features, outputs fair market valuations, and evaluates prediction metrics.

## ADDED Requirements

### Requirement: Multimodal regression training
The system SHALL train regression models (Gradient Boosted Decision Trees / LightGBM / XGBoost) on multimodal art representations predicting target log price.

#### Scenario: Successful model training
- **WHEN** training data containing multimodal features and log target prices are provided
- **THEN** model trains to convergence and reports validation metrics (MAE, RMSE, R², MAPE)

### Requirement: Valuation inference and price back-transformation
The system SHALL accept artwork features, predict log price, and invert the prediction back into native currency market valuation.

#### Scenario: Predicting artwork value
- **WHEN** multimodal features of an unpriced artwork are passed to the trained model
- **THEN** the system outputs estimated dollar valuation along with lower and upper confidence bounds

### Requirement: Valuation ablation evaluation
The system SHALL provide evaluation metrics comparing tabular-only baseline, vision-only baseline, and multimodal fusion performance.

#### Scenario: Run ablation evaluation
- **WHEN** the evaluation pipeline is run across all three configurations
- **THEN** a comparative performance table is returned showing R² and MAPE metrics for each configuration
