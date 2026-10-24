# Spec Delta

## Purpose

Provides an interactive web user interface for uploading artwork images, inputting metadata, and viewing valuation estimates.

## ADDED Requirements

### Requirement: Interactive artwork appraisal form
The system SHALL provide a web form allowing users to upload a painting image and specify artist name, creation year, medium, and canvas dimensions.

#### Scenario: User submits artwork details
- **WHEN** user uploads an artwork image and fills in artist, year, medium, and dimensions and clicks Appraise
- **THEN** the system executes the valuation pipeline and displays the estimated fair market value

### Requirement: Valuation report and feature breakdown
The system SHALL render an appraisal breakdown displaying predicted market value, estimated valuation range, and key contributing factors.

#### Scenario: Display valuation results
- **WHEN** appraisal inference completes successfully
- **THEN** the interface renders the predicted price, confidence interval, and aesthetic/metadata attributes summary
