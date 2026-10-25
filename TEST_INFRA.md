# CanvasIQ Test Infrastructure Specification (TEST_INFRA.md)

## 1. Executive Summary & Purpose

CanvasIQ is a multimodal machine learning system estimating fair market fine art auction valuations from visual artwork imagery and historical auction metadata. This document establishes the **Opaque-Box, Requirement-Driven 4-Tier End-to-End (E2E) Test Architecture** for CanvasIQ.

The testing track guarantees system reliability, statistical robustness, regression protection, and boundary resiliency across all 28 system features cataloged in `PROJECT.md`, without coupling tests to internal implementation details.

---

## 2. 4-Tier Opaque-Box E2E Testing Methodology

The test suite is structured into four distinct, progressive verification tiers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Tier 4: Real-World Scenarios                   │
│   (Appraisal flows, Streamlit form emulation, Latency budgets, Memory) │
├────────────────────────────────────────────────────────────────────────┤
│                 Tier 3: Cross-Feature Combinations                     │
│    (Late fusion synergy, Multimodal vs Unimodal ablations, Scalability)│
├────────────────────────────────────────────────────────────────────────┤
│                   Tier 2: Boundary & Corner Cases                      │
│ (>=5 tests/feature at domain limits: extreme sizes, corrupt data, NaNs)│
├────────────────────────────────────────────────────────────────────────┤
│                     Tier 1: Feature Coverage                           │
│  (>=5 tests/feature covering all 28 features in PROJECT.md happy-path) │
└────────────────────────────────────────────────────────────────────────┘
```

### Tier 1: Feature Coverage (Core Functional Paths)
- **Target**: All 28 features cataloged in `PROJECT.md` § Feature Inventory.
- **Coverage Requirement**: $\ge 5$ distinct test cases per feature ($28 \times 5 = 140$ test cases).
- **Scope**: Exercises the primary contract behavior (happy-path), input-to-output conversions, mathematical identities, vector dimensions, and data schemas across each module.

### Tier 2: Boundary & Corner Cases (Domain Limits)
- **Target**: All 28 features cataloged in `PROJECT.md` at operational and physical limits.
- **Coverage Requirement**: $\ge 5$ distinct test cases per feature ($28 \times 5 = 140$ test cases).
- **Scope**: Evaluates edge cases: miniature canvases ($0.1\text{ cm}$), colossal murals ($10,000\text{ cm}$), sub-dollar prices ($\$0.50$), nine-figure auctions ($\$250\text{M}$), corrupted JPEG byte streams, all-black/all-white canvases, unseen artists, unseen media types, and empty inputs.

### Tier 3: Cross-Feature Combinations (Pairwise & System Interactions)
- **Target**: Multi-component pipelines and architectural interfaces.
- **Coverage Requirement**: $\ge 15$ pairwise interaction test cases.
- **Scope**: Verifies late fusion synergy, multimodal superior/equal performance over unimodal baselines ($R^2_{\text{MM}} \ge R^2_{\text{Tabular}}, R^2_{\text{Vision}}$), synthetic generation to training propagation, fallback resilience during batch training, and pipeline save-to-reload prediction invariance.

### Tier 4: Real-World Application & Performance Scenarios
- **Target**: User-facing workflows, Streamlit UI interactions, and operational constraints.
- **Coverage Requirement**: $\ge 10$ comprehensive application scenarios.
- **Scope**: Simulates end-to-end user appraisal journeys (Monet impressionist appraisals, contemporary digital art, auction catalog batch valuation), and validates strict latency budgets: ResNet50 extraction $\le 150\text{ ms}$, tabular inference $\le 15\text{ ms}$, total CPU latency $\le 250\text{ ms}$, and cold-start pipeline load $\le 1.5\text{ s}$.

---

## 3. Directory Layout & File Inventory

```
tests/
├── __init__.py                           # Test package initializer
├── conftest.py                           # Pytest fixtures and progressive testability helpers
├── run_tests.py                          # Modular test runner CLI (supports standard Python & pytest)
├── e2e/
│   ├── __init__.py                       # E2E test subpackage
│   ├── test_tier1_feature_coverage.py   # Tier 1 suite: 140 feature coverage tests (F1 - F28)
│   ├── test_tier2_boundaries.py         # Tier 2 suite: 140 boundary & corner tests (F1 - F28)
│   ├── test_tier3_interactions.py       # Tier 3 suite: 15 cross-feature interaction tests
│   └── test_tier4_real_world.py         # Tier 4 suite: 10 real-world appraisal & latency benchmarks
└── unit/
    ├── __init__.py                       # Unit test subpackage
    ├── test_data.py                      # Data validation, synthetic generation, and cleaning tests
    ├── test_features.py                  # Feature extraction unit tests
    └── test_models.py                    # Valuation engine & evaluation unit tests
```

---

## 4. Feature Coverage Matrix (Tiers 1 & 2)

| # | Feature Name | Tier 1 Tests | Tier 2 Tests | Target Module |
|---|--------------|--------------|--------------|---------------|
| 1 | Dataset Schema Validator | 5 tests (`test_f1_*`) | 5 tests (`test_f1_b*`) | `src/data/validator.py` |
| 2 | Clean & Sanitize Records | 5 tests (`test_f2_*`) | 5 tests (`test_f2_b*`) | `src/data/cleaner.py` |
| 3 | Surface Area Calculation | 5 tests (`test_f3_*`) | 5 tests (`test_f3_b*`) | `src/data/cleaner.py` |
| 4 | Aspect Ratio Calculation | 5 tests (`test_f4_*`) | 5 tests (`test_f4_b*`) | `src/data/cleaner.py` |
| 5 | Log Target Transformation | 5 tests (`test_f5_*`) | 5 tests (`test_f5_b*`) | `src/data/cleaner.py` |
| 6 | Inverse Price Back-Transformation | 5 tests (`test_f6_*`) | 5 tests (`test_f6_b*`) | `src/data/cleaner.py` |
| 7 | Procedural Artwork Image Generator | 5 tests (`test_f7_*`) | 5 tests (`test_f7_b*`) | `src/data/synthetic.py` |
| 8 | Synthetic Metadata Generator | 5 tests (`test_f8_*`) | 5 tests (`test_f8_b*`) | `src/data/synthetic.py` |
| 9 | Frozen Pretrained ResNet50 Backbone | 5 tests (`test_f9_*`) | 5 tests (`test_f9_b*`) | `src/features/vision.py` |
| 10 | Image Preprocessing & Transforms | 5 tests (`test_f10_*`) | 5 tests (`test_f10_b*`) | `src/features/vision.py` |
| 11 | Missing/Corrupt Image Fallback | 5 tests (`test_f11_*`) | 5 tests (`test_f11_b*`) | `src/features/vision.py` |
| 12 | Device Handling (CPU/GPU) | 5 tests (`test_f12_*`) | 5 tests (`test_f12_b*`) | `src/features/vision.py` |
| 13 | Medium Categorical Encoding | 5 tests (`test_f13_*`) | 5 tests (`test_f13_b*`) | `src/features/tabular.py` |
| 14 | Artist Statistics & Pedigree Encoding | 5 tests (`test_f14_*`) | 5 tests (`test_f14_b*`) | `src/features/tabular.py` |
| 15 | Dimensional Scaling & Era Normalization | 5 tests (`test_f15_*`) | 5 tests (`test_f15_b*`) | `src/features/tabular.py` |
| 16 | Late Concatenation Layer | 5 tests (`test_f16_*`) | 5 tests (`test_f16_b*`) | `src/features/multimodal.py` |
| 17 | Modality Slicing for Ablations | 5 tests (`test_f17_*`) | 5 tests (`test_f17_b*`) | `src/features/multimodal.py` |
| 18 | LightGBM / XGBoost Regressor | 5 tests (`test_f18_*`) | 5 tests (`test_f18_b*`) | `src/models/train.py` |
| 19 | Price Inference Pipeline | 5 tests (`test_f19_*`) | 5 tests (`test_f19_b*`) | `src/models/predict.py` |
| 20 | Confidence Interval Estimation | 5 tests (`test_f20_*`) | 5 tests (`test_f20_b*`) | `src/models/predict.py` |
| 21 | Ablation Comparison Protocol | 5 tests (`test_f21_*`) | 5 tests (`test_f21_b*`) | `src/models/evaluate.py` |
| 22 | Model Pipeline Serialization | 5 tests (`test_f22_*`) | 5 tests (`test_f22_b*`) | `src/models/io.py` |
| 23 | Reload Verification Test | 5 tests (`test_f23_*`) | 5 tests (`test_f23_b*`) | `src/models/io.py` |
| 24 | Interactive Appraisal Dashboard | 5 tests (`test_f24_*`) | 5 tests (`test_f24_b*`) | `app/app.py` |
| 25 | Valuation Report & Visual Breakdown | 5 tests (`test_f25_*`) | 5 tests (`test_f25_b*`) | `app/app.py` |
| 26 | Feature-Wise Atomic Commits & Push | 5 tests (`test_f26_*`) | 5 tests (`test_f26_b*`) | Git repository & commits |
| 27 | OpenSpec Task Checklist Completion | 5 tests (`test_f27_*`) | 5 tests (`test_f27_b*`) | `tasks.md` |
| 28 | Change Archiving Workflow | 5 tests (`test_f28_*`) | 5 tests (`test_f28_b*`) | OpenSpec change state |
| **Total** | **28 System Features** | **140 Tests** | **140 Tests** | **Full System Scope** |

---

## 5. Expected Output Derivation & Verification Oracles

For every test case across all tiers, expected values are derived strictly from authoritative sources:
1. **Mathematical Truth & Geometric Properties**:
   - Surface area: $A = \text{height} \times \text{width}$.
   - Aspect ratio: $R = \text{width} / \text{height}$.
   - Log transformation: $y = \log(1 + \text{price})$ (`np.log1p`).
   - Inverse log transformation: $\hat{P} = \max(0.0, \exp(\hat{y}) - 1)$ (`np.expm1`).
   - Roundtrip identity: $\text{inverse}(\text{transform}(P)) = P$ within $\pm 10^{-6}$.
2. **Deterministic Pretrained Model Invariants**:
   - ResNet50 penultimate feature vector dimensionality is invariant: $\mathbb{R}^{2048}$.
   - Missing/corrupt image fallback: zero-vector $\mathbf{0} \in \mathbb{R}^{2048}$ with $\Vert\mathbf{0}\Vert_2 = 0.0$.
3. **Statistical Regression Constraints**:
   - Predicted price non-negativity: $\hat{P} \ge 0.0$.
   - Confidence bounds monotonicity: $L \le \hat{P} \le U$, with $L \ge 0.0$.
4. **Repository & Specification Artifacts**:
   - OpenSpec tasks structure derived from `openspec/changes/multimodal-art-valuation/tasks.md`.
   - Git remote tracking derived from `origin` remote config.

---

## 6. Progressive Testability & Milestone Integration

To allow tests to run cleanly throughout development without crashing before future milestones are implemented:
- **Component Loader (`tests.conftest.get_component`)**:
  Inspects whether the target module and symbol exist.
  - If the module is implemented, the test executes real logic and asserts expected outputs.
  - If the module is not yet implemented, the test is marked `SKIPPED` with an informative message indicating the pending milestone (e.g. `[Pending Milestone M2: src.features.vision]`).
- **Milestone Mapping**:
  - Features 1–8: Active during Milestone 1 (`src/data/`).
  - Features 9–17: Active during Milestone 2 (`src/features/`).
  - Features 18–23: Active during Milestone 3 (`src/models/`).
  - Features 24–25: Active during Milestone 4 (`app/app.py`).
  - Features 26–28: Active across Milestones 1–5 (Git, OpenSpec, Documentation).
  - Tiers 3 & 4: Progressively activate as cross-module pipelines are implemented.
  - Final Milestone (M5): 100% of all 305 tests must execute and pass with 0 skips and 0 failures.

---

## 7. Test Execution Instructions

### Running via the Modular Test Runner
A dedicated CLI runner `tests/run_tests.py` is provided for running tests under standard Python or pytest:

```bash
# Run all E2E tiers (default)
python3 tests/run_tests.py

# Run a specific tier
python3 tests/run_tests.py --tier 1
python3 tests/run_tests.py --tier 2
python3 tests/run_tests.py --tier 3
python3 tests/run_tests.py --tier 4

# Run tests for a specific milestone
python3 tests/run_tests.py --milestone M1
python3 tests/run_tests.py --milestone M2

# Run tests for a specific feature (1 - 28)
python3 tests/run_tests.py --feature 3

# Run with verbose output and JSON test summary
python3 tests/run_tests.py --verbose --json-report test_report.json
```

### Running via Pytest Directly
Once pytest is installed in the virtual environment:

```bash
# Run all E2E test files
pytest tests/e2e/ -v

# Run individual tiers
pytest tests/e2e/test_tier1_feature_coverage.py -v
pytest tests/e2e/test_tier2_boundaries.py -v
pytest tests/e2e/test_tier3_interactions.py -v
pytest tests/e2e/test_tier4_real_world.py -v

# Run by marker
pytest -m tier1
pytest -m "tier1 or tier2"
```

---

## 8. Quality Gate & Acceptance Criteria

A milestone passes the E2E quality gate when:
1. **Zero Collection Failures**: Test suites discover cleanly without syntax or import crashes.
2. **Feature Coverage Threshold**: All features implemented in the current milestone pass 100% of their Tier 1 and Tier 2 tests.
3. **No Regressions**: Previously implemented milestone features continue passing with zero regressions.
4. **M5 Victory Threshold**: In Milestone 5, all 305 test cases across Tiers 1–4 execute with 0 failures, 0 errors, and 0 skips.
