# ML Drift Monitoring System

**Status:** ✅ Completed — Production Ready

A production-grade ML drift monitoring and detection system built with strict Test-Driven Development (TDD) methodology. This system implements custom drift detection algorithms using industry-standard statistical techniques without dependency on third-party drift detection libraries.

---

## Executive Summary

**ML drift** refers to changes in data distributions or model predictions over time that degrade model performance. In production ML systems, drift can occur due to:
- Evolving user behavior
- Seasonal patterns
- Data quality issues
- Upstream pipeline changes

This system detects **feature drift** (changes in input distributions) and **prediction drift** (changes in model output distributions) using statistical hypothesis testing and distribution comparison techniques. It provides deterministic, configurable monitoring suitable for batch-based production ML workflows.

---

## Why This Project Exists

Machine learning models degrade over time as the statistical properties of production data diverge from training data. Without monitoring:
- **Performance degradation** goes undetected until business impact occurs
- **Regulatory compliance** risks emerge (model validation requirements)
- **Operational incidents** happen due to silent failures

Batch-based statistical drift monitoring is the industry-standard approach for:
- Scheduled model retraining decisions
- Data quality validation
- Compliance reporting
- Production health checks

This system provides the core statistical engine for such monitoring, designed for integration into MLOps pipelines.

---

## Key Capabilities

✅ **Implemented Features:**
- **PSI (Population Stability Index)** for numerical feature drift detection
- **Kolmogorov-Smirnov (KS) test** for distribution shift detection
- **Chi-Square test** for categorical feature drift
- **Configurable thresholds** (per-metric defaults + per-feature overrides)
- **Severity-based alerts** (warning / critical)
- **Deterministic pipeline execution** (no randomness, reproducible results)
- **CLI interface** with standard exit codes (0=no drift, 2=drift, 1=error)
- **Python API** for programmatic integration
- **100% test pass rate** (148/148 tests passing)
- **Configuration-driven design** (no hardcoded thresholds)

---

## Architecture Overview

The system follows a **modular, layered architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────┐
│                   CLI / API Layer                    │
│                  (drift/cli.py)                      │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              Pipeline Orchestration                  │
│               (drift/pipeline.py)                    │
└──┬────────┬─────────┬──────────┬────────────────────┘
   │        │         │          │
   ▼        ▼         ▼          ▼
┌─────┐ ┌────────┐ ┌──────────┐ ┌────────┐
│Wind-│ │Metrics │ │Detectors │ │Alerts  │
│owing│ │(PSI/KS)│ │(Thresh.) │ │(Sever.)│
└─────┘ └────────┘ └──────────┘ └────────┘
           │
           ▼
    ┌──────────────┐
    │Configuration │
    │  (Thresholds)│
    └──────────────┘
```

**Module Responsibilities:**
- `drift/windows.py` - Data windowing strategies (reference vs. current)
- `drift/metrics.py` - Statistical drift calculations (PSI, KS, Chi-Square)
- `drift/detectors.py` - Threshold-based drift decision logic
- `drift/alerts.py` - Alert generation with severity classification
- `drift/pipeline.py` - End-to-end orchestration
- `drift/cli.py` - Command-line interface
- `drift/config.py` - Configuration resolution (defaults + overrides)

---

## Project Structure

```
ml-drift-monitoring/
├── drift/
│   ├── __init__.py
│   ├── alerts.py          # Alert generation logic
│   ├── cli.py             # CLI runner
│   ├── config.py          # Configuration management
│   ├── detectors.py       # Drift detection decisions
│   ├── metrics.py         # Statistical calculations
│   ├── pipeline.py        # Pipeline orchestration
│   └── windows.py         # Windowing logic
├── tests/
│   ├── test_alerts.py     # Alert generation tests
│   ├── test_architecture.py  # Dependency validation
│   ├── test_cli.py        # CLI behavior tests
│   ├── test_config.py     # Configuration tests
│   ├── test_detectors.py  # Detector logic tests
│   ├── test_metrics.py    # Metric calculation tests
│   ├── test_pipeline.py   # End-to-end tests
│   └── test_windows.py    # Windowing tests
├── data/
│   └── sample_reference.csv
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
└── requirements.txt
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/SourabhaKK/ml-model-monitoring-drift-detection.git
cd ml-model-monitoring-drift-detection

# Install dependencies
pip install -r requirements.txt

# Install in development mode (optional)
pip install -e .
```

**Requirements:**
- Python ≥ 3.9
- numpy ≥ 1.24.0
- pandas ≥ 2.0.0
- scipy ≥ 1.10.0

---

## Usage Examples

### CLI Usage

```bash
# Detect drift using PSI metric
python -m drift.cli reference.csv current.csv \
  --metric psi \
  --threshold 0.1 \
  --feature-type numerical

# Detect drift using KS test
python -m drift.cli reference.csv current.csv \
  --metric ks \
  --threshold 0.3 \
  --feature-type numerical

# Detect drift in categorical features
python -m drift.cli reference.csv current.csv \
  --metric chi_square \
  --threshold 0.05 \
  --feature-type categorical
```

**Exit Codes:**
- `0` - No drift detected
- `2` - Drift detected
- `1` - Error (invalid input, missing file, etc.)

### Python API Usage

```python
import pandas as pd
from drift.pipeline import run_drift_pipeline

# Load data
reference_data = pd.read_csv("reference.csv")
current_data = pd.read_csv("current.csv")

# Run drift detection
result = run_drift_pipeline(
    reference_data,
    current_data,
    feature_type="numerical",
    metric="psi",
    threshold=0.1
)

# Check results
if result["drift_detected"]:
    print(f"Drift detected! Alerts: {result['alerts']}")
else:
    print("No drift detected")
```

### Configuration-Driven Usage

```python
from drift.config import get_threshold

# Define configuration
config = {
    "metrics": {
        "psi": {
            "default_threshold": 0.1,
            "feature_thresholds": {
                "age": 0.15,
                "income": 0.2
            }
        }
    }
}

# Resolve threshold for specific feature
threshold = get_threshold(config, metric="psi", feature="age")
# Returns: 0.15 (feature-specific override)

threshold = get_threshold(config, metric="psi", feature="other")
# Returns: 0.1 (default)
```

---

## Drift Metrics Implemented

### PSI (Population Stability Index)
**Purpose:** Detects shifts in numerical feature distributions  
**Method:** Compares binned distributions using KL-divergence-like formula  
**Interpretation:**
- PSI < 0.1: No significant drift
- 0.1 ≤ PSI < 0.2: Moderate drift (warning)
- PSI ≥ 0.2: Significant drift (critical)

**Use Case:** Monitoring continuous features like age, income, credit score

### Kolmogorov-Smirnov (KS) Test
**Purpose:** Detects distribution shifts in numerical data  
**Method:** Two-sample KS test comparing empirical CDFs  
**Interpretation:**
- Statistic measures maximum distance between CDFs
- Higher statistic = greater distribution difference

**Use Case:** Detecting shape changes in distributions (not just mean/variance)

### Chi-Square Test
**Purpose:** Detects shifts in categorical feature distributions  
**Method:** Chi-square test of independence on contingency table  
**Interpretation:**
- Low p-value (< threshold) indicates significant drift
- Tests whether category proportions have changed

**Use Case:** Monitoring categorical features like product category, user segment, device type

---

## Testing Strategy

This project was built using **strict Test-Driven Development (TDD)**:

### RED → GREEN Cycles
All features were developed through disciplined TDD cycles:
1. **RED:** Write failing tests defining behavior
2. **GREEN:** Implement minimal code to pass tests
3. **REFACTOR:** (optional) Improve code while maintaining tests

**Commit History Evidence:**
- 7 RED commits (test definitions)
- 7 GREEN commits (implementations)
- Perfect sequencing maintained throughout

### Test Characteristics
- **Behavior-focused:** Tests assert outputs, not implementation details
- **Deterministic:** No flaky tests, reproducible results
- **Fast:** Full suite runs in ~1.5 seconds
- **Comprehensive:** 148 tests covering all modules
- **CI-friendly:** No external dependencies (databases, APIs)

### Test Coverage
```
Total Tests: 148
Pass Rate: 100%
Modules Tested:
  - Windows: 13 tests
  - Metrics: 35 tests
  - Detectors: 33 tests
  - Alerts: 18 tests
  - Pipeline: 17 tests
  - CLI: 11 tests
  - Config: 11 tests
  - Architecture: 10 tests
```

**Run Tests:**
```bash
pytest -v                    # Verbose output
pytest --tb=short            # Short traceback
pytest --durations=10        # Show slowest tests
```

---

## Design Decisions & Trade-offs

### Why Not Use Evidently?
**Decision:** Implement custom drift metrics rather than depend on Evidently  
**Rationale:**
- **Learning objective:** Understand drift detection internals
- **Control:** Full control over metric calculations and thresholds
- **Simplicity:** Avoid heavyweight dependencies for core logic
- **Extensibility:** Easy to add custom metrics

**Note:** Evidently is excellent for production use; this project prioritizes educational value and architectural clarity.

### Why Manual Metric Implementation?
**Decision:** Implement PSI from scratch using NumPy  
**Rationale:**
- **Transparency:** Understand binning strategies and edge cases
- **Determinism:** Control over epsilon handling, bin boundaries
- **Interview defensibility:** Can explain every line of code

### Why No Visualization?
**Decision:** No dashboards or plotting  
**Rationale:**
- **Separation of concerns:** Monitoring engine ≠ visualization layer
- **Scope control:** Focus on statistical correctness
- **Integration flexibility:** Output JSON for any visualization tool

### Why Batch-Based?
**Decision:** No streaming/real-time processing  
**Rationale:**
- **Industry alignment:** Most drift monitoring is batch-scheduled
- **Simplicity:** Avoid distributed systems complexity
- **Determinism:** Easier to test and validate

---

## Scope & Limitations

This project **intentionally does not include:**

❌ **Streaming ingestion** - Batch processing only  
❌ **Data persistence** - No database or storage layer  
❌ **Dashboards/UI** - CLI and API only  
❌ **Alert delivery** - No Slack/PagerDuty/email integrations  
❌ **Multi-feature aggregation** - Single feature analysis only  
❌ **Model performance tracking** - Drift detection only, not accuracy monitoring  
❌ **Automated retraining** - Detection only, not remediation  
❌ **Historical trending** - No time-series analysis  

These omissions are **deliberate** to maintain focus on core drift detection logic. In a production system, these would be separate services/layers.

---

## Skills Demonstrated

This project showcases:
- ✅ **ML Systems Engineering** - Production-grade drift monitoring
- ✅ **Statistical Drift Detection** - PSI, KS, Chi-Square implementation
- ✅ **Test-Driven Development (TDD)** - Strict RED/GREEN discipline
- ✅ **Clean Architecture** - Modular design, separation of concerns
- ✅ **Configuration-Driven Design** - No hardcoded thresholds
- ✅ **Production-Grade Error Handling** - Explicit failures, clear messages
- ✅ **API Design** - CLI + Python API
- ✅ **Deterministic Systems** - Reproducible, testable behavior

**Suitable for roles:**
- Senior ML Engineer
- ML Platform Engineer
- MLOps Engineer

---

## License

MIT License - See [LICENSE](LICENSE) file for details.

---

## Development

### Running Tests
```bash
# Run all tests
pytest -v

# Run specific test file
pytest tests/test_metrics.py -v

# Run with coverage
pytest --cov=drift --cov-report=html
```

### Project Philosophy
- **TDD-first:** Tests define behavior before implementation
- **Simplicity:** Solve one problem well
- **Determinism:** No randomness, reproducible results
- **Explicitness:** Clear errors over silent failures

---

**Built with strict TDD discipline | 148/148 tests passing | Production-ready**
