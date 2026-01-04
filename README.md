# ML Drift Monitoring System

A production-grade ML drift monitoring and detection system built with strict Test-Driven Development (TDD) methodology.

## Project Status

**Current Phase:** Cycle 0 - RED (Project Initialization)

This project implements custom drift detection algorithms without relying on third-party drift detection libraries like Evidently. The system is designed to be modular, extensible, and production-ready.

## Architecture

The system is organized into the following modules:

- `drift/windows.py` - Windowing strategies for drift detection
- `drift/metrics.py` - Drift metrics (PSI, KS Test, Chi-Square, etc.)
- `drift/detectors.py` - Drift detection logic
- `drift/alerts.py` - Alert generation and management
- `drift/pipeline.py` - Main drift detection pipeline

## Installation

```bash
pip install -r requirements.txt
```

## Development

This project follows strict TDD methodology. All features are developed through RED → GREEN → REFACTOR cycles.

### Running Tests

```bash
pytest -v
```

## License

See LICENSE file for details.
