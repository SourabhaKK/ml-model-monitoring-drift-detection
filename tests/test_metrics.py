"""
Test suite for drift metrics.

This module tests that the metrics module is importable and ready for future implementation.
"""

import pytest


def test_metrics_module_importable():
    """Test that drift.metrics module can be imported."""
    try:
        import drift.metrics
    except ImportError as e:
        pytest.fail(f"Failed to import drift.metrics: {e}")


def test_metrics_module_exists():
    """Test that drift.metrics module exists and is accessible."""
    import drift.metrics
    assert drift.metrics is not None, "drift.metrics module should exist"
