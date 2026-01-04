"""
Test suite for drift detectors.

This module tests that the detectors module is importable and ready for future implementation.
"""

import pytest


def test_detectors_module_importable():
    """Test that drift.detectors module can be imported."""
    try:
        import drift.detectors
    except ImportError as e:
        pytest.fail(f"Failed to import drift.detectors: {e}")


def test_detectors_module_exists():
    """Test that drift.detectors module exists and is accessible."""
    import drift.detectors
    assert drift.detectors is not None, "drift.detectors module should exist"
