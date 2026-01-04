"""
Test suite for windowing strategies.

This module tests window selection behavior for drift detection:
- Reference window selection (first N rows)
- Current window selection (last N rows)
- Window size validation
- Deterministic behavior
"""

import pytest
import pandas as pd


def test_get_windows_function_exists():
    """Test that get_windows function exists in drift.windows module."""
    from drift.windows import get_windows
    
    assert callable(get_windows), "get_windows must be a callable function"


def test_reference_window_selects_first_n_rows():
    """Test that reference window consists of the FIRST N rows."""
    from drift.windows import get_windows
    
    # Create test dataset with identifiable rows
    data = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feature2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    })
    
    reference_size = 4
    current_size = 3
    
    reference_window, current_window = get_windows(data, reference_size, current_size)
    
    # Reference window should be first 4 rows
    assert len(reference_window) == 4, "Reference window must have correct size"
    assert list(reference_window["feature1"]) == [1, 2, 3, 4], \
        "Reference window must contain first N rows in order"


def test_current_window_selects_last_n_rows():
    """Test that current window consists of the LAST N rows."""
    from drift.windows import get_windows
    
    # Create test dataset with identifiable rows
    data = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feature2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    })
    
    reference_size = 4
    current_size = 3
    
    reference_window, current_window = get_windows(data, reference_size, current_size)
    
    # Current window should be last 3 rows
    assert len(current_window) == 3, "Current window must have correct size"
    assert list(current_window["feature1"]) == [8, 9, 10], \
        "Current window must contain last N rows in order"


def test_windows_preserve_order():
    """Test that both windows preserve row order from original dataset."""
    from drift.windows import get_windows
    
    data = pd.DataFrame({
        "feature1": [100, 200, 300, 400, 500],
        "feature2": ["a", "b", "c", "d", "e"]
    })
    
    reference_window, current_window = get_windows(data, 2, 2)
    
    # Reference: first 2 rows in order
    assert list(reference_window["feature1"]) == [100, 200]
    assert list(reference_window["feature2"]) == ["a", "b"]
    
    # Current: last 2 rows in order
    assert list(current_window["feature1"]) == [400, 500]
    assert list(current_window["feature2"]) == ["d", "e"]


def test_get_windows_raises_error_for_zero_reference_size():
    """Test that ValueError is raised when reference_size <= 0."""
    from drift.windows import get_windows
    
    data = pd.DataFrame({"feature1": [1, 2, 3, 4, 5]})
    
    with pytest.raises(ValueError, match="reference_size must be greater than 0"):
        get_windows(data, reference_size=0, current_size=2)


def test_get_windows_raises_error_for_negative_reference_size():
    """Test that ValueError is raised when reference_size is negative."""
    from drift.windows import get_windows
    
    data = pd.DataFrame({"feature1": [1, 2, 3, 4, 5]})
    
    with pytest.raises(ValueError, match="reference_size must be greater than 0"):
        get_windows(data, reference_size=-5, current_size=2)


def test_get_windows_raises_error_for_zero_current_size():
    """Test that ValueError is raised when current_size <= 0."""
    from drift.windows import get_windows
    
    data = pd.DataFrame({"feature1": [1, 2, 3, 4, 5]})
    
    with pytest.raises(ValueError, match="current_size must be greater than 0"):
        get_windows(data, reference_size=2, current_size=0)


def test_get_windows_raises_error_for_negative_current_size():
    """Test that ValueError is raised when current_size is negative."""
    from drift.windows import get_windows
    
    data = pd.DataFrame({"feature1": [1, 2, 3, 4, 5]})
    
    with pytest.raises(ValueError, match="current_size must be greater than 0"):
        get_windows(data, reference_size=2, current_size=-3)


def test_get_windows_raises_error_when_reference_size_exceeds_data_size():
    """Test that ValueError is raised when reference_size > dataset size."""
    from drift.windows import get_windows
    
    data = pd.DataFrame({"feature1": [1, 2, 3]})
    
    with pytest.raises(ValueError, match="reference_size.*exceeds.*data"):
        get_windows(data, reference_size=5, current_size=2)


def test_get_windows_raises_error_when_current_size_exceeds_data_size():
    """Test that ValueError is raised when current_size > dataset size."""
    from drift.windows import get_windows
    
    data = pd.DataFrame({"feature1": [1, 2, 3, 4]})
    
    with pytest.raises(ValueError, match="current_size.*exceeds.*data"):
        get_windows(data, reference_size=2, current_size=10)


def test_get_windows_is_deterministic():
    """Test that calling get_windows multiple times returns identical results."""
    from drift.windows import get_windows
    
    data = pd.DataFrame({
        "feature1": [10, 20, 30, 40, 50, 60],
        "feature2": [1, 2, 3, 4, 5, 6]
    })
    
    # Call get_windows multiple times
    ref1, curr1 = get_windows(data, 3, 2)
    ref2, curr2 = get_windows(data, 3, 2)
    ref3, curr3 = get_windows(data, 3, 2)
    
    # All reference windows should be identical
    assert ref1.equals(ref2), "Reference windows must be deterministic"
    assert ref2.equals(ref3), "Reference windows must be deterministic"
    
    # All current windows should be identical
    assert curr1.equals(curr2), "Current windows must be deterministic"
    assert curr2.equals(curr3), "Current windows must be deterministic"


def test_get_windows_returns_tuple():
    """Test that get_windows returns a tuple."""
    from drift.windows import get_windows
    
    data = pd.DataFrame({"feature1": [1, 2, 3, 4, 5]})
    
    result = get_windows(data, 2, 2)
    
    assert isinstance(result, tuple), "get_windows must return a tuple"
    assert len(result) == 2, "get_windows must return a tuple of length 2"


def test_get_windows_with_single_column_dataframe():
    """Test that get_windows works with single-column DataFrames."""
    from drift.windows import get_windows
    
    data = pd.DataFrame({"col": [1, 2, 3, 4, 5, 6, 7]})
    
    reference_window, current_window = get_windows(data, 3, 2)
    
    assert list(reference_window["col"]) == [1, 2, 3]
    assert list(current_window["col"]) == [6, 7]


def test_get_windows_with_multi_column_dataframe():
    """Test that get_windows works with multi-column DataFrames."""
    from drift.windows import get_windows
    
    data = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [10, 20, 30, 40, 50],
        "c": ["x", "y", "z", "w", "v"]
    })
    
    reference_window, current_window = get_windows(data, 2, 2)
    
    # Check all columns are preserved
    assert list(reference_window.columns) == ["a", "b", "c"]
    assert list(current_window.columns) == ["a", "b", "c"]
    
    # Check reference window (first 2 rows)
    assert list(reference_window["a"]) == [1, 2]
    assert list(reference_window["b"]) == [10, 20]
    assert list(reference_window["c"]) == ["x", "y"]
    
    # Check current window (last 2 rows)
    assert list(current_window["a"]) == [4, 5]
    assert list(current_window["b"]) == [40, 50]
    assert list(current_window["c"]) == ["w", "v"]


def test_get_windows_with_equal_reference_and_current_sizes():
    """Test that get_windows works when reference_size == current_size."""
    from drift.windows import get_windows
    
    data = pd.DataFrame({"feature": [1, 2, 3, 4, 5, 6]})
    
    reference_window, current_window = get_windows(data, 3, 3)
    
    assert len(reference_window) == 3
    assert len(current_window) == 3
    assert list(reference_window["feature"]) == [1, 2, 3]
    assert list(current_window["feature"]) == [4, 5, 6]

