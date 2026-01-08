"""
Test suite for drift metrics.

This module tests behavioral contracts for drift metrics:
- PSI (Population Stability Index)
- KS (Kolmogorov-Smirnov test)
- Chi-Square test
"""

import pytest
import numpy as np


# ============================================================================
# PSI (Population Stability Index) Tests
# ============================================================================

def test_calculate_psi_function_exists():
    """Test that calculate_psi function exists in drift.metrics module."""
    from drift.metrics import calculate_psi
    
    assert callable(calculate_psi), "calculate_psi must be a callable function"


def test_psi_returns_float():
    """Test that PSI returns a float value."""
    from drift.metrics import calculate_psi
    
    reference = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    current = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    
    result = calculate_psi(reference, current)
    
    assert isinstance(result, (float, np.floating)), "PSI must return a float"


def test_psi_returns_non_negative_value():
    """Test that PSI returns a value >= 0."""
    from drift.metrics import calculate_psi
    
    reference = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    current = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    
    result = calculate_psi(reference, current)
    
    assert result >= 0, "PSI must be non-negative"


def test_psi_is_deterministic():
    """Test that PSI returns same result for same inputs."""
    from drift.metrics import calculate_psi
    
    reference = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    current = np.array([15.0, 25.0, 35.0, 45.0, 55.0])
    
    result1 = calculate_psi(reference, current)
    result2 = calculate_psi(reference, current)
    result3 = calculate_psi(reference, current)
    
    assert result1 == result2, "PSI must be deterministic"
    assert result2 == result3, "PSI must be deterministic"


def test_psi_raises_error_for_empty_reference():
    """Test that PSI raises ValueError for empty reference data."""
    from drift.metrics import calculate_psi
    
    reference = np.array([])
    current = np.array([1.0, 2.0, 3.0])
    
    with pytest.raises(ValueError, match="reference.*empty|empty.*reference"):
        calculate_psi(reference, current)


def test_psi_raises_error_for_empty_current():
    """Test that PSI raises ValueError for empty current data."""
    from drift.metrics import calculate_psi
    
    reference = np.array([1.0, 2.0, 3.0])
    current = np.array([])
    
    with pytest.raises(ValueError, match="current.*empty|empty.*current"):
        calculate_psi(reference, current)


def test_psi_rejects_categorical_data():
    """Test that PSI raises TypeError for categorical/string data."""
    from drift.metrics import calculate_psi
    
    reference = np.array(["a", "b", "c", "d"])
    current = np.array(["e", "f", "g", "h"])
    
    with pytest.raises((TypeError, ValueError)):
        calculate_psi(reference, current)


# ============================================================================
# KS (Kolmogorov-Smirnov) Test
# ============================================================================

def test_calculate_ks_function_exists():
    """Test that calculate_ks function exists in drift.metrics module."""
    from drift.metrics import calculate_ks
    
    assert callable(calculate_ks), "calculate_ks must be a callable function"


def test_ks_returns_dict():
    """Test that KS returns a dictionary."""
    from drift.metrics import calculate_ks
    
    reference = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    current = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    
    result = calculate_ks(reference, current)
    
    assert isinstance(result, dict), "KS must return a dictionary"


def test_ks_dict_has_required_keys():
    """Test that KS result contains 'statistic' and 'p_value' keys."""
    from drift.metrics import calculate_ks
    
    reference = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    current = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    
    result = calculate_ks(reference, current)
    
    assert "statistic" in result, "KS result must contain 'statistic' key"
    assert "p_value" in result, "KS result must contain 'p_value' key"


def test_ks_statistic_is_float():
    """Test that KS statistic is a float."""
    from drift.metrics import calculate_ks
    
    reference = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    current = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    
    result = calculate_ks(reference, current)
    
    assert isinstance(result["statistic"], (float, np.floating)), \
        "KS statistic must be a float"


def test_ks_p_value_is_float():
    """Test that KS p_value is a float."""
    from drift.metrics import calculate_ks
    
    reference = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    current = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    
    result = calculate_ks(reference, current)
    
    assert isinstance(result["p_value"], (float, np.floating)), \
        "KS p_value must be a float"


def test_ks_statistic_in_valid_range():
    """Test that KS statistic is in [0, 1]."""
    from drift.metrics import calculate_ks
    
    reference = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    current = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    
    result = calculate_ks(reference, current)
    
    assert 0 <= result["statistic"] <= 1, "KS statistic must be in [0, 1]"


def test_ks_p_value_in_valid_range():
    """Test that KS p_value is in [0, 1]."""
    from drift.metrics import calculate_ks
    
    reference = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    current = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    
    result = calculate_ks(reference, current)
    
    assert 0 <= result["p_value"] <= 1, "KS p_value must be in [0, 1]"


def test_ks_is_deterministic():
    """Test that KS returns same result for same inputs."""
    from drift.metrics import calculate_ks
    
    reference = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    current = np.array([15.0, 25.0, 35.0, 45.0, 55.0])
    
    result1 = calculate_ks(reference, current)
    result2 = calculate_ks(reference, current)
    
    assert result1["statistic"] == result2["statistic"], "KS must be deterministic"
    assert result1["p_value"] == result2["p_value"], "KS must be deterministic"


def test_ks_raises_error_for_empty_reference():
    """Test that KS raises ValueError for empty reference data."""
    from drift.metrics import calculate_ks
    
    reference = np.array([])
    current = np.array([1.0, 2.0, 3.0])
    
    with pytest.raises(ValueError, match="reference.*empty|empty.*reference"):
        calculate_ks(reference, current)


def test_ks_raises_error_for_empty_current():
    """Test that KS raises ValueError for empty current data."""
    from drift.metrics import calculate_ks
    
    reference = np.array([1.0, 2.0, 3.0])
    current = np.array([])
    
    with pytest.raises(ValueError, match="current.*empty|empty.*current"):
        calculate_ks(reference, current)


def test_ks_rejects_categorical_data():
    """Test that KS raises TypeError for categorical/string data."""
    from drift.metrics import calculate_ks
    
    reference = np.array(["a", "b", "c", "d"])
    current = np.array(["e", "f", "g", "h"])
    
    with pytest.raises((TypeError, ValueError)):
        calculate_ks(reference, current)


# ============================================================================
# Chi-Square Test
# ============================================================================

def test_calculate_chi_square_function_exists():
    """Test that calculate_chi_square function exists in drift.metrics module."""
    from drift.metrics import calculate_chi_square
    
    assert callable(calculate_chi_square), "calculate_chi_square must be a callable function"


def test_chi_square_returns_dict():
    """Test that Chi-Square returns a dictionary."""
    from drift.metrics import calculate_chi_square
    
    reference = np.array(["a", "b", "c", "a", "b"])
    current = np.array(["a", "b", "c", "c", "b"])
    
    result = calculate_chi_square(reference, current)
    
    assert isinstance(result, dict), "Chi-Square must return a dictionary"


def test_chi_square_dict_has_required_keys():
    """Test that Chi-Square result contains 'statistic' and 'p_value' keys."""
    from drift.metrics import calculate_chi_square
    
    reference = np.array(["a", "b", "c", "a", "b"])
    current = np.array(["a", "b", "c", "c", "b"])
    
    result = calculate_chi_square(reference, current)
    
    assert "statistic" in result, "Chi-Square result must contain 'statistic' key"
    assert "p_value" in result, "Chi-Square result must contain 'p_value' key"


def test_chi_square_statistic_is_float():
    """Test that Chi-Square statistic is a float."""
    from drift.metrics import calculate_chi_square
    
    reference = np.array(["a", "b", "c", "a", "b"])
    current = np.array(["a", "b", "c", "c", "b"])
    
    result = calculate_chi_square(reference, current)
    
    assert isinstance(result["statistic"], (float, np.floating)), \
        "Chi-Square statistic must be a float"


def test_chi_square_p_value_is_float():
    """Test that Chi-Square p_value is a float."""
    from drift.metrics import calculate_chi_square
    
    reference = np.array(["a", "b", "c", "a", "b"])
    current = np.array(["a", "b", "c", "c", "b"])
    
    result = calculate_chi_square(reference, current)
    
    assert isinstance(result["p_value"], (float, np.floating)), \
        "Chi-Square p_value must be a float"


def test_chi_square_p_value_in_valid_range():
    """Test that Chi-Square p_value is in [0, 1]."""
    from drift.metrics import calculate_chi_square
    
    reference = np.array(["a", "b", "c", "a", "b"])
    current = np.array(["a", "b", "c", "c", "b"])
    
    result = calculate_chi_square(reference, current)
    
    assert 0 <= result["p_value"] <= 1, "Chi-Square p_value must be in [0, 1]"


def test_chi_square_is_deterministic():
    """Test that Chi-Square returns same result for same inputs."""
    from drift.metrics import calculate_chi_square
    
    reference = np.array(["x", "y", "z", "x", "y"])
    current = np.array(["x", "y", "z", "z", "y"])
    
    result1 = calculate_chi_square(reference, current)
    result2 = calculate_chi_square(reference, current)
    
    assert result1["statistic"] == result2["statistic"], "Chi-Square must be deterministic"
    assert result1["p_value"] == result2["p_value"], "Chi-Square must be deterministic"


def test_chi_square_raises_error_for_empty_reference():
    """Test that Chi-Square raises ValueError for empty reference data."""
    from drift.metrics import calculate_chi_square
    
    reference = np.array([])
    current = np.array(["a", "b", "c"])
    
    with pytest.raises(ValueError, match="reference.*empty|empty.*reference"):
        calculate_chi_square(reference, current)


def test_chi_square_raises_error_for_empty_current():
    """Test that Chi-Square raises ValueError for empty current data."""
    from drift.metrics import calculate_chi_square
    
    reference = np.array(["a", "b", "c"])
    current = np.array([])
    
    with pytest.raises(ValueError, match="current.*empty|empty.*current"):
        calculate_chi_square(reference, current)


def test_chi_square_rejects_continuous_numerical_data():
    """Test that Chi-Square raises TypeError for continuous numerical data."""
    from drift.metrics import calculate_chi_square
    
    reference = np.array([1.5, 2.7, 3.9, 4.2, 5.1])
    current = np.array([1.8, 2.3, 3.5, 4.7, 5.9])
    
    with pytest.raises((TypeError, ValueError)):
        calculate_chi_square(reference, current)

