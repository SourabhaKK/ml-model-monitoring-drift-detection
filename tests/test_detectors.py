"""
Test suite for drift detectors.

This module tests behavioral contracts for drift detectors:
- PSI detector
- KS detector
- Chi-Square detector
"""

import pytest


# ============================================================================
# PSI Detector Tests
# ============================================================================

def test_detect_psi_drift_function_exists():
    """Test that detect_psi_drift function exists in drift.detectors module."""
    from drift.detectors import detect_psi_drift
    
    assert callable(detect_psi_drift), "detect_psi_drift must be a callable function"


def test_psi_detector_returns_dict():
    """Test that PSI detector returns a dictionary."""
    from drift.detectors import detect_psi_drift
    
    result = detect_psi_drift(psi_value=0.15, threshold=0.1)
    
    assert isinstance(result, dict), "PSI detector must return a dictionary"


def test_psi_detector_has_required_keys():
    """Test that PSI detector result contains required keys."""
    from drift.detectors import detect_psi_drift
    
    result = detect_psi_drift(psi_value=0.15, threshold=0.1)
    
    assert "drift_detected" in result, "Result must contain 'drift_detected' key"
    assert "metric" in result, "Result must contain 'metric' key"
    assert "value" in result, "Result must contain 'value' key"
    assert "threshold" in result, "Result must contain 'threshold' key"


def test_psi_detector_drift_detected_is_bool():
    """Test that drift_detected is a boolean."""
    from drift.detectors import detect_psi_drift
    
    result = detect_psi_drift(psi_value=0.15, threshold=0.1)
    
    assert isinstance(result["drift_detected"], bool), "drift_detected must be a boolean"


def test_psi_detector_metric_name():
    """Test that metric field is 'psi'."""
    from drift.detectors import detect_psi_drift
    
    result = detect_psi_drift(psi_value=0.15, threshold=0.1)
    
    assert result["metric"] == "psi", "metric must be 'psi'"


def test_psi_detector_detects_drift_when_value_exceeds_threshold():
    """Test that PSI detector detects drift when value > threshold."""
    from drift.detectors import detect_psi_drift
    
    result = detect_psi_drift(psi_value=0.15, threshold=0.1)
    
    assert result["drift_detected"] is True, "Should detect drift when psi_value > threshold"


def test_psi_detector_no_drift_when_value_below_threshold():
    """Test that PSI detector does not detect drift when value <= threshold."""
    from drift.detectors import detect_psi_drift
    
    result = detect_psi_drift(psi_value=0.05, threshold=0.1)
    
    assert result["drift_detected"] is False, "Should not detect drift when psi_value <= threshold"


def test_psi_detector_no_drift_when_value_equals_threshold():
    """Test that PSI detector does not detect drift when value equals threshold."""
    from drift.detectors import detect_psi_drift
    
    result = detect_psi_drift(psi_value=0.1, threshold=0.1)
    
    assert result["drift_detected"] is False, "Should not detect drift when psi_value == threshold"


def test_psi_detector_returns_correct_value():
    """Test that detector returns the input psi_value."""
    from drift.detectors import detect_psi_drift
    
    result = detect_psi_drift(psi_value=0.25, threshold=0.1)
    
    assert result["value"] == 0.25, "value must match input psi_value"


def test_psi_detector_returns_correct_threshold():
    """Test that detector returns the input threshold."""
    from drift.detectors import detect_psi_drift
    
    result = detect_psi_drift(psi_value=0.25, threshold=0.1)
    
    assert result["threshold"] == 0.1, "threshold must match input threshold"


def test_psi_detector_raises_error_for_negative_threshold():
    """Test that PSI detector raises ValueError for threshold <= 0."""
    from drift.detectors import detect_psi_drift
    
    with pytest.raises(ValueError, match="threshold must be greater than 0"):
        detect_psi_drift(psi_value=0.15, threshold=-0.1)


def test_psi_detector_raises_error_for_zero_threshold():
    """Test that PSI detector raises ValueError for threshold = 0."""
    from drift.detectors import detect_psi_drift
    
    with pytest.raises(ValueError, match="threshold must be greater than 0"):
        detect_psi_drift(psi_value=0.15, threshold=0)


def test_psi_detector_is_deterministic():
    """Test that PSI detector returns same result for same inputs."""
    from drift.detectors import detect_psi_drift
    
    result1 = detect_psi_drift(psi_value=0.2, threshold=0.15)
    result2 = detect_psi_drift(psi_value=0.2, threshold=0.15)
    
    assert result1 == result2, "PSI detector must be deterministic"


# ============================================================================
# KS Detector Tests
# ============================================================================

def test_detect_ks_drift_function_exists():
    """Test that detect_ks_drift function exists in drift.detectors module."""
    from drift.detectors import detect_ks_drift
    
    assert callable(detect_ks_drift), "detect_ks_drift must be a callable function"


def test_ks_detector_returns_dict():
    """Test that KS detector returns a dictionary."""
    from drift.detectors import detect_ks_drift
    
    result = detect_ks_drift(statistic=0.3, p_value=0.01, threshold=0.2)
    
    assert isinstance(result, dict), "KS detector must return a dictionary"


def test_ks_detector_has_required_keys():
    """Test that KS detector result contains required keys."""
    from drift.detectors import detect_ks_drift
    
    result = detect_ks_drift(statistic=0.3, p_value=0.01, threshold=0.2)
    
    assert "drift_detected" in result, "Result must contain 'drift_detected' key"
    assert "metric" in result, "Result must contain 'metric' key"
    assert "statistic" in result, "Result must contain 'statistic' key"
    assert "p_value" in result, "Result must contain 'p_value' key"
    assert "threshold" in result, "Result must contain 'threshold' key"


def test_ks_detector_metric_name():
    """Test that metric field is 'ks'."""
    from drift.detectors import detect_ks_drift
    
    result = detect_ks_drift(statistic=0.3, p_value=0.01, threshold=0.2)
    
    assert result["metric"] == "ks", "metric must be 'ks'"


def test_ks_detector_detects_drift_when_statistic_exceeds_threshold():
    """Test that KS detector detects drift when statistic > threshold."""
    from drift.detectors import detect_ks_drift
    
    result = detect_ks_drift(statistic=0.3, p_value=0.01, threshold=0.2)
    
    assert result["drift_detected"] is True, "Should detect drift when statistic > threshold"


def test_ks_detector_no_drift_when_statistic_below_threshold():
    """Test that KS detector does not detect drift when statistic <= threshold."""
    from drift.detectors import detect_ks_drift
    
    result = detect_ks_drift(statistic=0.1, p_value=0.5, threshold=0.2)
    
    assert result["drift_detected"] is False, "Should not detect drift when statistic <= threshold"


def test_ks_detector_no_drift_when_statistic_equals_threshold():
    """Test that KS detector does not detect drift when statistic equals threshold."""
    from drift.detectors import detect_ks_drift
    
    result = detect_ks_drift(statistic=0.2, p_value=0.1, threshold=0.2)
    
    assert result["drift_detected"] is False, "Should not detect drift when statistic == threshold"


def test_ks_detector_returns_correct_values():
    """Test that detector returns the input values."""
    from drift.detectors import detect_ks_drift
    
    result = detect_ks_drift(statistic=0.35, p_value=0.02, threshold=0.25)
    
    assert result["statistic"] == 0.35, "statistic must match input"
    assert result["p_value"] == 0.02, "p_value must match input"
    assert result["threshold"] == 0.25, "threshold must match input"


def test_ks_detector_raises_error_for_negative_threshold():
    """Test that KS detector raises ValueError for threshold <= 0."""
    from drift.detectors import detect_ks_drift
    
    with pytest.raises(ValueError, match="threshold must be greater than 0"):
        detect_ks_drift(statistic=0.3, p_value=0.01, threshold=-0.1)


def test_ks_detector_raises_error_for_zero_threshold():
    """Test that KS detector raises ValueError for threshold = 0."""
    from drift.detectors import detect_ks_drift
    
    with pytest.raises(ValueError, match="threshold must be greater than 0"):
        detect_ks_drift(statistic=0.3, p_value=0.01, threshold=0)


def test_ks_detector_is_deterministic():
    """Test that KS detector returns same result for same inputs."""
    from drift.detectors import detect_ks_drift
    
    result1 = detect_ks_drift(statistic=0.4, p_value=0.03, threshold=0.3)
    result2 = detect_ks_drift(statistic=0.4, p_value=0.03, threshold=0.3)
    
    assert result1 == result2, "KS detector must be deterministic"


# ============================================================================
# Chi-Square Detector Tests
# ============================================================================

def test_detect_chi_square_drift_function_exists():
    """Test that detect_chi_square_drift function exists in drift.detectors module."""
    from drift.detectors import detect_chi_square_drift
    
    assert callable(detect_chi_square_drift), "detect_chi_square_drift must be a callable function"


def test_chi_square_detector_returns_dict():
    """Test that Chi-Square detector returns a dictionary."""
    from drift.detectors import detect_chi_square_drift
    
    result = detect_chi_square_drift(statistic=15.5, p_value=0.02, threshold=0.05)
    
    assert isinstance(result, dict), "Chi-Square detector must return a dictionary"


def test_chi_square_detector_has_required_keys():
    """Test that Chi-Square detector result contains required keys."""
    from drift.detectors import detect_chi_square_drift
    
    result = detect_chi_square_drift(statistic=15.5, p_value=0.02, threshold=0.05)
    
    assert "drift_detected" in result, "Result must contain 'drift_detected' key"
    assert "metric" in result, "Result must contain 'metric' key"
    assert "statistic" in result, "Result must contain 'statistic' key"
    assert "p_value" in result, "Result must contain 'p_value' key"
    assert "threshold" in result, "Result must contain 'threshold' key"


def test_chi_square_detector_metric_name():
    """Test that metric field is 'chi_square'."""
    from drift.detectors import detect_chi_square_drift
    
    result = detect_chi_square_drift(statistic=15.5, p_value=0.02, threshold=0.05)
    
    assert result["metric"] == "chi_square", "metric must be 'chi_square'"


def test_chi_square_detector_detects_drift_when_p_value_below_threshold():
    """Test that Chi-Square detector detects drift when p_value < threshold."""
    from drift.detectors import detect_chi_square_drift
    
    result = detect_chi_square_drift(statistic=15.5, p_value=0.02, threshold=0.05)
    
    assert result["drift_detected"] is True, "Should detect drift when p_value < threshold"


def test_chi_square_detector_no_drift_when_p_value_above_threshold():
    """Test that Chi-Square detector does not detect drift when p_value >= threshold."""
    from drift.detectors import detect_chi_square_drift
    
    result = detect_chi_square_drift(statistic=5.2, p_value=0.1, threshold=0.05)
    
    assert result["drift_detected"] is False, "Should not detect drift when p_value >= threshold"


def test_chi_square_detector_no_drift_when_p_value_equals_threshold():
    """Test that Chi-Square detector does not detect drift when p_value equals threshold."""
    from drift.detectors import detect_chi_square_drift
    
    result = detect_chi_square_drift(statistic=10.0, p_value=0.05, threshold=0.05)
    
    assert result["drift_detected"] is False, "Should not detect drift when p_value == threshold"


def test_chi_square_detector_returns_correct_values():
    """Test that detector returns the input values."""
    from drift.detectors import detect_chi_square_drift
    
    result = detect_chi_square_drift(statistic=20.5, p_value=0.01, threshold=0.05)
    
    assert result["statistic"] == 20.5, "statistic must match input"
    assert result["p_value"] == 0.01, "p_value must match input"
    assert result["threshold"] == 0.05, "threshold must match input"


def test_chi_square_detector_raises_error_for_negative_threshold():
    """Test that Chi-Square detector raises ValueError for threshold <= 0."""
    from drift.detectors import detect_chi_square_drift
    
    with pytest.raises(ValueError, match="threshold must be greater than 0"):
        detect_chi_square_drift(statistic=15.5, p_value=0.02, threshold=-0.05)


def test_chi_square_detector_raises_error_for_zero_threshold():
    """Test that Chi-Square detector raises ValueError for threshold = 0."""
    from drift.detectors import detect_chi_square_drift
    
    with pytest.raises(ValueError, match="threshold must be greater than 0"):
        detect_chi_square_drift(statistic=15.5, p_value=0.02, threshold=0)


def test_chi_square_detector_is_deterministic():
    """Test that Chi-Square detector returns same result for same inputs."""
    from drift.detectors import detect_chi_square_drift
    
    result1 = detect_chi_square_drift(statistic=18.3, p_value=0.03, threshold=0.05)
    result2 = detect_chi_square_drift(statistic=18.3, p_value=0.03, threshold=0.05)
    
    assert result1 == result2, "Chi-Square detector must be deterministic"

