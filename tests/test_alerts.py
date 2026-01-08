"""
Test suite for alert generation.

This module tests behavioral contracts for alert generation:
- No drift → no alert
- Drift detected → alert generated
- Severity mapping
"""

import pytest


def test_generate_alert_function_exists():
    """Test that generate_alert function exists in drift.alerts module."""
    from drift.alerts import generate_alert
    
    assert callable(generate_alert), "generate_alert must be a callable function"


def test_no_alert_when_no_drift_detected():
    """Test that no alert is generated when drift_detected is False."""
    from drift.alerts import generate_alert
    
    detector_output = {
        "drift_detected": False,
        "metric": "psi",
        "value": 0.05,
        "threshold": 0.1
    }
    
    result = generate_alert(detector_output)
    
    assert result is None, "Should return None when drift_detected is False"


def test_alert_generated_when_drift_detected():
    """Test that alert is generated when drift_detected is True."""
    from drift.alerts import generate_alert
    
    detector_output = {
        "drift_detected": True,
        "metric": "psi",
        "value": 0.15,
        "threshold": 0.1
    }
    
    result = generate_alert(detector_output)
    
    assert result is not None, "Should return alert when drift_detected is True"
    assert isinstance(result, dict), "Alert must be a dictionary"


def test_alert_has_required_keys():
    """Test that alert contains required keys."""
    from drift.alerts import generate_alert
    
    detector_output = {
        "drift_detected": True,
        "metric": "psi",
        "value": 0.15,
        "threshold": 0.1
    }
    
    result = generate_alert(detector_output)
    
    assert "alert" in result, "Alert must contain 'alert' key"
    assert "severity" in result, "Alert must contain 'severity' key"
    assert "metric" in result, "Alert must contain 'metric' key"
    assert "message" in result, "Alert must contain 'message' key"
    assert "details" in result, "Alert must contain 'details' key"


def test_alert_field_is_always_true():
    """Test that alert field is always True when alert is generated."""
    from drift.alerts import generate_alert
    
    detector_output = {
        "drift_detected": True,
        "metric": "psi",
        "value": 0.15,
        "threshold": 0.1
    }
    
    result = generate_alert(detector_output)
    
    assert result["alert"] is True, "alert field must be True"


def test_alert_metric_matches_detector_metric():
    """Test that alert metric matches detector output metric."""
    from drift.alerts import generate_alert
    
    detector_output = {
        "drift_detected": True,
        "metric": "ks",
        "statistic": 0.3,
        "p_value": 0.01,
        "threshold": 0.2
    }
    
    result = generate_alert(detector_output)
    
    assert result["metric"] == "ks", "Alert metric must match detector metric"


def test_alert_message_exists_and_contains_metric():
    """Test that alert message exists and contains metric name."""
    from drift.alerts import generate_alert
    
    detector_output = {
        "drift_detected": True,
        "metric": "psi",
        "value": 0.15,
        "threshold": 0.1
    }
    
    result = generate_alert(detector_output)
    
    assert isinstance(result["message"], str), "Message must be a string"
    assert len(result["message"]) > 0, "Message must not be empty"
    assert "psi" in result["message"].lower(), "Message should contain metric name"


def test_alert_details_is_dict():
    """Test that alert details is a dictionary."""
    from drift.alerts import generate_alert
    
    detector_output = {
        "drift_detected": True,
        "metric": "psi",
        "value": 0.15,
        "threshold": 0.1
    }
    
    result = generate_alert(detector_output)
    
    assert isinstance(result["details"], dict), "Details must be a dictionary"


def test_severity_warning_when_value_below_double_threshold():
    """Test that severity is 'warning' when value <= 2 * threshold."""
    from drift.alerts import generate_alert
    
    detector_output = {
        "drift_detected": True,
        "metric": "psi",
        "value": 0.15,
        "threshold": 0.1
    }
    
    result = generate_alert(detector_output)
    
    assert result["severity"] == "warning", "Severity should be 'warning' when value <= 2 * threshold"


def test_severity_critical_when_value_exceeds_double_threshold():
    """Test that severity is 'critical' when value > 2 * threshold."""
    from drift.alerts import generate_alert
    
    detector_output = {
        "drift_detected": True,
        "metric": "psi",
        "value": 0.25,
        "threshold": 0.1
    }
    
    result = generate_alert(detector_output)
    
    assert result["severity"] == "critical", "Severity should be 'critical' when value > 2 * threshold"


def test_severity_warning_at_boundary():
    """Test that severity is 'warning' when value exactly equals 2 * threshold."""
    from drift.alerts import generate_alert
    
    detector_output = {
        "drift_detected": True,
        "metric": "psi",
        "value": 0.2,
        "threshold": 0.1
    }
    
    result = generate_alert(detector_output)
    
    assert result["severity"] == "warning", "Severity should be 'warning' when value == 2 * threshold"


def test_generate_alert_is_deterministic():
    """Test that generate_alert returns same result for same input."""
    from drift.alerts import generate_alert
    
    detector_output = {
        "drift_detected": True,
        "metric": "psi",
        "value": 0.18,
        "threshold": 0.1
    }
    
    result1 = generate_alert(detector_output)
    result2 = generate_alert(detector_output)
    
    assert result1 == result2, "generate_alert must be deterministic"


def test_generate_alert_raises_error_for_missing_drift_detected_key():
    """Test that generate_alert raises ValueError when drift_detected key is missing."""
    from drift.alerts import generate_alert
    
    detector_output = {
        "metric": "psi",
        "value": 0.15,
        "threshold": 0.1
    }
    
    with pytest.raises(ValueError, match="drift_detected.*required|required.*drift_detected"):
        generate_alert(detector_output)


def test_generate_alert_raises_error_for_missing_metric_key():
    """Test that generate_alert raises ValueError when metric key is missing."""
    from drift.alerts import generate_alert
    
    detector_output = {
        "drift_detected": True,
        "value": 0.15,
        "threshold": 0.1
    }
    
    with pytest.raises(ValueError, match="metric.*required|required.*metric"):
        generate_alert(detector_output)


def test_generate_alert_raises_error_for_non_dict_input():
    """Test that generate_alert raises TypeError for non-dict input."""
    from drift.alerts import generate_alert
    
    with pytest.raises(TypeError):
        generate_alert("not a dict")


def test_generate_alert_raises_error_for_none_input():
    """Test that generate_alert raises TypeError for None input."""
    from drift.alerts import generate_alert
    
    with pytest.raises(TypeError):
        generate_alert(None)


def test_alert_for_ks_detector_output():
    """Test that alert works correctly for KS detector output."""
    from drift.alerts import generate_alert
    
    detector_output = {
        "drift_detected": True,
        "metric": "ks",
        "statistic": 0.35,
        "p_value": 0.02,
        "threshold": 0.2
    }
    
    result = generate_alert(detector_output)
    
    assert result is not None
    assert result["metric"] == "ks"
    assert result["severity"] == "warning"  # 0.35 <= 2 * 0.2


def test_alert_for_chi_square_detector_output():
    """Test that alert works correctly for Chi-Square detector output."""
    from drift.alerts import generate_alert
    
    detector_output = {
        "drift_detected": True,
        "metric": "chi_square",
        "statistic": 20.5,
        "p_value": 0.01,
        "threshold": 0.05
    }
    
    result = generate_alert(detector_output)
    
    assert result is not None
    assert result["metric"] == "chi_square"

