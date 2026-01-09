"""
Test suite for drift detection pipeline.

This module tests the end-to-end drift monitoring pipeline behavior.
"""

import pytest
import pandas as pd
import numpy as np


def test_run_drift_pipeline_function_exists():
    """Test that run_drift_pipeline function exists in drift.pipeline module."""
    from drift.pipeline import run_drift_pipeline
    
    assert callable(run_drift_pipeline), "run_drift_pipeline must be a callable function"


def test_pipeline_returns_dict():
    """Test that pipeline returns a dictionary."""
    from drift.pipeline import run_drift_pipeline
    
    reference_data = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0, 5.0]})
    current_data = pd.DataFrame({"feature": [1.5, 2.5, 3.5, 4.5, 5.5]})
    
    result = run_drift_pipeline(
        reference_data,
        current_data,
        feature_type="numerical",
        metric="psi",
        threshold=0.1
    )
    
    assert isinstance(result, dict), "Pipeline must return a dictionary"


def test_pipeline_has_required_keys():
    """Test that pipeline result contains required keys."""
    from drift.pipeline import run_drift_pipeline
    
    reference_data = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0, 5.0]})
    current_data = pd.DataFrame({"feature": [1.5, 2.5, 3.5, 4.5, 5.5]})
    
    result = run_drift_pipeline(
        reference_data,
        current_data,
        feature_type="numerical",
        metric="psi",
        threshold=0.1
    )
    
    assert "drift_detected" in result, "Result must contain 'drift_detected' key"
    assert "alerts" in result, "Result must contain 'alerts' key"
    assert "metrics" in result, "Result must contain 'metrics' key"
    assert "window" in result, "Result must contain 'window' key"


def test_drift_detected_is_bool():
    """Test that drift_detected is a boolean."""
    from drift.pipeline import run_drift_pipeline
    
    reference_data = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0, 5.0]})
    current_data = pd.DataFrame({"feature": [1.5, 2.5, 3.5, 4.5, 5.5]})
    
    result = run_drift_pipeline(
        reference_data,
        current_data,
        feature_type="numerical",
        metric="psi",
        threshold=0.1
    )
    
    assert isinstance(result["drift_detected"], bool), "drift_detected must be a boolean"


def test_alerts_is_list():
    """Test that alerts is a list."""
    from drift.pipeline import run_drift_pipeline
    
    reference_data = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0, 5.0]})
    current_data = pd.DataFrame({"feature": [1.5, 2.5, 3.5, 4.5, 5.5]})
    
    result = run_drift_pipeline(
        reference_data,
        current_data,
        feature_type="numerical",
        metric="psi",
        threshold=0.1
    )
    
    assert isinstance(result["alerts"], list), "alerts must be a list"


def test_metrics_is_dict():
    """Test that metrics is a dictionary."""
    from drift.pipeline import run_drift_pipeline
    
    reference_data = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0, 5.0]})
    current_data = pd.DataFrame({"feature": [1.5, 2.5, 3.5, 4.5, 5.5]})
    
    result = run_drift_pipeline(
        reference_data,
        current_data,
        feature_type="numerical",
        metric="psi",
        threshold=0.1
    )
    
    assert isinstance(result["metrics"], dict), "metrics must be a dictionary"


def test_window_is_dict():
    """Test that window is a dictionary."""
    from drift.pipeline import run_drift_pipeline
    
    reference_data = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0, 5.0]})
    current_data = pd.DataFrame({"feature": [1.5, 2.5, 3.5, 4.5, 5.5]})
    
    result = run_drift_pipeline(
        reference_data,
        current_data,
        feature_type="numerical",
        metric="psi",
        threshold=0.1
    )
    
    assert isinstance(result["window"], dict), "window must be a dictionary"


def test_no_drift_scenario():
    """Test pipeline behavior when no drift is detected."""
    from drift.pipeline import run_drift_pipeline
    
    # Identical distributions - no drift
    reference_data = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0, 5.0]})
    current_data = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0, 5.0]})
    
    result = run_drift_pipeline(
        reference_data,
        current_data,
        feature_type="numerical",
        metric="psi",
        threshold=0.5
    )
    
    assert result["drift_detected"] is False, "Should not detect drift for identical distributions"
    assert result["alerts"] == [], "Should have no alerts when no drift detected"


def test_drift_detected_scenario():
    """Test pipeline behavior when drift is detected."""
    from drift.pipeline import run_drift_pipeline
    
    # Very different distributions - drift expected
    reference_data = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0, 5.0]})
    current_data = pd.DataFrame({"feature": [10.0, 20.0, 30.0, 40.0, 50.0]})
    
    result = run_drift_pipeline(
        reference_data,
        current_data,
        feature_type="numerical",
        metric="psi",
        threshold=0.01
    )
    
    assert result["drift_detected"] is True, "Should detect drift for very different distributions"
    assert len(result["alerts"]) > 0, "Should have alerts when drift detected"


def test_alert_has_severity_when_drift_detected():
    """Test that alerts contain severity field."""
    from drift.pipeline import run_drift_pipeline
    
    reference_data = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0, 5.0]})
    current_data = pd.DataFrame({"feature": [10.0, 20.0, 30.0, 40.0, 50.0]})
    
    result = run_drift_pipeline(
        reference_data,
        current_data,
        feature_type="numerical",
        metric="psi",
        threshold=0.01
    )
    
    if result["drift_detected"]:
        assert len(result["alerts"]) > 0
        alert = result["alerts"][0]
        assert "severity" in alert, "Alert must contain severity"
        assert alert["severity"] in ["warning", "critical"], "Severity must be warning or critical"


def test_metrics_contains_metric_name():
    """Test that metrics dict contains the requested metric."""
    from drift.pipeline import run_drift_pipeline
    
    reference_data = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0, 5.0]})
    current_data = pd.DataFrame({"feature": [1.5, 2.5, 3.5, 4.5, 5.5]})
    
    result = run_drift_pipeline(
        reference_data,
        current_data,
        feature_type="numerical",
        metric="psi",
        threshold=0.1
    )
    
    assert "psi" in result["metrics"], "Metrics should contain the requested metric name"


def test_pipeline_is_deterministic():
    """Test that pipeline returns same result for same inputs."""
    from drift.pipeline import run_drift_pipeline
    
    reference_data = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0, 5.0]})
    current_data = pd.DataFrame({"feature": [1.5, 2.5, 3.5, 4.5, 5.5]})
    
    result1 = run_drift_pipeline(
        reference_data,
        current_data,
        feature_type="numerical",
        metric="psi",
        threshold=0.1
    )
    
    result2 = run_drift_pipeline(
        reference_data,
        current_data,
        feature_type="numerical",
        metric="psi",
        threshold=0.1
    )
    
    assert result1["drift_detected"] == result2["drift_detected"], "Pipeline must be deterministic"
    assert len(result1["alerts"]) == len(result2["alerts"]), "Pipeline must be deterministic"


def test_pipeline_raises_error_for_empty_reference_data():
    """Test that pipeline raises ValueError for empty reference data."""
    from drift.pipeline import run_drift_pipeline
    
    reference_data = pd.DataFrame({"feature": []})
    current_data = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    
    with pytest.raises(ValueError, match="reference.*empty|empty.*reference"):
        run_drift_pipeline(
            reference_data,
            current_data,
            feature_type="numerical",
            metric="psi",
            threshold=0.1
        )


def test_pipeline_raises_error_for_empty_current_data():
    """Test that pipeline raises ValueError for empty current data."""
    from drift.pipeline import run_drift_pipeline
    
    reference_data = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    current_data = pd.DataFrame({"feature": []})
    
    with pytest.raises(ValueError, match="current.*empty|empty.*current"):
        run_drift_pipeline(
            reference_data,
            current_data,
            feature_type="numerical",
            metric="psi",
            threshold=0.1
        )


def test_pipeline_raises_error_for_unsupported_metric():
    """Test that pipeline raises ValueError for unsupported metric."""
    from drift.pipeline import run_drift_pipeline
    
    reference_data = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0, 5.0]})
    current_data = pd.DataFrame({"feature": [1.5, 2.5, 3.5, 4.5, 5.5]})
    
    with pytest.raises(ValueError, match="unsupported.*metric|metric.*not.*supported"):
        run_drift_pipeline(
            reference_data,
            current_data,
            feature_type="numerical",
            metric="invalid_metric",
            threshold=0.1
        )


def test_pipeline_with_ks_metric():
    """Test pipeline works with KS metric."""
    from drift.pipeline import run_drift_pipeline
    
    reference_data = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0, 5.0]})
    current_data = pd.DataFrame({"feature": [1.5, 2.5, 3.5, 4.5, 5.5]})
    
    result = run_drift_pipeline(
        reference_data,
        current_data,
        feature_type="numerical",
        metric="ks",
        threshold=0.3
    )
    
    assert "ks" in result["metrics"], "Metrics should contain KS metric"
    assert isinstance(result["drift_detected"], bool)


def test_pipeline_with_chi_square_metric():
    """Test pipeline works with Chi-Square metric."""
    from drift.pipeline import run_drift_pipeline
    
    reference_data = pd.DataFrame({"feature": ["a", "b", "c", "a", "b"]})
    current_data = pd.DataFrame({"feature": ["a", "b", "c", "c", "b"]})
    
    result = run_drift_pipeline(
        reference_data,
        current_data,
        feature_type="categorical",
        metric="chi_square",
        threshold=0.05
    )
    
    assert "chi_square" in result["metrics"], "Metrics should contain Chi-Square metric"
    assert isinstance(result["drift_detected"], bool)
