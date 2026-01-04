"""
Test suite for drift detection pipeline.

This module tests the core pipeline contract and ensures:
- Pipeline entry point exists
- Correct input/output contracts
- Expected return structure
"""

import pytest


def test_run_pipeline_exists():
    """Test that run_pipeline function exists in drift.pipeline module."""
    from drift.pipeline import run_pipeline
    
    assert callable(run_pipeline), "run_pipeline must be a callable function"


def test_run_pipeline_accepts_reference_and_current_data():
    """Test that run_pipeline accepts reference_data and current_data parameters."""
    from drift.pipeline import run_pipeline
    import pandas as pd
    
    # Create minimal test data
    reference_data = pd.DataFrame({"feature1": [1, 2, 3]})
    current_data = pd.DataFrame({"feature1": [4, 5, 6]})
    
    # This should not raise TypeError about missing arguments
    try:
        result = run_pipeline(reference_data, current_data)
    except TypeError as e:
        if "missing" in str(e) and "required positional argument" in str(e):
            pytest.fail(f"run_pipeline missing required parameters: {e}")
        # Other TypeErrors are acceptable at this stage
        pass


def test_run_pipeline_returns_dict_with_required_keys():
    """Test that run_pipeline returns a dict with drift_detected, metrics, and alerts."""
    from drift.pipeline import run_pipeline
    import pandas as pd
    
    reference_data = pd.DataFrame({"feature1": [1, 2, 3]})
    current_data = pd.DataFrame({"feature1": [4, 5, 6]})
    
    result = run_pipeline(reference_data, current_data)
    
    assert isinstance(result, dict), "run_pipeline must return a dictionary"
    assert "drift_detected" in result, "Result must contain 'drift_detected' key"
    assert "metrics" in result, "Result must contain 'metrics' key"
    assert "alerts" in result, "Result must contain 'alerts' key"


def test_run_pipeline_drift_detected_is_bool():
    """Test that drift_detected value is a boolean."""
    from drift.pipeline import run_pipeline
    import pandas as pd
    
    reference_data = pd.DataFrame({"feature1": [1, 2, 3]})
    current_data = pd.DataFrame({"feature1": [4, 5, 6]})
    
    result = run_pipeline(reference_data, current_data)
    
    assert isinstance(result["drift_detected"], bool), "drift_detected must be a boolean"


def test_run_pipeline_metrics_is_dict():
    """Test that metrics value is a dictionary."""
    from drift.pipeline import run_pipeline
    import pandas as pd
    
    reference_data = pd.DataFrame({"feature1": [1, 2, 3]})
    current_data = pd.DataFrame({"feature1": [4, 5, 6]})
    
    result = run_pipeline(reference_data, current_data)
    
    assert isinstance(result["metrics"], dict), "metrics must be a dictionary"


def test_run_pipeline_alerts_is_list():
    """Test that alerts value is a list."""
    from drift.pipeline import run_pipeline
    import pandas as pd
    
    reference_data = pd.DataFrame({"feature1": [1, 2, 3]})
    current_data = pd.DataFrame({"feature1": [4, 5, 6]})
    
    result = run_pipeline(reference_data, current_data)
    
    assert isinstance(result["alerts"], list), "alerts must be a list"
