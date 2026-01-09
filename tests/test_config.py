"""
Test suite for configuration-driven drift threshold resolution.

This module tests behavioral contracts for:
- Default threshold resolution
- Feature-specific overrides
- Configuration validation
- Deterministic behavior
"""

import pytest


def test_get_threshold_function_exists():
    """Test that get_threshold function exists in drift.config module."""
    from drift.config import get_threshold
    
    assert callable(get_threshold), "get_threshold must be a callable function"


def test_default_threshold_used_when_no_override():
    """Test that default threshold is used when feature has no override."""
    from drift.config import get_threshold
    
    config = {
        "metrics": {
            "psi": {
                "default_threshold": 0.1
            }
        }
    }
    
    threshold = get_threshold(config, metric="psi", feature="feature1")
    
    assert threshold == 0.1, "Should use default threshold when no feature override"


def test_feature_specific_override_supersedes_default():
    """Test that feature-specific threshold overrides default."""
    from drift.config import get_threshold
    
    config = {
        "metrics": {
            "psi": {
                "default_threshold": 0.1,
                "feature_thresholds": {
                    "feature1": 0.2
                }
            }
        }
    }
    
    threshold = get_threshold(config, metric="psi", feature="feature1")
    
    assert threshold == 0.2, "Should use feature-specific threshold over default"


def test_default_threshold_for_different_feature():
    """Test that default is used for features without specific override."""
    from drift.config import get_threshold
    
    config = {
        "metrics": {
            "psi": {
                "default_threshold": 0.1,
                "feature_thresholds": {
                    "feature1": 0.2
                }
            }
        }
    }
    
    threshold = get_threshold(config, metric="psi", feature="feature2")
    
    assert threshold == 0.1, "Should use default for features without override"


def test_raises_error_for_missing_metric_config():
    """Test that clear error is raised when metric config is missing."""
    from drift.config import get_threshold
    
    config = {
        "metrics": {
            "psi": {
                "default_threshold": 0.1
            }
        }
    }
    
    with pytest.raises(ValueError, match="metric.*not.*configured|unsupported.*metric"):
        get_threshold(config, metric="nonexistent_metric", feature="feature1")


def test_raises_error_for_missing_default_threshold():
    """Test that error is raised when default_threshold is missing."""
    from drift.config import get_threshold
    
    config = {
        "metrics": {
            "psi": {
                "feature_thresholds": {
                    "feature1": 0.2
                }
            }
        }
    }
    
    with pytest.raises(ValueError, match="default_threshold.*required|missing.*default"):
        get_threshold(config, metric="psi", feature="feature2")


def test_get_threshold_is_deterministic():
    """Test that get_threshold returns same result for same inputs."""
    from drift.config import get_threshold
    
    config = {
        "metrics": {
            "psi": {
                "default_threshold": 0.1,
                "feature_thresholds": {
                    "feature1": 0.2
                }
            }
        }
    }
    
    threshold1 = get_threshold(config, metric="psi", feature="feature1")
    threshold2 = get_threshold(config, metric="psi", feature="feature1")
    
    assert threshold1 == threshold2, "get_threshold must be deterministic"


def test_works_with_ks_metric():
    """Test that threshold resolution works with KS metric."""
    from drift.config import get_threshold
    
    config = {
        "metrics": {
            "ks": {
                "default_threshold": 0.3
            }
        }
    }
    
    threshold = get_threshold(config, metric="ks", feature="feature1")
    
    assert threshold == 0.3, "Should work with KS metric"


def test_works_with_chi_square_metric():
    """Test that threshold resolution works with Chi-Square metric."""
    from drift.config import get_threshold
    
    config = {
        "metrics": {
            "chi_square": {
                "default_threshold": 0.05
            }
        }
    }
    
    threshold = get_threshold(config, metric="chi_square", feature="feature1")
    
    assert threshold == 0.05, "Should work with Chi-Square metric"


def test_multiple_feature_overrides():
    """Test that multiple feature-specific overrides work correctly."""
    from drift.config import get_threshold
    
    config = {
        "metrics": {
            "psi": {
                "default_threshold": 0.1,
                "feature_thresholds": {
                    "feature1": 0.2,
                    "feature2": 0.15,
                    "feature3": 0.25
                }
            }
        }
    }
    
    assert get_threshold(config, metric="psi", feature="feature1") == 0.2
    assert get_threshold(config, metric="psi", feature="feature2") == 0.15
    assert get_threshold(config, metric="psi", feature="feature3") == 0.25
    assert get_threshold(config, metric="psi", feature="feature4") == 0.1


def test_config_with_multiple_metrics():
    """Test configuration with multiple metrics."""
    from drift.config import get_threshold
    
    config = {
        "metrics": {
            "psi": {
                "default_threshold": 0.1
            },
            "ks": {
                "default_threshold": 0.3
            },
            "chi_square": {
                "default_threshold": 0.05
            }
        }
    }
    
    assert get_threshold(config, metric="psi", feature="f1") == 0.1
    assert get_threshold(config, metric="ks", feature="f1") == 0.3
    assert get_threshold(config, metric="chi_square", feature="f1") == 0.05
