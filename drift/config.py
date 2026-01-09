# Configuration management for drift monitoring


def get_threshold(config, metric, feature):
    """
    Resolve drift threshold from configuration.
    
    Resolution order:
    1. Feature-specific override (if exists)
    2. Metric default threshold
    
    Args:
        config: Configuration dictionary
        metric: Metric name (e.g., "psi", "ks", "chi_square")
        feature: Feature name
        
    Returns:
        float: Resolved threshold value
        
    Raises:
        ValueError: If metric is not configured or default_threshold is missing
    """
    # Validate metric exists in config
    if "metrics" not in config:
        raise ValueError("metrics configuration is required")
    
    if metric not in config["metrics"]:
        raise ValueError(f"metric '{metric}' is not configured")
    
    metric_config = config["metrics"][metric]
    
    # Check for feature-specific override
    if "feature_thresholds" in metric_config:
        if feature in metric_config["feature_thresholds"]:
            return metric_config["feature_thresholds"][feature]
    
    # Fall back to default threshold
    if "default_threshold" not in metric_config:
        raise ValueError(f"default_threshold is required for metric '{metric}'")
    
    return metric_config["default_threshold"]
