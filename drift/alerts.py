# Alert generation and management


def generate_alert(detector_output):
    """
    Generate alert from drift detector output.
    
    Args:
        detector_output: Dictionary containing detector results
        
    Returns:
        dict | None: Alert dictionary if drift detected, None otherwise
        
    Raises:
        TypeError: If detector_output is not a dict
        ValueError: If required keys are missing
    """
    # Validate input type
    if not isinstance(detector_output, dict):
        raise TypeError("detector_output must be a dictionary")
    
    # Validate required keys
    if "drift_detected" not in detector_output:
        raise ValueError("drift_detected key is required")
    if "metric" not in detector_output:
        raise ValueError("metric key is required")
    
    # No alert if no drift detected
    if not detector_output["drift_detected"]:
        return None
    
    # Extract metric type
    metric = detector_output["metric"]
    
    # Determine severity based on value vs threshold
    # For PSI: use "value" and "threshold"
    # For KS: use "statistic" and "threshold"
    # For Chi-Square: p_value based, but we still need value for severity
    
    # Get the relevant value for severity calculation
    if "value" in detector_output:
        value = detector_output["value"]
    elif "statistic" in detector_output:
        value = detector_output["statistic"]
    elif "p_value" in detector_output:
        # For chi-square, p_value is used for drift detection
        # but we can't use it for severity mapping the same way
        # We'll use a default or the statistic if available
        value = detector_output.get("statistic", 0)
    else:
        value = 0
    
    threshold = detector_output.get("threshold", 0)
    
    # Determine severity
    if value <= 2 * threshold:
        severity = "warning"
    else:
        severity = "critical"
    
    # Create alert message
    message = f"Drift detected using {metric} metric"
    
    # Create details dict from detector output (excluding drift_detected)
    details = {k: v for k, v in detector_output.items() if k != "drift_detected"}
    
    return {
        "alert": True,
        "severity": severity,
        "metric": metric,
        "message": message,
        "details": details
    }

