# Drift detection logic


def detect_psi_drift(psi_value, threshold):
    """
    Detect drift using PSI (Population Stability Index).
    
    Args:
        psi_value: PSI metric value
        threshold: Threshold for drift detection
        
    Returns:
        dict: Drift detection result with keys:
            - drift_detected (bool): True if psi_value > threshold
            - metric (str): "psi"
            - value (float): Input psi_value
            - threshold (float): Input threshold
            
    Raises:
        ValueError: If threshold <= 0
    """
    # Validate threshold
    if threshold <= 0:
        raise ValueError("threshold must be greater than 0")
    
    # Determine drift
    drift_detected = psi_value > threshold
    
    return {
        "drift_detected": drift_detected,
        "metric": "psi",
        "value": psi_value,
        "threshold": threshold
    }


def detect_ks_drift(statistic, p_value, threshold):
    """
    Detect drift using KS (Kolmogorov-Smirnov) test.
    
    Args:
        statistic: KS test statistic
        p_value: KS test p-value
        threshold: Threshold for drift detection
        
    Returns:
        dict: Drift detection result with keys:
            - drift_detected (bool): True if statistic > threshold
            - metric (str): "ks"
            - statistic (float): Input statistic
            - p_value (float): Input p_value
            - threshold (float): Input threshold
            
    Raises:
        ValueError: If threshold <= 0
    """
    # Validate threshold
    if threshold <= 0:
        raise ValueError("threshold must be greater than 0")
    
    # Determine drift (based on statistic, not p_value)
    drift_detected = statistic > threshold
    
    return {
        "drift_detected": drift_detected,
        "metric": "ks",
        "statistic": statistic,
        "p_value": p_value,
        "threshold": threshold
    }


def detect_chi_square_drift(statistic, p_value, threshold):
    """
    Detect drift using Chi-Square test.
    
    Args:
        statistic: Chi-Square test statistic
        p_value: Chi-Square test p-value
        threshold: Threshold for drift detection
        
    Returns:
        dict: Drift detection result with keys:
            - drift_detected (bool): True if p_value < threshold
            - metric (str): "chi_square"
            - statistic (float): Input statistic
            - p_value (float): Input p_value
            - threshold (float): Input threshold
            
    Raises:
        ValueError: If threshold <= 0
    """
    # Validate threshold
    if threshold <= 0:
        raise ValueError("threshold must be greater than 0")
    
    # Determine drift (based on p_value)
    drift_detected = p_value < threshold
    
    return {
        "drift_detected": drift_detected,
        "metric": "chi_square",
        "statistic": statistic,
        "p_value": p_value,
        "threshold": threshold
    }

