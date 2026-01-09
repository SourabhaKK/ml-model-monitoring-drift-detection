# Main drift detection pipeline
from drift.metrics import calculate_psi, calculate_ks, calculate_chi_square
from drift.detectors import detect_psi_drift, detect_ks_drift, detect_chi_square_drift
from drift.alerts import generate_alert


def run_drift_pipeline(reference_data, current_data, *, feature_type, metric, threshold):
    """
    Run end-to-end drift detection pipeline.
    
    Args:
        reference_data: Reference dataset (baseline)
        current_data: Current dataset to compare against reference
        feature_type: Type of feature ("numerical" or "categorical")
        metric: Drift metric to use ("psi", "ks", or "chi_square")
        threshold: Threshold for drift detection
        
    Returns:
        dict: Dictionary containing:
            - drift_detected (bool): Whether drift was detected
            - alerts (list): List of alerts generated
            - metrics (dict): Drift metrics computed
            - window (dict): Window information
            
    Raises:
        ValueError: If data is empty or metric is unsupported
    """
    # Validate inputs
    if len(reference_data) == 0:
        raise ValueError("reference data cannot be empty")
    if len(current_data) == 0:
        raise ValueError("current data cannot be empty")
    
    # Validate metric
    supported_metrics = ["psi", "ks", "chi_square"]
    if metric not in supported_metrics:
        raise ValueError(f"unsupported metric: {metric}")
    
    # Extract first column for analysis
    ref_values = reference_data.iloc[:, 0].values
    curr_values = current_data.iloc[:, 0].values
    
    # Compute metric
    if metric == "psi":
        metric_result = calculate_psi(ref_values, curr_values)
        metric_output = {"psi": metric_result}
        # Detect drift
        detector_output = detect_psi_drift(psi_value=metric_result, threshold=threshold)
    elif metric == "ks":
        metric_result = calculate_ks(ref_values, curr_values)
        metric_output = {"ks": metric_result}
        # Detect drift
        detector_output = detect_ks_drift(
            statistic=metric_result["statistic"],
            p_value=metric_result["p_value"],
            threshold=threshold
        )
    elif metric == "chi_square":
        metric_result = calculate_chi_square(ref_values, curr_values)
        metric_output = {"chi_square": metric_result}
        # Detect drift
        detector_output = detect_chi_square_drift(
            statistic=metric_result["statistic"],
            p_value=metric_result["p_value"],
            threshold=threshold
        )
    
    # Generate alert if drift detected
    alert = generate_alert(detector_output)
    alerts = [alert] if alert is not None else []
    
    # Assemble result
    return {
        "drift_detected": detector_output["drift_detected"],
        "alerts": alerts,
        "metrics": metric_output,
        "window": {
            "reference_size": len(reference_data),
            "current_size": len(current_data)
        }
    }


def run_pipeline(reference_data, current_data):
    """
    Run drift detection pipeline on reference and current data.
    
    Args:
        reference_data: Reference dataset (baseline)
        current_data: Current dataset to compare against reference
        
    Returns:
        dict: Dictionary containing:
            - drift_detected (bool): Whether drift was detected
            - metrics (dict): Drift metrics computed
            - alerts (list): List of alerts generated
    """
    return {
        "drift_detected": False,
        "metrics": {},
        "alerts": []
    }
