# Drift metrics implementation
import numpy as np
from scipy import stats


def calculate_psi(reference, current):
    """
    Calculate Population Stability Index (PSI) for numerical features.
    
    Args:
        reference: Reference data (baseline)
        current: Current data to compare against reference
        
    Returns:
        float: PSI value (>= 0)
        
    Raises:
        ValueError: If inputs are empty
        TypeError: If inputs are categorical/string data
    """
    # Convert to numpy arrays
    reference = np.asarray(reference)
    current = np.asarray(current)
    
    # Validate non-empty
    if len(reference) == 0:
        raise ValueError("reference data cannot be empty")
    if len(current) == 0:
        raise ValueError("current data cannot be empty")
    
    # Validate numerical data (reject categorical)
    if reference.dtype.kind in ('U', 'S', 'O'):  # Unicode, byte string, or object
        raise TypeError("PSI requires numerical data, not categorical")
    if current.dtype.kind in ('U', 'S', 'O'):
        raise TypeError("PSI requires numerical data, not categorical")
    
    # Use fixed number of bins for deterministic behavior
    n_bins = 10
    
    # Create bins based on reference data
    bins = np.linspace(reference.min(), reference.max(), n_bins + 1)
    bins[0] = -np.inf  # Extend first bin to catch all values
    bins[-1] = np.inf  # Extend last bin to catch all values
    
    # Calculate histograms
    ref_counts, _ = np.histogram(reference, bins=bins)
    curr_counts, _ = np.histogram(current, bins=bins)
    
    # Convert to proportions
    ref_props = ref_counts / len(reference)
    curr_props = curr_counts / len(current)
    
    # Avoid division by zero - add small epsilon
    epsilon = 1e-10
    ref_props = np.where(ref_props == 0, epsilon, ref_props)
    curr_props = np.where(curr_props == 0, epsilon, curr_props)
    
    # Calculate PSI
    psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
    
    return float(psi)


def calculate_ks(reference, current):
    """
    Calculate Kolmogorov-Smirnov test statistic for numerical features.
    
    Args:
        reference: Reference data (baseline)
        current: Current data to compare against reference
        
    Returns:
        dict: {"statistic": float, "p_value": float}
        
    Raises:
        ValueError: If inputs are empty
        TypeError: If inputs are categorical/string data
    """
    # Convert to numpy arrays
    reference = np.asarray(reference)
    current = np.asarray(current)
    
    # Validate non-empty
    if len(reference) == 0:
        raise ValueError("reference data cannot be empty")
    if len(current) == 0:
        raise ValueError("current data cannot be empty")
    
    # Validate numerical data (reject categorical)
    if reference.dtype.kind in ('U', 'S', 'O'):
        raise TypeError("KS test requires numerical data, not categorical")
    if current.dtype.kind in ('U', 'S', 'O'):
        raise TypeError("KS test requires numerical data, not categorical")
    
    # Perform KS test
    statistic, p_value = stats.ks_2samp(reference, current)
    
    return {
        "statistic": float(statistic),
        "p_value": float(p_value)
    }


def calculate_chi_square(reference, current):
    """
    Calculate Chi-Square test statistic for categorical features.
    
    Args:
        reference: Reference data (baseline) - categorical
        current: Current data to compare against reference - categorical
        
    Returns:
        dict: {"statistic": float, "p_value": float}
        
    Raises:
        ValueError: If inputs are empty
        TypeError: If inputs are continuous numerical data
    """
    # Convert to numpy arrays
    reference = np.asarray(reference)
    current = np.asarray(current)
    
    # Validate non-empty
    if len(reference) == 0:
        raise ValueError("reference data cannot be empty")
    if len(current) == 0:
        raise ValueError("current data cannot be empty")
    
    # Validate categorical data (reject continuous numerical)
    # Check if data is floating point (continuous)
    if reference.dtype.kind == 'f':
        raise TypeError("Chi-Square test requires categorical data, not continuous numerical")
    if current.dtype.kind == 'f':
        raise TypeError("Chi-Square test requires categorical data, not continuous numerical")
    
    # Get unique categories from both datasets
    all_categories = np.unique(np.concatenate([reference, current]))
    
    # Count occurrences in each dataset
    ref_counts = np.array([np.sum(reference == cat) for cat in all_categories])
    curr_counts = np.array([np.sum(current == cat) for cat in all_categories])
    
    # Create contingency table
    contingency_table = np.array([ref_counts, curr_counts])
    
    # Perform chi-square test
    chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
    
    return {
        "statistic": float(chi2),
        "p_value": float(p_value)
    }

