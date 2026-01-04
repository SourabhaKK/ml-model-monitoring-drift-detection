# Windowing strategies for drift detection


def get_windows(data, reference_size, current_size):
    """
    Extract reference and current windows from data.
    
    Args:
        data: Input dataset (pandas DataFrame)
        reference_size: Number of rows for reference window (first N rows)
        current_size: Number of rows for current window (last N rows)
        
    Returns:
        tuple: (reference_window, current_window)
        
    Raises:
        ValueError: If window sizes are invalid or exceed data size
    """
    # Validate reference_size
    if reference_size <= 0:
        raise ValueError("reference_size must be greater than 0")
    
    # Validate current_size
    if current_size <= 0:
        raise ValueError("current_size must be greater than 0")
    
    # Get data size
    data_size = len(data)
    
    # Validate reference_size doesn't exceed data size
    if reference_size > data_size:
        raise ValueError(f"reference_size ({reference_size}) exceeds data size ({data_size})")
    
    # Validate current_size doesn't exceed data size
    if current_size > data_size:
        raise ValueError(f"current_size ({current_size}) exceeds data size ({data_size})")
    
    # Extract reference window (first N rows)
    reference_window = data.iloc[:reference_size]
    
    # Extract current window (last N rows)
    current_window = data.iloc[-current_size:]
    
    return (reference_window, current_window)

