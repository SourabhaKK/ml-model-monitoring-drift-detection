# CLI runner for drift monitoring
import argparse
import json
import sys
import pandas as pd
from drift.pipeline import run_drift_pipeline


def main():
    """
    Main CLI entry point for drift monitoring.
    
    Returns:
        int: Exit code
            0 - No drift detected
            1 - Error occurred
            2 - Drift detected
    """
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(description='Drift monitoring CLI')
        parser.add_argument('reference', help='Path to reference CSV file')
        parser.add_argument('current', help='Path to current CSV file')
        parser.add_argument('--metric', required=True, help='Drift metric (psi, ks, chi_square)')
        parser.add_argument('--threshold', type=float, required=True, help='Drift detection threshold')
        parser.add_argument('--feature-type', default='numerical', help='Feature type (numerical, categorical)')
        
        args = parser.parse_args()
        
        # Load CSV files
        reference_data = pd.read_csv(args.reference)
        current_data = pd.read_csv(args.current)
        
        # Run drift pipeline
        result = run_drift_pipeline(
            reference_data,
            current_data,
            feature_type=args.feature_type,
            metric=args.metric,
            threshold=args.threshold
        )
        
        # Print JSON output
        print(json.dumps(result))
        
        # Return appropriate exit code
        if result["drift_detected"]:
            return 2
        else:
            return 0
            
    except Exception as e:
        # Return error exit code for any failure
        return 1


if __name__ == '__main__':
    sys.exit(main())
