"""
Test suite for CLI execution behavior.

This module tests the drift monitoring CLI:
- Exit codes for different scenarios
- JSON output structure
- Error handling
"""

import pytest
import json
import tempfile
import os
from io import StringIO
from unittest.mock import patch
import pandas as pd


def test_main_function_exists():
    """Test that main function exists in drift.cli module."""
    from drift.cli import main
    
    assert callable(main), "main must be a callable function"


def test_main_returns_int():
    """Test that main returns an integer exit code."""
    from drift.cli import main
    
    # Create temporary CSV files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as ref_file:
        ref_file.write("feature\n1.0\n2.0\n3.0\n4.0\n5.0\n")
        ref_path = ref_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as curr_file:
        curr_file.write("feature\n1.0\n2.0\n3.0\n4.0\n5.0\n")
        curr_path = curr_file.name
    
    try:
        with patch('sys.argv', ['cli', ref_path, curr_path, '--metric', 'psi', '--threshold', '0.5']):
            result = main()
        
        assert isinstance(result, int), "main must return an integer exit code"
    finally:
        os.unlink(ref_path)
        os.unlink(curr_path)


def test_successful_execution_no_drift_returns_zero():
    """Test that main returns 0 when no drift is detected."""
    from drift.cli import main
    
    # Create identical datasets - no drift
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as ref_file:
        ref_file.write("feature\n1.0\n2.0\n3.0\n4.0\n5.0\n")
        ref_path = ref_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as curr_file:
        curr_file.write("feature\n1.0\n2.0\n3.0\n4.0\n5.0\n")
        curr_path = curr_file.name
    
    try:
        with patch('sys.argv', ['cli', ref_path, curr_path, '--metric', 'psi', '--threshold', '0.5']):
            exit_code = main()
        
        assert exit_code == 0, "Should return 0 when no drift detected"
    finally:
        os.unlink(ref_path)
        os.unlink(curr_path)


def test_drift_detected_returns_two():
    """Test that main returns 2 when drift is detected."""
    from drift.cli import main
    
    # Create very different datasets - drift expected
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as ref_file:
        ref_file.write("feature\n1.0\n2.0\n3.0\n4.0\n5.0\n")
        ref_path = ref_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as curr_file:
        curr_file.write("feature\n10.0\n20.0\n30.0\n40.0\n50.0\n")
        curr_path = curr_file.name
    
    try:
        with patch('sys.argv', ['cli', ref_path, curr_path, '--metric', 'psi', '--threshold', '0.01']):
            exit_code = main()
        
        assert exit_code == 2, "Should return 2 when drift detected"
    finally:
        os.unlink(ref_path)
        os.unlink(curr_path)


def test_json_output_produced_on_success():
    """Test that JSON output is produced to stdout."""
    from drift.cli import main
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as ref_file:
        ref_file.write("feature\n1.0\n2.0\n3.0\n4.0\n5.0\n")
        ref_path = ref_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as curr_file:
        curr_file.write("feature\n1.0\n2.0\n3.0\n4.0\n5.0\n")
        curr_path = curr_file.name
    
    try:
        with patch('sys.argv', ['cli', ref_path, curr_path, '--metric', 'psi', '--threshold', '0.5']):
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                main()
                output = mock_stdout.getvalue()
        
        # Should be valid JSON
        result = json.loads(output)
        assert isinstance(result, dict), "Output should be valid JSON dict"
    finally:
        os.unlink(ref_path)
        os.unlink(curr_path)


def test_json_output_contains_drift_detected_field():
    """Test that JSON output contains drift_detected field."""
    from drift.cli import main
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as ref_file:
        ref_file.write("feature\n1.0\n2.0\n3.0\n4.0\n5.0\n")
        ref_path = ref_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as curr_file:
        curr_file.write("feature\n10.0\n20.0\n30.0\n40.0\n50.0\n")
        curr_path = curr_file.name
    
    try:
        with patch('sys.argv', ['cli', ref_path, curr_path, '--metric', 'psi', '--threshold', '0.01']):
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                main()
                output = mock_stdout.getvalue()
        
        result = json.loads(output)
        assert "drift_detected" in result, "JSON output must contain drift_detected field"
        assert result["drift_detected"] is True, "drift_detected should be True for drifted data"
    finally:
        os.unlink(ref_path)
        os.unlink(curr_path)


def test_invalid_file_path_returns_one():
    """Test that main returns 1 for invalid file path."""
    from drift.cli import main
    
    with patch('sys.argv', ['cli', '/nonexistent/file.csv', '/another/nonexistent.csv', '--metric', 'psi', '--threshold', '0.1']):
        exit_code = main()
    
    assert exit_code == 1, "Should return 1 for invalid file path"


def test_invalid_metric_returns_one():
    """Test that main returns 1 for unsupported metric."""
    from drift.cli import main
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as ref_file:
        ref_file.write("feature\n1.0\n2.0\n3.0\n4.0\n5.0\n")
        ref_path = ref_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as curr_file:
        curr_file.write("feature\n1.0\n2.0\n3.0\n4.0\n5.0\n")
        curr_path = curr_file.name
    
    try:
        with patch('sys.argv', ['cli', ref_path, curr_path, '--metric', 'invalid_metric', '--threshold', '0.1']):
            exit_code = main()
        
        assert exit_code == 1, "Should return 1 for unsupported metric"
    finally:
        os.unlink(ref_path)
        os.unlink(curr_path)


def test_cli_is_deterministic():
    """Test that CLI returns same result for same inputs."""
    from drift.cli import main
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as ref_file:
        ref_file.write("feature\n1.0\n2.0\n3.0\n4.0\n5.0\n")
        ref_path = ref_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as curr_file:
        curr_file.write("feature\n1.5\n2.5\n3.5\n4.5\n5.5\n")
        curr_path = curr_file.name
    
    try:
        with patch('sys.argv', ['cli', ref_path, curr_path, '--metric', 'psi', '--threshold', '0.1']):
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout1:
                exit_code1 = main()
                output1 = mock_stdout1.getvalue()
        
        with patch('sys.argv', ['cli', ref_path, curr_path, '--metric', 'psi', '--threshold', '0.1']):
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout2:
                exit_code2 = main()
                output2 = mock_stdout2.getvalue()
        
        assert exit_code1 == exit_code2, "CLI must be deterministic (same exit code)"
        
        result1 = json.loads(output1)
        result2 = json.loads(output2)
        assert result1["drift_detected"] == result2["drift_detected"], "CLI must be deterministic (same drift decision)"
    finally:
        os.unlink(ref_path)
        os.unlink(curr_path)


def test_cli_with_ks_metric():
    """Test that CLI works with KS metric."""
    from drift.cli import main
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as ref_file:
        ref_file.write("feature\n1.0\n2.0\n3.0\n4.0\n5.0\n")
        ref_path = ref_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as curr_file:
        curr_file.write("feature\n1.5\n2.5\n3.5\n4.5\n5.5\n")
        curr_path = curr_file.name
    
    try:
        with patch('sys.argv', ['cli', ref_path, curr_path, '--metric', 'ks', '--threshold', '0.3']):
            exit_code = main()
        
        assert exit_code in [0, 2], "Should return valid exit code for KS metric"
    finally:
        os.unlink(ref_path)
        os.unlink(curr_path)


def test_cli_with_chi_square_metric():
    """Test that CLI works with Chi-Square metric."""
    from drift.cli import main
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as ref_file:
        ref_file.write("feature\na\nb\nc\na\nb\n")
        ref_path = ref_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as curr_file:
        curr_file.write("feature\na\nb\nc\nc\nb\n")
        curr_path = curr_file.name
    
    try:
        with patch('sys.argv', ['cli', ref_path, curr_path, '--metric', 'chi_square', '--threshold', '0.05', '--feature-type', 'categorical']):
            exit_code = main()
        
        assert exit_code in [0, 2], "Should return valid exit code for Chi-Square metric"
    finally:
        os.unlink(ref_path)
        os.unlink(curr_path)
