"""
Architectural boundary tests.

This module ensures that the project maintains its architectural constraints:
- No dependency on Evidently at the core level
- Custom drift detection implementation
"""

import pytest
import os
import glob


def test_no_evidently_imports_in_drift_modules():
    """Test that no drift module imports Evidently."""
    drift_dir = os.path.join(os.path.dirname(__file__), "..", "drift")
    
    # Get all Python files in drift directory
    python_files = glob.glob(os.path.join(drift_dir, "*.py"))
    
    for filepath in python_files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Check for evidently imports
        assert "import evidently" not in content.lower(), \
            f"{filepath} contains Evidently import - violates architectural constraint"
        assert "from evidently" not in content.lower(), \
            f"{filepath} contains Evidently import - violates architectural constraint"


def test_run_pipeline_does_not_use_evidently():
    """Test that run_pipeline implementation does not depend on Evidently."""
    from drift import pipeline
    import inspect
    
    # Get the source code of the pipeline module
    source = inspect.getsource(pipeline)
    
    # Check that Evidently is not mentioned
    assert "evidently" not in source.lower(), \
        "pipeline module must not depend on Evidently"


def test_no_evidently_in_requirements():
    """Test that Evidently is not listed in requirements.txt."""
    requirements_path = os.path.join(
        os.path.dirname(__file__), "..", "requirements.txt"
    )
    
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = f.read().lower()
    
    assert "evidently" not in requirements, \
        "Evidently must not be in core requirements - custom implementation required"
