"""
Pytest configuration for quantum-classification-train tests.
"""
import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def pytest_configure(config):
    """Configure pytest."""
    # Suppress PennyLane warnings for cleaner test output
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="pennylane")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
