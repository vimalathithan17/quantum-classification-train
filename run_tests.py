#!/usr/bin/env python3
"""Simple test runner to verify fixes."""
import sys
import subprocess

print("Running pytest test suite...")
print("=" * 70)

result = subprocess.run(
    [sys.executable, "-m", "pytest", "-v", "--tb=short"],
    capture_output=False,
    text=True
)

sys.exit(result.returncode)
