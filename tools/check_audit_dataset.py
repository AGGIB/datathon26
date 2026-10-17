#!/usr/bin/env python3
"""Compatibility entry point for the synthetic pytest audit suite; no real data."""

from pathlib import Path
import sys


def main():
    try:
        import pytest
    except ImportError:
        print("AUDIT CHECK FAILED: module=pytest", file=sys.stderr)
        return 2
    tests = Path(__file__).resolve().parent / "tests" / "test_audit_dataset.py"
    return int(pytest.main(["-q", "-p", "no:cacheprovider", str(tests)]))


if __name__ == "__main__":
    sys.exit(main())
