#!/usr/bin/env python3
"""Run the pytest-collected synthetic external23 tests; never reads the real dataset."""
from pathlib import Path
import sys


def main():
    try:
        import pytest
    except ImportError:
        print('ERROR: pytest_required', file=sys.stderr)
        return 1
    return pytest.main(['-q', str(Path(__file__).resolve().parents[1] / 'ml/tests/test_external23.py')])


if __name__ == '__main__':
    raise SystemExit(main())
