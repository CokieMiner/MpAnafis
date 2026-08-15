#!/usr/bin/env python3
"""Inventory Rust allow attributes in production and test-oriented code."""

import sys
from pathlib import Path

# Add tools/ root to Python search path
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from audit.allow_rules import run_check_allows

if __name__ == "__main__":
    sys.exit(run_check_allows())
