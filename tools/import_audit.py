#!/usr/bin/env python3
"""Audit Rust imports and module-boundary usage."""

import sys
from pathlib import Path

# Add tools/ root to Python search path
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from audit.import_rules import run_import_audit

if __name__ == "__main__":
    sys.exit(run_import_audit())
