#!/usr/bin/env python3
"""Generate the externally reachable integer API inventory from rustdoc JSON."""

import sys
from pathlib import Path

# Add tools/ root to Python search path
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from api_inventory.renderer import run_api_inventory

if __name__ == "__main__":
    sys.exit(run_api_inventory())
