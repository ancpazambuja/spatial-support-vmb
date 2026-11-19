"""Shared pytest configuration.

Makes the historical ``src`` namespace package importable (the repository
root is the import root, matching how the apps are run with
``PYTHONPATH=<repo>``) and forces a headless matplotlib backend before any
source module imports matplotlib/seaborn.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
