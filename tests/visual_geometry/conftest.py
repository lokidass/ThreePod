# tests/visual_geometry/conftest.py
# Makes the repo root (production modules) and this directory (fixtures,
# helpers, render, quality) importable for every test in this suite,
# regardless of how pytest is invoked.
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
