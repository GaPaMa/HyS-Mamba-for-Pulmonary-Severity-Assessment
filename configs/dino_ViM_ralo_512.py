"""RALO with the 512-unit head described for the recovered statistical checkpoint.

Use only with a matching checkpoint. This structural configuration variant does
not supply or certify the unavailable checkpoint or its training history.
"""
import runpy as _runpy
from pathlib import Path as _Path

_config = _runpy.run_path(str(_Path(__file__).with_name("dino_ViM_ralo.py")))
globals().update({key: value for key, value in _config.items() if not key.startswith("_")})
model["head_hidden"] = 512
del _runpy, _Path, _config

