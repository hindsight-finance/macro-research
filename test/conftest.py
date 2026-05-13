"""Pytest environment setup."""

from __future__ import annotations

import site
import sys
import types
from pathlib import Path


def _prefer_venv_mpl_toolkits() -> None:
    for site_dir in site.getsitepackages():
        toolkit_dir = Path(site_dir) / "mpl_toolkits"
        if (toolkit_dir / "mplot3d").is_dir():
            module = types.ModuleType("mpl_toolkits")
            module.__path__ = [str(toolkit_dir)]
            sys.modules["mpl_toolkits"] = module
            return


_prefer_venv_mpl_toolkits()
