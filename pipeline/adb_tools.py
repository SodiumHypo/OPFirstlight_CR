"""Locate host tooling via environment overrides (no machine-specific paths).

Env vars:
  CR_ADB           — full path to adb.exe (falls back to `adb` on PATH)
  CR_MUMU_MANAGER  — full path to MuMuManager.exe (required by instances.py)
  CR_ADB_SERIAL    — default adb serial for CLI tools (fallback 127.0.0.1:5555)
"""
from __future__ import annotations

import os
import shutil


def adb_path() -> str:
    return os.environ.get("CR_ADB") or shutil.which("adb") or "adb"


def adb_serial_default() -> str:
    return os.environ.get("CR_ADB_SERIAL", "127.0.0.1:5555")


def mumu_manager_path() -> str:
    path = os.environ.get("CR_MUMU_MANAGER")
    if not path:
        raise RuntimeError(
            "set CR_MUMU_MANAGER to the full path of MuMuManager.exe "
            "(e.g. <MuMu install>/nx_main/MuMuManager.exe)"
        )
    return path
