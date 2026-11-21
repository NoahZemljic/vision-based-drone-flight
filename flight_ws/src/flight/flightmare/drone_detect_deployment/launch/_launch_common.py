"""Python-environment resolution for the detector launch (same pattern as
flightpilot's ``_launch_common.py``, ros2/ros2#1094)."""

import os
import sys
from pathlib import Path


def resolve_python() -> str:
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if conda_prefix:
        p = Path(conda_prefix) / 'bin' / 'python3'
        if p.is_file():
            return str(p)
    virtual_env = os.environ.get('VIRTUAL_ENV', '')
    if virtual_env:
        p = Path(virtual_env) / 'bin' / 'python3'
        if p.is_file():
            return str(p)
    return sys.executable


def resolve_detector_env() -> dict:
    """PYTHONPATH / LD_LIBRARY_PATH so the node runs under the active python."""
    active_python = resolve_python()
    python_prefix = str(Path(active_python).resolve().parent.parent)
    site_packages = f'{python_prefix}/lib/python3.12/site-packages'

    extra_pythonpath = site_packages
    pythonpath = os.environ.get('PYTHONPATH', '')
    if pythonpath:
        extra_pythonpath = extra_pythonpath + os.pathsep + pythonpath

    extra_ld = f'{python_prefix}/lib'
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if ld_path:
        extra_ld = extra_ld + os.pathsep + ld_path

    return {
        'PYTHONPATH': extra_pythonpath,
        'LD_LIBRARY_PATH': extra_ld,
    }
