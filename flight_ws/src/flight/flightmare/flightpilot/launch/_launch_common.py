"""Shared launch helpers for flightpilot launch files.

Contains the Python-environment resolution logic needed by all launch files
to work around the ros2/ros2#1094 shebang issue.
"""

import os
import sys
from pathlib import Path


def resolve_python() -> str:
    """Return the path to the active conda/venv Python interpreter.

    ``ros2 launch`` itself runs under /usr/bin/python3 (hardcoded shebang),
    so sys.executable is NOT the conda Python even when the env is active.
    Instead we read $CONDA_PREFIX or $VIRTUAL_ENV to locate the correct
    interpreter.
    """
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


def resolve_flightmare_env() -> dict:
    """Build the additional environment variables for flightmare.

    Mirrors set_vars.sh: FLIGHTMARE_PATH, AGI_PATH, PYTHONPATH,
    LD_LIBRARY_PATH.
    """
    launch_dir = Path(__file__).resolve().parent          # .../launch
    flightpilot_dir = launch_dir.parent                   # .../flightpilot
    flightmare_dir = flightpilot_dir.parent               # .../flightmare
    agilicious_dir = flightmare_dir.parent / 'agilicious'

    flightmare = str(flightmare_dir)
    agi = str(agilicious_dir)

    active_python = resolve_python()
    python_prefix = str(Path(active_python).resolve().parent.parent)
    site_packages = f'{python_prefix}/lib/python3.12/site-packages'

    extra_pythonpath = site_packages
    pythonpath = os.environ.get('PYTHONPATH', '')
    if pythonpath:
        extra_pythonpath = extra_pythonpath + os.pathsep + pythonpath

    extra_ld = os.pathsep.join([
        f'{agi}/agilib/externals/acados-src/lib',
        f'{python_prefix}/lib',
    ])
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if ld_path:
        extra_ld = extra_ld + os.pathsep + ld_path

    return {
        'FLIGHTMARE_PATH': flightmare,
        'AGI_PATH': agi,
        'PYTHONPATH': extra_pythonpath,
        'LD_LIBRARY_PATH': extra_ld,
    }
