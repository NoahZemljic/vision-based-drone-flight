"""Warp kernels of the rpg quadrotor simulator, one file per component.

Importing this package registers every kernel for float32 and float64 (see
``registry.register``); the glue classes fetch them from ``<module>.KERNELS``.

Two modes:
- source tree: the kernel files are imported and JIT-compiled by Warp;
- binary tree (no kernel files, ``_bin/`` present): the modules are
  synthesized from ``_bin/signatures.json`` and ``ensure_loaded`` binds the
  shipped PTX by kernel name.
"""

import os as _os

from rpg_simulator.kernels import registry as _registry

if _os.path.isfile(_os.path.join(_os.path.dirname(__file__), "model_motor.py")):
    from rpg_simulator.kernels import (  # noqa: F401  registration side effects
        cmd_queue,
        controller_geo,
        integrator_euler,
        integrator_rk4,
        low_level_controller_betaflight,
        model_battery,
        model_motor,
        model_rigid_body,
        model_rotor_simple_augmentation,
        model_scale,
        model_thrust_torque_simple,
        quad_state,
    )
else:
    _bin = _registry.bin_dir()
    if _bin is None:
        raise RuntimeError(
            "rpg_simulator kernel sources are absent and no precompiled binaries were found "
            f"({_registry.default_bin_dir()} or ${_registry.BIN_DIR_ENV})"
        )
    _registry.synthesize_stub_modules(_bin)
