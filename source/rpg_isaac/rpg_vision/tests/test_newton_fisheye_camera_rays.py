"""Tests for the RPG Newton polynomial f-theta camera rays."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp

_MODULE_PATH = Path(__file__).parents[1] / "rpg_newton_renderer.py"
_SPEC = importlib.util.spec_from_file_location("rpg_newton_renderer_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
_compute_fisheye_polynomial_camera_rays = _MODULE._compute_fisheye_polynomial_camera_rays


def _generate_test_rays(max_fov_degrees: float = 360.0) -> np.ndarray:
    output = wp.empty((1, 4, 4, 2), dtype=wp.vec3f, device="cpu")
    wp.launch(
        kernel=_compute_fisheye_polynomial_camera_rays,
        dim=(1, 4, 4),
        inputs=[
            4,
            4,
            8.0,
            4.0,
            4.0,
            2.0,
            math.radians(max_fov_degrees),
            0.0,
            0.01,
            0.0,
            0.0,
            0.0,
            0.0,
            output,
        ],
        device="cpu",
    )
    return output.numpy()


def test_fisheye_uses_isotropic_radial_scale_for_non_nominal_aspect_ratio():
    rays = _generate_test_rays()
    radius = math.sqrt(10.0)
    theta = 0.01 * radius
    expected = np.array([math.sin(theta) / radius, -3.0 * math.sin(theta) / radius, -math.cos(theta)])

    assert rays[0, 3, 2, 0] == pytest.approx(np.zeros(3))
    assert rays[0, 3, 2, 1] == pytest.approx(expected, abs=1.0e-6)
    assert np.linalg.norm(rays[0, 3, 2, 1]) == pytest.approx(1.0, abs=1.0e-6)


def test_fisheye_rejects_rays_outside_half_max_fov():
    rays = _generate_test_rays(max_fov_degrees=1.0)
    assert rays[0, 3, 2, 1] == pytest.approx(np.zeros(3))
