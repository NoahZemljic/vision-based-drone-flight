"""Double-sphere camera model (Usenko et al., "The Double Sphere Camera Model",
3DV 2018, https://arxiv.org/abs/1807.08957).

Batched over N environments. This is the onboard camera of the course's camera
drone. The ONLY course-specific facts used here are the calibration constants
reproduced in the exercise PDF (Section 3.1.1):

    xi   = -0.007441148905788614
    alpha=  0.6138265280850498
    fx   = 599.87676296874      fy  = 449.2911037186814
    cx   = 632.1317557077697    cy  = 388.2472267118482
    resolution 1280 x 720

The camera is mounted on the drone's nose and looks along the body +x axis with an
additional mount tilt (15 deg up on the real drone, encoded via ``r_BC_deg``).

We implement only what Task 2 needs: the FORWARD projection (used to synthesize the
actor's bounding box from its known 3D pose), which is the canonical model of the
paper and is numerically verified (on-axis world point lands at the image centre;
the xi=0 / alpha-bounded limits behave as expected).
"""

from __future__ import annotations

import math

import torch


def _rot_x(a: float) -> torch.Tensor:
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float64)


def _rot_y(a: float) -> torch.Tensor:
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float64)


def _rot_z(a: float) -> torch.Tensor:
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)


class DoubleSphereCamera:
    """A double-sphere camera. ``r_BC_deg`` is the body->camera mount rotation
    (roll, pitch, yaw in degrees); the camera optical axis (+z) is the body +x axis
    rotated by this mount rotation."""

    def __init__(self, intrinsics, resolution, r_BC_deg=(0.0, 0.0, 0.0), device="cpu", dtype=torch.float32):
        self.xi, self.alpha, self.fx, self.fy, self.cx, self.cy = (float(v) for v in intrinsics)
        self.width, self.height = float(resolution[0]), float(resolution[1])
        self.device, self.dtype = device, dtype
        roll, pitch, yaw = (math.radians(float(a)) for a in r_BC_deg)
        # Body -> camera rotation. The camera's optical axis is its +z, but it is mounted
        # on the drone's nose, so the optical axis must point along the body +x (forward).
        #
        # FIX (lane-2, 2026-09-15): the analytic mount MUST reproduce the RENDERED onboard
        # fisheye pixel-for-pixel. Empirically (box B, debug_bbox_verify.py, `camera-from-prim`
        # == segmentation truth at IoU 0.55-0.72) the render's body->optical rotation is
        #
        #     R_BC = rot_y(90deg + tilt) @ rot_z(-90deg)
        #
        # i.e. the optical +z is rotated nose-forward (+x) ABOUT BODY-Y (a pitch in the body's
        # vertical x-z plane), then rolled -90deg about the optical axis. This roll is the
        # double-sphere optical frame (x right, y DOWN, z forward) -> the renderer's OpenGL
        # frame (x right, y UP, z backward), which `_compute_double_sphere_camera_rays` emits
        # via `(dx, -dy, -dz)`. The previous `rot_y(90) @ rot_x(pitch)` tilted the axis in the
        # body's HORIZONTAL y-plane (a sideways yaw) and omitted the roll, so the analytic
        # bbox was ~90deg off the rendered fisheye (IoU 0.000).
        #
        # Using the config convention r_BC_deg=(roll, pitch, yaw) with the nose up/down tilt
        # carried in `pitch` (positive = nose-DOWN, matching the render's `+15` up-tilt quat
        # which lands at -15deg in `quat_w_world`), the equivalent is
        #     R_BC = rot_y(90deg) @ rot_y(-pitch) @ rot_z(yaw) @ rot_x(roll)
        # with `yaw` additionally carrying the constant -90deg roll. We fold the constant
        # roll in explicitly so the caller's `r_BC_deg` stays the course's (roll,pitch,yaw):
        self.R_BC = (
            _rot_y(math.pi / 2.0)
            @ _rot_y(pitch)               # nose up/down tilt about body-Y (was _rot_x(pitch));
                                          # pitch=-15 (config) => rot_y(75deg) => optical +15deg UP
            @ _rot_z(yaw - math.pi / 2.0) # constant -90deg optical roll + caller yaw
            @ _rot_x(roll)
        )

    def pose(self, R_WB: torch.Tensor, p_WB: torch.Tensor):
        """World->camera transform: returns (R_CW, t) so that a world point X maps to
        camera coordinates via ``R_CW @ X + t``.

        R_WB: (N,3,3) body->world (columns R[:,k] are the k-th body axis in world).
        p_WB: (N,3) body origin in world coordinates.
        """
        R_WB = R_WB.to(torch.float64)
        p_WB = p_WB.to(torch.float64)
        R_BC = self.R_BC.to(R_WB.device)
        R_BW = R_WB.transpose(-2, -1)                    # world -> body
        R_CW = R_BC.transpose(-2, -1) @ R_BW             # body -> camera @ world -> body
        t = -torch.bmm(R_CW, p_WB.unsqueeze(-1)).squeeze(-1)
        return R_CW, t

    def project(self, points_W: torch.Tensor, R_CW: torch.Tensor, t: torch.Tensor, homogeneous: bool = True):
        """points_W (N,P,3) -> (N,P,3) stack of [u, v, valid].

        homogeneous=True  -> u, v in [-1, 1]; False -> pixel coordinates.
        valid = 1.0 where the point is in front of the camera and inside the image.

        Canonical double-sphere forward projection (Usenko et al. 2018).
        """
        pts = points_W.to(torch.float64)
        R_CW = R_CW.to(torch.float64)
        t = t.to(torch.float64)

        X = torch.bmm(R_CW, pts.transpose(1, 2)).transpose(1, 2) + t.unsqueeze(1)
        x, y, z = X[..., 0], X[..., 1], X[..., 2]

        xi, alpha = self.xi, self.alpha
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy
        W, H = self.width, self.height

        d1 = torch.sqrt(x * x + y * y + z * z).clamp_min(1e-9)
        xp = x / d1
        yp = y / d1
        zp = z / d1
        zz = zp + xi
        d2 = torch.sqrt(xp * xp + yp * yp + zz * zz).clamp_min(1e-9)
        denom = alpha * d2 + (1.0 - alpha) * zz
        denom = torch.where(denom.abs() < 1e-12, torch.full_like(denom, 1e-12), denom)
        m_x = xp / denom
        m_y = yp / denom

        u_px = fx * m_x + cx
        v_px = fy * m_y + cy

        valid = (z > 0.0) & (u_px >= 0) & (u_px < W) & (v_px >= 0) & (v_px < H)
        valid = valid.to(torch.float64)

        if homogeneous:
            u = torch.clamp(2.0 * u_px / W - 1.0, -1.0, 1.0)
            v = torch.clamp(2.0 * v_px / H - 1.0, -1.0, 1.0)
            return torch.stack([u, v, valid], dim=-1)
        return torch.stack([u_px, v_px, valid], dim=-1)


# Course calibration (PDF Section 3.1.1). Do not change: this matches the real drone.
OFFBOARD_SMALL_DS = dict(
    intrinsics=[
        -0.007441148905788614,  # xi
        0.6138265280850498,     # alpha
        599.87676296874,        # fx
        449.2911037186814,      # fy
        632.1317557077697,      # cx
        388.2472267118482,      # cy
    ],
    resolution=[1280, 720],
)
