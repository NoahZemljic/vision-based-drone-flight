#!/bin/sh
''':'
# Re-exec under Isaac Sim's python so ./this_file.py works from any shell.
exec "${ISAAC_PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}" "$0" "$@"
'''
"""Render camera-drone images for a list of poses (VBDF Task 1).

The rpg_isaac stand-in for the course's shared ROS renderer: you supply the poses of
the camera drone and of the actor drone, this script returns the images (double-sphere
``offboard_small`` lens, 1280x720, RPG Newton raytracer, kit-less). What you render,
and how you label it, is up to you.

``--poses`` is a CSV with one row per image and a header line::

    cam_x,cam_y,cam_z,cam_qw,cam_qx,cam_qy,cam_qz,actor_x,actor_y,actor_z,actor_qw,actor_qx,actor_qy,actor_qz

(positions [m] in the world frame, attitudes as wxyz quaternions). Images are written
to ``--out`` as ``img_XXXXX.png``, numbered by row (+ ``--start_index``). Rendering the
same pose file once with ``--stage arena`` (or ``winti``) and once with ``--stage empty``
gives pixel-aligned image pairs.

  /workspace/isaaclab/_isaac_sim/python.sh run/vbdf/detection/render_images.py \
      --poses my_poses.csv --out dataset/train --stage arena --device cuda:0
"""

import argparse
import csv
import os
import sys

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--poses", required=True, help="CSV of poses, see the module docstring")
parser.add_argument("--out", required=True)
parser.add_argument("--stage", default="arena", help="arena | winti | winti_notrack | empty | path to a stage USD")
parser.add_argument("--actor", default="offboard_4S_vision", help="offboard_4S_vision | kolibri | path to a drone USD")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--start_index", type=int, default=0)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import rpg_tasks.tasks  # noqa: F401, E402
from rpg_tasks.tasks.direct.rpg_vbdf.vbdf_render_env import RPGVbdfRenderEnv, RPGVbdfRenderEnvCfg  # noqa: E402

COLUMNS = [
    "cam_x", "cam_y", "cam_z", "cam_qw", "cam_qx", "cam_qy", "cam_qz",
    "actor_x", "actor_y", "actor_z", "actor_qw", "actor_qx", "actor_qy", "actor_qz",
]


def read_poses(path: str) -> np.ndarray:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: no pose rows")
    missing = [c for c in COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(f"{path}: missing columns {missing} (expected {COLUMNS})")
    return np.array([[float(r[c]) for c in COLUMNS] for r in rows])


def main():
    poses = read_poses(args_cli.poses)
    env_cfg = RPGVbdfRenderEnvCfg()
    env_cfg.scene_num_envs = args_cli.num_envs
    env_cfg.image_width, env_cfg.image_height = args_cli.width, args_cli.height
    env_cfg.stage, env_cfg.actor_drone = args_cli.stage, args_cli.actor
    env_cfg.test_env = True
    env_cfg.__post_init__()
    # not going through resolve_task_config: pick the preset defaults by hand
    from isaaclab_tasks.utils import PresetCfg

    if isinstance(env_cfg.sim.physics, PresetCfg):
        env_cfg.sim.physics = env_cfg.sim.physics.default
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    os.makedirs(args_cli.out, exist_ok=True)
    print(f"[render_images] {len(poses)} poses -> {args_cli.out} (stage={args_cli.stage})")
    with launch_simulation(env_cfg, args_cli):
        env = RPGVbdfRenderEnv(env_cfg)
        env.reset()
        n = env.num_envs
        dev = env.device

        def norm_quat(q):
            return q / np.linalg.norm(q, axis=1, keepdims=True)

        for lo in range(0, len(poses), n):
            batch = poses[lo : lo + n]
            pad = n - len(batch)
            if pad:
                batch = np.concatenate([batch, np.repeat(batch[-1:], pad, axis=0)])
            cam_pos = torch.as_tensor(batch[:, 0:3], device=dev, dtype=torch.float32)
            cam_att = torch.as_tensor(norm_quat(batch[:, 3:7]), device=dev, dtype=torch.float32)
            act_pos = torch.as_tensor(batch[:, 7:10], device=dev, dtype=torch.float32)
            act_att = torch.as_tensor(norm_quat(batch[:, 10:14]), device=dev, dtype=torch.float32)
            rgb, _ = env.render_poses(cam_pos, cam_att, act_pos, act_att)
            rgb = rgb.cpu().numpy().astype(np.uint8)
            for j in range(n - pad):
                idx = args_cli.start_index + lo + j
                Image.fromarray(rgb[j]).save(os.path.join(args_cli.out, f"img_{idx:05d}.png"))
            if (lo // n) % 25 == 0:
                print(f"[render_images] {min(lo + n, len(poses))}/{len(poses)}")
        env.close()
    print(f"[render_images] done: {len(poses)} images in {args_cli.out}")


if __name__ == "__main__":
    main()
