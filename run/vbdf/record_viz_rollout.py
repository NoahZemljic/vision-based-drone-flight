"""Record a fixed 3rd-person rollout video for a VBDF tracking Viz env.

Faithful to the proven recording pattern in run/vbdf/play_tracking_viz_kit.py and
play_s1_batch.py: AppLauncher first (before any isaaclab/rpg_tasks import), explicit
fps from env.step_dt, and a monkey-patch of RecordVideo.stop_recording to encode with
the system ffmpeg directly (the stock moviepy path fails: render_fps is None).

Usage (from the container, via /workspace/isaaclab/_isaac_sim/python.sh):
  record_viz_rollout.py --task Rpg-Vbdf-SB-Tracking-Viz-v0 \
      --checkpoint run/vbdf/logs/sb3/Rpg-Vbdf-SB-Tracking-v0/2026-09-14_15-12-23_SB-i01/model.zip \
      --out_name sb_i01 --video_length 1200
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import subprocess
import sys

# --- 1. SimulationApp before any isaaclab / pxr / rpg_tasks import ---
from isaaclab.app import AppLauncher

_orig_argv = list(sys.argv)  # preserve --task/--checkpoint/--out_name for the main parser

_boot = argparse.ArgumentParser(add_help=False)
AppLauncher.add_app_launcher_args(_boot)
_boot_args, _ = _boot.parse_known_args()
_boot_args.enable_cameras = True
if getattr(_boot_args, "headless", None) is True:
    _boot_args.headless = False
app_launcher = AppLauncher(_boot_args)
simulation_app = app_launcher.app
sys.argv = _orig_argv  # restore our args (main argparse is below)

# --- 2. now imports are safe ---
import gymnasium as gym
import numpy as np
import torch
from pathlib import Path

# tanh_policy lives in scripts/sb3 (unpickle helper for TanhMlpPolicy checkpoints)
_SB3 = Path(__file__).resolve().parents[2] / "scripts" / "sb3"
if _SB3.is_dir() and str(_SB3) not in sys.path:
    sys.path.insert(0, str(_SB3))

from gymnasium.wrappers.rendering import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.utils.seed import configure_seed
from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
from isaaclab_tasks.utils import launch_simulation, resolve_task_config

import isaaclab_tasks  # noqa: F401
import rpg_tasks.tasks  # noqa: F401
import tanh_policy  # noqa: F401

# --- camera fix (2026-09-16): the raked 3rd-person pose must reach the LIVE Newton
# viewer camera, not just cfg.viewer / cfg.video_recorder. VideoRecorder._sync_newton_camera
# reads viewer.camera.pos/yaw/pitch every frame, and the Newton visualizer is initialised
# from its OWN cfg.eye/lookat (base VisualizerCfg default (4,-4,3)->(0,0,0)), so the raked
# VIZ_EYE/VIZ_LOOKAT never reached the rendered perspective. We re-pose the live viewer
# camera right after sim.reset() (viewer initialised) using the same constants the env cfg
# already carries, so the rendered frames match VIZ_EYE->VIZ_LOOKAT. Verified ON PIXELS:
# room wall extent 143x40 px (old flat sliver) -> 1109x361 px (raked, 87% of frame width).
# See report/camera_framing_analysis.md APPENDIX.
from rpg_tasks.tasks.direct.rpg_vbdf.vbdf_render_env import VIZ_EYE, VIZ_LOOKAT


def _set_third_person_newton_camera(env, *, eye=VIZ_EYE, lookat=VIZ_LOOKAT) -> None:
    """Re-pose the live Newton viewer camera to the raked ``eye``/``lookat``.

    Must run AFTER the env's newton visualizer is initialised (i.e. after ``env.reset()``,
    which triggers ``sim.reset()``). This is the ONLY place that actually reaches the
    rendered 3rd-person perspective -- ``cfg.viewer`` / ``cfg.video_recorder`` are copied
    but then overwritten every frame by VideoRecorder._sync_newton_camera() from this live
    camera.
    """
    try:
        sim = getattr(env.unwrapped, "sim", None)
        viz = next(
            (v for v in getattr(sim, "visualizers", [])
             if getattr(getattr(v, "cfg", None), "visualizer_type", None) == "newton"),
            None,
        )
    except Exception as exc:  # noqa: BLE001 - best effort diagnostic
        print(f"[CAM] could not introspect newton visualizer: {exc!r}", flush=True)
        return
    if viz is None:
        print("[CAM] WARNING: no newton visualizer found; raked pose NOT applied", flush=True)
        return
    viewer = getattr(viz, "_viewer", None)
    if viewer is None:
        print("[CAM] WARNING: newton viewer not initialised; raked pose NOT applied", flush=True)
        return
    from pyglet.math import Vec3 as PygletVec3

    before = tuple(float(x) for x in viewer.camera.pos)
    viewer.camera.pos = PygletVec3(*eye)
    viewer.camera.look_at(lookat)
    after = tuple(float(x) for x in viewer.camera.pos)
    print(
        f"[CAM] newton viewer camera {before} -> {after} (eye={tuple(eye)} lookat={tuple(lookat)})",
        flush=True,
    )


def _patch_ffmpeg() -> None:
    """Override RecordVideo.stop_recording to shell out to system ffmpeg (moviepy fps=None bug)."""

    def stop_recording(self):
        assert self.recording
        if len(self.recorded_frames) == 0:
            self.recording = False
            return
        fps = float(self.frames_per_sec or 62.5)
        path = os.path.join(self.video_folder, f"{self._video_name}.mp4")
        os.makedirs(self.video_folder, exist_ok=True)
        h, w = np.asarray(self.recorded_frames[0]).shape[:2]
        pad_h, pad_w = h % 2, w % 2
        enc_h, enc_w = h + pad_h, w + pad_w
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{enc_w}x{enc_h}",
            "-r", f"{fps:.6f}", "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for frame in self.recorded_frames:
                arr = np.ascontiguousarray(frame, dtype=np.uint8)
                if pad_h or pad_w:
                    arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)))
                proc.stdin.write(arr.tobytes())
            proc.stdin.close()
            err = proc.stderr.read() if proc.stderr is not None else b""
            if proc.wait() != 0:
                raise RuntimeError(err.decode(errors="replace"))
        finally:
            if proc.poll() is None:
                proc.kill()
        print(f"[INFO] wrote {len(self.recorded_frames)} frames @ {fps:g} fps -> {path}", flush=True)
        self.recorded_frames = []
        self.recording = False
        self._video_name = None
        if self.gc_trigger and self.gc_trigger(self.episode_id):
            gc.collect()

    RecordVideo.stop_recording = stop_recording


def _write_frames_ffmpeg(frames, fps: float, path: str) -> None:
    """Write an iterator/list of RGB frames (numpy uint8 HxWx3) to an mp4 via system ffmpeg."""
    frames = list(frames)
    if not frames:
        raise RuntimeError("no frames to encode")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h, w = np.asarray(frames[0]).shape[:2]
    pad_h, pad_w = h % 2, w % 2
    enc_h, enc_w = h + pad_h, w + pad_w
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{enc_w}x{enc_h}",
        "-r", f"{fps:.6f}", "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            arr = np.ascontiguousarray(frame, dtype=np.uint8)
            if pad_h or pad_w:
                arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)))
            proc.stdin.write(arr.tobytes())
        proc.stdin.close()
        err = proc.stderr.read() if proc.stderr is not None else b""
        if proc.wait() != 0:
            raise RuntimeError(err.decode(errors="replace"))
    finally:
        if proc.poll() is None:
            proc.kill()
    print(f"[RECORD] wrote {len(frames)} frames @ {fps:g} fps -> {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=("--manifest" not in sys.argv))
    parser.add_argument("--checkpoint", required=("--manifest" not in sys.argv))
    parser.add_argument("--out_name", required=("--manifest" not in sys.argv), help="video name_prefix (writes <out_name>-step-0.mp4)")
    parser.add_argument("--video_length", type=int, default=1200)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--manifest", default=None, help="path to a JSON list of clip-spec dicts; renders ALL clips inside ONE launch_simulation boot (the cold-start win). When given, --task/--checkpoint/--out_name are read per-spec from the manifest and need NOT be passed on the CLI.")
    parser.add_argument("--alpha", type=float, default=0.0, help="env.act_lpf_alpha (action low-pass filter). MUST match the training plant: render SB-Smooth-LPF / BB-Smooth-LPF with --alpha 0.3 (PROBE_PLAN PHASE-1 SMOOTHNESS invariant).")
    parser.add_argument("--thrust_delta", type=float, default=0.0, help="env.thrust_delta_coeff (lever-2 normalized collective-thrust delta cost). Reward-only term: pass the train value (0.02) so the render env cfg is bit-identical to train; has no effect on the greedy rollout trajectory.")
    parser.add_argument("--pov", action="store_true", help="record the onboard (POV) fisheye camera instead of the 3rd-person viewer")
    parser.add_argument("--overlay", dest="overlay", action="store_true", help="3rd-person: draw the actor body-box + stats HUD overlay (post-render)")
    parser.add_argument("--no-overlay", dest="overlay", action="store_false", help="3rd-person: leave the raw viewer frames un-overlaid")
    parser.set_defaults(overlay=True)
    args, _unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]]  # don't let hydra (inside resolve_task_config) see our custom args

    # Manifest mode: render EVERY spec inside a SINGLE launch_simulation boot. The cold
    # boot is ~30x the render time, so pay it once for the whole batch. Each spec may
    # carry a DIFFERENT checkpoint/task/pov/alpha/thrust_delta/out_name/overlay.
    if args.manifest:
        return _run_manifest(args)

    _, env_cfg, agent_cfg = _build_env_cfg(args)
    launch_args = _build_launch_args(args)
    with launch_simulation(env_cfg, launch_args):
        path = _render_one(args, env_cfg, agent_cfg)
    print(f"[RECORD] done -> {path}", flush=True)


_MANIFEST_DEFAULTS = argparse.Namespace(
    task="Rpg-Vbdf-SB-Tracking-Viz-v0", checkpoint=None, out_name=None,
    video_length=1200, num_envs=1, seed=123, device="cuda:0",
    alpha=0.0, thrust_delta=0.0, pov=False, overlay=True,
)
_MANIFEST_KEYS = {
    "task", "checkpoint", "out_name", "video_length", "num_envs", "seed",
    "device", "alpha", "thrust_delta", "pov", "overlay",
}


def _spec_namespace(spec: dict) -> argparse.Namespace:
    """Coerce one manifest spec dict into a Namespace mirroring the CLI args, applying
    the manifest defaults for any omitted field."""
    ns = copy.copy(_MANIFEST_DEFAULTS)
    unknown = set(spec) - _MANIFEST_KEYS
    if unknown:
        raise ValueError(
            f"--manifest: unknown spec key(s) {sorted(unknown)} (allowed: {sorted(_MANIFEST_KEYS)})"
        )
    for k, v in spec.items():
        setattr(ns, k, v)
    if not ns.checkpoint:
        raise ValueError("--manifest: every spec needs a 'checkpoint' (model.zip path)")
    if not ns.out_name:
        raise ValueError("--manifest: every spec needs an 'out_name'")
    return ns


def _run_manifest(args: argparse.Namespace) -> None:
    """Render N clips (each with possibly a different task/checkpoint/pov/alpha/overlay)
    inside ONE launch_simulation boot. The engine boots ONCE on the first spec's cfg; we
    then resolve+plate a FRESH env_cfg and gym.make a FRESH env per spec (the num_envs /
    manager env state is not carried between specs), load that spec's VecNormalize+PPO,
    record to the spec's out_name, close the env, and move on WITHOUT re-running
    launch_simulation. Key win: one boot, N gym.make/record cycles."""
    with open(args.manifest) as f:
        raw = json.load(f)
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError("--manifest must name a JSON file containing a NON-EMPTY list of clip-spec dicts")
    specs = [_spec_namespace(s) for s in raw]

    boot = specs[0]
    _, env_cfg, agent_cfg = _build_env_cfg(boot)
    launch_args = _build_launch_args(boot)

    paths = []
    with launch_simulation(env_cfg, launch_args):
        for spec in specs:
            print(f"[BATCH] clip {spec.out_name} (task={spec.task} checkpoint={spec.checkpoint})", flush=True)
            # Fresh env_cfg PER SPEC inside the SAME boot: task/alpha/thrust_delta/pov
            # can all differ spec-to-spec. We only re-enter resolve_task_config + plating
            # + gym.make/record — NOT launch_simulation, which stays open for the batch.
            _, e, a = _build_env_cfg(spec)
            paths.append(_render_one(spec, e, a))
    print(f"[BATCH] rendered {len(paths)} clips: {paths}", flush=True)


def _build_env_cfg(args_ns: argparse.Namespace):
    """resolve_task_config for ONE clip's task + plate the plant-fidelity invariants
    (act_lpf_alpha, thrust_delta_coeff, POV viz, log_dir) onto env_cfg. Returns
    (task_str, env_cfg, agent_cfg). Shared by the single-clip CLI path and the manifest loop."""
    env_cfg, agent_cfg = resolve_task_config(args_ns.task, "sb3_cfg_entry_point")
    # PHASE-1 SMOOTHNESS invariant: the LPF changes plant dynamics. An LPF-trained policy
    # rendered WITHOUT the LPF is a WRONG plant (spurious in_view 0.35 / crash ~380 seen).
    # Set the same act_lpf_alpha used at training (SB/BB-Smooth-LPF: 0.3) before env build.
    if args_ns.alpha != 0.0:
        print(f"[RECORD] plating env.act_lpf_alpha={args_ns.alpha} (render plant == train plant)", flush=True)
        setattr(env_cfg, "act_lpf_alpha", args_ns.alpha)
    if args_ns.thrust_delta != 0.0:
        # lever-2 knob (SB-ThrustDelta-01 / BB-Final-01 trained with thrust_delta_coeff=0.02).
        # Reward-only term (does NOT change the greedy rollout trajectory), plumbed so the
        # render env cfg is identical to the train env cfg.
        print(f"[RECORD] plating env.thrust_delta_coeff={args_ns.thrust_delta} (train cfg match)", flush=True)
        setattr(env_cfg, "thrust_delta_coeff", args_ns.thrust_delta)
    env_cfg.scene.num_envs = args_ns.num_envs
    # Do NOT set env_cfg.seed here: DirectRLEnv.__init__ calls self.seed() whenever
    # cfg.seed is not None, and that path calls omni.replicator rep.set_global_seed,
    # which raises "Unable to retrieve replicator graph" once the replicator graph is
    # torn down after the first env.close() in a single warm boot. Trajectory divergence
    # across seeds is driven instead by configure_seed(args_ns.seed) in _render_one
    # (torch.manual_seed -> torch.rand used in apply_random_state), NOT by replicator.
    env_cfg.seed = None
    env_cfg.sim.device = args_ns.device
    env_cfg.stage = "arena"
    env_cfg.test_env = True
    if args_ns.pov:
        # Apply the POV branch of the shared viz-cfg helper: attach the onboard
        # double-sphere fisheye (15 deg up-tilt) AND hide the ego drone's mesh so the
        # fisheye view isn't self-occluded. resolve_task_config already ran __post_init__,
        # so we invoke the helper directly to get the exact post_init POV semantics.
        from rpg_tasks.tasks.direct.rpg_vbdf.vbdf_render_env import _apply_tracking_viz_cfg

        _apply_tracking_viz_cfg(env_cfg, pov=True)
    log_dir = str(Path(args_ns.checkpoint).resolve().parent)
    env_cfg.log_dir = log_dir
    print(
        f"[RECORD] task={args_ns.task} stage={env_cfg.stage} pov={args_ns.pov} "
        f"act_lpf_alpha={getattr(env_cfg, 'act_lpf_alpha', 0.0)} "
        f"thrust_delta_coeff={getattr(env_cfg, 'thrust_delta_coeff', 0.0)} "
        f"camera={tuple(env_cfg.viewer.eye)}->{tuple(env_cfg.viewer.lookat)} "
        f"fov={env_cfg.video_recorder.horiz_fov_deg}",
        flush=True,
    )
    return args_ns.task, env_cfg, agent_cfg


def _build_launch_args(args_ns: argparse.Namespace) -> argparse.Namespace:
    """launch_simulation's boot config. In batch mode this is built from the FIRST spec
    (the app boots once on that spec's task/cfg); later specs only re-make the env."""
    return argparse.Namespace(
        task=args_ns.task,
        agent="sb3_cfg_entry_point",
        num_envs=args_ns.num_envs,
        seed=args_ns.seed,
        device=args_ns.device,
        video=True,
        video_length=args_ns.video_length,
        enable_cameras=True,
        visualizer=["newton"] if not args_ns.pov else [],
        checkpoint=args_ns.checkpoint,
        use_pretrained_checkpoint=False,
        real_time=False,
        keep_all_info=False,
        disable_fabric=False,
        deterministic=True,
    )


def _render_one(args_ns: argparse.Namespace, env_cfg, agent_cfg) -> str:
    """ONE clip INSIDE an already-open launch_simulation block: gym.make -> load
    VecNormalize -> PPO.load -> record -> env.close(). Returns the written mp4 path."""
    env = gym.make(args_ns.task, cfg=env_cfg, render_mode="rgb_array" if not args_ns.pov else None)

    if isinstance(agent_cfg.get("learning_rate"), dict):
        agent_cfg["learning_rate"] = float(agent_cfg["learning_rate"]["start"])
    agent_cfg = process_sb3_cfg(agent_cfg, env.unwrapped.num_envs)

    fps = 1.0 / float(env.unwrapped.step_dt)
    print(f"[RECORD] step_dt={float(env.unwrapped.step_dt):.5f} fps={fps:.3f}", flush=True)
    log_dir = str(Path(args_ns.checkpoint).resolve().parent)
    video_folder = os.path.join(log_dir, "videos", "play")
    if not args_ns.pov and not args_ns.overlay:
        _patch_ffmpeg()
        env = RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix=args_ns.out_name,
            step_trigger=lambda step: step == 0,
            video_length=args_ns.video_length,
            disable_logger=True,
            fps=fps,
        )
    env = Sb3VecEnvWrapper(env, fast_variant=True)

    # load VecNormalize if the checkpoint's run saved one
    vec = Path(args_ns.checkpoint).with_name("model_vecnormalize.pkl")
    if vec.exists():
        print(f"[RECORD] loading normalization {vec}", flush=True)
        env = VecNormalize.load(str(vec), env)
        env.training = False
        env.norm_reward = False

    print(f"[RECORD] loading checkpoint {args_ns.checkpoint}", flush=True)
    agent = PPO.load(args_ns.checkpoint, env, print_system_info=True)
    configure_seed(args_ns.seed, True)

    if args_ns.pov and args_ns.overlay:
        _record_pov_overlay(env, agent, args_ns, fps, log_dir)
    elif args_ns.pov:
        _record_pov(env, agent, args_ns, fps, log_dir)
    elif args_ns.overlay:
        _record_third_person_overlay(env, agent, args_ns, fps, log_dir)
    else:
        _record_third_person(env, agent, args_ns)
    env.close()
    return os.path.join(log_dir, "videos", "play", f"{args_ns.out_name}-step-0.mp4")


def _record_third_person(env, agent, args) -> None:
    """Record the newton viewer rgb_array frames (the 3rd-person fixed shot) via RecordVideo."""
    obs = env.reset()
    _set_third_person_newton_camera(env)
    for i in range(args.video_length):
        with torch.inference_mode():
            actions, _ = agent.predict(obs, deterministic=True)
            obs, _, _, _ = env.step(actions)
        if (i + 1) % 200 == 0:
            print(f"[RECORD] step {i+1}/{args.video_length}", flush=True)


# Unified v4 HUD palette (OpenCV is BGR)
_HUD_WHITE = (238, 238, 238)
_HUD_SB = (255, 199, 51)     # cyan-ish, state signal
_HUD_BB = (0, 235, 0)        # green, vision signal
_HUD_RED = (60, 60, 235)     # LOST badge
_HUD_PANEL = (28, 28, 20)


class VizOverlay:
    """Unified v4 telemetry HUD for all four rollout clips (documents/TELEMETRY_HUD_DESIGN.md).

    One telemetry band on every view: a brand/camera chip, the COMMON row (duration,
    ego<->actor distance, signed yaw error, ego speed) and ONE story row -- SB shows the ego
    world position (the state signal), BB shows the vision bbox (normalised x,y,w,h) with an
    IN VIEW / LOST badge. The bbox *rect* is drawn only on the BB POV view (that is the only
    clip where the detection is visible in the image itself).

    The rect is the actor's 8-corner 26 cm body box projected through the SAME double-sphere
    camera model that feeds ``last_bbox``, posed from the policy's analytic
    ``u.camera.pose(qs.R, qs.pos)`` (the analytic mount == the rendered fisheye / the drone's
    silhouette to ~3-5 px since the camera fix). Drawn as the green "vision detection" rect
    + corner ticks. Everything is a light 2D overlay baked into the recorded frames (cv2) -
    no scene primitives, no frustum, no camera motion.
    """

    def __init__(self, env, *, width: int, height: int, pov: bool, is_bb: bool):
        self.env = env
        self.w, self.h = width, height
        self.pov = pov
        self.is_bb = is_bb

    # ------------------------------------------------------------------ state
    def _ego_pos(self) -> np.ndarray:
        u = self.env.unwrapped
        return u.quad_state.pos[0].detach().cpu().numpy()

    def _ego_speed(self) -> float:
        u = self.env.unwrapped
        v = u.quad_state.vel[0].detach().cpu().numpy()
        return float(np.linalg.norm(v))

    def _dist_yaw(self):
        """Ego<->actor distance [m] and SIGNED yaw error [deg] (actor off the nose, + = left)."""
        u = self.env.unwrapped
        qs = u.quad_state
        dist = float(u.dist_to_actor[0].detach().cpu().item())
        rel = u.other_pos[0] - qs.pos[0]
        rel_n = rel / torch.linalg.norm(rel).clamp_min(1e-9)
        body_x = qs.R[0, :, 0]
        dot = float((body_x * rel_n).sum().detach().cpu().item())
        cross_z = float((torch.cross(body_x, rel_n)[2]).detach().cpu().item())
        yaw = float(np.degrees(np.arctan2(cross_z, np.clip(dot, -1.0, 1.0))))
        return dist, yaw

    def _project_corners(self):
        """Project the actor's 8 body corners through the double-sphere camera model,
        returning normalized (u,v) in [-1,1] plus a per-corner front/valid flag.

        Posed from the POLICY's analytic ``DoubleSphereCamera.pose`` -- the exact
        ``R_CW, t = u.camera.pose(qs.R, qs.pos)`` the BB policy uses to build
        ``last_bbox`` -- so the rendered rect is literally the detection the policy
        acted on. (The camera mount is verified to match the rendered fisheye / the
        drone's silhouette to ~3-5 px; the earlier "pose from the camera prim"
        workaround was a reaction to a ~287 px analytic/rendered mismatch that the
        camera fix eliminated. Never a cosmetic correction at render time.)"""
        u = self.env.unwrapped
        qs = u.quad_state
        corners_w = u.other_pos.unsqueeze(1) + torch.einsum(
            "nij,kj->nki", u.other_R, u._actor_corners
        )  # (N,8,3)
        R_CW, t = u.camera.pose(qs.R, qs.pos)
        uvv = u.camera.project(corners_w, R_CW, t, homogeneous=True)  # (N,8,3)
        un = uvv[0, :, 0].detach().cpu().numpy()
        vn = uvv[0, :, 1].detach().cpu().numpy()
        valid = uvv[0, :, 2].detach().cpu().numpy() > 0
        return un, vn, valid

    def _bbox_normalized(self):
        """BBox in normalized (x,y,w,h) units -- SAME projection as the drawn rect, so the
        HUD numbers always describe the green box on screen."""
        un, vn, valid = self._project_corners()
        if not valid.any():
            return np.zeros(4), False
        return np.array([
            un[valid].min(), vn[valid].min(),
            un[valid].max() - un[valid].min(), vn[valid].max() - vn[valid].min(),
        ]), True

    def _ds_corner_pixels(self):
        """Pixel (u,v) of the actor's 8 body corners from ``_project_corners``."""
        un, vn, valid = self._project_corners()
        upx = (un + 1.0) * 0.5 * self.w
        vpx = (vn + 1.0) * 0.5 * self.h
        return upx, vpx, valid

    # ------------------------------------------------------------------ draw
    def draw(self, frame: np.ndarray, t_s: float) -> np.ndarray:
        import cv2

        img = np.ascontiguousarray(frame, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.5, 1

        # ---- 1. BB POV only: the vision-detection rect (green) ----
        if self.pov and self.is_bb:
            upx, vpx, valid = self._ds_corner_pixels()
            if valid.any():
                xs = upx[valid]
                ys = vpx[valid]
                x0, y0 = int(round(xs.min())), int(round(ys.min()))
                x1, y1 = int(round(xs.max())), int(round(ys.max()))
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(self.w - 1, x1), min(self.h - 1, y1)
                cv2.rectangle(img, (x0, y0), (x1, y1), _HUD_BB, 2, cv2.LINE_AA)
                for (cx, cy, sx, sy) in (
                    (x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1),
                ):
                    cv2.line(img, (cx, cy), (cx + 14 * sx, cy), _HUD_BB, 2)
                    cv2.line(img, (cx, cy), (cx, cy + 14 * sy), _HUD_BB, 2)

        # ---- 2. telemetry rows ----
        dist, yaw = self._dist_yaw()
        speed = self._ego_speed()
        brand = "[BB] vision signal" if self.is_bb else "[SB] state signal"
        brand_col = _HUD_BB if self.is_bb else _HUD_SB
        cam = "onboard fisheye" if self.pov else "3rd-person"

        rows = []  # (items, accent_col, badge) per story row; common row accent = None
        rows.append(([("t", "%.2f s" % t_s), ("d", "%.2f m" % dist),
                      ("yaw", "%+5.1f deg" % yaw), ("v", "%.2f m/s" % speed)], None, None))
        if self.is_bb:
            bb, in_view = self._bbox_normalized()
            val = "(%.2f, %.2f, %.2f, %.2f)" % (bb[0], bb[1], bb[2], bb[3])
            badge = "IN VIEW" if in_view else "LOST"
            badge_col = _HUD_BB if in_view else _HUD_RED
            rows.append(([("bbox", val)], _HUD_BB, (badge, badge_col)))
        else:
            p = self._ego_pos()
            val = "(%6.2f, %6.2f, %5.2f) m" % (p[0], p[1], p[2])
            rows.append(([("pos", val)], _HUD_SB, None))

        return self._hud_panel(img, rows, brand, brand_col, cam, font, scale, thick)

    def _hud_panel(self, img, rows, brand, brand_col, cam, font, scale, thick):
        import cv2

        # layout: 3 rows (chip, common, story) in one top-left panel
        pad = 10
        row_h = cv2.getTextSize("Xg", font, scale, thick)[0][1] + 6
        # measure story row width for panel sizing
        lens = [cv2.getTextSize(brand + "  " + cam, font, scale, thick)[0][0]]
        for row in rows:
            text = "  ".join(f"{k} {v}" for k, v in row[0]) + (f"   {row[2][0]}" if row[2] else "")
            lens.append(cv2.getTextSize(text, font, scale, thick)[0][0])
        panel_w = max(lens) + 2 * pad
        panel_h = 3 * row_h + pad
        x0, y0 = 12, 12
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), _HUD_PANEL, -1)
        img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)

        yy = y0 + row_h
        img = cv2.putText(img, brand, (x0 + pad, yy), font, scale, brand_col, thick, cv2.LINE_AA)
        bw = cv2.getTextSize(brand, font, scale, thick)[0][0]
        img = cv2.putText(img, "   " + cam, (x0 + pad + bw, yy), font, scale, _HUD_WHITE, thick, cv2.LINE_AA)
        yy += row_h
        for row in rows:
            xs = x0 + pad
            for k, v in row[0]:
                img = cv2.putText(img, k, (xs, yy), font, scale, _HUD_WHITE, thick, cv2.LINE_AA)
                kw = cv2.getTextSize(k, font, scale, thick)[0][0]
                vcol = row[1] if row[1] is not None else _HUD_WHITE
                img = cv2.putText(img, " " + v, (xs + kw, yy), font, scale, vcol, thick, cv2.LINE_AA)
                xs += kw + cv2.getTextSize(" " + v, font, scale, thick)[0][0] + 14
            if row[2]:
                img = cv2.putText(img, "   " + row[2][0], (xs, yy), font, scale, row[2][1], thick, cv2.LINE_AA)
            yy += row_h
        return img


def _record_third_person_overlay(env, agent, args, fps, log_dir) -> None:
    """Record the 3rd-person viewer frames + the unified v4 telemetry HUD.

    Common row (duration / distance / yaw error / speed) on every clip; story row =
    ego world position for SB, vision bbox numbers for BB. No rect on the external shot.
    """
    unwrapped = env.unwrapped
    is_bb = ("Bbox" in type(unwrapped).__name__) or ("BB" in args.task)

    obs = env.reset()
    _set_third_person_newton_camera(env)
    frame0 = unwrapped.render()
    if frame0 is None:
        raise RuntimeError("overlay 3rd-person recording needs rgb_array render from the viewer")
    h, w = frame0.shape[:2]
    overlay = VizOverlay(env, width=w, height=h, pov=False, is_bb=is_bb)
    print(f"[OVERLAY] 3rd-person size={w}x{h} tracker={'BB' if is_bb else 'SB'} unified HUD", flush=True)

    frames = [overlay.draw(frame0, 0.0)]
    for i in range(args.video_length):
        with torch.inference_mode():
            actions, _ = agent.predict(obs, deterministic=True)
            obs, _, _, _ = env.step(actions)
        frame = unwrapped.render()
        if frame is not None:
            frames.append(overlay.draw(frame, (i + 1) / fps))
        if (i + 1) % 200 == 0:
            print(f"[RECORD] overlay step {i+1}/{args.video_length} ({len(frames)} frames)", flush=True)

    out_path = os.path.join(log_dir, "videos", "play", f"{args.out_name}-step-0.mp4")
    _write_frames_ffmpeg(frames, fps, out_path)


def _record_pov_overlay(env, agent, args, fps, log_dir) -> None:
    """Record the onboard double-sphere fisheye (POV) frames + the unified v4 HUD.

    BB draws the vision detection rect + bbox story row; SB draws the position story row.
    Derives the frame dims from the first onboard rgb frame."""
    unwrapped = env.unwrapped
    camera = getattr(unwrapped, "_onboard_camera", None)
    if camera is None:
        raise RuntimeError("--pov overlay requested but the env has no onboard camera")

    def _read_rgb():
        out = camera.data.output
        rgb = out.get("rgb", None) or out.get(camera.cfg.rgb_aov, None)
        if rgb is None:
            return None
        arr = rgb[0].detach().cpu().numpy()[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    is_bb = ("Bbox" in type(unwrapped).__name__) or ("BB" in args.task)

    obs = env.reset()
    rgb0 = _read_rgb()
    if rgb0 is None:
        raise RuntimeError("no rgb frame from onboard camera on reset")
    h, w = rgb0.shape[:2]
    overlay = VizOverlay(env, width=w, height=h, pov=True, is_bb=is_bb)
    print(f"[OVERLAY] POV size={w}x{h} tracker={'BB' if is_bb else 'SB'} unified HUD "
          f"bbox={'ON' if is_bb else 'OFF'}", flush=True)

    frames = [overlay.draw(rgb0, 0.0)]
    step = 0
    while step < args.video_length:
        with torch.inference_mode():
            actions, _ = agent.predict(obs, deterministic=True)
            obs, _, _, _ = env.step(actions)
        step += 1
        rgb = _read_rgb()
        if rgb is not None:
            frames.append(overlay.draw(rgb, step / fps))
        if step % 200 == 0:
            print(f"[RECORD] POV overlay step {step}/{args.video_length} ({len(frames)} frames)", flush=True)

    out_path = os.path.join(log_dir, "videos", "play", f"{args.out_name}-step-0.mp4")
    _write_frames_ffmpeg(frames, fps, out_path)


def _record_pov(env, agent, args, fps, log_dir) -> None:
    """Record the onboard double-sphere fisheye (POV) camera, bypassing the viewer.

    RecordVideo is viewer-centric (it captures ``render()`` frames); the onboard camera
    instead lands in ``env.unwrapped._onboard_camera.data.output["rgb"]`` after each
    physics step. We read that tensor and pipe frames to ffmpeg directly.
    """
    unwrapped = env.unwrapped
    camera = getattr(unwrapped, "_onboard_camera", None)
    if camera is None:
        raise RuntimeError("--pov requested but the env has no onboard camera (pov not enabled on cfg?)")

    obs = env.reset()
    frames = []
    step = 0
    while step < args.video_length:
        with torch.inference_mode():
            actions, _ = agent.predict(obs, deterministic=True)
            obs, _, _, _ = env.step(actions)
        step += 1
        # the onboard camera is rendered every sim.render_interval sub-steps; read its
        # latest output, convert RGB (uint8) -> numpy, append one frame per control step.
        out = camera.data.output
        rgb = out.get("rgb", None)
        if rgb is None and camera.cfg.rgb_aov in out:
            rgb = out[camera.cfg.rgb_aov]
        if rgb is None:
            continue
        arr = rgb[0].detach().cpu().numpy()[..., :3]
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 255)).astype(np.uint8)
        frames.append(arr)
        if step % 200 == 0:
            print(f"[RECORD] POV step {step}/{args.video_length} ({len(frames)} frames)", flush=True)

    out_path = os.path.join(log_dir, "videos", "play", f"{args.out_name}-step-0.mp4")
    _write_frames_ffmpeg(frames, fps, out_path)


if __name__ == "__main__":
    main()
