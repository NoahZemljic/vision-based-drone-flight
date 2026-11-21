#!/bin/sh
''':'
# Re-exec under Isaac Sim's python so ./this_file.py works from any shell.
exec "${ISAAC_PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}" "$0" "$@"
'''
"""Seeded evaluation used for the final report.

Reseed after PPO.load and before every env.reset: loading a policy changes global
RNG state. Distance and measured body-rate variation use the second half;
angular error and crash counts use the full rollout. CSV output is environment 0;
all_envs_* JSON values aggregate the fleet. Use run_fleet_eval.py to select the
saved per-model overrides, evaluation seeds and delay seed from the report.
"""

import argparse
import hashlib
import csv
import json
import os
import sys

import numpy as np


class Table(dict):
    """Minimal column store: Table[col] -> np.ndarray, Table[[c1, c2]] -> (n, k) array."""

    def __getitem__(self, key):
        if isinstance(key, list):
            return np.stack([dict.__getitem__(self, k) for k in key], axis=1)
        return dict.__getitem__(self, key)

    def __len__(self):
        return len(next(iter(self.values())))

    @staticmethod
    def read_csv(path):
        with open(path) as f:
            rows = list(csv.DictReader(f))
        cols = [c for c in rows[0].keys() if c]
        return Table({c: np.array([float(r[c]) for r in rows]) for c in cols})

    @staticmethod
    def from_rows(rows):
        return Table({c: np.array([r[c] for r in rows], dtype=float) for c in rows[0]})

    def to_csv(self, path):
        cols = list(self.keys())
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for i in range(len(self)):
                w.writerow([self[c][i] for c in cols])

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def score_hover(df, target=(0.0, 0.0, 1.0)):
    p = df[["px", "py", "pz"]]
    d = np.linalg.norm(p - np.asarray(target), axis=1)
    v = np.linalg.norm(df[["vx", "vy", "vz"]], axis=1)
    half = len(df) // 2
    out = {
        "steps": len(df),
        "dist_mean": d.mean(),
        "dist_mean_2nd_half": d[half:].mean(),
        "dist_final": d[-1],
        "pos_std_2nd_half": p[half:].std(axis=0).mean(),
        "speed_mean_2nd_half": v[half:].mean(),
        "cmd_thrust_std_2nd_half": df["c"][half:].std(),
        "cmd_rate_std_2nd_half": df[["wx", "wy", "wz"]][half:].std(axis=0).mean(),
    }
    if "total_rew" in df:  # absent in deployment rollouts (flightpilot.rollout_log)
        out["rew_per_step"] = df["total_rew"].mean()
    return out


def score_tracking(df):
    p = df[["px", "py", "pz"]]
    o = df[["other_pos_x", "other_pos_y", "other_pos_z"]]
    d = np.linalg.norm(p - o, axis=1)
    # camera axis = body x (r_BC = 0)
    qw, qx, qy, qz = (df[c] for c in ("qw", "qx", "qy", "qz"))
    xb = np.stack([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qw * qz), 2 * (qx * qz - qw * qy)], axis=1)
    dirs = (o - p) / np.maximum(d[:, None], 1e-9)
    ang = np.degrees(np.arccos(np.clip((xb * dirs).sum(1), -1, 1)))
    out = {
        "steps": len(df),
        "dist_mean": d.mean(),
        "dist_err_mean": np.abs(d - 1.5).mean(),
        "dist_err_p95": np.percentile(np.abs(d - 1.5), 95),
        "view_angle_mean_deg": ang.mean(),
        "in_view_45deg_frac": (ang < 45).mean(),
    }
    if "total_rew" in df:  # absent in deployment rollouts (flightpilot.rollout_log)
        out["rew_per_step"] = df["total_rew"].mean()
    # per-component reward means: any column that is not state / command / bookkeeping
    known = {
        "t", "px", "py", "pz", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "omex", "omey", "omez",
        "accx", "accy", "accz", "c", "wx", "wy", "wz", "flightmode", "done", "total_rew",
        "other_pos_x", "other_pos_y", "other_pos_z", "other_att_w", "other_att_x", "other_att_y",
        "other_att_z", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    }
    for c in df:
        if c not in known:
            out[f"rew_{c}"] = df[c].mean()
    return out


def write_flightplot(args_cli, raw, df, obs, step):
    """flightpy ``PPO.eval`` layout so flightplot's streamlit app can browse rpg_isaac runs."""
    run_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
    ev = os.path.join(run_dir, "Eval")
    os.makedirs(ev, exist_ok=True)
    cols = list(df.keys())
    with open(os.path.join(ev, f"traj_{step:05d}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + cols)  # leading index column ("Unnamed: 0" in pandas), as pandas.to_csv writes
        for i in range(len(df)):
            w.writerow([i] + [df[c][i] for c in cols])
    with open(os.path.join(ev, f"obs_{step:05d}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + ["obs_%2i" % i for i in range(obs.shape[1])])
        for i in range(obs.shape[0]):
            w.writerow([i] + list(obs[i]))
    cfg_path = os.path.join(run_dir, "config_tracking.yaml")
    if not os.path.exists(cfg_path):  # flightplot reads rewards.components; keep the flightpy shape
        with open(cfg_path, "w") as f:
            f.write("# written by eval_seeded.py --flightplot (rpg_isaac run; not a flightpy config)\n")
            f.write("rewards:\n  components: [%s]\n" % ", ".join(raw.reward_names))
            f.write("main:\n  task: %s\n" % args_cli.task)
    print(f"[eval] flightplot files -> {ev}/{{traj,obs}}_{step:05d}.csv")


def _is_scalar(v):
    return isinstance(v, (int, float, str, bool)) or v is None


def print_scores(title, s):
    print(f"[eval] {title}")
    for k, v in s.items():
        if not _is_scalar(v):
            continue
        print(f"    {k:26s} {v:.4f}" if isinstance(v, float) else f"    {k:26s} {v}")


def _term_ids(term):
    a = np.asarray(term.detach().cpu().numpy()).reshape(-1)
    return np.flatnonzero(a > 0.5)


def _nearest_wall(p, lo, hi):
    return float(min(p[0] - lo[0], hi[0] - p[0],
                     p[1] - lo[1], hi[1] - p[1],
                     p[2] - lo[2], hi[2] - p[2]))


def _popstd(a, axis=0):
    return np.asarray(a, dtype=float).std(axis=axis)


def score_env0_from_df(df, tracking):
    """2nd-half env-0 rates / dist_err — population standard deviations and tracking distance error."""
    n = len(df)
    h = n // 2
    out = {}
    if "omex" in df:
        out["omex_std2"] = float(_popstd(df["omex"][h:]))
        out["omey_std2"] = float(_popstd(df["omey"][h:]))
        out["omez_std2"] = float(_popstd(df["omez"][h:]))
    if tracking and "other_pos_x" in df:
        p = df[["px", "py", "pz"]]
        o = df[["other_pos_x", "other_pos_y", "other_pos_z"]]
        d = np.linalg.norm(p - o, axis=1)
        out["dist_err"] = float(np.abs(d[h:] - 1.5).mean())
        if "bbox_w" in df:
            out["bbox_lost_frac"] = float((np.asarray(df["bbox_w"][h:]) <= 1e-9).mean())
    if "done" in df:
        out["crash_env0"] = float((np.asarray(df["done"]) > 0.5).any())
    return out


def score_fleet(dist_tn, ome_tn, view_tn=None, bboxw_tn=None, tracking=True):
    """32-env (or N-env) metrics matching env-0 methodology.

    dist_err = mean |d-1.5| over all envs × 2nd-half steps (tracking only).
    omex/y/z_std2 = mean over envs of that env's 2nd-half population std.
    """
    dist_tn = np.asarray(dist_tn, dtype=float)
    ome_tn = np.asarray(ome_tn, dtype=float)
    t = dist_tn.shape[0]
    h = t // 2
    d2 = dist_tn[h:]
    out = {
        "all_envs_dist_mean": float(dist_tn.mean()),
        "all_envs_dist_mean_2nd_half": float(d2.mean()),
    }
    if tracking:
        out["all_envs_dist_err"] = float(np.abs(d2 - 1.5).mean())
        out["all_envs_dist_err_full"] = float(np.abs(dist_tn - 1.5).mean())
    ome2 = ome_tn[h:]
    per_env = _popstd(ome2, axis=0)  # [N, 3]
    out["all_envs_omex_std2"] = float(per_env[:, 0].mean())
    out["all_envs_omey_std2"] = float(per_env[:, 1].mean())
    out["all_envs_omez_std2"] = float(per_env[:, 2].mean())
    if view_tn is not None:
        ang = np.degrees(np.asarray(view_tn, dtype=float))
        out["all_envs_in_view_45"] = float((ang < 45.0).mean())
        out["all_envs_view_angle_deg"] = float(ang.mean())
    if bboxw_tn is not None:
        out["all_envs_bbox_lost_frac"] = float((np.asarray(bboxw_tn[h:], dtype=float) <= 1e-9).mean())
    return out


def write_scores_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"[eval] wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Rpg-Vbdf-Hover-v0")
    parser.add_argument("--checkpoint", default=None, help="model.zip; default: latest run of --task under run/vbdf/logs/sb3/")
    parser.add_argument("--csv", default=None, help="score an existing rollout CSV and exit")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--out_json", default=None, help="write the score dict as JSON (single-seed)")
    parser.add_argument("--out_dir", default=None, help="write {out_name}_{seed}.{csv,json} here (multi-seed)")
    parser.add_argument("--out_name", default=None, help="file prefix when using --out_dir (default: task name)")
    parser.add_argument("--write_csv", action="store_true", help="also dump env-0 CSVs when using --out_dir")
    parser.add_argument("--seed", type=int, default=123, help="seed for the eval env (re-seeded after model load)")
    parser.add_argument("--seeds", default=None, help="comma-separated seeds; one Isaac session, reset per seed")
    parser.add_argument("--random_init", action="store_true", help="train-mode resets instead of the test env")
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of the tanh mean")
    parser.add_argument("--flightplot", action="store_true", help="write flightpy-style Eval/ files next to --checkpoint")
    # consistent-random eval: pin the command-latency draw to this seed (0 = off).
    # When > 0, every eval run of every model draws the SAME [0, random_delay) delay
    # sequence regardless of --seed / model, so compared runs share one delay fixture.
    parser.add_argument("--delay_seed", type=int, default=0)
    if any(a == "--csv" for a in sys.argv):
        args, _ = parser.parse_known_args()
        df = Table.read_csv(args.csv)
        scorer = score_hover if "other_pos_x" not in df else score_tracking
        print_scores(args.csv, scorer(df))
        return

    from isaaclab_tasks.utils import (
        add_launcher_args,
        get_checkpoint_path,
        launch_simulation,
        resolve_task_config,
        setup_preset_cli,
    )

    add_launcher_args(parser)
    args_cli, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0]] + hydra_args
    import gymnasium as gym
    import torch
    import isaaclab_tasks  # noqa: F401
    import rpg_tasks.tasks  # noqa: F401

    sys.path.insert(0, os.path.join(REPO, "scripts", "sb3"))
    import tanh_policy  # noqa: F401, E402

    if args_cli.checkpoint is None:
        # same resolution as scripts/sb3/play.py: latest run dir, model.zip preferred
        # over the newest periodic model_<n>.zip (short or interrupted runs)
        log_root = os.path.join(REPO, "run", "vbdf", "logs", "sb3", args_cli.task)
        args_cli.checkpoint = get_checkpoint_path(
            log_root, ".*", r"model_.*\.zip", sort_alpha=False, preferred_checkpoint=r"model\.zip"
        )
        print(f"[eval] latest checkpoint: {args_cli.checkpoint}")

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, "sb3_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.test_env = not args_cli.random_init
    env_cfg.seed = args_cli.seed
    if args_cli.delay_seed > 0:
        env_cfg.delay_seed = args_cli.delay_seed
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        raw = env.unwrapped
        dev = raw.device

        from stable_baselines3 import PPO

        model = PPO.load(args_cli.checkpoint, device="cpu")
        print(f"[eval] policy log_std {model.policy.log_std.detach().numpy().round(3)}")
        # Sit next to model.zip. Do NOT str.replace("/model", ...) — that hits
        # deliverable/models/ and silently drops vecnormalize (raw obs → crash).
        vn_path = os.path.join(os.path.dirname(args_cli.checkpoint), "model_vecnormalize.pkl")
        obs_rms = None
        if os.path.exists(vn_path):
            import pickle

            with open(vn_path, "rb") as f:
                vn = pickle.load(f)
            obs_rms = vn.obs_rms
            clip_obs = vn.clip_obs
            print(f"[eval] obs normalisation from {vn_path}")
        else:
            raise FileNotFoundError(f"Matching observation normalization is required: {vn_path}")

        def norm_fn(obs):
            o = obs.cpu().numpy()
            if obs_rms is not None:
                o = np.clip((o - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -clip_obs, clip_obs)
            return o

        def act_fn(obs):
            a, _ = model.predict(norm_fn(obs), deterministic=not args_cli.stochastic)
            return torch.as_tensor(a, device=dev)

        # re-seed AFTER model load so env.seed takes effect (loading the SB3 model
        # resets the global RNG, which otherwise neutralises the seed)
        from isaaclab.utils.seed import configure_seed

        tracking = hasattr(raw, "other_pos")
        seed_list = [int(x) for x in args_cli.seeds.split(",")] if args_cli.seeds else [args_cli.seed]
        out_name = args_cli.out_name or os.path.basename(os.path.dirname(os.path.abspath(args_cli.checkpoint)))

        n_env = int(raw.num_envs)
        lo = raw.world_lo.detach().cpu().numpy()
        hi = raw.world_hi.detach().cpu().numpy()

        def rollout(seed):
            configure_seed(seed)
            print(f"[eval] re-seeded to {seed} before reset")
            obs, _ = env.reset()
            rows = []
            obs_rows = []
            crashes = 0
            fleet_dist, fleet_ome, fleet_view, fleet_bbox = [], [], [], []
            agg = {"rew": [], "crash": 0, "eps": 0}
            crash_per_env = np.zeros(n_env, dtype=int)
            crash_events = []
            for step in range(args_cli.steps):
                qs_pre = raw.quad_state
                pos_pre = qs_pre.pos.detach().cpu().numpy().copy()
                vel_pre = qs_pre.vel.detach().cpu().numpy().copy()
                other_pre = raw.other_pos.detach().cpu().numpy().copy() if tracking else None
                act = act_fn(obs["policy"])
                obs, rew, term, trunc, _ = env.step(act)
                if args_cli.flightplot:
                    obs_rows.append(np.asarray(norm_fn(obs["policy"]))[0])
                qs = raw.quad_state
                if n_env > 1:
                    if tracking:
                        fleet_dist.append(raw.dist_to_actor.detach().cpu().numpy().copy())
                        fleet_view.append(raw.view_angle.detach().cpu().numpy().copy())
                        fleet_bbox.append(raw.last_bbox[:, 2].detach().cpu().numpy().copy())
                    else:
                        fleet_dist.append(torch.linalg.norm(qs.pos - raw.target_point, dim=1).detach().cpu().numpy().copy())
                    fleet_ome.append(qs.ome.detach().cpu().numpy().copy())
                    agg["rew"].append(rew.mean().item())
                    agg["crash"] += int(term.sum().item())
                    agg["eps"] += int((term | trunc).sum().item())
                for ei in _term_ids(term):
                    crash_per_env[int(ei)] += 1
                    p = pos_pre[int(ei)]
                    ev = {
                        "env": int(ei),
                        "step": int(step),
                        "px": float(p[0]), "py": float(p[1]), "pz": float(p[2]),
                        "speed": float(np.linalg.norm(vel_pre[int(ei)])),
                        "nearest_wall": _nearest_wall(p, lo, hi),
                    }
                    if other_pre is not None:
                        ev["dist_actor"] = float(np.linalg.norm(p - other_pre[int(ei)]))
                    crash_events.append(ev)
                e = 0
                row = {
                    "px": qs.pos[e, 0].item(), "py": qs.pos[e, 1].item(), "pz": qs.pos[e, 2].item(),
                    "qw": qs.att[e, 0].item(), "qx": qs.att[e, 1].item(), "qy": qs.att[e, 2].item(), "qz": qs.att[e, 3].item(),
                    "vx": qs.vel[e, 0].item(), "vy": qs.vel[e, 1].item(), "vz": qs.vel[e, 2].item(),
                    "omex": qs.ome[e, 0].item(), "omey": qs.ome[e, 1].item(), "omez": qs.ome[e, 2].item(),
                    "accx": qs.acc[e, 0].item(), "accy": qs.acc[e, 1].item(), "accz": qs.acc[e, 2].item(),
                    "c": raw.pi_act[e, 0].item(), "wx": raw.pi_act[e, 1].item(), "wy": raw.pi_act[e, 2].item(), "wz": raw.pi_act[e, 3].item(),
                    "flightmode": int(raw.flight_mode[e].item()),
                    "done": int(bool(term[e]) or bool(trunc[e])),
                }
                for i, name in enumerate(raw.reward_names):
                    row[name] = raw.reward_components[e, i].item()
                if tracking:
                    row.update({
                        "other_pos_x": raw.other_pos[e, 0].item(), "other_pos_y": raw.other_pos[e, 1].item(), "other_pos_z": raw.other_pos[e, 2].item(),
                        "other_att_w": raw.other_att[e, 0].item(), "other_att_x": raw.other_att[e, 1].item(), "other_att_y": raw.other_att[e, 2].item(), "other_att_z": raw.other_att[e, 3].item(),
                        "bbox_x": raw.last_bbox[e, 0].item(), "bbox_y": raw.last_bbox[e, 1].item(), "bbox_w": raw.last_bbox[e, 2].item(), "bbox_h": raw.last_bbox[e, 3].item(),
                    })
                rows.append(row)
                if bool(term[e]):
                    crashes += 1
            df = Table.from_rows(rows)
            s = score_tracking(df) if tracking else score_hover(df, target=tuple(raw.target_point.cpu().numpy()))
            s.update(score_env0_from_df(df, tracking))
            s["crash_episodes"] = crashes
            s["seed"] = seed
            s["steps"] = args_cli.steps
            s["num_envs"] = n_env
            s["all_envs_crash_per_env"] = [int(x) for x in crash_per_env]
            s["all_envs_crash_n_envs"] = int((crash_per_env > 0).sum())
            s["all_envs_crash_max_per_env"] = int(crash_per_env.max()) if n_env else 0
            s["crash_events"] = crash_events
            if fleet_dist:
                s.update(score_fleet(
                    np.stack(fleet_dist),
                    np.stack(fleet_ome),
                    view_tn=np.stack(fleet_view) if fleet_view else None,
                    bboxw_tn=np.stack(fleet_bbox) if fleet_bbox else None,
                    tracking=tracking,
                ))
                s["all_envs_rew_per_step"] = float(np.mean(agg["rew"]))
                s["all_envs_crash_episodes"] = agg["crash"]
                s["all_envs_episodes_ended"] = agg["eps"]
            else:
                s["all_envs_crash_episodes"] = int(crash_per_env.sum())
            return df, s, (np.asarray(obs_rows) if obs_rows else None)

        with open(args_cli.checkpoint, "rb") as f:
            checkpoint_sha256 = hashlib.sha256(f.read()).hexdigest()
        with open(vn_path, "rb") as f:
            normalization_sha256 = hashlib.sha256(f.read()).hexdigest()
        for seed in seed_list:
            df, s, obs_arr = rollout(seed)
            csv_path = args_cli.out_csv
            json_path = args_cli.out_json
            if args_cli.out_dir:
                os.makedirs(args_cli.out_dir, exist_ok=True)
                stem = os.path.join(args_cli.out_dir, f"{out_name}_{seed}")
                csv_path = stem + ".csv" if args_cli.write_csv else None
                json_path = stem + ".json"
            elif csv_path and not json_path:
                json_path = os.path.splitext(csv_path)[0] + ".json"
            if csv_path:
                df.to_csv(csv_path)
                print(f"[eval] wrote {csv_path}")
            if args_cli.flightplot and obs_arr is not None:
                write_flightplot(args_cli, raw, df, obs_arr, int(model.num_timesteps))
            if json_path:
                payload = {
                    "cell": out_name,
                    "task": args_cli.task,
                    "checkpoint": args_cli.checkpoint,
                    "checkpoint_sha256": checkpoint_sha256,
                    "normalization_sha256": normalization_sha256,
                    "environment_overrides": [t for t in hydra_args if t.startswith("env.")],
                    "delay_seed": args_cli.delay_seed,
                    **s,
                }
                write_scores_json(json_path, payload)
            print_scores(f"{args_cli.task} seed={seed} {args_cli.checkpoint}", s)
            print(
                f"[eval] crashes: total={s.get('all_envs_crash_episodes', 0)} "
                f"envs={s.get('all_envs_crash_n_envs', 0)} "
                f"max/env={s.get('all_envs_crash_max_per_env', 0)} "
                f"events={len(s.get('crash_events') or [])}"
            )
            print(
                "EVAL_DONE "
                + " ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in s.items() if _is_scalar(v)
                )
            )
        env.close()


if __name__ == "__main__":
    main()
