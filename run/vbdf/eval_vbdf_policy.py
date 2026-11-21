#!/bin/sh
''':'
# Re-exec under Isaac Sim's python so ./this_file.py works from any shell.
exec "${ISAAC_PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}" "$0" "$@"
'''
"""Headless rollout of a VBDF policy in the test env (flightmare ``main.test_env``),
same metrics as the flightmare Eval/Rollout CSVs.

Policy sources:
  (default)                                     latest run of --task under run/vbdf/logs/sb3/
  --checkpoint model.zip                        a specific SB3 run (+ model_vecnormalize.pkl next to it)
  --csv traj_04000.csv                          only score an existing rollout CSV (--out_csv or flightmare Eval)
  --flightplot                                  also write the flightpy Eval/ layout next to the checkpoint
                                                (Eval/traj_<step>.csv, Eval/obs_<step>.csv, config_tracking.yaml)
                                                for flight_ws/src/flight/flightmare/flightplot (streamlit)

Usage (container):
  /workspace/isaaclab/_isaac_sim/python.sh run/vbdf/eval_vbdf_policy.py --task Rpg-Vbdf-Hover-v0 \
      --steps 1000 --device cuda:0 physics=agisim_newton_mjwarp
"""

import argparse
import csv
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
            f.write("# written by eval_vbdf_policy.py --flightplot (rpg_isaac run; not a flightpy config)\n")
            f.write("rewards:\n  components: [%s]\n" % ", ".join(raw.reward_names))
            f.write("main:\n  task: %s\n" % args_cli.task)
    print(f"[eval] flightplot files -> {ev}/{{traj,obs}}_{step:05d}.csv")


def print_scores(title, s):
    print(f"[eval] {title}")
    for k, v in s.items():
        print(f"    {k:26s} {v:.4f}" if isinstance(v, float) else f"    {k:26s} {v}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Rpg-Vbdf-Hover-v0")
    parser.add_argument("--checkpoint", default=None, help="model.zip; default: latest run of --task under run/vbdf/logs/sb3/")
    parser.add_argument("--csv", default=None, help="score an existing rollout CSV and exit")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--random_init", action="store_true", help="train-mode resets instead of the test env")
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of the tanh mean")
    parser.add_argument("--flightplot", action="store_true", help="write flightpy-style Eval/ files next to --checkpoint")
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
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        raw = env.unwrapped
        dev = raw.device

        from stable_baselines3 import PPO

        model = PPO.load(args_cli.checkpoint, device="cpu")
        print(f"[eval] policy log_std {model.policy.log_std.detach().numpy().round(3)}")
        vn_path = args_cli.checkpoint.replace("/model", "/model_vecnormalize").replace(".zip", ".pkl")
        obs_rms = None
        if os.path.exists(vn_path):
            import pickle

            with open(vn_path, "rb") as f:
                vn = pickle.load(f)
            obs_rms = vn.obs_rms
            clip_obs = vn.clip_obs
            print(f"[eval] obs normalisation from {vn_path}")

        def norm_fn(obs):
            o = obs.cpu().numpy()
            if obs_rms is not None:
                o = np.clip((o - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -clip_obs, clip_obs)
            return o

        def act_fn(obs):
            a, _ = model.predict(norm_fn(obs), deterministic=not args_cli.stochastic)
            return torch.as_tensor(a, device=dev)

        obs, _ = env.reset()
        rows = []
        obs_rows = []
        crashes = 0
        tracking = hasattr(raw, "other_pos")
        agg = {"dist": [], "rew": [], "crash": 0, "eps": 0, "view": []}
        for step in range(args_cli.steps):
            act = act_fn(obs["policy"])
            obs, rew, term, trunc, _ = env.step(act)
            if args_cli.flightplot:
                obs_rows.append(np.asarray(norm_fn(obs["policy"]))[0])
            qs = raw.quad_state
            if raw.num_envs > 1:  # batch aggregate over all envs
                if tracking:
                    agg["dist"].append(raw.dist_to_actor.mean().item())
                    agg["view"].append((raw.view_angle * 180.0 / 3.141592653589793).mean().item())
                else:
                    agg["dist"].append(torch.linalg.norm(qs.pos - raw.target_point, dim=1).mean().item())
                agg["rew"].append(rew.mean().item())
                agg["crash"] += int(term.sum().item())
                agg["eps"] += int((term | trunc).sum().item())
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
        if args_cli.out_csv:
            df.to_csv(args_cli.out_csv)
            print(f"[eval] wrote {args_cli.out_csv}")
        if args_cli.flightplot:
            write_flightplot(args_cli, raw, df, np.asarray(obs_rows), int(model.num_timesteps))
        s = score_tracking(df) if tracking else score_hover(df, target=tuple(raw.target_point.cpu().numpy()))
        s["crash_episodes"] = crashes
        if raw.num_envs > 1:
            half = len(agg["dist"]) // 2
            s["all_envs_dist_mean"] = float(np.mean(agg["dist"]))
            s["all_envs_dist_mean_2nd_half"] = float(np.mean(agg["dist"][half:]))
            s["all_envs_rew_per_step"] = float(np.mean(agg["rew"]))
            s["all_envs_crash_episodes"] = agg["crash"]
            s["all_envs_episodes_ended"] = agg["eps"]
            if tracking:
                s["all_envs_view_angle_deg"] = float(np.mean(agg["view"]))
        print_scores(f"{args_cli.task} {args_cli.checkpoint}", s)
        print("EVAL_DONE " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items()))
        env.close()


if __name__ == "__main__":
    main()
