# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train an RL agent with Stable Baselines3."""

import argparse
import contextlib
import inspect
import logging
import os
import random
import shutil
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import rpg_tasks.tasks  # noqa: F401
import torch
from rpg_tasks.tasks.direct.rpg_base.base_env_cfg import resolve_drone
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import DirectMARLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import (
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

logger = logging.getLogger(__name__)

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="sb3_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--log_interval", type=int, default=100_000, help="Log data every n timesteps.")
parser.add_argument("--checkpoint", type=str, default=None, help="Continue the training from checkpoint.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--keep_all_info",
    action="store_true",
    default=False,
    help="Use a slower SB3 wrapper but keep all the extra training info.",
)
parser.add_argument("--log_project_name", type=str, default="rpg_isaac", help="WandB project name.")
parser.add_argument("--run_name", type=str, default=None, help="Run name suffix for the log directory and wandb run.")
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

if args_cli.video:
    args_cli.enable_cameras = True


class FreezeLogStdCallback(BaseCallback):
    """Keep policy log_std at its init value for the first ``freeze_iters`` updates."""

    def __init__(self, freeze_iters: int, verbose=0):
        super().__init__(verbose)
        self._freeze_iters = int(freeze_iters)
        self._updates = 0

    def _on_training_start(self) -> None:
        self.model.policy.log_std.requires_grad_(False)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # fires before update k+1 trains; > keeps updates 1..freeze_iters frozen
        self._updates += 1
        if self._updates > self._freeze_iters:
            self.model.policy.log_std.requires_grad_(True)


class ExtrasLogCallback(BaseCallback):
    """Calls the env's ``on_train_log(it, logger)`` hook after each rollout for wandb viz."""

    def __init__(self, traj_log_interval: int = 50, verbose=0):
        super().__init__(verbose)
        self._update_count = 0
        self.traj_log_interval = traj_log_interval
        self._disable_train_log_hook = False

    def _unwrap_env(self):
        """Resolve to the underlying task env across VecEnv and Gym wrappers."""
        env = self.training_env
        seen = set()
        while env is not None and id(env) not in seen:
            seen.add(id(env))
            if hasattr(env, "venv") and getattr(env, "venv") is not env:
                env = getattr(env, "venv")
                continue
            if hasattr(env, "env") and getattr(env, "env") is not env:
                env = getattr(env, "env")
                continue
            break
        return getattr(env, "unwrapped", env)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        env = self._unwrap_env()
        self._update_count += 1
        if self._disable_train_log_hook:
            return
        try:
            on_train_log = getattr(env, "on_train_log", None)
            if not callable(on_train_log):
                return
            viz_interval = int(getattr(env, "_viz_interval", 0))
            should_eval_log = viz_interval > 0 and self._update_count % viz_interval == 0
            policy = getattr(self.model, "policy", None)
            if should_eval_log and policy is not None and callable(getattr(policy, "train", None)):
                was_training = bool(getattr(policy, "training", False))
                policy.eval()
                try:
                    on_train_log(self._update_count, getattr(self.model, "logger", None) or SimpleNamespace(tot_timesteps=int(self.num_timesteps)))
                finally:
                    policy.train(was_training)
            else:
                on_train_log(self._update_count, getattr(self.model, "logger", None) or SimpleNamespace(tot_timesteps=int(self.num_timesteps)))
        except Exception as exc:
            self._disable_train_log_hook = True
            logger.warning("Disabling env.on_train_log hook for SB3 due to callback error: %s", exc)


def cleanup_pbar(*args):
    """Stop training and clean up the progress bar on ctrl+c."""
    import gc

    tqdm_objects = [obj for obj in gc.get_objects() if "tqdm" in type(obj).__name__]
    for tqdm_object in tqdm_objects:
        if "tqdm_rich" in type(tqdm_object).__name__:
            tqdm_object.close()
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, cleanup_pbar)


def main():
    """Train with stable-baselines agent."""
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
    with launch_simulation(env_cfg, args_cli):
        import wandb
        from wandb.integration.sb3 import WandbCallback

        # randomly sample a seed if seed = -1
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)

        # override configurations with non-hydra CLI arguments
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]

        # set the environment seed
        env_cfg.seed = agent_cfg["seed"]
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # directory for logging into
        run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args_cli.run_name:
            run_info += f"_{args_cli.run_name}"
        log_root_path = os.path.abspath(os.path.join("logs", "sb3", args_cli.task))
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        print(f"Exact experiment name requested from command line: {run_info}")
        log_dir = os.path.join(log_root_path, run_info)
        # The env constructor resolves the airframe, so dumping before that
        # records kolibri's sim params for every drone. resolve_drone rebuilds
        # from DRONES each call, so doing it here just moves it earlier.
        os.makedirs(log_dir, exist_ok=True)
        (Path(log_dir) / "command.txt").write_text(" ".join(sys.orig_argv))

        # Handle an adaptive learning-rate schedule. Pop it before process_sb3_cfg (which only accepts
        # float/str) and re-inject the callable afterwards. YAML form:
        #   learning_rate: {start: 3e-4, end: 1e-5, end_fraction: 1.0}
        custom_lr_schedule = None
        lr_cfg = agent_cfg.get("learning_rate")
        if isinstance(lr_cfg, dict):
            lr_start = float(lr_cfg["start"])
            lr_end = float(lr_cfg.get("end", lr_start))
            lr_end_fraction = float(lr_cfg.get("end_fraction", 1.0))
            custom_lr_schedule = LinearSchedule(lr_start, lr_end, lr_end_fraction)
            agent_cfg["learning_rate"] = lr_start

        agent_cfg = process_sb3_cfg(agent_cfg, env_cfg.scene.num_envs)
        if custom_lr_schedule is not None:
            agent_cfg["learning_rate"] = custom_lr_schedule

        policy_arch = agent_cfg.pop("policy")
        if policy_arch == "TanhMlpPolicy":
            from tanh_policy import TanhMlpPolicy

            policy_arch = TanhMlpPolicy
        n_timesteps = agent_cfg.pop("n_timesteps", None)
        freeze_std_iters = int(agent_cfg.pop("freeze_std_iters", 0))
        agent_cfg.pop("max_iterations", None)
        if n_timesteps is None:
            raise KeyError("'n_timesteps' must be set in agent config.")

        # set the IO descriptors export flag if requested
        if isinstance(env_cfg, ManagerBasedRLEnvCfg):
            env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        else:
            logger.warning(
                "IO descriptors are only supported for manager based RL environments."
                " No IO descriptors will be exported."
            )

        # set the log directory for the environment
        env_cfg.log_dir = log_dir

        # step-derived curricula need the agent's rollout length to convert env steps to iterations;
        # 0 disables all curricula (env stays at nominal)
        if (
            hasattr(env_cfg, "curriculum_steps_per_iteration")
            and env_cfg.curriculum_steps_per_iteration > 0
            and agent_cfg.get("n_steps")
        ):
            env_cfg.curriculum_steps_per_iteration = int(agent_cfg["n_steps"])

        resolve_drone(env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        shutil.copy2(inspect.getfile(type(env_cfg)), os.path.join(log_dir, "params", "env_cfg.py"))

        # create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        # wrap for video recording
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        start_time = time.time()

        # wrap around environment for stable baselines
        env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

        norm_keys = {"normalize_input", "normalize_value", "clip_obs"}
        norm_args = {}
        for key in norm_keys:
            if key in agent_cfg:
                norm_args[key] = agent_cfg.pop(key)

        # value normalization is useful on its own: it rescales the critic's
        # targets without touching observations, so a policy trained with it
        # still runs unchanged against a bare env
        if norm_args.get("normalize_input") or norm_args.get("normalize_value"):
            print(f"Normalizing, {norm_args=}")
            env = VecNormalize(
                env,
                training=True,
                norm_obs=norm_args.get("normalize_input", False),
                norm_reward=norm_args.get("normalize_value", False),
                clip_obs=norm_args.get("clip_obs", 100.0),
                gamma=agent_cfg["gamma"],
                clip_reward=np.inf,
            )

        # wandb run
        run = wandb.init(
            project=args_cli.log_project_name,
            name=run_info,
            config={**agent_cfg, "task": args_cli.task, "num_envs": env_cfg.scene.num_envs},
            sync_tensorboard=True,
            monitor_gym=False,
            save_code=True,
            dir=log_dir,
        )

        # create agent from stable baselines
        agent = PPO(policy_arch, env, verbose=1, tensorboard_log=log_dir, **agent_cfg)
        if args_cli.checkpoint is not None:
            agent = agent.load(args_cli.checkpoint, env, print_system_info=True)
        callbacks = [
            CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix="model", verbose=1),
            WandbCallback(gradient_save_freq=0, model_save_path=log_dir, verbose=1),
            ExtrasLogCallback(),
        ]
        if freeze_std_iters > 0:
            callbacks.append(FreezeLogStdCallback(freeze_std_iters))

        # train the agent
        with contextlib.suppress(KeyboardInterrupt):
            agent.learn(total_timesteps=n_timesteps, callback=callbacks, progress_bar=True, log_interval=1)

        # save the final model
        agent.save(os.path.join(log_dir, "model"))
        print("Saving to:", os.path.join(log_dir, "model.zip"))
        if isinstance(env, VecNormalize):
            print("Saving normalization")
            env.save(os.path.join(log_dir, "model_vecnormalize.pkl"))
        print(f"Training time: {round(time.time() - start_time, 2)} seconds")

        env.close()
        wandb.finish()


if __name__ == "__main__":
    main()
