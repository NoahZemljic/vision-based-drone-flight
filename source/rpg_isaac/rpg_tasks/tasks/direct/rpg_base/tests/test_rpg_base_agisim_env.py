# test_rpg_base_agisim_env.py
from __future__ import annotations

import argparse

# --------------------------------------------------------------
# 1. Import AppLauncher first (safe to import before sim startup)
# --------------------------------------------------------------
from isaaclab.app import AppLauncher

# Add CLI args
parser = argparse.ArgumentParser(description="Barebones RPGBaseAgisimEnv test.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Ensure headless and single environment
args_cli.enable_cameras = False

# --------------------------------------------------------------
# 2. Launch the simulator BEFORE importing any isaaclab modules
# --------------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import time

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------
# 3. Now safely import IsaacLab/Omni modules
# --------------------------------------------------------------
import torch

from rpg_tasks.tasks.direct.rpg_base.base_env import RPGBaseEnv
from rpg_tasks.tasks.direct.rpg_base.base_env_cfg import RPGBaseEnvCfg

dtype_str = "float32"
dtype = torch.float64 if dtype_str == "float64" else torch.float32


def main():
    print("[INFO] Creating RPGBaseEnv (agisim)...")

    cfg = RPGBaseEnvCfg(ctrl_dt=0.02, integrator_dt=0.001, quad_state_dtype=dtype_str)
    cfg.scene.num_envs = 1
    cfg.rpg_sim_params.integrator = "IntegratorRK4"
    cfg.episode_length_s = 100000.0

    cfg.backend = "agisim"
    env = RPGBaseEnv(cfg=cfg, render_mode="human")
    print(f"joint names: {env.robot.joint_names}")

    # Dictionary to record states for plotting
    state_history = {
        "time": [],
        "pos": [],
        "vel": [],
        "att": [],
    }

    # List to store logging data for poses:
    # [time, x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz, ax, ay, az,
    #  tau_x, tau_y, tau_z, mot_0..3, motdes_0..3, E, ucap, ubat, bfli_0..2]
    log_poses = []

    # Benchmarking setup
    benchmark_duration = 15.0  # seconds real time
    step = 0
    start_time = time.time()
    start_sim_time = env._sim_step_counter * env.physics_dt
    print(f"[BENCHMARK] Starting benchmark for {benchmark_duration}s real time...")
    print(f"[BENCHMARK] Number of environments: {env.num_envs}")
    print(f"[BENCHMARK] Physics dt: {env.physics_dt}s, Step dt: {env.step_dt}s")

    timing_dict = {
        "[step] pre_decimation_t": 0.0,
        "[step] decimation_apply_action_t": 0.0,
        "[step] decimation_write_data_t": 0.0,
        "[step] decimation_sim_step_t": 0.0,
        "[step] decimation_render_t": 0.0,
        "[step] decimation_scene_update_t": 0.0,
        "[step] post_step_t": 0.0,
        "[test_script] step_t": 0.0,
        "[test_script] log_t": 0.0,
        "[test_script] contact_check": 0.0,
    }

    while simulation_app.is_running():
        action = torch.zeros(env.num_envs, 4, device=env.device)  # ctbr commands
        action[:, 0] = 10.0  # collective thrust for hover

        sim_time = step * env.ctrl_dt

        # Log current state before step
        log_start_t = time.time()
        pos_data = env.quad_state.pos[0].cpu().numpy()
        att_data = env.quad_state.att[0].cpu().numpy()  # quaternion [qw, qx, qy, qz]
        vel_data = env.quad_state.vel[0].cpu().numpy()
        ome_data = env.quad_state.ome[0].cpu().numpy()
        acc_data = env.quad_state.acc[0].cpu().numpy()
        tau_data = env.quad_state.tau[0].cpu().numpy()
        mot_data = env.quad_state.mot[0].cpu().numpy()
        motdes_data = env.quad_state.motdes[0].cpu().numpy()
        E_data = env.quad_state.E[0].cpu().numpy()
        ucap_data = env.quad_state.ucap[0].cpu().numpy()
        ubat_data = env.quad_state.ubat[0].cpu().numpy()
        bfli_data = env.quad_state.bfli[0].cpu().numpy()

        log_poses.append(
            [
                sim_time,
                pos_data[0],
                pos_data[1],
                pos_data[2],
                att_data[1],
                att_data[2],
                att_data[3],
                att_data[0],  # reorder to qx, qy, qz, qw
                vel_data[0],
                vel_data[1],
                vel_data[2],
                ome_data[0],
                ome_data[1],
                ome_data[2],
                acc_data[0],
                acc_data[1],
                acc_data[2],
                tau_data[0],
                tau_data[1],
                tau_data[2],
                mot_data[0],
                mot_data[1],
                mot_data[2],
                mot_data[3],
                motdes_data[0],
                motdes_data[1],
                motdes_data[2],
                motdes_data[3],
                E_data[0],
                ucap_data[0],
                ubat_data[0],
                bfli_data[0],
                bfli_data[1],
                bfli_data[2],
            ]
        )
        timing_dict["[test_script] log_t"] += time.time() - log_start_t

        step_start_time = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        timing_dict["[test_script] step_t"] += time.time() - step_start_time

        # Record state history (first environment only)
        log_start_t = time.time()
        state_history["time"].append(sim_time)
        state_history["pos"].append(env.quad_state.pos[0].cpu().numpy().copy())
        state_history["vel"].append(env.quad_state.vel[0].cpu().numpy().copy())
        state_history["att"].append(env.quad_state.att[0].cpu().numpy().copy())
        timing_dict["[test_script] log_t"] += time.time() - log_start_t

        for key in info.keys():
            timing_dict[key] = timing_dict.get(key, 0.0) + info[key]

        step += 1

        # Calculate benchmark metrics
        current_real_time = time.time()
        elapsed_real_time = (
            current_real_time
            - start_time
            - timing_dict["[test_script] log_t"]
            - timing_dict["[test_script] contact_check"]
        )
        elapsed_sim_time = step * env.ctrl_dt - start_sim_time

        if step % 100 == 0:
            speed_up = (
                (env.num_envs * elapsed_sim_time) / elapsed_real_time
                if elapsed_real_time > 0
                else 0
            )
            print(
                f"[Step {step}] Real time: {elapsed_real_time:.2f}s, Sim time: {elapsed_sim_time:.2f}s, Speed-up: {speed_up:.2f}x"
            )
            print(
                f"           reward={reward}, terminated={terminated}, truncated={truncated}"
            )
            print(f"time: {sim_time:.3f}s")
            print(
                f"pos: {env.quad_state.pos[-1].cpu().numpy()}, vel: {env.quad_state.vel[-1].cpu().numpy()}"
            )
            print(
                f"att (quat): {env.quad_state.att[-1].cpu().numpy()}, ang_vel: {env.quad_state.ome[-1].cpu().numpy()}"
            )
            print(
                f"mot: {env.quad_state.mot[-1].cpu().numpy()}, motdes: {env.quad_state.motdes[-1].cpu().numpy()}"
            )
            print("")

        contact_start_t = time.time()
        if not (env.contact_sensor.data.net_forces_w == 0.0).all():
            print("[INFO] Contact detected.")
        timing_dict["[test_script] contact_check"] += time.time() - contact_start_t

        if benchmark_duration > 0 and elapsed_real_time >= benchmark_duration:
            final_speed_up = (env.num_envs * elapsed_sim_time) / elapsed_real_time
            print(f"\n[BENCHMARK COMPLETE]")
            print(f"Real time elapsed: {elapsed_real_time:.2f}s")
            print(f"Simulation time elapsed: {elapsed_sim_time:.2f}s")
            print(f"Total simulation steps: {step * env.num_envs}")
            print(f"Physics steps per env step: {env.cfg.decimation}")
            print(f"Total physics steps: {env._sim_step_counter}")
            print(f"Final speed-up: {final_speed_up:.2f}x")
            print(
                f"Steps per second (real time): {step * env.num_envs / elapsed_real_time:.1f}"
            )
            print(
                f"Physics steps per second (real time): {env._sim_step_counter / elapsed_real_time:.1f}"
            )
            print(f"BASE ENV timing info: {timing_dict}")
            break

    # Convert lists to numpy arrays for plotting
    time_arr = np.array(state_history["time"])
    pos = np.array(state_history["pos"])
    vel = np.array(state_history["vel"])
    att = np.array(state_history["att"])

    # Create plots
    fig, axs = plt.subplots(3, 3, figsize=(18, 15))

    # 1. XY plane
    axs[0, 0].plot(pos[:, 0], pos[:, 1], "b-", linewidth=2)
    axs[0, 0].set_xlabel("X Position [m]")
    axs[0, 0].set_ylabel("Y Position [m]")
    axs[0, 0].set_title("XY Plane Trajectory")
    axs[0, 0].grid(True)
    axs[0, 0].axis("equal")

    # 2. X position
    axs[0, 1].plot(time_arr, pos[:, 0], "b-", linewidth=2)
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("X Position [m]")
    axs[0, 1].set_title("X Position vs Time")
    axs[0, 1].grid(True)

    # 3. Y position
    axs[0, 2].plot(time_arr, pos[:, 1], "b-", linewidth=2)
    axs[0, 2].set_xlabel("Time [s]")
    axs[0, 2].set_ylabel("Y Position [m]")
    axs[0, 2].set_title("Y Position vs Time")
    axs[0, 2].grid(True)

    # 4. Z position
    axs[1, 0].plot(time_arr, pos[:, 2], "b-", linewidth=2)
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Z Position [m]")
    axs[1, 0].set_title("Z Position vs Time")
    axs[1, 0].grid(True)

    # 5. X velocity
    axs[1, 1].plot(time_arr, vel[:, 0], "b-", linewidth=2)
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].set_ylabel("X Velocity [m/s]")
    axs[1, 1].set_title("X Velocity vs Time")
    axs[1, 1].grid(True)

    # 6. Y velocity
    axs[1, 2].plot(time_arr, vel[:, 1], "b-", linewidth=2)
    axs[1, 2].set_xlabel("Time [s]")
    axs[1, 2].set_ylabel("Y Velocity [m/s]")
    axs[1, 2].set_title("Y Velocity vs Time")
    axs[1, 2].grid(True)

    # 7. Z velocity
    axs[2, 0].plot(time_arr, vel[:, 2], "b-", linewidth=2)
    axs[2, 0].set_xlabel("Time [s]")
    axs[2, 0].set_ylabel("Z Velocity [m/s]")
    axs[2, 0].set_title("Z Velocity vs Time")
    axs[2, 0].grid(True)

    # 8. Attitude (quaternion components)
    axs[2, 1].plot(time_arr, att[:, 0], "b-", label="w", linewidth=1.5)
    axs[2, 1].plot(time_arr, att[:, 1], "g-", label="x", linewidth=1.5)
    axs[2, 1].plot(time_arr, att[:, 2], "r-", label="y", linewidth=1.5)
    axs[2, 1].plot(time_arr, att[:, 3], "c-", label="z", linewidth=1.5)
    axs[2, 1].set_xlabel("Time [s]")
    axs[2, 1].set_ylabel("Quaternion Components")
    axs[2, 1].set_title("Attitude (Quaternion) vs Time")
    axs[2, 1].legend()
    axs[2, 1].grid(True)

    # 9. Hide unused subplot
    axs[2, 2].axis("off")

    plt.tight_layout()

    file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(file_dir, exist_ok=True)

    plot_file = os.path.join(file_dir, "hover_trajectory.png")
    plt.savefig(plot_file, dpi=150)
    print(f"\n[INFO] Plot saved to {plot_file}")
    plt.show()

    # Save pose log
    poses_file = os.path.join(file_dir, "isaac_agisim_poses.txt")
    poses_array = np.array(log_poses)
    np.savetxt(poses_file, poses_array, fmt="%.15e", delimiter=" ")
    print(f"[INFO] Poses saved to {poses_file}")
    print(
        f"[INFO] Poses format: time x y z qx qy qz qw vx vy vz wx wy wz ax ay az tau_x tau_y tau_z mot_0 mot_1 mot_2 mot_3 motdes_0 motdes_1 motdes_2 motdes_3 E ucap ubat bfli_0 bfli_1 bfli_2 (columns)"
    )
    print(f"[INFO] Total logged pose points: {len(log_poses)}")

    env.close()
    print("[INFO] Environment closed.")


if __name__ == "__main__":
    main()
    simulation_app.close()
