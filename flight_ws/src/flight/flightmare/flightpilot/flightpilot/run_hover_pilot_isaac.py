#!/usr/bin/env python3
"""Deploy an rpg_isaac-trained ``Rpg-Vbdf-Hover-v0`` policy (sb3 PPO, 15-dim state obs) via ROS 2.

Isaac-checkpoint twin of ``run_hover_pilot.py`` (which loads the course's flightpy format).
Reads ``model.zip`` (policy weights) + ``model_vecnormalize.pkl`` (obs running stats) from
``scripts/sb3/train.py`` and rebuilds the actor as a plain torch MLP via
``flightpilot.isaac_policy``, so neither sb3 nor rpg_tasks is needed on the vehicle. Obs layout / action mapping mirror
``rpg_vbdf/vbdf_hover_env.py`` and ``vbdf_base_env.py``:

    obs = [p(3), R[:, :2] column-major (6), v_world(3), omega_body(3)]
    cmd = clip(tanh(mlp(norm(obs))), -1, 1) * act_std + act_mean
    act_mean = [4*thrust_max/(2m), 0, 0, 0], act_std = [4*thrust_max/(2m), omega_max]

Publishes ``agiros_pilot/feedthrough_command`` once ``start_policy`` (std_msgs/Bool) is true.
While the policy runs the rollout is logged and written on stop / shutdown in the layout the
course's streamlit viewer reads (``flightpilot.rollout_log``; ``save_rollout:=false`` to disable).
"""
import os

import numpy as np
import rclpy
import torch
import torch.nn as nn
from agiros_msgs.msg import Command, QuadState
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Bool

from flightpilot.isaac_policy import load_isaac_actor, load_obs_norm
from flightpilot.rollout_log import RolloutLog, state_row

N_OBS = 15


class IsaacHoverPilot:
    def __init__(self, node: Node, actor: nn.Module, obs_norm, mass: float, thrust_max: float, omega_max, log=None):
        self.node = node
        self.actor = actor
        self.log = log
        self.mean_obs, self.var_obs, self.clip_obs, self.eps = obs_norm
        self.start = False
        self.seq = 0
        half = 4.0 * thrust_max / (2.0 * mass)  # vbdf_base_env: quad_max_force / (2 m)
        self.act_mean = np.array([half, 0.0, 0.0, 0.0])
        self.act_std = np.array([half, *omega_max])
        node.get_logger().info("mass %.3f thrust_max %.2f N/motor omega_max %s" % (mass, thrust_max, list(omega_max)))
        state_qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        cmd_qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.cmd_pub = node.create_publisher(Command, "agiros_pilot/feedthrough_command", cmd_qos)
        node.create_subscription(Bool, "start_policy", self.start_cb, 1)
        node.create_subscription(QuadState, "agiros_pilot/state", self.state_cb, state_qos)

    def start_cb(self, msg):
        self.start = msg.data
        self.node.get_logger().info("RL policy %s" % ("started" if self.start else "stopped"))
        if not self.start:
            self.save()

    def save(self):
        run = self.log.save() if self.log is not None else None
        if run is not None:
            self.node.get_logger().info("rollout -> %s" % run)

    def observation(self, s: QuadState) -> np.ndarray:
        p = [s.pose.position.x, s.pose.position.y, s.pose.position.z]
        q = [s.pose.orientation.x, s.pose.orientation.y, s.pose.orientation.z, s.pose.orientation.w]
        v = [s.velocity.linear.x, s.velocity.linear.y, s.velocity.linear.z]
        w = [s.velocity.angular.x, s.velocity.angular.y, s.velocity.angular.z]
        rot6 = R.from_quat(q).as_matrix().reshape((9,), order="F")[:6]  # columns 0, 1 == qs.R[:, :, :2] in the env
        return np.concatenate((p, rot6, v, w))

    def act(self, obs: np.ndarray):
        obs = np.clip((obs - self.mean_obs) / np.sqrt(self.var_obs + self.eps), -self.clip_obs, self.clip_obs)
        with torch.no_grad():
            a = self.actor(torch.as_tensor(obs, dtype=torch.float32)[None]).numpy()[0]
        return np.clip(a, -1.0, 1.0) * self.act_std + self.act_mean, obs

    def state_cb(self, s: QuadState):
        cmd, norm_obs = self.act(self.observation(s))
        if not self.start:
            return
        if self.log is not None:
            self.log.append(s.t, state_row(s, cmd), norm_obs)
        msg = Command()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.is_single_rotor_thrust = False
        msg.collective_thrust = float(cmd[0])
        msg.bodyrates.x, msg.bodyrates.y, msg.bodyrates.z = map(float, cmd[1:])
        self.cmd_pub.publish(msg)
        self.seq += 1


def main():
    rclpy.init()
    node = rclpy.create_node("flightpilot")
    torch.set_num_threads(1)
    node.declare_parameter("quad_name", "kolibri")
    node.declare_parameter("checkpoint", "")  # .../logs/sb3/Rpg-Vbdf-Hover-v0/<run>/model.zip
    # VBDF kolibri (rpg_vbdf/vbdf_sim_params.py + vbdf_env_cfg.py): the env's action mapping
    node.declare_parameter("mass", 0.220)
    node.declare_parameter("thrust_max", 2.5)  # per motor [N]; RPGVbdfBaseEnvCfg.derated_thrust_max
    node.declare_parameter("omega_max", [10.0, 10.0, 4.0])
    node.declare_parameter("save_rollout", True)
    node.declare_parameter("rollout_dir", "")  # default: <checkpoint dir>/rollout
    ckpt = str(node.get_parameter("checkpoint").value)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"checkpoint {ckpt!r} not found (expected .../model.zip)")
    actor, n_obs, _ = load_isaac_actor(ckpt)
    if n_obs != N_OBS:
        raise ValueError(f"checkpoint has {n_obs}-dim obs, expected {N_OBS} (hover)")
    vecnorm = ckpt[: -len(".zip")] + "_vecnormalize.pkl"
    obs_norm = load_obs_norm(vecnorm, N_OBS)
    node.get_logger().info("loaded %s (obs stats: %s)" % (ckpt, vecnorm if os.path.isfile(vecnorm) else "none"))
    log = None
    if bool(node.get_parameter("save_rollout").value):
        root = str(node.get_parameter("rollout_dir").value) or os.path.join(os.path.dirname(ckpt), "rollout")
        log = RolloutLog(root, "Rpg-Vbdf-Hover-v0")
        node.get_logger().info("logging rollouts to %s/<timestamp>/Eval" % root)
    pilot = IsaacHoverPilot(
        node,
        actor,
        obs_norm,
        float(node.get_parameter("mass").value),
        float(node.get_parameter("thrust_max").value),
        [float(x) for x in node.get_parameter("omega_max").value],
        log,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        pilot.save()


if __name__ == "__main__":
    main()
