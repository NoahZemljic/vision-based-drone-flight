#!/usr/bin/env python3
"""Deploy an rpg_isaac-trained ``Rpg-Vbdf-Tracking-v0`` policy (sb3 PPO) via ROS 2.

Builds the observation ``[state (15), prev_cmd (4), 8 x (x, y, w, h) bounding-box
history]`` and publishes one command per received ``/drone_detector/detections``
message once ``start_policy`` is true. If you change the observation in the
environment, change it here too.

While the policy runs the rollout is logged and written on stop / shutdown in the
layout the course's streamlit viewer reads (``flightpilot.rollout_log``;
``save_rollout:=false`` to disable).
"""
import os

import numpy as np
import rclpy
import torch
from agiros_msgs.msg import Command, QuadState
from detection_msgs.msg import BoundingBox, BoundingBoxes
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Bool

from flightpilot.isaac_policy import load_isaac_actor, load_obs_norm
from flightpilot.rollout_log import RolloutLog, state_row

N_HIST = 8
IMG_W, IMG_H = 1280, 720


class IsaacTrackingPilot:
    def __init__(self, node: Node, actor, obs_norm, mass: float, thrust_max: float, omega_max, log=None):
        self.node = node
        self.actor = actor
        self.log = log
        self.mean_obs, self.var_obs, self.clip_obs, self.eps = obs_norm
        self.start = False
        half = 4.0 * thrust_max / (2.0 * mass)  # vbdf_base_env: quad_max_force / (2 m)
        self.act_mean = np.array([half, 0.0, 0.0, 0.0])
        self.act_std = np.array([half, *omega_max])
        self.state_obs = np.zeros(15)
        self.state_msg = None
        self.other_pos = np.zeros(3)
        self.cmd = np.zeros(4)
        self.bb_hist = np.zeros((N_HIST, 4))
        node.get_logger().info("mass %.3f thrust_max %.2f N/motor omega_max %s" % (mass, thrust_max, list(omega_max)))
        state_qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        cmd_qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.cmd_pub = node.create_publisher(Command, "agiros_pilot/feedthrough_command", cmd_qos)
        node.create_subscription(Bool, "start_policy", self.start_cb, 1)
        node.create_subscription(QuadState, "agiros_pilot/state", self.state_cb, state_qos)
        node.create_subscription(BoundingBoxes, "/drone_detector/detections", self.detection_cb, state_qos)
        node.create_subscription(PoseStamped, "/vbdf/actor/pose", self.actor_cb, 1)

    def start_cb(self, msg):
        self.start = msg.data
        self.node.get_logger().info("RL policy %s" % ("started" if self.start else "stopped"))
        if not self.start:
            self.save()

    def save(self):
        run = self.log.save() if self.log is not None else None
        if run is not None:
            self.node.get_logger().info("rollout -> %s" % run)

    def actor_cb(self, msg: PoseStamped):
        self.other_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

    def state_cb(self, s: QuadState):
        self.state_msg = s
        p = [s.pose.position.x, s.pose.position.y, s.pose.position.z]
        q = [s.pose.orientation.x, s.pose.orientation.y, s.pose.orientation.z, s.pose.orientation.w]
        v = [s.velocity.linear.x, s.velocity.linear.y, s.velocity.linear.z]
        w = [s.velocity.angular.x, s.velocity.angular.y, s.velocity.angular.z]
        rot6 = R.from_quat(q).as_matrix().reshape((9,), order="F")[:6]
        self.state_obs = np.concatenate((p, rot6, v, w))

    def detection_cb(self, msg: BoundingBoxes):
        """Called whenever a new detection is received. Upon each received detection
        a command is published."""
        bbox = msg.bounding_boxes[0] if msg.bounding_boxes else BoundingBox()
        # pixels -> normalised [-1, 1], (x, y, w, h) with (x, y) the min corner (course convention)
        bb = np.array([bbox.xmin / IMG_W, bbox.xmax / IMG_W, bbox.ymin / IMG_H, bbox.ymax / IMG_H]) * 2 - 1
        self.bb_hist[1:] = self.bb_hist[:-1]
        self.bb_hist[0] = [bb[0], bb[2], bb[1] - bb[0], bb[3] - bb[2]]

        obs = np.concatenate([self.state_obs, self.cmd, self.bb_hist.flatten()])
        obs = np.clip((obs - self.mean_obs) / np.sqrt(self.var_obs + self.eps), -self.clip_obs, self.clip_obs)
        with torch.no_grad():
            a = self.actor(torch.as_tensor(obs, dtype=torch.float32)[None]).numpy()[0]
        self.cmd = np.clip(a, -1.0, 1.0) * self.act_std + self.act_mean

        if not self.start:
            return
        if self.log is not None and self.state_msg is not None:
            s = self.state_msg
            row = state_row(s, self.cmd)
            row.update(dict(zip(("other_pos_x", "other_pos_y", "other_pos_z"), self.other_pos)))
            row.update(dict(zip(("bbox_x", "bbox_y", "bbox_w", "bbox_h"), self.bb_hist[0])))
            self.log.append(s.t, row, obs)
        out = Command()
        out.header.stamp = self.node.get_clock().now().to_msg()
        out.is_single_rotor_thrust = False
        out.collective_thrust = float(self.cmd[0])
        out.bodyrates.x, out.bodyrates.y, out.bodyrates.z = map(float, self.cmd[1:])
        self.cmd_pub.publish(out)


def main():
    rclpy.init()
    node = rclpy.create_node("flightpilot")
    torch.set_num_threads(1)
    node.declare_parameter("quad_name", "kolibri")
    node.declare_parameter("checkpoint", "")  # .../logs/sb3/Rpg-Vbdf-Tracking-v0/<run>/model.zip
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
    if n_obs != 15 + 4 + 4 * N_HIST:
        node.get_logger().warning(
            f"checkpoint has {n_obs}-dim obs but this node builds {15 + 4 + 4 * N_HIST} — "
            "adapt the observation here to match your environment"
        )
    vecnorm = ckpt[: -len(".zip")] + "_vecnormalize.pkl"
    obs_norm = load_obs_norm(vecnorm, n_obs)
    node.get_logger().info("loaded %s (obs stats: %s)" % (ckpt, vecnorm if os.path.isfile(vecnorm) else "none"))
    log = None
    if bool(node.get_parameter("save_rollout").value):
        root = str(node.get_parameter("rollout_dir").value) or os.path.join(os.path.dirname(ckpt), "rollout")
        log = RolloutLog(root, "Rpg-Vbdf-Tracking-v0",
                         extra_cols=["other_pos_x", "other_pos_y", "other_pos_z",
                                     "bbox_x", "bbox_y", "bbox_w", "bbox_h"])
        node.get_logger().info("logging rollouts to %s/<timestamp>/Eval" % root)
    pilot = IsaacTrackingPilot(
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
