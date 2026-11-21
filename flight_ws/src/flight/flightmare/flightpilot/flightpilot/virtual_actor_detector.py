#!/usr/bin/env python3
"""Sim-only stand-in for the course's actor drone + YOLO detector (rpg_isaac addition,
NOT part of the ROS 1 flightpilot).

Moves a virtual actor drone on a smooth path, projects its 26 cm x 8 cm body into the
camera drone's double-sphere camera (offboard_small_ds, camera along body x, via
``rpg_vbdf``'s ``DoubleSphereCamera``) using the camera drone's agiros state, and
publishes ``detection_msgs/BoundingBoxes`` in pixels on ``/drone_detector/detections``
at 60 Hz, plus the actor pose on ``/vbdf/actor/pose`` for logging.
"""
import math

import numpy as np
import rclpy
import torch
from agiros_msgs.msg import QuadState
from detection_msgs.msg import BoundingBox, BoundingBoxes
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from scipy.spatial.transform import Rotation as R

# the instructor tree has the full camera; the student tree falls back to the
# skeleton in double_sphere_camera.py — this node starts working (publishing
# real boxes instead of empty detections) the moment you implement it
try:
    from rpg_tasks.tasks.direct.rpg_vbdf.utils.double_sphere_camera_solution import OFFBOARD_SMALL_DS, DoubleSphereCamera
except ImportError:
    from rpg_tasks.tasks.direct.rpg_vbdf.utils.double_sphere_camera import OFFBOARD_SMALL_DS, DoubleSphereCamera

CAMERA = DoubleSphereCamera(OFFBOARD_SMALL_DS["intrinsics"], OFFBOARD_SMALL_DS["resolution"], (0.0, 0.0, 0.0), device="cpu", dtype=torch.float64)
WIDTH, HEIGHT = OFFBOARD_SMALL_DS["resolution"]
T_BM = np.array([[0.12, -0.12, 0.0], [-0.12, 0.12, 0.0], [-0.12, -0.12, 0.0], [0.12, 0.12, 0.0]])
DRONE_HEIGHT = 0.08


def project(points_w, R_WB, p_WB):
    """Double-sphere projection of world points -> (u, v, valid) in pixels."""
    pts = torch.as_tensor(points_w, dtype=torch.float64)[None]
    R_CW, t = CAMERA.pose(torch.as_tensor(R_WB, dtype=torch.float64)[None], torch.as_tensor(p_WB, dtype=torch.float64)[None])
    uvv = CAMERA.project(pts, R_CW, t, homogeneous=False)[0].numpy()
    return uvv[:, 0], uvv[:, 1], uvv[:, 2] > 0.5


class VirtualActorDetector(Node):
    def __init__(self):
        super().__init__("virtual_actor_detector")
        self.declare_parameter("quad_name", "kolibri")
        self.declare_parameter("actor_start", [2.0, 0.0, 1.5])
        self.declare_parameter("amplitude", [1.5, 1.0, 0.3])  # m
        self.declare_parameter("omega", [0.25, 0.17, 0.11])  # rad/s
        self.declare_parameter("hold_s", 3.0)  # actor hovers this long after start
        self.declare_parameter("rate_hz", 60.0)
        self.declare_parameter("require_valid", True)  # empty detection when the actor leaves the image
        quad = str(self.get_parameter("quad_name").value)
        self.p0 = np.array(self.get_parameter("actor_start").value, dtype=float)
        self.amp = np.array(self.get_parameter("amplitude").value, dtype=float)
        self.om = np.array(self.get_parameter("omega").value, dtype=float)
        self.hold = float(self.get_parameter("hold_s").value)
        self.require_valid = bool(self.get_parameter("require_valid").value)
        self.state = None
        self.t_start = None
        state_qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(QuadState, f"/{quad}/agiros_pilot/state", self.state_cb, state_qos)
        self.det_pub = self.create_publisher(BoundingBoxes, "/drone_detector/detections", 1)
        self.pose_pub = self.create_publisher(PoseStamped, "/vbdf/actor/pose", 1)
        self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self.tick)
        self.get_logger().info(f"virtual actor at {self.p0.tolist()} amp={self.amp.tolist()} omega={self.om.tolist()}")

    def state_cb(self, s):
        self.state = s

    def actor(self, t):
        t = max(0.0, t - self.hold)
        pos = self.p0 + self.amp * np.sin(self.om * t)
        vel = self.amp * self.om * np.cos(self.om * t)
        yaw = math.atan2(vel[1], vel[0]) if np.linalg.norm(vel[:2]) > 1e-3 else 0.0
        return pos, R.from_euler("z", yaw).as_matrix()

    def tick(self):
        if self.state is None:
            return
        now = self.get_clock().now()
        if self.t_start is None:
            self.t_start = now
        t = (now - self.t_start).nanoseconds * 1e-9
        pos, R_a = self.actor(t)
        s = self.state
        p = np.array([s.pose.position.x, s.pose.position.y, s.pose.position.z])
        R_WB = R.from_quat([s.pose.orientation.x, s.pose.orientation.y, s.pose.orientation.z, s.pose.orientation.w]).as_matrix()
        mid = pos + T_BM @ R_a.T
        up = R_a[:, 2] * DRONE_HEIGHT / 2.0
        corners = np.vstack([mid + up, mid - up])
        u, v, valid = project(corners, R_WB, p)
        msg = BoundingBoxes()
        msg.header.stamp = now.to_msg()
        msg.image_header.stamp = now.to_msg()
        if (not self.require_valid) or valid.any():
            bb = BoundingBox()
            bb.class_name = "drone"
            bb.probability = 1.0
            bb.xmin = int(np.clip(u.min(), 0, WIDTH - 1))
            bb.xmax = int(np.clip(u.max(), 0, WIDTH - 1))
            bb.ymin = int(np.clip(v.min(), 0, HEIGHT - 1))
            bb.ymax = int(np.clip(v.max(), 0, HEIGHT - 1))
            msg.bounding_boxes.append(bb)
        self.det_pub.publish(msg)
        ps = PoseStamped()
        ps.header.stamp = now.to_msg()
        ps.header.frame_id = "world"
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, pos)
        q = R.from_matrix(R_a).as_quat()
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = map(float, q)
        self.pose_pub.publish(ps)


def main():
    rclpy.init()
    rclpy.spin(VirtualActorDetector())


if __name__ == "__main__":
    main()
