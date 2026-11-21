"""Require advancing AgiSim state and report whether the binary emits /clock."""
import os
import math
import signal
import subprocess
import tempfile
import time

import rclpy
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from agiros_msgs.msg import QuadState
from rosgraph_msgs.msg import Clock
from rclpy.qos import qos_profile_sensor_data


def main():
    for package in ("agiros", "agilib", "agiros_gui", "flightpilot", "detection_msgs", "drone_detect_deployment"):
        print(f"{package}: {get_package_prefix(package)}", flush=True)
    # Isolate the test from any running course simulation on the host network.
    os.environ["ROS_DOMAIN_ID"] = "172"
    rclpy.init()
    node = rclpy.create_node("course_smoke_test")
    received = set()
    timestamps = []

    def state_callback(message):
        values = (message.t, message.pose.position.x, message.pose.position.y, message.pose.position.z)
        if all(math.isfinite(value) for value in values):
            timestamps.append(message.t)

    node.create_subscription(QuadState, "/course_smoke/agiros_pilot/state", state_callback, qos_profile_sensor_data)
    node.create_subscription(Clock, "/clock", lambda _: received.add("clock"), qos_profile_sensor_data)
    command = [
        get_package_prefix("agiros") + "/lib/agiros/agisim_node", "--ros-args",
        "-r", "__ns:=/course_smoke", "-r", "__node:=agiros_pilot",
        "-p", "agi_param_dir:=" + get_package_share_directory("agilib") + "/params",
        "-p", "ros_param_dir:=" + get_package_share_directory("agiros") + "/parameters",
        "-p", "simulator_config:=kolibri_simulation.yaml",
        "-p", "pilot_config:=simple_sim_pilot.yaml", "-p", "is_server:=true",
    ]
    with tempfile.TemporaryFile(mode="w+") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and process.poll() is None:
                rclpy.spin_once(node, timeout_sec=0.2)
                if len(timestamps) >= 100 and timestamps[-1] > timestamps[0]:
                    break
            if process.poll() is not None or len(timestamps) < 100 or timestamps[-1] <= timestamps[0]:
                log.seek(0)
                raise RuntimeError(f"Agisim smoke test failed; received {len(timestamps)} valid states\n{log.read()}")
            print(f"PASS: AgiSim published {len(timestamps)} finite states with advancing timestamps.", flush=True)
            if "clock" not in received:
                print("NOTE: This course binary did not emit /clock; keep use_sim_time=false (the deployment default).", flush=True)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
