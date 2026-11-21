"""Sim-only actor + detector stand-in (``flightpilot/virtual_actor_detector.py``).

Runs under Isaac's python (``python_executable``, default ``$ISAAC_PYTHON``) because the
double-sphere projection comes from ``rpg_vbdf``'s ``DoubleSphereCamera`` (torch)."""
import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

sys.path.insert(0, os.path.dirname(__file__))
from _launch_common import resolve_flightmare_env  # noqa: E402


def generate_launch_description():
    default_python = os.environ.get("ISAAC_PYTHON", "/workspace/isaaclab/_isaac_sim/python.sh")
    arguments = [
        DeclareLaunchArgument("quad_name", default_value="kolibri"),
        DeclareLaunchArgument("rate_hz", default_value="60.0"),
        DeclareLaunchArgument("python_executable", default_value=default_python),
    ]
    node = Node(
        package="flightpilot",
        executable="virtual_actor_detector",
        name="virtual_actor_detector",
        output="screen",
        prefix=[LaunchConfiguration("python_executable"), " "],
        additional_env=resolve_flightmare_env(),
        parameters=[{
            "use_sim_time": False,
            "quad_name": LaunchConfiguration("quad_name"),
            "rate_hz": LaunchConfiguration("rate_hz"),
        }],
    )
    return LaunchDescription(arguments + [node])
