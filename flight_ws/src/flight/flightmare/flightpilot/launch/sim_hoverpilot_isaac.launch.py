"""sim_hoverpilot.launch.py twin for an rpg_isaac sb3 checkpoint (``checkpoint`` = path to
model.zip; ``model_vecnormalize.pkl`` next to it is picked up automatically)."""
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
        DeclareLaunchArgument("checkpoint", default_value=""),
        DeclareLaunchArgument("mass", default_value="0.220"),
        DeclareLaunchArgument("thrust_max", default_value="2.5"),
        DeclareLaunchArgument("save_rollout", default_value="true"),
        DeclareLaunchArgument("rollout_dir", default_value=""),  # default: <checkpoint dir>/rollout
        DeclareLaunchArgument("python_executable", default_value=default_python),
    ]
    node = Node(
        package="flightpilot",
        executable="run_hover_pilot_isaac",
        name="flightpilot",
        namespace=LaunchConfiguration("quad_name"),
        output="screen",
        prefix=[LaunchConfiguration("python_executable"), " "],
        additional_env=resolve_flightmare_env(),
        parameters=[{
            "use_sim_time": False,
            "quad_name": LaunchConfiguration("quad_name"),
            "checkpoint": LaunchConfiguration("checkpoint"),
            "mass": LaunchConfiguration("mass"),
            "thrust_max": LaunchConfiguration("thrust_max"),
            "save_rollout": LaunchConfiguration("save_rollout"),
            "rollout_dir": LaunchConfiguration("rollout_dir"),
        }],
    )
    return LaunchDescription(arguments + [node])
