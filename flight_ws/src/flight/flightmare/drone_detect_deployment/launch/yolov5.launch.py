"""YOLOv5 detector (``drone_detect_deployment/yolo_detector.py``).

Runs under Isaac's python (``python_executable``, default ``$ISAAC_PYTHON``) — that
interpreter has torch and the vendored fork's dependencies.

    ros2 launch drone_detect_deployment yolov5.launch.py \
        weights:=/path/to/best.pt input_image_topic:=/kolibri/onboard_view
"""
import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

sys.path.insert(0, os.path.dirname(__file__))
from _launch_common import resolve_detector_env  # noqa: E402


def generate_launch_description():
    default_python = os.environ.get("ISAAC_PYTHON", "/workspace/isaaclab/_isaac_sim/python.sh")
    arguments = [
        DeclareLaunchArgument("weights", description="path to the trained checkpoint (best.pt)"),
        DeclareLaunchArgument("input_image_topic", default_value="/image_topic"),
        DeclareLaunchArgument("compressed", default_value="false"),
        DeclareLaunchArgument("output_topic", default_value="/drone_detector/detections"),
        DeclareLaunchArgument("conf_thres", default_value="0.25"),
        DeclareLaunchArgument("iou_thres", default_value="0.45"),
        DeclareLaunchArgument("img_size", default_value="1280"),
        DeclareLaunchArgument("device", default_value=""),
        DeclareLaunchArgument("publish_image", default_value="false"),
        DeclareLaunchArgument("output_image_topic", default_value="/drone_detector/image_out"),
        DeclareLaunchArgument("line_thickness", default_value="3"),
        DeclareLaunchArgument("compressed_output", default_value="false"),
        DeclareLaunchArgument("python_executable", default_value=default_python),
    ]
    node = Node(
        package="drone_detect_deployment",
        executable="yolo_detector",
        name="yolo_detector",
        output="screen",
        prefix=[LaunchConfiguration("python_executable"), " "],
        additional_env=resolve_detector_env(),
        parameters=[{
            "use_sim_time": False,
            "weights": LaunchConfiguration("weights"),
            "input_image_topic": LaunchConfiguration("input_image_topic"),
            "compressed": LaunchConfiguration("compressed"),
            "output_topic": LaunchConfiguration("output_topic"),
            "conf_thres": LaunchConfiguration("conf_thres"),
            "iou_thres": LaunchConfiguration("iou_thres"),
            "img_size": LaunchConfiguration("img_size"),
            "device": LaunchConfiguration("device"),
            "publish_image": LaunchConfiguration("publish_image"),
            "output_image_topic": LaunchConfiguration("output_image_topic"),
            "line_thickness": LaunchConfiguration("line_thickness"),
            "compressed_output": LaunchConfiguration("compressed_output"),
        }],
    )
    return LaunchDescription(arguments + [node])
