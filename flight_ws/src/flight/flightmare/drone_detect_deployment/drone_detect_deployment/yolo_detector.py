#!/usr/bin/env python3
"""ROS 2 YOLOv5 detector for the VBDF course (port of the ROS 1 yolov5_ros wrapper).

Subscribes to an image topic (``sensor_msgs/Image``, or ``CompressedImage`` with
``compressed:=true``), runs the vendored ``drone_detect_yolov5`` fork on each frame and
publishes ``detection_msgs/BoundingBoxes`` (pixels; empty message when nothing beats
``conf_thres``). With ``publish_image:=true`` it also draws the boxes on the frame and
publishes the annotated image on a separate topic (``sensor_msgs/Image``, or
``CompressedImage`` when ``compressed_output:=true``) — handy for ``rqt_image_view`` and
for recording overlaid detections in a rosbag. Runs under Isaac's python (see
``launch/yolov5.launch.py``) so the fork's torch environment is available.
"""
import os

import numpy as np
import rclpy
import torch
from detection_msgs.msg import BoundingBox, BoundingBoxes
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image

DEFAULT_REPO = "/workspace/isaaclab/source/rpg_isaac/run/vbdf/detection/drone_detect_yolov5"


def image_to_rgb(msg: Image) -> np.ndarray:
    enc = msg.encoding.lower()
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if enc in ("rgb8", "bgr8"):
        img = buf.reshape(msg.height, msg.width, 3)
        return img if enc == "rgb8" else img[:, :, ::-1]
    if enc in ("rgba8", "bgra8"):
        img = buf.reshape(msg.height, msg.width, 4)[:, :, :3]
        return img if enc == "rgba8" else img[:, :, ::-1]
    if enc == "mono8":
        return np.repeat(buf.reshape(msg.height, msg.width, 1), 3, axis=2)
    raise ValueError(f"unsupported image encoding {msg.encoding!r}")


class YoloDetector(Node):
    def __init__(self):
        super().__init__("yolo_detector")
        self.declare_parameter("weights", "")
        self.declare_parameter("yolo_repo", DEFAULT_REPO)
        self.declare_parameter("input_image_topic", "/image_topic")
        self.declare_parameter("compressed", False)
        self.declare_parameter("output_topic", "/drone_detector/detections")
        self.declare_parameter("conf_thres", 0.25)
        self.declare_parameter("iou_thres", 0.45)
        self.declare_parameter("img_size", 1280)
        self.declare_parameter("device", "")  # "" = cuda if available, else cpu
        self.declare_parameter("publish_image", False)  # also publish the annotated frame (yolov5_ros ~publish_image)
        self.declare_parameter("output_image_topic", "/drone_detector/image_out")  # yolov5_ros ~output_image_topic
        self.declare_parameter("line_thickness", 3)  # box line width (yolov5_ros ~line_thickness)
        self.declare_parameter("compressed_output", False)  # publish CompressedImage instead of Image

        weights = str(self.get_parameter("weights").value)
        if not os.path.isfile(weights):
            raise FileNotFoundError(f"weights {weights!r} not found (expected .../best.pt)")
        repo = str(self.get_parameter("yolo_repo").value)
        # YOLOv5's select_device wants "", a GPU index ("0", "0,1") or "cpu" — never "cuda"
        device = str(self.get_parameter("device").value).strip()
        if not device:
            device = "0" if torch.cuda.is_available() else "cpu"
        elif device.startswith("cuda"):
            device = device.split(":", 1)[1] if ":" in device else "0"
        self.model = torch.hub.load(repo, "custom", path=weights, source="local", device=device, _verbose=False)
        self.model.conf = float(self.get_parameter("conf_thres").value)
        self.model.iou = float(self.get_parameter("iou_thres").value)
        self.img_size = int(self.get_parameter("img_size").value)
        self.get_logger().info(f"loaded {weights} on {device} (conf {self.model.conf}, iou {self.model.iou}, img {self.img_size})")

        self.pub = self.create_publisher(BoundingBoxes, str(self.get_parameter("output_topic").value), 1)
        self.publish_image = bool(self.get_parameter("publish_image").value)
        self.compressed_output = bool(self.get_parameter("compressed_output").value)
        self.line_thickness = int(self.get_parameter("line_thickness").value)
        self.img_pub = None
        if self.publish_image:
            from ultralytics.utils.plotting import Annotator, colors  # noqa: F401 (import check)

            self._Annotator, self._colors = Annotator, colors
            out_topic = str(self.get_parameter("output_image_topic").value)
            self.img_pub = self.create_publisher(
                CompressedImage if self.compressed_output else Image, out_topic, 1
            )
            self.get_logger().info(f"publishing annotated images on {out_topic}")
        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        topic = str(self.get_parameter("input_image_topic").value)
        if bool(self.get_parameter("compressed").value):
            self.create_subscription(CompressedImage, topic, self.compressed_cb, qos)
        else:
            self.create_subscription(Image, topic, self.image_cb, qos)
        self.get_logger().info(f"listening on {topic} -> {self.get_parameter('output_topic').value}")
        self._frames = 0

    def compressed_cb(self, msg: CompressedImage):
        import cv2

        img = cv2.imdecode(np.frombuffer(bytes(msg.data), dtype=np.uint8), cv2.IMREAD_COLOR)
        self.detect(img[:, :, ::-1], msg.header)

    def image_cb(self, msg: Image):
        self.detect(image_to_rgb(msg), msg.header)

    def detect(self, rgb: np.ndarray, image_header):
        with torch.no_grad():
            results = self.model(np.ascontiguousarray(rgb), size=self.img_size)
        dets = results.xyxy[0].cpu().numpy()
        out = BoundingBoxes()
        out.header.stamp = self.get_clock().now().to_msg()
        out.image_header = image_header
        for *xyxy, conf, cls in dets:
            bb = BoundingBox()
            bb.class_name = str(self.model.names[int(cls)])
            bb.probability = float(conf)
            bb.xmin, bb.ymin, bb.xmax, bb.ymax = (int(v) for v in xyxy)
            out.bounding_boxes.append(bb)
        self.pub.publish(out)
        if self.img_pub is not None:
            # draw boxes exactly like the ROS 1 yolov5_ros wrapper (Annotator on a BGR
            # frame, box_label with per-class colour), publish as bgr8
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            ann = self._Annotator(bgr, line_width=self.line_thickness, example=str(self.model.names))
            for *xyxy, conf, cls in dets:
                c = int(cls)
                ann.box_label(xyxy, f"{self.model.names[c]} {conf:.2f}", color=self._colors(c, True))
            self.img_pub.publish(self._image_msg(ann.result(), image_header))
        self._frames += 1
        if self._frames == 1:
            self.get_logger().info(f"first frame: {len(out.bounding_boxes)} detection(s)")

    def _image_msg(self, bgr: np.ndarray, header):
        """Annotated frame as bgr8 (like yolov5_ros' cv2_to_imgmsg), or jpeg CompressedImage."""
        bgr = np.ascontiguousarray(bgr, dtype=np.uint8)
        if self.compressed_output:
            import cv2

            msg = CompressedImage()
            msg.header = header
            msg.format = "jpeg"
            msg.data = cv2.imencode(".jpg", bgr)[1].tobytes()
            return msg
        msg = Image()
        msg.header = header
        msg.height, msg.width = bgr.shape[0], bgr.shape[1]
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = bgr.shape[1] * 3
        msg.data = bgr.tobytes()
        return msg


def main():
    rclpy.init()
    torch.set_num_threads(1)
    rclpy.spin(YoloDetector())


if __name__ == "__main__":
    main()
