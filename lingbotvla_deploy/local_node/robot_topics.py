#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robot ROS2 topic config for LingBot-VLA local deployment.

Reads:
    configs/topics.yaml

Supports:
- ROS state topics
- ROS command topics
- TF topics
- ROS camera topics when cameras.*.source == "ros"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import yaml

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)

from sensor_msgs.msg import JointState, CompressedImage, Image
from geometry_msgs.msg import PoseStamped
from tf2_msgs.msg import TFMessage


@dataclass(frozen=True)
class Topic:
    channel: str
    msg_type: type
    resize_to: Optional[Tuple[int, int]] = None


def default_topics_yaml_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "topics.yaml"


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Empty YAML config: {path}")

    return cfg


def _state_msg_type(name: str) -> type:
    if "ee_pose" in name:
        return PoseStamped
    return JointState


def _command_msg_type(name: str) -> type:
    if "ee_pose" in name:
        return PoseStamped
    return JointState


def _image_msg_type(msg_type_name: str) -> type:
    msg_type_name = str(msg_type_name).lower()

    if msg_type_name in ("compressedimage", "sensor_msgs/msg/compressedimage"):
        return CompressedImage

    if msg_type_name in ("image", "sensor_msgs/msg/image"):
        return Image

    raise ValueError(f"Unsupported ROS image msg_type: {msg_type_name}")


@dataclass
class RobotTopicsConfig:
    config_path: str | Path | None = None

    state: Dict[str, Topic] = field(init=False)
    tf: Dict[str, Topic] = field(init=False)
    images: Dict[str, Topic] = field(init=False)
    action: Dict[str, Topic] = field(init=False)
    qos: Dict[str, QoSProfile] = field(init=False)

    camera_deque_length: int = 5
    state_deque_length: int = 80
    tf_deque_length: int = 30

    def __post_init__(self):
        path = (
            Path(self.config_path).expanduser().resolve()
            if self.config_path is not None
            else default_topics_yaml_path()
        )

        cfg = load_yaml_config(path)

        ros_cfg = cfg.get("ros_topics", {})
        states_cfg = ros_cfg.get("states", {})
        tf_cfg = ros_cfg.get("tf", {})
        commands_cfg = ros_cfg.get("commands", {})

        if not states_cfg:
            raise KeyError(f"No ros_topics.states found in {path}")

        if not commands_cfg:
            raise KeyError(f"No ros_topics.commands found in {path}")

        self.state = {
            name: Topic(channel=str(channel), msg_type=_state_msg_type(name))
            for name, channel in states_cfg.items()
        }

        self.tf = {
            name: Topic(channel=str(channel), msg_type=TFMessage)
            for name, channel in tf_cfg.items()
        }

        self.action = {
            name: Topic(channel=str(channel), msg_type=_command_msg_type(name))
            for name, channel in commands_cfg.items()
        }

        self.images = {}
        for logical_name, cam_cfg in cfg.get("cameras", {}).items():
            source = str(cam_cfg.get("source", "")).lower().strip()
            if source != "ros":
                continue

            output_name = cam_cfg.get("output_name", logical_name)
            ros_cfg = cam_cfg.get("ros", {})

            topic = ros_cfg.get("topic", None)
            if not topic:
                raise KeyError(f"ROS camera {logical_name} missing ros.topic")

            resize_to = ros_cfg.get("resize_to", None)
            if resize_to is not None:
                resize_to = tuple(resize_to)

            self.images[output_name] = Topic(
                channel=str(topic),
                msg_type=_image_msg_type(ros_cfg.get("msg_type", "CompressedImage")),
                resize_to=resize_to,
            )

        self.qos = {
            "sub": QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                durability=DurabilityPolicy.VOLATILE,
            ),
            "image": QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                durability=DurabilityPolicy.VOLATILE,
            ),
            "pub": QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                durability=DurabilityPolicy.VOLATILE,
            ),
            "tf": QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                durability=DurabilityPolicy.VOLATILE,
            ),
            "tf_static": QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        }


def main():
    cfg = RobotTopicsConfig()

    print("state topics:")
    for name, topic in cfg.state.items():
        print(f"  {name}: {topic.channel}, {topic.msg_type.__name__}")

    print("tf topics:")
    for name, topic in cfg.tf.items():
        print(f"  {name}: {topic.channel}, {topic.msg_type.__name__}")

    print("ros image topics:")
    for name, topic in cfg.images.items():
        print(f"  {name}: {topic.channel}, {topic.msg_type.__name__}, resize_to={topic.resize_to}")

    print("action topics:")
    for name, topic in cfg.action.items():
        print(f"  {name}: {topic.channel}, {topic.msg_type.__name__}")


if __name__ == "__main__":
    main()
