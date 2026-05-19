#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS2 bridge for LingBot-VLA local deployment.

Responsibilities:
- Subscribe robot feedback topics.
- Subscribe optional ROS image topics when cameras.*.source == "ros".
- Decode ROS CompressedImage / Image to RGB numpy arrays.
- Keep latest / nearest buffers by timestamp.
- Publish robot command topics.

Time:
- use_recv_time=True:
    message_time = local receive time, time.time()
- use_recv_time=False:
    message_time = ROS header.stamp
"""

from __future__ import annotations

import time
import threading
import logging
from collections import deque
from typing import Any, Dict, Optional, Sequence

import numpy as np

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import JointState, CompressedImage, Image
from geometry_msgs.msg import PoseStamped
from tf2_msgs.msg import TFMessage


try:
    from .robot_topics import RobotTopicsConfig
except ImportError:
    from robot_topics import RobotTopicsConfig


logger = logging.getLogger("Ros2Bridge")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)


def stamp_to_sec(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def get_msg_timestamp(msg: Any, fallback_now: Optional[float] = None) -> float:
    if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
        return stamp_to_sec(msg.header.stamp)
    return time.time() if fallback_now is None else fallback_now


def pose_to_7d_array(pose_msg) -> np.ndarray:
    p = pose_msg.position
    q = pose_msg.orientation

    return np.array(
        [p.x, p.y, p.z, q.x, q.y, q.z, q.w],
        dtype=np.float32,
    )


def array_to_joint_state(
    values: Sequence[float],
    stamp=None,
    names: Optional[Sequence[str]] = None,
    frame_id: str = "",
) -> JointState:
    msg = JointState()

    if stamp is not None:
        msg.header.stamp = stamp

    msg.header.frame_id = frame_id

    if names is not None:
        msg.name = list(names)

    msg.position = [float(x) for x in values]
    msg.velocity = []
    msg.effort = []

    return msg


def array_to_pose_stamped(
    values: Sequence[float],
    stamp=None,
    frame_id: str = "base_link",
) -> PoseStamped:
    if len(values) != 7:
        raise ValueError(f"Pose command must have 7 values, got {len(values)}")

    msg = PoseStamped()

    if stamp is not None:
        msg.header.stamp = stamp

    msg.header.frame_id = frame_id

    msg.pose.position.x = float(values[0])
    msg.pose.position.y = float(values[1])
    msg.pose.position.z = float(values[2])

    msg.pose.orientation.x = float(values[3])
    msg.pose.orientation.y = float(values[4])
    msg.pose.orientation.z = float(values[5])
    msg.pose.orientation.w = float(values[6])

    return msg


def compressed_image_to_rgb_array(
    msg: CompressedImage,
    resize_to=None,
) -> np.ndarray:
    import cv2

    arr = np.frombuffer(msg.data, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise RuntimeError("Failed to decode CompressedImage")

    if resize_to is not None:
        out_w, out_h = resize_to
        image_bgr = cv2.resize(image_bgr, (int(out_w), int(out_h)))

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb.astype(np.uint8)


def raw_image_to_rgb_array(
    msg: Image,
    resize_to=None,
) -> np.ndarray:
    import cv2

    h = int(msg.height)
    w = int(msg.width)
    step = int(msg.step)
    encoding = str(msg.encoding).lower()

    raw = np.frombuffer(msg.data, dtype=np.uint8)

    if encoding in ("rgb8", "bgr8"):
        row = raw.reshape(h, step)
        image = row[:, : w * 3].reshape(h, w, 3)

        if encoding == "bgr8":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    elif encoding in ("mono8", "8uc1"):
        row = raw.reshape(h, step)
        gray = row[:, :w].reshape(h, w)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    else:
        raise ValueError(f"Unsupported Image encoding: {msg.encoding}")

    if resize_to is not None:
        out_w, out_h = resize_to
        image = cv2.resize(image, (int(out_w), int(out_h)))

    return image.astype(np.uint8)


class MessageBuffer:
    def __init__(self, maxlen: int):
        self._buf = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, item: Dict[str, Any]) -> None:
        with self._lock:
            self._buf.append(item)

    def latest(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self._buf:
                return None
            return self._buf[-1]

    def nearest(self, target_time: float) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self._buf:
                return None

            return min(
                self._buf,
                key=lambda item: abs(float(item["message_time"]) - float(target_time)),
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


class Ros2Bridge:
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        use_recv_time: bool = True,
        num_threads: Optional[int] = None,
        node_name: str = "lingbotvla_ros2_bridge",
    ):
        self.config = config or {}
        self.robot_config = self.config.get("robot", {})
        self.config_path = self.config.get("config_path", None)

        self.hardware = self.robot_config.get("hardware", "R1_PRO")
        self.enable_publish = set(self.robot_config.get("enable_publish", []))

        if self.hardware == "R1_LITE":
            self.dof_of_arm = 6
        elif self.hardware == "R1_PRO":
            self.dof_of_arm = 7
        else:
            raise ValueError(f"Unknown hardware: {self.hardware}")

        self.use_recv_time = bool(use_recv_time)
        self.topics_config = RobotTopicsConfig(config_path=self.config_path)

        self.obs_buffer: Dict[str, MessageBuffer] = {}
        self.tf_buffer: Dict[str, MessageBuffer] = {}
        self.subscribers = {}
        self.publishers = {}

        # IMPORTANT:
        # rclpy must be initialized before creating Executor / GuardCondition.
        # Otherwise Humble may raise:
        #   AttributeError: __enter__
        if not rclpy.ok():
            rclpy.init(args=None)

        self.node = rclpy.create_node(node_name)

        self.callback_group = ReentrantCallbackGroup()
        self.executor = MultiThreadedExecutor(num_threads=num_threads)

        self._init_subscribers()
        self._init_publishers()

        self.executor.add_node(self.node)

        self._executor_thread = threading.Thread(
            target=self._run_executor,
            daemon=True,
        )
        self._executor_thread.start()

        logger.info(
            "Ros2Bridge initialized. hardware=%s dof=%s use_recv_time=%s ros_images=%s enable_publish=%s",
            self.hardware,
            self.dof_of_arm,
            self.use_recv_time,
            list(self.topics_config.images.keys()),
            sorted(self.enable_publish),
        )

    def now(self) -> float:
        return time.time()

    def _create_data_dict(
        self,
        msg_time: float,
        data: Any,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        recv_time = self.now()

        item = {
            "message_time": recv_time if self.use_recv_time else msg_time,
            "header_time": msg_time,
            "receive_time": recv_time,
            "data": data,
        }

        if extra:
            item.update(extra)

        return item

    def _init_subscribers(self) -> None:
        for name, topic in self.topics_config.state.items():
            self.obs_buffer[name] = MessageBuffer(
                maxlen=self.topics_config.state_deque_length
            )

            if "ee_pose" in name:
                cb = lambda msg, n=name: self._pose_callback(msg, n)
            else:
                cb = lambda msg, n=name: self._joint_state_callback(msg, n)

            self.subscribers[name] = self.node.create_subscription(
                topic.msg_type,
                topic.channel,
                cb,
                self.topics_config.qos["sub"],
                callback_group=self.callback_group,
            )

            logger.info("Subscribed state: %s -> %s", name, topic.channel)

        for name, topic in self.topics_config.images.items():
            self.obs_buffer[name] = MessageBuffer(
                maxlen=self.topics_config.camera_deque_length
            )

            self.subscribers[name] = self.node.create_subscription(
                topic.msg_type,
                topic.channel,
                lambda msg, n=name, r=topic.resize_to: self._image_callback(msg, n, r),
                self.topics_config.qos["image"],
                callback_group=self.callback_group,
            )

            logger.info(
                "Subscribed ROS image: %s -> %s resize_to=%s",
                name,
                topic.channel,
                topic.resize_to,
            )

        for name, topic in self.topics_config.tf.items():
            qos_name = "tf_static" if name == "tf_static" else "tf"

            self.tf_buffer[name] = MessageBuffer(
                maxlen=self.topics_config.tf_deque_length
            )

            self.subscribers[name] = self.node.create_subscription(
                topic.msg_type,
                topic.channel,
                lambda msg, n=name: self._tf_callback(msg, n),
                self.topics_config.qos[qos_name],
                callback_group=self.callback_group,
            )

            logger.info("Subscribed TF: %s -> %s", name, topic.channel)

    def _init_publishers(self) -> None:
        for name, topic in self.topics_config.action.items():
            self.publishers[name] = self.node.create_publisher(
                topic.msg_type,
                topic.channel,
                self.topics_config.qos["pub"],
            )

            logger.info("Created publisher: %s -> %s", name, topic.channel)

    def _joint_state_callback(self, msg: JointState, state_name: str) -> None:
        msg_time = get_msg_timestamp(msg)

        position = np.asarray(msg.position, dtype=np.float32)
        velocity = np.asarray(msg.velocity, dtype=np.float32) if msg.velocity else None
        effort = np.asarray(msg.effort, dtype=np.float32) if msg.effort else None

        if state_name in ("left_arm", "right_arm"):
            data = position[: self.dof_of_arm]
        else:
            data = position

        self.obs_buffer[state_name].append(
            self._create_data_dict(
                msg_time=msg_time,
                data=data,
                extra={
                    "name": list(msg.name),
                    "velocity": velocity,
                    "effort": effort,
                },
            )
        )

    def _pose_callback(self, msg: PoseStamped, state_name: str) -> None:
        msg_time = get_msg_timestamp(msg)

        self.obs_buffer[state_name].append(
            self._create_data_dict(
                msg_time=msg_time,
                data=pose_to_7d_array(msg.pose),
                extra={"frame_id": msg.header.frame_id},
            )
        )

    def _image_callback(self, msg, image_name: str, resize_to=None) -> None:
        msg_time = get_msg_timestamp(msg)

        try:
            if isinstance(msg, CompressedImage):
                image_rgb = compressed_image_to_rgb_array(msg, resize_to=resize_to)
                encoding = "compressed"

            elif isinstance(msg, Image):
                image_rgb = raw_image_to_rgb_array(msg, resize_to=resize_to)
                encoding = msg.encoding

            else:
                raise TypeError(f"Unsupported image message type: {type(msg)}")

            self.obs_buffer[image_name].append(
                self._create_data_dict(
                    msg_time=msg_time,
                    data=image_rgb,
                    extra={
                        "source": "ros",
                        "encoding": encoding,
                        "shape": tuple(image_rgb.shape),
                    },
                )
            )

        except Exception as exc:
            logger.warning("Failed to process ROS image %s: %s", image_name, exc)

    def _tf_callback(self, msg: TFMessage, tf_name: str) -> None:
        recv_time = self.now()
        msg_time = get_msg_timestamp(msg.transforms[-1]) if len(msg.transforms) > 0 else recv_time

        self.tf_buffer[tf_name].append(
            self._create_data_dict(
                msg_time=msg_time,
                data=msg,
                extra={"num_transforms": len(msg.transforms)},
            )
        )

    def has_state(self, name: str) -> bool:
        return name in self.obs_buffer and len(self.obs_buffer[name]) > 0

    def get_latest_item(self, name: str) -> Optional[Dict[str, Any]]:
        if name not in self.obs_buffer:
            return None
        return self.obs_buffer[name].latest()

    def get_nearest_item(self, name: str, target_time: float) -> Optional[Dict[str, Any]]:
        if name not in self.obs_buffer:
            return None
        return self.obs_buffer[name].nearest(target_time)

    def get_latest_raw_obs(self) -> Optional[Dict[str, Any]]:
        required = ["left_arm", "right_arm", "left_gripper", "right_gripper"]

        for key in required:
            if not self.has_state(key):
                return None

        state = {}
        state_time = {}

        for name, buffer in self.obs_buffer.items():
            if name in self.topics_config.images:
                continue

            item = buffer.latest()
            if item is None:
                continue

            state[name] = np.asarray(item["data"], dtype=np.float32)
            state_time[name] = float(item["message_time"])

        timestamp = max(state_time.get(k, 0.0) for k in required)

        return {
            "timestamp": timestamp,
            "state": state,
            "state_time": state_time,
        }

    def get_nearest_raw_obs(self, reference_time: float) -> Optional[Dict[str, Any]]:
        required = ["left_arm", "right_arm", "left_gripper", "right_gripper"]

        state = {}
        state_time = {}

        for key in required:
            item = self.get_nearest_item(key, reference_time)

            if item is None:
                return None

            state[key] = np.asarray(item["data"], dtype=np.float32)
            state_time[key] = float(item["message_time"])

        for key in ["joint_states", "torso", "left_ee_pose", "right_ee_pose"]:
            item = self.get_nearest_item(key, reference_time)

            if item is not None:
                state[key] = np.asarray(item["data"], dtype=np.float32)
                state_time[key] = float(item["message_time"])

        return {
            "timestamp": float(reference_time),
            "state": state,
            "state_time": state_time,
        }

    def get_lingbot_qpos_state(
        self,
        reference_time: Optional[float] = None,
        gripper_scalar_mode: str = "first",
    ) -> Optional[np.ndarray]:
        obs = (
            self.get_latest_raw_obs()
            if reference_time is None
            else self.get_nearest_raw_obs(reference_time)
        )

        if obs is None:
            return None

        state = obs["state"]

        left_arm = np.asarray(state["left_arm"], dtype=np.float32)[: self.dof_of_arm]
        right_arm = np.asarray(state["right_arm"], dtype=np.float32)[: self.dof_of_arm]

        left_g_raw = np.asarray(state["left_gripper"], dtype=np.float32).reshape(-1)
        right_g_raw = np.asarray(state["right_gripper"], dtype=np.float32).reshape(-1)

        if left_g_raw.size == 0 or right_g_raw.size == 0:
            return None

        if gripper_scalar_mode == "mean":
            left_g = np.array([float(np.mean(left_g_raw))], dtype=np.float32)
            right_g = np.array([float(np.mean(right_g_raw))], dtype=np.float32)
        else:
            left_g = np.array([float(left_g_raw[0])], dtype=np.float32)
            right_g = np.array([float(right_g_raw[0])], dtype=np.float32)

        return np.concatenate(
            [
                left_arm,
                left_g,
                right_arm,
                right_g,
            ],
            axis=0,
        ).astype(np.float32)

    def get_latest_ros_images(self, require_all: bool = False) -> Optional[Dict[str, Any]]:
        if not self.topics_config.images:
            return None

        images = {}
        timestamps = {}
        header_times = {}
        receive_times = {}

        for name in self.topics_config.images.keys():
            item = self.get_latest_item(name)

            if item is None:
                if require_all:
                    return None
                continue

            images[name] = item["data"]
            timestamps[name] = float(item["message_time"])
            header_times[name] = float(item["header_time"])
            receive_times[name] = float(item["receive_time"])

        if not images:
            return None

        reference_time = timestamps.get("camera_top", max(timestamps.values()))

        return {
            "images": images,
            "timestamps": timestamps,
            "header_times": header_times,
            "receive_times": receive_times,
            "reference_time": reference_time,
            "source": "ros",
        }

    def get_synced_ros_images(
        self,
        reference_camera: str = "camera_top",
        require_all: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        EFMNode-style ROS image synchronization.

        1. Use latest reference_camera frame as reference.
        2. Find nearest image frame from every other ROS camera buffer.
        """
        if not self.topics_config.images:
            return None

        if reference_camera not in self.topics_config.images:
            raise KeyError(
                f"reference_camera={reference_camera} is not in ROS image topics: "
                f"{list(self.topics_config.images.keys())}"
            )

        ref_item = self.get_latest_item(reference_camera)

        if ref_item is None:
            if require_all:
                return None
            return self.get_latest_ros_images(require_all=require_all)

        reference_time = float(ref_item["message_time"])

        images = {}
        timestamps = {}
        header_times = {}
        receive_times = {}
        time_diffs = {}

        for name in self.topics_config.images.keys():
            item = self.get_nearest_item(name, reference_time)

            if item is None:
                if require_all:
                    return None
                continue

            images[name] = item["data"]
            timestamps[name] = float(item["message_time"])
            header_times[name] = float(item["header_time"])
            receive_times[name] = float(item["receive_time"])
            time_diffs[name] = abs(float(item["message_time"]) - reference_time)

        if not images:
            return None

        return {
            "images": images,
            "timestamps": timestamps,
            "header_times": header_times,
            "receive_times": receive_times,
            "time_diffs": time_diffs,
            "reference_camera": reference_camera,
            "reference_time": reference_time,
            "source": "ros_synced",
        }


    def _can_publish(self, name: str) -> bool:
        return name in self.enable_publish

    def _default_joint_names(self, name: str, values: Sequence[float]) -> Optional[list[str]]:
        """
        Default JointState.name for R1Pro command topics.

        If your controller accepts empty name, this is not strictly required.
        But publishing names makes the message match your tested ros2 topic pub format.
        """
        n = len(values)

        if name == "left_arm":
            return [f"left_arm_joint{i}" for i in range(1, n + 1)]

        if name == "right_arm":
            return [f"right_arm_joint{i}" for i in range(1, n + 1)]

        if name == "left_gripper":
            return ["R1PRO_left_gripper_joint"]

        if name == "right_gripper":
            return ["R1PRO_right_gripper_joint"]

        if name == "torso":
            return [f"torso_joint{i}" for i in range(1, n + 1)]

        return None

    def _default_frame_id(self, name: str) -> str:
        if name in ("left_arm", "right_arm", "torso"):
            return "base_link"

        # Your gripper test command uses empty frame_id.
        if name in ("left_gripper", "right_gripper"):
            return ""

        return ""


    def publish_joint_command(
        self,
        name: str,
        values: Sequence[float],
        joint_names: Optional[Sequence[str]] = None,
    ) -> None:
        if name not in self.publishers:
            raise KeyError(f"No publisher for action name: {name}")

        if not self._can_publish(name):
            return

        values = np.asarray(values, dtype=np.float32).reshape(-1)

        if joint_names is None:
            joint_names = self._default_joint_names(name, values)

        frame_id = self._default_frame_id(name)

        msg = array_to_joint_state(
            values=values,
            stamp=self.node.get_clock().now().to_msg(),
            names=joint_names,
            frame_id=frame_id,
        )

        self.publishers[name].publish(msg)

    def publish_pose_command(
        self,
        name: str,
        pose_7d: Sequence[float],
        frame_id: str = "base_link",
    ) -> None:
        if name not in self.publishers:
            raise KeyError(f"No publisher for action name: {name}")

        if not self._can_publish(name):
            return

        msg = array_to_pose_stamped(
            values=pose_7d,
            stamp=self.node.get_clock().now().to_msg(),
            frame_id=frame_id,
        )

        self.publishers[name].publish(msg)

    def publish_qpos_action(
        self,
        left_arm,
        right_arm,
        left_gripper=None,
        right_gripper=None,
    ) -> None:
        self.publish_joint_command(
            "left_arm",
            np.asarray(left_arm, dtype=np.float32).reshape(-1)[: self.dof_of_arm],
        )

        self.publish_joint_command(
            "right_arm",
            np.asarray(right_arm, dtype=np.float32).reshape(-1)[: self.dof_of_arm],
        )

        if left_gripper is not None:
            lg = np.asarray(
                [left_gripper] if np.isscalar(left_gripper) else left_gripper,
                dtype=np.float32,
            ).reshape(-1)

            self.publish_joint_command("left_gripper", lg)

        if right_gripper is not None:
            rg = np.asarray(
                [right_gripper] if np.isscalar(right_gripper) else right_gripper,
                dtype=np.float32,
            ).reshape(-1)

            self.publish_joint_command("right_gripper", rg)

    def wait_until_ready(
        self,
        timeout_s: float = 10.0,
        required_keys: Optional[Sequence[str]] = None,
    ) -> bool:
        required_keys = list(
            required_keys or ["left_arm", "right_arm", "left_gripper", "right_gripper"]
        )

        start = time.time()

        while time.time() - start < timeout_s:
            if all(self.has_state(k) for k in required_keys):
                return True
            time.sleep(0.05)

        missing = [k for k in required_keys if not self.has_state(k)]
        logger.warning("Ros2Bridge not ready. Missing: %s", missing)
        return False

    def wait_until_images_ready(
        self,
        timeout_s: float = 10.0,
        require_all: bool = True,
    ) -> bool:
        if not self.topics_config.images:
            return True

        keys = list(self.topics_config.images.keys())
        start = time.time()

        while time.time() - start < timeout_s:
            ready = [self.has_state(k) for k in keys]

            if all(ready) if require_all else any(ready):
                return True

            time.sleep(0.05)

        missing = [k for k in keys if not self.has_state(k)]
        logger.warning("ROS images not ready. Missing: %s", missing)
        return False

    def _run_executor(self) -> None:
        try:
            self.executor.spin()
        except Exception as exc:
            logger.exception("ROS2 executor error: %s", exc)


    def get_used_topic_time_report(self, reference_time, image_timestamps=None):
        """
        Report time alignment for ONLY the topics actually used by LingBot-VLA input.

        Used topics:
          images:
            camera_top / camera_wrist_left / camera_wrist_right
          states:
            left_arm / right_arm / left_gripper / right_gripper

        Not included:
          joint_states / torso / ee_pose / tf / commands

        The timestamp used here is the same message_time used by the bridge:
          - receive_time when sync.time_source == receive_time
          - header.stamp when sync.time_source == header_stamp
        """
        reference_time = float(reference_time)
        image_timestamps = image_timestamps or {}

        report = {
            "reference_time": reference_time,
            "time_source": "receive_time" if getattr(self, "use_recv_time", True) else "header_stamp",
            "used_topics": {},
            "missing_topics": [],
            "max_abs_diff_s": None,
            "max_abs_diff_topic": None,
        }

        def add_item(name, kind, timestamp, topic=None):
            if timestamp is None:
                report["missing_topics"].append(name)
                return

            timestamp = float(timestamp)
            dt = timestamp - reference_time

            report["used_topics"][name] = {
                "kind": kind,
                "topic": topic or name,
                "timestamp": timestamp,
                "dt_s": dt,
                "abs_dt_s": abs(dt),
                "dt_ms": dt * 1000.0,
                "abs_dt_ms": abs(dt) * 1000.0,
            }

        # Images used by VLA.
        ros_image_topics = getattr(self, "ros_image_topics", {})
        for cam_name in ("camera_top", "camera_wrist_left", "camera_wrist_right"):
            topic = None
            try:
                topic_cfg = ros_image_topics.get(cam_name, None)
                if isinstance(topic_cfg, dict):
                    topic = topic_cfg.get("topic", None)
                elif hasattr(topic_cfg, "topic"):
                    topic = topic_cfg.topic
            except Exception:
                topic = None

            add_item(
                name=cam_name,
                kind="image",
                timestamp=image_timestamps.get(cam_name, None),
                topic=topic,
            )

        # State topics actually used to build 16D qpos.
        state_buffers = getattr(self, "state_buffers", None)
        if state_buffers is None:
            state_buffers = getattr(self, "buffers", {})

        state_topics = getattr(self, "state_topics", {})

        for state_name in ("left_arm", "right_arm", "left_gripper", "right_gripper"):
            item = None
            buf = state_buffers.get(state_name, None) if isinstance(state_buffers, dict) else None

            if buf is not None:
                try:
                    if hasattr(buf, "nearest"):
                        item = buf.nearest(reference_time)
                    elif hasattr(buf, "get_nearest"):
                        item = buf.get_nearest(reference_time)
                except Exception:
                    item = None

            timestamp = None
            if isinstance(item, dict):
                timestamp = item.get("message_time", item.get("timestamp", None))

            topic = None
            try:
                topic_cfg = state_topics.get(state_name, None)
                if isinstance(topic_cfg, dict):
                    topic = topic_cfg.get("topic", None)
                elif hasattr(topic_cfg, "topic"):
                    topic = topic_cfg.topic
            except Exception:
                topic = None

            add_item(
                name=state_name,
                kind="state",
                timestamp=timestamp,
                topic=topic,
            )

        if report["used_topics"]:
            max_name, max_item = max(
                report["used_topics"].items(),
                key=lambda kv: kv[1]["abs_dt_s"],
            )
            report["max_abs_diff_s"] = float(max_item["abs_dt_s"])
            report["max_abs_diff_topic"] = max_name

        return report


    def destroy(self) -> None:
        try:
            self.executor.shutdown()
        except Exception:
            pass

        if hasattr(self, "_executor_thread") and self._executor_thread.is_alive():
            self._executor_thread.join(timeout=2.0)

        try:
            self.executor.remove_node(self.node)
        except Exception:
            pass

        try:
            self.node.destroy_node()
        except Exception:
            pass

        if rclpy.ok():
            rclpy.shutdown()


def main():
    bridge = Ros2Bridge(
        config={
            "robot": {
                "hardware": "R1_PRO",
                "enable_publish": [],
            }
        },
        use_recv_time=True,
    )

    try:
        print("state ready:", bridge.wait_until_ready(timeout_s=10.0))
        print("ros image ready:", bridge.wait_until_images_ready(timeout_s=10.0, require_all=False))

        while rclpy.ok():
            state = bridge.get_lingbot_qpos_state()
            imgs = bridge.get_latest_ros_images(require_all=False)

            print(
                "state:",
                None if state is None else state.shape,
                "ros_images:",
                None if imgs is None else {k: v.shape for k, v in imgs["images"].items()},
            )

            time.sleep(1.0)

    except KeyboardInterrupt:
        pass

    finally:
        bridge.destroy()


if __name__ == "__main__":
    main()
