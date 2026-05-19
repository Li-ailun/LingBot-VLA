#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build synchronized observation from Python cameras or ROS cameras.

Pure Python mode:
    images come from CameraHub

Pure ROS mode:
    images come from Ros2Bridge.get_latest_ros_images()

Synchronization:
    reference_time = camera_top timestamp
    state = nearest robot state to reference_time
"""

from __future__ import annotations

import time
import logging
from typing import Any, Dict, Optional

import numpy as np


try:
    from .camera_sources import CameraHub
    from .ros2_bridge import Ros2Bridge
except ImportError:
    from camera_sources import CameraHub
    from ros2_bridge import Ros2Bridge


logger = logging.getLogger("ObservationBuilder")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)


class ObservationBuilder:
    def __init__(
        self,
        camera_hub: Optional[CameraHub],
        ros_bridge: Ros2Bridge,
        robot_name: str = "my_robot",
        instruction: str = "",
        require_all_cameras: bool = True,
        use_camera_time_for_state: bool = True,
        allow_latest_state_fallback: bool = True,
        expected_state_dim: Optional[int] = 16,
        include_debug_state: bool = True,
        reference_camera: str = "camera_top",
        required_cameras: Optional[list[str]] = None,
        max_state_time_diff_s: Optional[float] = 0.15,
    ):
        self.camera_hub = camera_hub
        self.ros_bridge = ros_bridge

        self.robot_name = robot_name
        self.instruction = instruction

        self.require_all_cameras = require_all_cameras
        self.use_camera_time_for_state = use_camera_time_for_state
        self.allow_latest_state_fallback = allow_latest_state_fallback
        self.expected_state_dim = expected_state_dim
        self.include_debug_state = include_debug_state

        self.reference_camera = reference_camera
        self.required_cameras = required_cameras or [
            "camera_top",
            "camera_wrist_left",
            "camera_wrist_right",
        ]
        self.max_state_time_diff_s = max_state_time_diff_s

        self.last_reference_time: Optional[float] = None
        self.frame_count = 0

    def set_instruction(self, instruction: str) -> None:
        self.instruction = instruction

    def set_robot_name(self, robot_name: str) -> None:
        self.robot_name = robot_name

    def _collect_images(self) -> Optional[Dict[str, Any]]:
        images: Dict[str, np.ndarray] = {}
        timestamps: Dict[str, float] = {}
        sources: Dict[str, str] = {}
        frame_indices: Dict[str, int] = {}

        if self.camera_hub is not None:
            py_data = self.camera_hub.get_lingbot_images(require_all=False)

            if py_data is not None:
                for name, img in py_data["images"].items():
                    images[name] = img
                    timestamps[name] = float(py_data["timestamps"][name])
                    sources[name] = "python"

                    if "frame_indices" in py_data:
                        frame_indices[name] = int(
                            py_data["frame_indices"].get(name, -1)
                        )

        if hasattr(self.ros_bridge, "get_synced_ros_images"):
            ros_data = self.ros_bridge.get_synced_ros_images(
                reference_camera=self.reference_camera,
                require_all=False,
            )
        else:
            ros_data = self.ros_bridge.get_latest_ros_images(require_all=False)

        if ros_data is not None:
            for name, img in ros_data["images"].items():
                images[name] = img
                timestamps[name] = float(ros_data["timestamps"][name])
                sources[name] = "ros"

        if not images:
            return None

        if self.require_all_cameras:
            missing = [name for name in self.required_cameras if name not in images]

            if missing:
                logger.warning("Missing cameras: %s", missing)
                return None

        if self.reference_camera in timestamps:
            reference_time = timestamps[self.reference_camera]
        else:
            reference_time = max(timestamps.values())

        return {
            "images": images,
            "timestamps": timestamps,
            "sources": sources,
            "frame_indices": frame_indices,
            "reference_time": reference_time,
        }

    def _get_state(self, reference_time: float) -> Optional[np.ndarray]:
        state = None

        if self.use_camera_time_for_state:
            state = self.ros_bridge.get_lingbot_qpos_state(
                reference_time=reference_time
            )

        if state is None and self.allow_latest_state_fallback:
            state = self.ros_bridge.get_lingbot_qpos_state(reference_time=None)

        return state

    def build(self) -> Optional[Dict[str, Any]]:
        image_data = self._collect_images()

        if image_data is None:
            return None

        reference_time = float(image_data["reference_time"])

        if self.last_reference_time is not None and reference_time == self.last_reference_time:
            return None

        state = self._get_state(reference_time)

        if state is None:
            logger.warning("Robot qpos state is not ready.")
            return None

        state = np.asarray(state, dtype=np.float32).reshape(-1)

        if self.expected_state_dim is not None and state.shape[0] != self.expected_state_dim:
            logger.warning(
                "Unexpected state dim. expected=%s actual=%s",
                self.expected_state_dim,
                state.shape[0],
            )
            return None

        obs: Dict[str, Any] = {
            "robot_name": self.robot_name,
            "instruction": self.instruction,
            "timestamp": reference_time,
            "state": state,
            "images": image_data["images"],
            "metadata": {
                "frame_count": self.frame_count,
                "camera_timestamps": image_data["timestamps"],
                "camera_sources": image_data["sources"],
                "camera_frame_indices": image_data["frame_indices"],
                "state_dim": int(state.shape[0]),
                "image_shapes": {
                    name: tuple(img.shape)
                    for name, img in image_data["images"].items()
                },
                "reference_camera": self.reference_camera,
            },
        }

        if self.include_debug_state:
            raw_ros_obs = self.ros_bridge.get_nearest_raw_obs(reference_time)

            if raw_ros_obs is None:
                raw_ros_obs = self.ros_bridge.get_latest_raw_obs()

            if raw_ros_obs is not None:
                state_time = raw_ros_obs.get("state_time", {})

                obs["metadata"]["ros_state_time"] = state_time
                obs["metadata"]["debug_state_keys"] = list(
                    raw_ros_obs.get("state", {}).keys()
                )
                obs["metadata"]["state_time_diff"] = {
                    k: abs(reference_time - float(v))
                    for k, v in state_time.items()
                }

                if self.max_state_time_diff_s is not None:
                    diffs = obs["metadata"]["state_time_diff"]
                    max_diff = max(diffs.values()) if diffs else 0.0
                    obs["metadata"]["max_state_time_diff"] = max_diff

                    if max_diff > self.max_state_time_diff_s:
                        logger.warning(
                            "Large state-camera time diff: %.4fs",
                            max_diff,
                        )

        self.last_reference_time = reference_time
        self.frame_count += 1


        # Unified timing report for ONLY currently used VLA input topics:
        #   images: camera_top / camera_wrist_left / camera_wrist_right
        #   states: left_arm / right_arm / left_gripper / right_gripper
        # This intentionally ignores unused topics such as joint_states, torso,
        # ee poses, tf, and command topics.
        try:
            metadata = obs.setdefault("metadata", {})
            reference_time_for_report = float(
                metadata.get("reference_time", obs.get("timestamp"))
            )
            image_timestamps_for_report = metadata.get("camera_timestamps", {})

            if self.ros_bridge is not None and hasattr(self.ros_bridge, "get_used_topic_time_report"):
                used_time_report = self.ros_bridge.get_used_topic_time_report(
                    reference_time=reference_time_for_report,
                    image_timestamps=image_timestamps_for_report,
                )

                metadata["used_time_report"] = used_time_report
                metadata["max_used_time_diff"] = used_time_report.get("max_abs_diff_s", None)
                metadata["max_used_time_diff_topic"] = used_time_report.get("max_abs_diff_topic", None)

                # Keep old key for terminal / website compatibility.
                if used_time_report.get("max_abs_diff_s", None) is not None:
                    metadata["max_state_time_diff"] = used_time_report["max_abs_diff_s"]

        except Exception as exc:
            metadata = obs.setdefault("metadata", {})
            metadata["used_time_report_error"] = str(exc)


        return obs

    def build_blocking(
        self,
        timeout_s: float = 10.0,
        sleep_s: float = 0.02,
    ) -> Optional[Dict[str, Any]]:
        start = time.time()

        while time.time() - start < timeout_s:
            obs = self.build()

            if obs is not None:
                return obs

            time.sleep(sleep_s)

        return None

    @staticmethod
    def summarize_observation(obs: Dict[str, Any]) -> str:
        if obs is None:
            return "Observation: None"

        lines = [
            f"robot_name: {obs.get('robot_name')}",
            f"instruction: {obs.get('instruction')}",
            f"timestamp: {obs.get('timestamp')}",
            f"state shape: {obs['state'].shape}, dtype={obs['state'].dtype}",
            "images:",
        ]

        for name, img in obs["images"].items():
            src = obs.get("metadata", {}).get("camera_sources", {}).get(name, "?")
            lines.append(
                f"  {name}: shape={img.shape}, dtype={img.dtype}, source={src}"
            )

        if "max_state_time_diff" in obs.get("metadata", {}):
            lines.append(
                f"max_state_time_diff: {obs['metadata']['max_state_time_diff']:.4f}s"
            )

        return "\n".join(lines)
