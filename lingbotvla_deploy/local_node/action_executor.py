#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Execute action chunks on local robot.

Default LingBot-VLA action interpretation:

R1_PRO action/state layout, 16 dim:
[
    left_arm_7,
    left_gripper_1,
    right_arm_7,
    right_gripper_1,
]

Default action_type = "absolute_qpos":

- arm action:
    target_arm = current_arm + predicted_arm_delta

- gripper action:
    target_gripper = predicted_gripper_absolute

Then:
- apply optional absolute limits
- clip max step delta
- low-pass smooth
- publish absolute target qpos through ROS2
"""

from __future__ import annotations

import time
import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


logger = logging.getLogger("ActionExecutor")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)


class ActionExecutor:
    """
    Local action executor.

    Recommended first version:
        action_type = "absolute_qpos"

    Meaning:
        arm output is delta qpos
        gripper output is absolute gripper target

    Final published command is always absolute target qpos.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        ros_bridge: Optional[Any] = None,
    ):
        self.config = config or {}
        self.ros_bridge = ros_bridge

        robot_cfg = self.config.get("robot", {})
        safety_cfg = self.config.get("safety", {})

        # For compatibility:
        # If config still says "absolute_qpos", we interpret it as:
        # arm delta + gripper absolute.
        self.action_type = robot_cfg.get("action_type", "absolute_qpos")
        if self.action_type != "absolute_qpos":
            raise ValueError(
                f"Only absolute_qpos is supported now. Got action_type={self.action_type}"
            )

        self.control_frequency = float(robot_cfg.get("control_frequency", 15.0))
        self.action_steps = int(robot_cfg.get("action_steps", 5))

        self.state_dim = robot_cfg.get("state_dim", None)
        self.action_dim = robot_cfg.get("action_dim", self.state_dim)

        if self.ros_bridge is not None and hasattr(self.ros_bridge, "dof_of_arm"):
            self.dof_of_arm = int(self.ros_bridge.dof_of_arm)
        else:
            if self.action_dim is None:
                self.dof_of_arm = int(robot_cfg.get("dof_of_arm", 7))
            else:
                self.dof_of_arm = int((int(self.action_dim) - 2) // 2)

        self.expected_dim = self.dof_of_arm * 2 + 2

        if self.action_dim is None:
            self.action_dim = self.expected_dim

        if int(self.action_dim) != self.expected_dim:
            raise ValueError(
                f"Invalid action_dim={self.action_dim}. "
                f"For dof_of_arm={self.dof_of_arm}, expected {self.expected_dim}."
            )

        # Safety and smoothing.
        self.enable_safety = bool(safety_cfg.get("enable", True))

        # 关节是弧度单位，每个控制周期最多变化 0.02 rad
        self.max_joint_delta_per_step = safety_cfg.get("max_joint_delta_per_step", 0.02)

        # 夹爪是 0-100 单位，每个控制周期最多变化 5
        self.max_gripper_delta_per_step = safety_cfg.get("max_gripper_delta_per_step", 5.0)

        self.low_pass_alpha = float(safety_cfg.get("low_pass_alpha", 0.3))
        self.low_pass_alpha = float(np.clip(self.low_pass_alpha, 0.0, 1.0))

        # Gripper debounce / deadband.
        # 小于 0.5 的夹爪变化忽略，防止在目标附近抖动
        self.gripper_deadband = float(safety_cfg.get("gripper_deadband", 0.5))
        self.gripper_min_interval_s = float(safety_cfg.get("gripper_min_interval_s", 0.0))

        # Optional absolute limits.
        self.joint_lower = safety_cfg.get("joint_lower", None)
        self.joint_upper = safety_cfg.get("joint_upper", None)

        # 夹爪绝对范围：0-100
        self.gripper_lower = safety_cfg.get("gripper_lower", 0.0)
        self.gripper_upper = safety_cfg.get("gripper_upper", 100.0)

        self.prev_target: Optional[np.ndarray] = None
        self.prev_gripper_values: Optional[np.ndarray] = None
        self.last_gripper_update_time: float = 0.0

        logger.info(
            "ActionExecutor initialized: action_type=%s, dim=%d, dof_of_arm=%d, "
            "control_frequency=%.2f, action_steps=%d",
            self.action_type,
            self.expected_dim,
            self.dof_of_arm,
            self.control_frequency,
            self.action_steps,
        )

    # ------------------------------------------------------------------
    # Vector layout
    # ------------------------------------------------------------------

    def _indices(self) -> Dict[str, Any]:
        d = self.dof_of_arm
        return {
            "left_arm": slice(0, d),
            "left_gripper": d,
            "right_arm": slice(d + 1, d + 1 + d),
            "right_gripper": d + 1 + d,
        }

    def split_action(self, action: Sequence[float]) -> Tuple[np.ndarray, float, np.ndarray, float]:
        vec = np.asarray(action, dtype=np.float32).reshape(-1)

        if vec.shape[0] != self.expected_dim:
            raise ValueError(
                f"Invalid action dim. Expected {self.expected_dim}, got {vec.shape[0]}"
            )

        idx = self._indices()

        left_arm = vec[idx["left_arm"]]
        left_gripper = float(vec[idx["left_gripper"]])
        right_arm = vec[idx["right_arm"]]
        right_gripper = float(vec[idx["right_gripper"]])

        return left_arm, left_gripper, right_arm, right_gripper

    # ------------------------------------------------------------------
    # Core conversion: raw model action -> absolute target
    # ------------------------------------------------------------------

    def raw_action_to_target(
        self,
        current_state: Sequence[float],
        raw_action: Sequence[float],
    ) -> np.ndarray:
        """
        Convert one raw model action to one absolute target.

        Supported action_type:

        1. "absolute_qpos"
           arm target = current arm + raw arm delta
           gripper target = raw gripper absolute

        2. "absolute_qpos"
           whole vector is absolute target

        3. "absolute_qpos"
           whole vector is delta, including gripper
        """
        current = np.asarray(current_state, dtype=np.float32).reshape(-1)
        raw = np.asarray(raw_action, dtype=np.float32).reshape(-1)

        if current.shape[0] != self.expected_dim:
            raise ValueError(
                f"Invalid current_state dim. Expected {self.expected_dim}, got {current.shape[0]}"
            )

        if raw.shape[0] != self.expected_dim:
            raise ValueError(
                f"Invalid raw_action dim. Expected {self.expected_dim}, got {raw.shape[0]}"
            )

        if self.action_type != "absolute_qpos":
            raise ValueError(
                f"Only absolute_qpos is supported now. Got action_type={self.action_type}"
            )

        # GM-100 GalaxeaR1Pro action format:
        # [left_arm_abs_7, left_gripper_abs, right_arm_abs_7, right_gripper_abs]
        return raw.astype(np.float32)

    def process_action_chunk(
        self,
        current_state: Sequence[float],
        action_chunk: Sequence[Sequence[float]] | Sequence[float],
    ) -> np.ndarray:
        """
        Convert raw model action chunk to safe absolute qpos target chunk.

        Input:
            current_state: [D]
            action_chunk: [D] or [T, D]

        Output:
            processed absolute target chunk: [T, D]
        """
        current_state = np.asarray(current_state, dtype=np.float32).reshape(-1)

        if current_state.shape[0] != self.expected_dim:
            raise ValueError(
                f"Invalid current_state dim. Expected {self.expected_dim}, got {current_state.shape[0]}"
            )

        raw_chunk = np.asarray(action_chunk, dtype=np.float32)

        if raw_chunk.ndim == 1:
            raw_chunk = raw_chunk[None, :]

        if raw_chunk.ndim != 2:
            raise ValueError(f"action_chunk must be [D] or [T,D], got {raw_chunk.shape}")

        if raw_chunk.shape[1] != self.expected_dim:
            raise ValueError(
                f"Invalid action_chunk dim. Expected {self.expected_dim}, got {raw_chunk.shape[1]}"
            )

        raw_chunk = raw_chunk[: self.action_steps]

        processed = []
        previous_for_clip = current_state.copy()

        for i, raw_action in enumerate(raw_chunk):
            target = self.raw_action_to_target(
                current_state=current_state,
                raw_action=raw_action,
            )

            if self.enable_safety:
                target = self._apply_absolute_limits(target)
                target = self._clip_step_delta(
                    current=previous_for_clip,
                    target=target,
                )

            if i == 0:
                target = self._low_pass_filter(target)
            else:
                target = (
                    self.low_pass_alpha * target
                    + (1.0 - self.low_pass_alpha) * processed[-1]
                ).astype(np.float32)

            target = self._apply_gripper_debounce(target)

            processed.append(target.astype(np.float32))
            previous_for_clip = target.astype(np.float32)

        processed_chunk = np.stack(processed, axis=0)
        self.prev_target = processed_chunk[-1].copy()

        return processed_chunk

    def process_one_action(
        self,
        current_state: Sequence[float],
        action: Sequence[float],
    ) -> np.ndarray:
        return self.process_action_chunk(current_state, action)[0]

    # ------------------------------------------------------------------
    # Safety helpers
    # ------------------------------------------------------------------

    def _make_per_dim_step_limit(self) -> np.ndarray:
        d = self.dof_of_arm
        limit = np.zeros((self.expected_dim,), dtype=np.float32)

        joint_limit = self._to_array_or_scalar(self.max_joint_delta_per_step, length=d)
        gripper_limit = self._to_array_or_scalar(self.max_gripper_delta_per_step, length=1)

        idx = self._indices()

        limit[idx["left_arm"]] = joint_limit
        limit[idx["right_arm"]] = joint_limit
        limit[idx["left_gripper"]] = gripper_limit[0]
        limit[idx["right_gripper"]] = gripper_limit[0]

        return limit

    @staticmethod
    def _to_array_or_scalar(value: Any, length: int) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)

        if arr.size == 1:
            return np.full((length,), float(arr[0]), dtype=np.float32)

        if arr.size != length:
            raise ValueError(f"Expected scalar or length {length}, got {arr.size}")

        return arr.astype(np.float32)

    def _clip_step_delta(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        limit = self._make_per_dim_step_limit()
        delta = target - current
        clipped_delta = np.clip(delta, -limit, limit)
        return (current + clipped_delta).astype(np.float32)

    def _low_pass_filter(self, target: np.ndarray) -> np.ndarray:
        if self.prev_target is None:
            return target.astype(np.float32)

        if self.low_pass_alpha >= 1.0:
            return target.astype(np.float32)

        if self.low_pass_alpha <= 0.0:
            return self.prev_target.astype(np.float32)

        smoothed = (
            self.low_pass_alpha * target
            + (1.0 - self.low_pass_alpha) * self.prev_target
        )

        return smoothed.astype(np.float32)

    def _apply_gripper_debounce(self, target: np.ndarray) -> np.ndarray:
        idx = self._indices()
        g_idx = np.array(
            [idx["left_gripper"], idx["right_gripper"]],
            dtype=np.int64,
        )

        now = time.time()
        current_grippers = target[g_idx].copy()

        if self.prev_gripper_values is None:
            self.prev_gripper_values = current_grippers
            self.last_gripper_update_time = now
            return target

        diff = np.abs(current_grippers - self.prev_gripper_values)
        hold = np.zeros_like(current_grippers, dtype=bool)

        if self.gripper_deadband > 0:
            hold |= diff < self.gripper_deadband

        if self.gripper_min_interval_s > 0:
            if now - self.last_gripper_update_time < self.gripper_min_interval_s:
                hold |= diff > self.gripper_deadband

        new_grippers = current_grippers.copy()
        new_grippers[hold] = self.prev_gripper_values[hold]

        if np.any(~hold):
            self.last_gripper_update_time = now

        self.prev_gripper_values = new_grippers.copy()
        target[g_idx] = new_grippers

        return target.astype(np.float32)

    def _apply_absolute_limits(self, target: np.ndarray) -> np.ndarray:
        target = target.copy()
        idx = self._indices()

        if self.joint_lower is not None or self.joint_upper is not None:
            lower = -np.inf if self.joint_lower is None else self.joint_lower
            upper = np.inf if self.joint_upper is None else self.joint_upper

            lower_arr = self._to_array_or_scalar(lower, self.dof_of_arm)
            upper_arr = self._to_array_or_scalar(upper, self.dof_of_arm)

            target[idx["left_arm"]] = np.clip(target[idx["left_arm"]], lower_arr, upper_arr)
            target[idx["right_arm"]] = np.clip(target[idx["right_arm"]], lower_arr, upper_arr)

        if self.gripper_lower is not None or self.gripper_upper is not None:
            lower = -np.inf if self.gripper_lower is None else float(self.gripper_lower)
            upper = np.inf if self.gripper_upper is None else float(self.gripper_upper)

            target[idx["left_gripper"]] = np.clip(target[idx["left_gripper"]], lower, upper)
            target[idx["right_gripper"]] = np.clip(target[idx["right_gripper"]], lower, upper)

        return target.astype(np.float32)

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    def publish(self, action: Sequence[float]) -> None:
        """
        Publish one absolute qpos target vector.

        Important:
        - process_action_chunk() returns absolute target.
        - publish() expects absolute target.
        """
        if self.ros_bridge is None:
            raise RuntimeError("ros_bridge is None. Cannot publish action.")

        left_arm, left_gripper, right_arm, right_gripper = self.split_action(action)

        self.ros_bridge.publish_qpos_action(
            left_arm=left_arm,
            right_arm=right_arm,
            left_gripper=left_gripper,
            right_gripper=right_gripper,
        )

    def execute_action_chunk(
        self,
        action_chunk: Sequence[Sequence[float]] | Sequence[float],
        current_state: Optional[Sequence[float]] = None,
        sleep: bool = True,
    ) -> np.ndarray:
        """
        Process and execute an action chunk.

        If current_state is None, get it from ros_bridge.
        """
        if current_state is None:
            if self.ros_bridge is None:
                raise RuntimeError("current_state is None and ros_bridge is unavailable.")

            current_state = self.ros_bridge.get_lingbot_qpos_state()

            if current_state is None:
                raise RuntimeError("Failed to get current qpos state from ros_bridge.")

        processed_chunk = self.process_action_chunk(
            current_state=current_state,
            action_chunk=action_chunk,
        )

        dt = 1.0 / max(self.control_frequency, 1e-6)

        for target in processed_chunk:
            self.publish(target)

            if sleep:
                time.sleep(dt)

        return processed_chunk

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def reset_filter(self) -> None:
        self.prev_target = None
        self.prev_gripper_values = None
        self.last_gripper_update_time = 0.0

    def summarize_action(self, action: Sequence[float]) -> str:
        left_arm, left_gripper, right_arm, right_gripper = self.split_action(action)

        return (
            f"left_arm={np.round(left_arm, 4)}\n"
            f"left_gripper={left_gripper:.4f}\n"
            f"right_arm={np.round(right_arm, 4)}\n"
            f"right_gripper={right_gripper:.4f}"
        )


def main():
    """
    Offline smoke test.

    Run:
        cd ~/LingBotVLA/lingbotvla_deploy
        python3 local_node/action_executor.py
    """

    config = {
        "robot": {
            "control_frequency": 15.0,
            "action_steps": 5,
            "action_type": "absolute_qpos",
            "state_dim": 16,
            "action_dim": 16,
            "dof_of_arm": 7,
        },
        "safety": {
            "enable": True,
            "max_joint_delta_per_step": 0.02,
            "max_gripper_delta_per_step": 5.0,
            "low_pass_alpha": 0.3,
            "gripper_deadband": 0.5,
            "gripper_min_interval_s": 0.0,
            "gripper_lower": 0.0,
            "gripper_upper": 100.0,
        }
    }

    executor = ActionExecutor(config=config, ros_bridge=None)

    current_state = np.zeros((16,), dtype=np.float32)

    # Fake model output:
    # arm dims are delta
    # gripper dims are absolute
    action_chunk = np.zeros((50, 16), dtype=np.float32)

    # left arm delta
    action_chunk[:, 0:7] = 0.1

    # left gripper absolute
    action_chunk[:, 7] = 60.0

    # right arm delta
    action_chunk[:, 8:15] = -0.1

    # right gripper absolute
    action_chunk[:, 15] = 40.0

    processed = executor.process_action_chunk(
        current_state=current_state,
        action_chunk=action_chunk,
    )

    print("processed shape:", processed.shape)
    print("first target:")
    print(executor.summarize_action(processed[0]))
    print("last target:")
    print(executor.summarize_action(processed[-1]))

    print("\nExplanation:")
    print("- arm raw action is delta, but clipped to max_joint_delta_per_step=0.02")
    print("- gripper raw action is absolute, but clipped by max_gripper_delta_per_step=5.0")


if __name__ == "__main__":
    main()
