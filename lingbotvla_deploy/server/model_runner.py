#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model runner for LingBot-VLA deployment server.

Input from local client:
{
    "type": "infer",
    "request_id": "...",
    "robot_name": "my_robot",
    "instruction": "...",
    "state": [16 floats],
    "images": {
        "camera_top": {"encoding": "jpeg_base64", "data": "..."},
        "camera_wrist_left": {...},
        "camera_wrist_right": {...}
    }
}

Output to local client:
{
    "type": "action_chunk",
    "request_id": "...",
    "action_type": "absolute_qpos",
    "actions": [[16 floats], ...]
}
"""

from __future__ import annotations

import base64
import importlib
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


logger = logging.getLogger("LingBotVLAModelRunner")


def decode_base64_jpeg_to_rgb(image_info: Dict[str, Any]) -> np.ndarray:
    import cv2

    if image_info.get("encoding") != "jpeg_base64":
        raise ValueError(f"Unsupported image encoding: {image_info.get('encoding')}")

    data = base64.b64decode(image_info["data"])
    arr = np.frombuffer(data, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise RuntimeError("Failed to decode jpeg_base64 image.")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb.astype(np.uint8)


def vector_action_from_official_output(
    official_output: Dict[str, Any],
    current_state: np.ndarray,
    dof_of_arm: int = 7,
    return_action_type: str = "absolute_qpos",
) -> np.ndarray:
    """
    Convert official LingBot output dict to [T, 16].

    Official output may be one of:
    1. {"action": [T,16]}
    2. {"action.arm.position": [T,14], "action.effector.position": [T,2]}
    3. Other robot-config-reversed keys.

    Because official FeatureTransform.unapply() adds state back when subtract_state=True,
    the returned arm action is usually absolute qpos. For our local executor we return:
        absolute_qpos
    by converting:
        arm_delta = arm_absolute - current_arm
        gripper = gripper_absolute
    """
    current_state = np.asarray(current_state, dtype=np.float32).reshape(-1)
    expected_dim = dof_of_arm * 2 + 2

    if current_state.shape[0] != expected_dim:
        raise ValueError(
            f"current_state dim mismatch: expected={expected_dim}, got={current_state.shape[0]}"
        )

    # Case 1: direct full action vector.
    if "action" in official_output:
        action = np.asarray(official_output["action"], dtype=np.float32)
        if action.ndim == 1:
            action = action[None, :]
        if action.shape[1] != expected_dim:
            raise ValueError(f"official action dim mismatch: {action.shape}")
        absolute = action

    else:
        # Case 2: split action features.
        arm_key = None
        eff_key = None

        for k in official_output.keys():
            if k.endswith("arm.position") or k == "action.arm.position":
                arm_key = k
            if k.endswith("effector.position") or k == "action.effector.position":
                eff_key = k

        if arm_key is None or eff_key is None:
            raise KeyError(
                f"Cannot find action keys in official output. keys={list(official_output.keys())}"
            )

        arm = np.asarray(official_output[arm_key], dtype=np.float32)
        eff = np.asarray(official_output[eff_key], dtype=np.float32)

        if arm.ndim == 1:
            arm = arm[None, :]
        if eff.ndim == 1:
            eff = eff[None, :]

        if arm.shape[1] != dof_of_arm * 2:
            raise ValueError(f"arm action shape mismatch: {arm.shape}")
        if eff.shape[1] != 2:
            raise ValueError(f"effector action shape mismatch: {eff.shape}")

        t = min(arm.shape[0], eff.shape[0])
        arm = arm[:t]
        eff = eff[:t]

        absolute = np.zeros((t, expected_dim), dtype=np.float32)
        absolute[:, 0:dof_of_arm] = arm[:, 0:dof_of_arm]
        absolute[:, dof_of_arm] = eff[:, 0]
        absolute[:, dof_of_arm + 1 : dof_of_arm + 1 + dof_of_arm] = arm[:, dof_of_arm : dof_of_arm * 2]
        absolute[:, -1] = eff[:, 1]

    # GM-100 GalaxeaR1Pro actions are absolute targets.
    # Always return absolute_qpos:
    # [left_arm_abs_7, left_gripper_abs, right_arm_abs_7, right_gripper_abs]
    return absolute.astype(np.float32)


class DummyModelRunner:
    """
    For communication test only.
    It returns zero arm deltas and keeps current gripper values.
    """

    def __init__(
        self,
        action_dim: int = 16,
        dof_of_arm: int = 7,
        use_length: int = 5,
        action_type: str = "absolute_qpos",
    ):
        self.action_dim = int(action_dim)
        self.dof_of_arm = int(dof_of_arm)
        self.use_length = int(use_length)
        self.action_type = action_type

    def infer(self, request: Dict[str, Any]) -> Dict[str, Any]:
        state = np.asarray(request["state"], dtype=np.float32).reshape(-1)

        actions = np.repeat(state[None, :], self.use_length, axis=0).astype(np.float32)

        if self.action_type == "absolute_qpos":
            actions[:] = state[None, :]

        else:
            raise ValueError(f"Unsupported dummy action_type: {self.action_type}")

        return {
            "actions": actions,
            "action_type": self.action_type,
            "extra": {"dummy": True},
        }


class LingBotVLAModelRunner:
    """
    Wrapper around official deploy.lingbot_vla_policy.LingbotVLAServer.

    This wrapper converts our JSON websocket observation to official observation dict,
    and converts official output back to local action chunk format.
    """

    def __init__(
        self,
        repo_dir: str,
        model_path: str,
        qwen25_path: str,
        robot_name: str = "my_robot",
        robot_config_path: Optional[str] = None,
        norm_path: Optional[str] = None,
        use_length: int = 25,
        num_denoising_step: int = 10,
        use_compile: bool = False,
        use_bf16: bool = True,
        use_fp32: bool = False,
        dof_of_arm: int = 7,
        return_action_type: str = "absolute_qpos",
    ):
        self.repo_dir = Path(repo_dir).expanduser().resolve()
        self.model_path = Path(model_path).expanduser().resolve()
        self.qwen25_path = Path(qwen25_path).expanduser().resolve()
        self.robot_name = robot_name
        self.robot_config_path = (
            Path(robot_config_path).expanduser().resolve() if robot_config_path else None
        )
        self.norm_path = norm_path
        self.use_length = int(use_length)
        self.num_denoising_step = int(num_denoising_step)
        self.use_compile = bool(use_compile)
        self.use_bf16 = bool(use_bf16)
        self.use_fp32 = bool(use_fp32)
        self.dof_of_arm = int(dof_of_arm)
        self.return_action_type = return_action_type

        self.policy = None
        self.current_robot_name: Optional[str] = None

        self._validate_paths()
        self._prepare_imports()
        self._load_policy()

    def _validate_paths(self) -> None:
        if not self.repo_dir.exists():
            raise FileNotFoundError(f"repo_dir not found: {self.repo_dir}")

        if not self.model_path.exists():
            raise FileNotFoundError(f"model_path not found: {self.model_path}")

        if not self.qwen25_path.exists():
            raise FileNotFoundError(f"qwen25_path not found: {self.qwen25_path}")

        cli_yaml = self.model_path / "lingbotvla_cli.yaml"
        if not cli_yaml.exists():
            raise FileNotFoundError(
                f"Missing {cli_yaml}. Official LingBot-VLA inference expects a post-training "
                f"hf_ckpt containing lingbotvla_cli.yaml. Your base pretrained checkpoint may "
                f"not be directly deployable. Use --dummy to test communication, or provide a "
                f"post-training checkpoint."
            )

    def _prepare_imports(self) -> None:
        os.environ["QWEN25_PATH"] = str(self.qwen25_path)

        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))

        if self.robot_config_path is not None:
            dst_dir = self.repo_dir / "configs" / "robot_configs"
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{self.robot_name}.yaml"
            if self.robot_config_path.resolve() != dst.resolve():
                shutil.copyfile(self.robot_config_path, dst)
                logger.info("Copied robot config to: %s", dst)

        robot_config = self.repo_dir / "configs" / "robot_configs" / f"{self.robot_name}.yaml"
        if not robot_config.exists():
            raise FileNotFoundError(
                f"Robot config not found: {robot_config}. "
                f"Create configs/robot_configs/{self.robot_name}.yaml first."
            )

    def _load_policy(self) -> None:
        old_cwd = os.getcwd()
        os.chdir(str(self.repo_dir))
        try:
            module = importlib.import_module("deploy.lingbot_vla_policy")
            LingbotVLAServer = getattr(module, "LingbotVLAServer")

            logger.info("Loading LingBot-VLA model from: %s", self.model_path)

            self.policy = LingbotVLAServer(
                path_to_pi_model=str(self.model_path),
                use_length=self.use_length,
                use_bf16=self.use_bf16,
                use_fp32=self.use_fp32,
                robot_norm_path=self.norm_path,
                num_denoising_step=self.num_denoising_step,
                use_compile=self.use_compile,
            )

            self.reset(self.robot_name)

        finally:
            os.chdir(old_cwd)

    def reset(self, robot_name: Optional[str] = None) -> None:
        robot_name = robot_name or self.robot_name

        old_cwd = os.getcwd()
        os.chdir(str(self.repo_dir))
        try:
            logger.info("Resetting policy with robot config: %s", robot_name)
            self.policy.reset(robot_name)
            self.current_robot_name = robot_name
        finally:
            os.chdir(old_cwd)

    def _build_official_observation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        state = np.asarray(request["state"], dtype=np.float32).reshape(-1)

        images_req = request.get("images", {})
        required = ["camera_top", "camera_wrist_left", "camera_wrist_right"]
        missing = [k for k in required if k not in images_req]
        if missing:
            raise KeyError(f"Missing images in request: {missing}")

        obs = {
            "task": request.get("instruction", ""),
            "observation.state": state,
            "observation.images.camera_top": decode_base64_jpeg_to_rgb(images_req["camera_top"]),
            "observation.images.camera_wrist_left": decode_base64_jpeg_to_rgb(images_req["camera_wrist_left"]),
            "observation.images.camera_wrist_right": decode_base64_jpeg_to_rgb(images_req["camera_wrist_right"]),
        }

        return obs

    def infer(self, request: Dict[str, Any]) -> Dict[str, Any]:
        robot_name = request.get("robot_name", self.robot_name)

        if robot_name != self.current_robot_name:
            self.reset(robot_name)

        state = np.asarray(request["state"], dtype=np.float32).reshape(-1)

        official_obs = self._build_official_observation(request)

        old_cwd = os.getcwd()
        os.chdir(str(self.repo_dir))

        try:
            t0 = time.time()
            official_output = self.policy.infer(official_obs)
            infer_s = time.time() - t0
        finally:
            os.chdir(old_cwd)

        actions = vector_action_from_official_output(
            official_output=official_output,
            current_state=state,
            dof_of_arm=self.dof_of_arm,
            return_action_type=self.return_action_type,
        )

        return {
            "actions": actions,
            "action_type": self.return_action_type,
            "extra": {
                "official_keys": list(official_output.keys()),
                "model_infer_s": infer_s,
            },
        }
