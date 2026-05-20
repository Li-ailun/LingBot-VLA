#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GM-100 Galaxea R1Pro adapter.

Local observation format used by our ros2_node:
{
    "instruction": "...",
    "state": [16],
    "images": {
        "camera_top": ...,
        "camera_wrist_left": ...,
        "camera_wrist_right": ...
    }
}

Official / GM-100-style fields:
    observation.images.head_rgb
    observation.images.left_wrist_rgb
    observation.images.right_wrist_rgb
    observation.state.left_arm
    observation.state.left_gripper
    observation.state.right_arm
    observation.state.right_gripper

Action vector layout:
    [left_arm_abs_7, left_gripper_abs, right_arm_abs_7, right_gripper_abs]
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np


DOF_OF_ARM = 7
ACTION_DIM = 16
STATE_DIM = 16


LOCAL_IMAGE_TO_GM100 = {
    "camera_top": "observation.images.head_rgb",
    "camera_wrist_left": "observation.images.left_wrist_rgb",
    "camera_wrist_right": "observation.images.right_wrist_rgb",
}


def as_state16(state: Any) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] != STATE_DIM:
        raise ValueError(f"Expected state dim {STATE_DIM}, got {state.shape}")
    return state


def split_state16(state: Any) -> Dict[str, np.ndarray]:
    """Split local 16D state into official GM-100 R1Pro state keys."""
    s = as_state16(state)
    return {
        "observation.state.left_arm": s[0:7].astype(np.float32),
        "observation.state.left_gripper": np.asarray([s[7]], dtype=np.float32),
        "observation.state.right_arm": s[8:15].astype(np.float32),
        "observation.state.right_gripper": np.asarray([s[15]], dtype=np.float32),
    }


def merge_action16(
    left_arm: Any,
    left_gripper: Any,
    right_arm: Any,
    right_gripper: Any,
) -> np.ndarray:
    """Merge official split action fields into our absolute_qpos 16D vector."""
    left_arm = np.asarray(left_arm, dtype=np.float32).reshape(-1)
    right_arm = np.asarray(right_arm, dtype=np.float32).reshape(-1)
    left_gripper = np.asarray(left_gripper, dtype=np.float32).reshape(-1)
    right_gripper = np.asarray(right_gripper, dtype=np.float32).reshape(-1)

    if left_arm.shape[0] != DOF_OF_ARM:
        raise ValueError(f"left_arm dim mismatch: {left_arm.shape}")
    if right_arm.shape[0] != DOF_OF_ARM:
        raise ValueError(f"right_arm dim mismatch: {right_arm.shape}")
    if left_gripper.shape[0] < 1 or right_gripper.shape[0] < 1:
        raise ValueError("gripper action must have at least one value")

    out = np.zeros((ACTION_DIM,), dtype=np.float32)
    out[0:7] = left_arm
    out[7] = float(left_gripper[0])
    out[8:15] = right_arm
    out[15] = float(right_gripper[0])
    return out


def map_images_to_gm100(
    images: Mapping[str, Any],
    *,
    duplicate_head_right: bool = False,
) -> Dict[str, Any]:
    """
    Map local 3-view images into official GM-100 keys.

    For now we use 3 views:
      camera_top -> head_rgb
      camera_wrist_left -> left_wrist_rgb
      camera_wrist_right -> right_wrist_rgb

    If the loaded official checkpoint requires head_right_rgb,
    set duplicate_head_right=True to copy camera_top into it.
    """
    out: Dict[str, Any] = {}

    for local_key, official_key in LOCAL_IMAGE_TO_GM100.items():
        if local_key not in images:
            raise KeyError(f"Missing local image: {local_key}")
        out[official_key] = images[local_key]

    if duplicate_head_right:
        out["observation.images.head_right_rgb"] = images["camera_top"]

    return out


def build_gm100_request(
    local_request: Mapping[str, Any],
    *,
    duplicate_head_right: bool = False,
) -> Dict[str, Any]:
    """
    Convert one local inference request to an official-style R1Pro request dict.
    This does not normalize; official model code should apply its own transforms.
    """
    if "state" not in local_request:
        raise KeyError("local_request missing 'state'")
    if "images" not in local_request:
        raise KeyError("local_request missing 'images'")

    out: Dict[str, Any] = {}

    out.update(split_state16(local_request["state"]))
    out.update(
        map_images_to_gm100(
            local_request["images"],
            duplicate_head_right=duplicate_head_right,
        )
    )

    instruction = (
        local_request.get("instruction")
        or local_request.get("task")
        or local_request.get("language_instruction")
        or "do the task"
    )

    # Keep several aliases because different official code paths may look for different names.
    out["instruction"] = instruction
    out["task"] = instruction
    out["language_instruction"] = instruction

    state16 = as_state16(local_request["state"]).astype(np.float32)

    out["robot_type"] = "galaxea_r1pro"
    out["action_type"] = "absolute_qpos"

    # Keep both names:
    # - "state" is our local/simple protocol
    # - "observation.state" is convenient for official FeatureTransform robot_config slicing
    out["state"] = state16
    out["observation.state"] = state16

    return out


def _to_2d_action_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def official_output_to_action_chunk(
    official_output: Mapping[str, Any],
    *,
    dof_of_arm: int = DOF_OF_ARM,
) -> np.ndarray:
    """
    Convert official model output to [T,16] absolute_qpos.

    Supported forms:
      1. {"actions": [T,16]}
      2. {"action": [T,16]}
      3. split:
         {"action.left_arm": [T,7],
          "action.left_gripper": [T,1] or [T],
          "action.right_arm": [T,7],
          "action.right_gripper": [T,1] or [T]}
      4. official unified:
         {"action.arm.position": [T,14],
          "action.effector.position": [T,2]}
    """
    if "actions" in official_output:
        arr = _to_2d_action_array(official_output["actions"])
        if arr.shape[1] != ACTION_DIM:
            raise ValueError(f"actions dim mismatch: {arr.shape}")
        return arr.astype(np.float32)

    if "action" in official_output:
        arr = _to_2d_action_array(official_output["action"])
        if arr.shape[1] != ACTION_DIM:
            raise ValueError(f"action dim mismatch: {arr.shape}")
        return arr.astype(np.float32)

    split_keys = [
        "action.left_arm",
        "action.left_gripper",
        "action.right_arm",
        "action.right_gripper",
    ]
    if all(k in official_output for k in split_keys):
        left_arm = _to_2d_action_array(official_output["action.left_arm"])
        right_arm = _to_2d_action_array(official_output["action.right_arm"])
        left_gripper = _to_2d_action_array(official_output["action.left_gripper"])
        right_gripper = _to_2d_action_array(official_output["action.right_gripper"])

        t = min(
            left_arm.shape[0],
            right_arm.shape[0],
            left_gripper.shape[0],
            right_gripper.shape[0],
        )
        rows = [
            merge_action16(
                left_arm[i],
                left_gripper[i],
                right_arm[i],
                right_gripper[i],
            )
            for i in range(t)
        ]
        return np.stack(rows).astype(np.float32)

    if "action.arm.position" in official_output and "action.effector.position" in official_output:
        arm = _to_2d_action_array(official_output["action.arm.position"])
        eff = _to_2d_action_array(official_output["action.effector.position"])

        if arm.shape[1] != dof_of_arm * 2:
            raise ValueError(f"action.arm.position dim mismatch: {arm.shape}")
        if eff.shape[1] != 2:
            raise ValueError(f"action.effector.position dim mismatch: {eff.shape}")

        t = min(arm.shape[0], eff.shape[0])
        out = np.zeros((t, ACTION_DIM), dtype=np.float32)
        out[:, 0:7] = arm[:t, 0:7]
        out[:, 7] = eff[:t, 0]
        out[:, 8:15] = arm[:t, 7:14]
        out[:, 15] = eff[:t, 1]
        return out

    raise KeyError(f"Unsupported official output keys: {list(official_output.keys())}")


def summarize_gm100_request(req: Mapping[str, Any]) -> str:
    lines = []
    lines.append(f"robot_type: {req.get('robot_type')}")
    lines.append(f"instruction: {req.get('instruction')}")
    for key in [
        "observation.state.left_arm",
        "observation.state.left_gripper",
        "observation.state.right_arm",
        "observation.state.right_gripper",
    ]:
        v = np.asarray(req[key])
        lines.append(f"{key}: shape={v.shape}, first={np.round(v.reshape(-1)[:4], 4)}")

    image_keys = [k for k in req.keys() if k.startswith("observation.images.")]
    lines.append("images: " + ", ".join(image_keys))
    return "\n".join(lines)
