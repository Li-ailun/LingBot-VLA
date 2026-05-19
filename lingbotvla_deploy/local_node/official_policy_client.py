#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
import numpy as np
import websockets.sync.client

try:
    from .msgpack_numpy import Packer, unpackb
except ImportError:
    from msgpack_numpy import Packer, unpackb


logger = logging.getLogger("OfficialPolicyClient")


def _as_action_chunk_response(
    response: Dict[str, Any],
    current_state: np.ndarray,
    dof_of_arm: int = 7,
    return_action_type: str = "arm_delta_gripper_absolute",
) -> Dict[str, Any]:
    """
    Normalize official LingBot-VLA response to our local format:

    {
        "actions": np.ndarray[T, 16],
        "action_type": "arm_delta_gripper_absolute",
        "raw_response": response
    }

    Supported response formats:
    1. {"actions": [T,16]}
    2. {"action": [T,16]}
    3. {"action.arm.position": [T,14], "action.effector.position": [T,2]}
    """
    current_state = np.asarray(current_state, dtype=np.float32).reshape(-1)
    action_dim = dof_of_arm * 2 + 2

    if "actions" in response:
        actions = np.asarray(response["actions"], dtype=np.float32)

    elif "action" in response:
        actions = np.asarray(response["action"], dtype=np.float32)

    else:
        arm_key = None
        eff_key = None

        for k in response.keys():
            if k.endswith("arm.position") or k == "action.arm.position":
                arm_key = k
            if k.endswith("effector.position") or k == "action.effector.position":
                eff_key = k

        if arm_key is None or eff_key is None:
            raise KeyError(
                f"Cannot normalize official response. keys={list(response.keys())}"
            )

        arm = np.asarray(response[arm_key], dtype=np.float32)
        eff = np.asarray(response[eff_key], dtype=np.float32)

        if arm.ndim == 1:
            arm = arm[None, :]
        if eff.ndim == 1:
            eff = eff[None, :]

        t = min(arm.shape[0], eff.shape[0])

        # Official unapply usually returns absolute arm target when robot_config uses subtract_state: true.
        absolute = np.zeros((t, action_dim), dtype=np.float32)
        absolute[:, 0:dof_of_arm] = arm[:t, 0:dof_of_arm]
        absolute[:, dof_of_arm] = eff[:t, 0]
        absolute[:, dof_of_arm + 1 : dof_of_arm + 1 + dof_of_arm] = arm[
            :t, dof_of_arm : dof_of_arm * 2
        ]
        absolute[:, -1] = eff[:t, 1]

        if return_action_type == "absolute_qpos":
            actions = absolute
        else:
            actions = absolute.copy()
            actions[:, 0:dof_of_arm] -= current_state[0:dof_of_arm]
            right_slice = slice(dof_of_arm + 1, dof_of_arm + 1 + dof_of_arm)
            actions[:, right_slice] -= current_state[right_slice]
            # gripper keeps absolute value

    if actions.ndim == 1:
        actions = actions[None, :]

    if actions.shape[1] != action_dim:
        raise ValueError(f"Action dim mismatch: expected={action_dim}, got={actions.shape}")

    return {
        "request_id": response.get("request_id", None),
        "action_type": return_action_type,
        "actions": actions.astype(np.float32),
        "raw_response": response,
        "server_timing": response.get("server_timing", {}),
    }


class OfficialPolicyClient:
    """
    LingBot official msgpack websocket client.

    It talks to official deploy.websocket_policy_server.WebsocketPolicyServer.

    Server URL should be like:
        ws://127.0.0.1:8000

    Not:
        ws://127.0.0.1:8000/ws
    """

    def __init__(
        self,
        server_url: str = "ws://127.0.0.1:8000",
        dof_of_arm: int = 7,
        return_action_type: str = "arm_delta_gripper_absolute",
        api_key: Optional[str] = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.dof_of_arm = int(dof_of_arm)
        self.return_action_type = return_action_type
        self.api_key = api_key
        self.packer = Packer()

        headers = {"Authorization": f"Api-Key {api_key}"} if api_key else None

        logger.info("Connecting to official policy server: %s", self.server_url)
        self.ws = websockets.sync.client.connect(
            self.server_url,
            compression=None,
            max_size=None,
            additional_headers=headers,
        )

        metadata_raw = self.ws.recv()
        self.server_metadata = unpackb(metadata_raw)
        logger.info("Connected. Server metadata: %s", self.server_metadata)

    def close(self):
        if self.ws is not None:
            self.ws.close()

    def build_official_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        state = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
        images = observation["images"]

        required = [
            "camera_top",
            "camera_wrist_left",
            "camera_wrist_right",
        ]

        missing = [k for k in required if k not in images]
        if missing:
            raise KeyError(f"Missing images: {missing}")

        return {
            "task": observation.get("instruction", ""),
            "observation.state": state,
            "observation.images.camera_top": np.asarray(images["camera_top"], dtype=np.uint8),
            "observation.images.camera_wrist_left": np.asarray(images["camera_wrist_left"], dtype=np.uint8),
            "observation.images.camera_wrist_right": np.asarray(images["camera_wrist_right"], dtype=np.uint8),
        }

    def infer(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        current_state = np.asarray(observation["state"], dtype=np.float32).reshape(-1)

        official_obs = self.build_official_observation(observation)

        self.ws.send(self.packer.pack(official_obs))
        response_raw = self.ws.recv()

        if isinstance(response_raw, str):
            raise RuntimeError(f"Server returned error string:\n{response_raw}")

        response = unpackb(response_raw)

        return _as_action_chunk_response(
            response=response,
            current_state=current_state,
            dof_of_arm=self.dof_of_arm,
            return_action_type=self.return_action_type,
        )

    def reset(self, robot_name: str):
        self.ws.send(
            self.packer.pack(
                {
                    "reset": True,
                    "robo_name": robot_name,
                }
            )
        )
        raw = self.ws.recv()
        if not isinstance(raw, str):
            return unpackb(raw)
        return raw
