#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WebSocket client for local LingBot-VLA deployment.

Responsibilities:
- Connect to remote LingBot-VLA policy server.
- Serialize observation:
    - state: numpy -> list
    - images: RGB uint8 numpy -> JPEG bytes -> base64 string
- Send observation.
- Receive action chunk.
- Convert action chunk to np.ndarray.

Expected outgoing message:

{
    "type": "infer",
    "request_id": "...",
    "robot_name": "my_robot",
    "instruction": "...",
    "timestamp": 123.456,
    "state": [...],
    "images": {
        "camera_top": {
            "encoding": "jpeg_base64",
            "shape": [H, W, 3],
            "dtype": "uint8",
            "data": "..."
        },
        "camera_wrist_left": {...},
        "camera_wrist_right": {...}
    },
    "metadata": {...}
}

Expected incoming message:

{
    "type": "action_chunk",
    "request_id": "...",
    "action_type": "arm_delta_gripper_absolute",
    "actions": [
        [a0, a1, ..., a15],
        ...
    ]
}
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

import numpy as np


logger = logging.getLogger("PolicyClient")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)


def _json_safe(obj: Any) -> Any:
    """
    Convert common numpy/python objects to JSON-safe objects.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    return str(obj)


def encode_rgb_image_to_base64_jpeg(image_rgb: np.ndarray, quality: int = 80) -> Dict[str, Any]:
    """
    Encode RGB uint8 image to JPEG base64 dict.

    Input:
        image_rgb: RGB uint8 image, shape [H, W, 3]

    Output:
        {
            "encoding": "jpeg_base64",
            "shape": [H, W, 3],
            "dtype": "uint8",
            "quality": 80,
            "data": "..."
        }
    """
    import cv2

    image_rgb = np.asarray(image_rgb)

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image shape [H,W,3], got {image_rgb.shape}")

    if image_rgb.dtype != np.uint8:
        image_rgb = image_rgb.astype(np.uint8)

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    ok, encoded = cv2.imencode(
        ".jpg",
        image_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )

    if not ok:
        raise RuntimeError("Failed to encode image to JPEG")

    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")

    return {
        "encoding": "jpeg_base64",
        "shape": list(image_rgb.shape),
        "dtype": "uint8",
        "quality": int(quality),
        "data": b64,
    }


def decode_action_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decode server response and return normalized action result.

    Accepts either:
        response["actions"]
    or:
        response["action_chunk"]
    """
    if "error" in response:
        raise RuntimeError(f"Policy server returned error: {response['error']}")

    if response.get("type") not in (None, "action_chunk", "infer_result"):
        logger.warning("Unexpected response type: %s", response.get("type"))

    actions = response.get("actions", None)
    if actions is None:
        actions = response.get("action_chunk", None)

    if actions is None:
        raise KeyError(
            "Server response has no 'actions' or 'action_chunk'. "
            f"Response keys: {list(response.keys())}"
        )

    action_chunk = np.asarray(actions, dtype=np.float32)

    if action_chunk.ndim == 1:
        action_chunk = action_chunk[None, :]

    if action_chunk.ndim != 2:
        raise ValueError(f"Expected action_chunk shape [T,D], got {action_chunk.shape}")

    return {
        "request_id": response.get("request_id", None),
        "action_type": response.get("action_type", "arm_delta_gripper_absolute"),
        "actions": action_chunk,
        "raw_response": response,
    }


class PolicyClient:
    """
    WebSocket policy client.

    Two usage modes:

    1. Simple blocking one-shot mode:
        client = PolicyClient("ws://server:8000/ws")
        result = client.infer(observation)

    2. Async persistent mode:
        client = PolicyClient("ws://server:8000/ws")
        await client.connect()
        result = await client.infer_async(observation)
        await client.close()
    """

    def __init__(
        self,
        server_url: str,
        timeout_s: float = 30.0,
        jpeg_quality: int = 80,
        expected_action_dim: Optional[int] = 16,
        verbose: bool = False,
    ):
        self.server_url = server_url
        self.timeout_s = float(timeout_s)
        self.jpeg_quality = int(jpeg_quality)
        self.expected_action_dim = expected_action_dim
        self.verbose = verbose

        self.websocket = None
        self._connected = False

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def build_request(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert local observation into JSON-serializable request.
        """
        request_id = str(uuid.uuid4())

        state = np.asarray(observation["state"], dtype=np.float32).reshape(-1)

        if self.expected_action_dim is not None and state.shape[0] != self.expected_action_dim:
            raise ValueError(
                f"State dim mismatch. Expected {self.expected_action_dim}, got {state.shape[0]}"
            )

        images = observation.get("images", {})
        encoded_images = {}

        for name, img in images.items():
            encoded_images[name] = encode_rgb_image_to_base64_jpeg(
                img,
                quality=self.jpeg_quality,
            )

        request = {
            "type": "infer",
            "request_id": request_id,
            "robot_name": observation.get("robot_name", "my_robot"),
            "instruction": observation.get("instruction", ""),
            "timestamp": float(observation.get("timestamp", time.time())),
            "state": state.tolist(),
            "images": encoded_images,
            "metadata": _json_safe(observation.get("metadata", {})),
        }

        return request

    # ------------------------------------------------------------------
    # Async persistent connection
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Open persistent WebSocket connection.
        """
        if self._connected and self.websocket is not None:
            return

        import websockets

        logger.info("Connecting to policy server: %s", self.server_url)

        self.websocket = await asyncio.wait_for(
            websockets.connect(
                self.server_url,
                max_size=None,
                ping_interval=20,
                ping_timeout=20,
            ),
            timeout=self.timeout_s,
        )

        self._connected = True
        logger.info("Connected to policy server.")

    async def close(self) -> None:
        """
        Close persistent WebSocket connection.
        """
        if self.websocket is not None:
            await self.websocket.close()

        self.websocket = None
        self._connected = False
        logger.info("Policy client closed.")

    async def infer_async(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send one observation and receive action chunk using persistent connection.
        """
        if not self._connected or self.websocket is None:
            await self.connect()

        request = self.build_request(observation)
        payload = json.dumps(request, ensure_ascii=False)

        if self.verbose:
            image_shapes = {
                k: v.get("shape") for k, v in request["images"].items()
            }
            logger.info(
                "Sending request_id=%s state_dim=%d image_shapes=%s",
                request["request_id"],
                len(request["state"]),
                image_shapes,
            )

        await asyncio.wait_for(
            self.websocket.send(payload),
            timeout=self.timeout_s,
        )

        raw_reply = await asyncio.wait_for(
            self.websocket.recv(),
            timeout=self.timeout_s,
        )

        response = json.loads(raw_reply)
        result = decode_action_response(response)

        if self.expected_action_dim is not None:
            if result["actions"].shape[1] != self.expected_action_dim:
                raise ValueError(
                    f"Action dim mismatch. Expected {self.expected_action_dim}, "
                    f"got {result['actions'].shape[1]}"
                )

        return result

    # ------------------------------------------------------------------
    # Blocking one-shot connection
    # ------------------------------------------------------------------

    async def _infer_once_async(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        import websockets

        request = self.build_request(observation)
        payload = json.dumps(request, ensure_ascii=False)

        async with websockets.connect(
            self.server_url,
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
        ) as ws:
            await asyncio.wait_for(ws.send(payload), timeout=self.timeout_s)
            raw_reply = await asyncio.wait_for(ws.recv(), timeout=self.timeout_s)

        response = json.loads(raw_reply)
        result = decode_action_response(response)

        if self.expected_action_dim is not None:
            if result["actions"].shape[1] != self.expected_action_dim:
                raise ValueError(
                    f"Action dim mismatch. Expected {self.expected_action_dim}, "
                    f"got {result['actions'].shape[1]}"
                )

        return result

    def infer(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Blocking one-shot inference.

        This is easiest for first smoke test.
        Later ros2_node.py can use persistent async mode for lower latency.
        """
        return asyncio.run(self._infer_once_async(observation))


# ----------------------------------------------------------------------
# Offline serialization smoke test
# ----------------------------------------------------------------------

def _make_fake_observation() -> Dict[str, Any]:
    """
    Build fake observation for local serialization test.
    This does not connect to robot.
    """
    return {
        "robot_name": "my_robot",
        "instruction": "test instruction",
        "timestamp": time.time(),
        "state": np.zeros((16,), dtype=np.float32),
        "images": {
            "camera_top": np.zeros((180, 640, 3), dtype=np.uint8),
            "camera_wrist_left": np.zeros((480, 640, 3), dtype=np.uint8),
            "camera_wrist_right": np.zeros((480, 640, 3), dtype=np.uint8),
        },
        "metadata": {
            "test": True,
            "image_shapes": {
                "camera_top": (180, 640, 3),
                "camera_wrist_left": (480, 640, 3),
                "camera_wrist_right": (480, 640, 3),
            },
        },
    }


def main():
    """
    Smoke tests.

    1. Serialization only:
        python3 local_node/client_ws.py

    2. Send to server:
        python3 local_node/client_ws.py ws://127.0.0.1:8000/ws
    """
    import sys

    obs = _make_fake_observation()

    if len(sys.argv) == 1:
        client = PolicyClient(
            server_url="ws://127.0.0.1:8000/ws",
            expected_action_dim=16,
            jpeg_quality=80,
            verbose=True,
        )
        request = client.build_request(obs)

        print("serialization ok")
        print("request keys:", list(request.keys()))
        print("state dim:", len(request["state"]))
        print("image keys:", list(request["images"].keys()))
        for k, v in request["images"].items():
            print(k, "shape:", v["shape"], "base64 length:", len(v["data"]))
        return

    server_url = sys.argv[1]
    client = PolicyClient(
        server_url=server_url,
        expected_action_dim=16,
        jpeg_quality=80,
        verbose=True,
    )

    result = client.infer(obs)

    print("infer ok")
    print("request_id:", result["request_id"])
    print("action_type:", result["action_type"])
    print("actions shape:", result["actions"].shape)
    print("first action:", result["actions"][0])


if __name__ == "__main__":
    main()
