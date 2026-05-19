#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Monitor WebSocket server for LingBot-VLA.

Purpose:
- Push robot telemetry to website/index.html.
- Receive simple UI commands from website.
- Do NOT participate in the real-time action control loop.

Default endpoint:
    ws://127.0.0.1:9000/monitor

Browser page:
    website/index.html

Typical future usage inside ros2_node.py:
    monitor = MonitorServer(host="127.0.0.1", port=9000)
    monitor.start_in_thread()

    ...
    monitor.update_from_observation(
        obs=obs,
        actions=actions,
        result=result,
        camera_mode=camera_mode,
        protocol=args.protocol,
        policy_connected=True,
    )
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np


logger = logging.getLogger("MonitorServer")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------
# JSON / image helpers
# ---------------------------------------------------------------------

def json_safe(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    return str(obj)


def encode_rgb_image_to_base64_jpeg(
    image_rgb: np.ndarray,
    quality: int = 60,
    max_width: Optional[int] = 640,
) -> Dict[str, Any]:
    """
    Convert RGB uint8 image to JPEG base64 for browser preview.

    Input:
        image_rgb: np.ndarray, shape [H,W,3], RGB

    Output:
        {
            "encoding": "jpeg_base64",
            "shape": [H,W,3],
            "dtype": "uint8",
            "quality": 60,
            "data": "..."
        }
    """
    import cv2

    image_rgb = np.asarray(image_rgb)

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected image shape [H,W,3], got {image_rgb.shape}")

    if image_rgb.dtype != np.uint8:
        image_rgb = image_rgb.astype(np.uint8)

    h, w = image_rgb.shape[:2]

    if max_width is not None and w > max_width:
        new_w = int(max_width)
        new_h = int(round(h * new_w / w))
        image_rgb = cv2.resize(image_rgb, (new_w, new_h))
        h, w = image_rgb.shape[:2]

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    ok, encoded = cv2.imencode(
        ".jpg",
        image_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )

    if not ok:
        raise RuntimeError("Failed to encode image to JPEG")

    return {
        "encoding": "jpeg_base64",
        "shape": [int(h), int(w), 3],
        "dtype": "uint8",
        "quality": int(quality),
        "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
    }


def build_telemetry_from_observation(
    obs: Optional[Dict[str, Any]] = None,
    actions: Optional[np.ndarray | Sequence[Sequence[float]]] = None,
    result: Optional[Dict[str, Any]] = None,
    camera_mode: str = "-",
    protocol: str = "-",
    policy_connected: Optional[bool] = None,
    jpeg_quality: int = 60,
    max_preview_width: int = 640,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convert ros2_node runtime data to website telemetry JSON.
    """
    telemetry: Dict[str, Any] = {
        "type": "telemetry",
        "timestamp": time.time(),
        "camera_mode": camera_mode,
        "protocol": protocol,
        "policy_connected": policy_connected,
    }

    if obs is not None:
        telemetry["instruction"] = obs.get("instruction", "")

        state = np.asarray(obs.get("state", []), dtype=np.float32).reshape(-1)
        telemetry["state"] = state.tolist()

        metadata = obs.get("metadata", {})
        telemetry["image_shapes"] = metadata.get("image_shapes", {})
        telemetry["camera_sources"] = metadata.get("camera_sources", {})
        telemetry["camera_timestamps"] = metadata.get("camera_timestamps", {})

        if "max_state_time_diff" in metadata:
            telemetry["max_state_time_diff"] = float(metadata["max_state_time_diff"])

        images = {}
        for name, img in obs.get("images", {}).items():
            try:
                images[name] = encode_rgb_image_to_base64_jpeg(
                    img,
                    quality=jpeg_quality,
                    max_width=max_preview_width,
                )
            except Exception as exc:
                logger.warning("Failed to encode monitor image %s: %s", name, exc)

        telemetry["images"] = images

    if actions is not None:
        arr = np.asarray(actions, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        telemetry["actions"] = arr.tolist()

    if result is not None:
        if "server_timing" in result:
            telemetry["server_timing"] = dict(result["server_timing"])
        elif "raw_response" in result and isinstance(result["raw_response"], dict):
            telemetry["server_timing"] = result["raw_response"].get("server_timing", {})

        if "action_type" in result:
            telemetry["action_type"] = result["action_type"]

    if extra:
        telemetry.update(json_safe(extra))

    return telemetry


# ---------------------------------------------------------------------
# Monitor server
# ---------------------------------------------------------------------

class MonitorServer:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9000,
        path: str = "/monitor",
        broadcast_hz: float = 5.0,
        jpeg_quality: int = 60,
        max_preview_width: int = 640,
        command_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.host = host
        self.port = int(port)
        self.path = path
        self.broadcast_hz = float(broadcast_hz)
        self.jpeg_quality = int(jpeg_quality)
        self.max_preview_width = int(max_preview_width)
        self.command_callback = command_callback

        self._clients = set()
        self._clients_lock = asyncio.Lock()

        self._latest_telemetry: Dict[str, Any] = {
            "type": "telemetry",
            "timestamp": time.time(),
            "camera_mode": "-",
            "protocol": "-",
            "policy_connected": None,
            "state": [],
            "actions": [],
            "server_timing": {},
            "images": {},
            "image_shapes": {},
        }

        self._telemetry_lock = threading.Lock()
        self._commands = deque(maxlen=100)
        self._commands_lock = threading.Lock()

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._server = None

    # ------------------------------------------------------------------
    # Public API for ros2_node.py
    # ------------------------------------------------------------------

    def start_in_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logger.info("MonitorServer already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop_in_thread,
            name="monitor_server_thread",
            daemon=True,
        )
        self._thread.start()
        logger.info("MonitorServer thread started.")

    def stop(self) -> None:
        self._stop_event.set()

        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        logger.info("MonitorServer stopped.")

    def update(self, telemetry: Dict[str, Any]) -> None:
        """
        Thread-safe telemetry update.
        """
        telemetry = dict(telemetry)
        telemetry.setdefault("type", "telemetry")
        telemetry.setdefault("timestamp", time.time())

        with self._telemetry_lock:
            self._latest_telemetry = json_safe(telemetry)

    def update_from_observation(
        self,
        obs: Optional[Dict[str, Any]],
        actions: Optional[np.ndarray | Sequence[Sequence[float]]] = None,
        result: Optional[Dict[str, Any]] = None,
        camera_mode: str = "-",
        protocol: str = "-",
        policy_connected: Optional[bool] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        telemetry = build_telemetry_from_observation(
            obs=obs,
            actions=actions,
            result=result,
            camera_mode=camera_mode,
            protocol=protocol,
            policy_connected=policy_connected,
            jpeg_quality=self.jpeg_quality,
            max_preview_width=self.max_preview_width,
            extra=extra,
        )
        self.update(telemetry)

    def get_latest_command(self) -> Optional[Dict[str, Any]]:
        """
        Pop one command from website.
        """
        with self._commands_lock:
            if not self._commands:
                return None
            return self._commands.popleft()

    def get_all_commands(self) -> list[Dict[str, Any]]:
        """
        Pop all queued commands.
        """
        with self._commands_lock:
            cmds = list(self._commands)
            self._commands.clear()
            return cmds

    # ------------------------------------------------------------------
    # Async internals
    # ------------------------------------------------------------------

    def _run_loop_in_thread(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self.run_async())
        except RuntimeError as exc:
            # Expected when stopping loop.
            logger.info("Monitor loop stopped: %s", exc)
        except Exception as exc:
            logger.exception("MonitorServer error: %s", exc)
        finally:
            try:
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except Exception:
                pass

            self._loop.close()

    async def run_async(self) -> None:
        import websockets

        logger.info(
            "Starting monitor server at ws://%s:%d%s",
            self.host,
            self.port,
            self.path,
        )

        async with websockets.serve(
            self._handler,
            self.host,
            self.port,
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
        ):
            broadcaster = asyncio.create_task(self._broadcast_loop())

            while not self._stop_event.is_set():
                await asyncio.sleep(0.1)

            broadcaster.cancel()

    async def _handler(self, websocket, *args):
        """
        Compatible with different websockets versions.
        """
        req_path = self._get_request_path(websocket, args)

        if req_path not in (self.path, "/", None):
            logger.warning("Client connected with unexpected path: %s", req_path)

        peer = getattr(websocket, "remote_address", None)
        logger.info("Monitor client connected: %s path=%s", peer, req_path)

        async with self._clients_lock:
            self._clients.add(websocket)

        try:
            await websocket.send(
                json.dumps(
                    {
                        "type": "monitor_hello",
                        "server": "lingbotvla_monitor_server",
                        "time": time.time(),
                        "path": self.path,
                    },
                    ensure_ascii=False,
                )
            )

            async for raw in websocket:
                await self._handle_client_message(websocket, raw)

        except Exception as exc:
            logger.info("Monitor client disconnected/error: %s", exc)

        finally:
            async with self._clients_lock:
                self._clients.discard(websocket)

    def _get_request_path(self, websocket, args) -> Optional[str]:
        if args:
            return args[0]

        if hasattr(websocket, "path"):
            return websocket.path

        request = getattr(websocket, "request", None)
        if request is not None and hasattr(request, "path"):
            return request.path

        return None

    async def _handle_client_message(self, websocket, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except Exception as exc:
            await websocket.send(
                json.dumps(
                    {
                        "type": "error",
                        "error": f"invalid json: {exc}",
                    },
                    ensure_ascii=False,
                )
            )
            return

        msg_type = msg.get("type", "")

        with self._commands_lock:
            self._commands.append(msg)

        if self.command_callback is not None:
            try:
                self.command_callback(msg)
            except Exception as exc:
                logger.exception("Command callback failed: %s", exc)

        await websocket.send(
            json.dumps(
                {
                    "type": "ack",
                    "command": msg_type,
                    "time": time.time(),
                },
                ensure_ascii=False,
            )
        )

        logger.info("Received monitor command: %s", msg)

    async def _broadcast_loop(self) -> None:
        dt = 1.0 / max(self.broadcast_hz, 1e-6)

        while True:
            await asyncio.sleep(dt)

            with self._telemetry_lock:
                msg = dict(self._latest_telemetry)

            msg.setdefault("type", "telemetry")
            msg["monitor_time"] = time.time()

            raw = json.dumps(json_safe(msg), ensure_ascii=False)

            async with self._clients_lock:
                clients = list(self._clients)

            if not clients:
                continue

            dead = []

            for ws in clients:
                try:
                    await ws.send(raw)
                except Exception:
                    dead.append(ws)

            if dead:
                async with self._clients_lock:
                    for ws in dead:
                        self._clients.discard(ws)


# ---------------------------------------------------------------------
# Demo telemetry
# ---------------------------------------------------------------------

def make_demo_image(h: int, w: int, t: float, offset: int = 0) -> np.ndarray:
    """
    Generate simple RGB demo image.

    Important:
    Use a bounded phase instead of raw time.time(), otherwise the large Unix timestamp
    can overflow small numpy integer types.
    """
    y = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    x = np.linspace(0, 255, w, dtype=np.uint8)[None, :]

    r = np.repeat(x, h, axis=0)
    g = np.repeat(y, w, axis=1)

    phase = int((t * 30.0) % 255)
    b = (
        r.astype(np.int32)
        + g.astype(np.int32)
        + phase
        + int(offset)
    ) % 255
    b = b.astype(np.uint8)

    return np.stack([r, g, b], axis=-1)


def run_demo(args) -> None:
    server = MonitorServer(
        host=args.host,
        port=args.port,
        path=args.path,
        broadcast_hz=args.broadcast_hz,
        jpeg_quality=args.jpeg_quality,
        max_preview_width=args.max_preview_width,
    )
    server.start_in_thread()

    logger.info("Demo monitor server running.")
    logger.info("Open website/index.html and connect to ws://%s:%d%s", args.host, args.port, args.path)

    try:
        step = 0

        while True:
            t = time.time()
            step += 1

            state = np.sin(np.linspace(0, 2 * np.pi, 16) + t).astype(np.float32)
            actions = np.zeros((5, 16), dtype=np.float32)
            actions[:, 7] = 50.0 + 20.0 * np.sin(t)
            actions[:, 15] = 50.0 + 20.0 * np.cos(t)

            obs = {
                "robot_name": "my_robot",
                "instruction": "demo instruction",
                "timestamp": t,
                "state": state,
                "images": {
                    "camera_top": make_demo_image(180, 640, t, 0),
                    "camera_wrist_left": make_demo_image(240, 320, t, 30),
                    "camera_wrist_right": make_demo_image(240, 320, t, 80),
                },
                "metadata": {
                    "frame_count": step,
                    "camera_sources": {
                        "camera_top": "demo",
                        "camera_wrist_left": "demo",
                        "camera_wrist_right": "demo",
                    },
                    "camera_timestamps": {
                        "camera_top": t,
                        "camera_wrist_left": t - 0.01,
                        "camera_wrist_right": t - 0.02,
                    },
                    "image_shapes": {
                        "camera_top": [180, 640, 3],
                        "camera_wrist_left": [240, 320, 3],
                        "camera_wrist_right": [240, 320, 3],
                    },
                    "max_state_time_diff": 0.025,
                },
            }

            result = {
                "action_type": "arm_delta_gripper_absolute",
                "server_timing": {
                    "infer_ms": 120.0 + 10.0 * np.sin(t),
                    "roundtrip_ms": 180.0 + 20.0 * np.cos(t),
                },
            }

            server.update_from_observation(
                obs=obs,
                actions=actions,
                result=result,
                camera_mode="demo",
                protocol="official",
                policy_connected=True,
            )

            # Print commands from website.
            for cmd in server.get_all_commands():
                logger.info("Demo received command: %s", cmd)

            time.sleep(0.2)

    except KeyboardInterrupt:
        logger.info("Demo stopped by user.")

    finally:
        server.stop()


def parse_args():
    parser = argparse.ArgumentParser("LingBot-VLA monitor server")

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--path", type=str, default="/monitor")
    parser.add_argument("--broadcast-hz", type=float, default=5.0)
    parser.add_argument("--jpeg-quality", type=int, default=60)
    parser.add_argument("--max-preview-width", type=int, default=640)
    parser.add_argument("--demo", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.demo:
        run_demo(args)
        return

    server = MonitorServer(
        host=args.host,
        port=args.port,
        path=args.path,
        broadcast_hz=args.broadcast_hz,
        jpeg_quality=args.jpeg_quality,
        max_preview_width=args.max_preview_width,
    )

    server.start_in_thread()

    try:
        logger.info(
            "Monitor server is running at ws://%s:%d%s",
            args.host,
            args.port,
            args.path,
        )
        logger.info("No telemetry source attached. Use --demo to test with fake data.")

        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        pass

    finally:
        server.stop()


if __name__ == "__main__":
    main()
