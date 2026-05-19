#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Remote LingBot-VLA policy server.

Protocol:
- JSON over WebSocket
- Compatible with local_node/client_ws.py

Request:
{
    "type": "infer",
    "request_id": "...",
    "robot_name": "my_robot",
    "instruction": "...",
    "state": [...],
    "images": {
        "camera_top": {"encoding": "jpeg_base64", "data": "..."},
        "camera_wrist_left": {...},
        "camera_wrist_right": {...}
    }
}

Response:
{
    "type": "action_chunk",
    "request_id": "...",
    "action_type": "arm_delta_gripper_absolute",
    "actions": [[...], ...],
    "server_timing": {...}
}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import socket
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
import websockets


try:
    from .model_runner import LingBotVLAModelRunner, DummyModelRunner
except ImportError:
    from model_runner import LingBotVLAModelRunner, DummyModelRunner


logger = logging.getLogger("LingBotPolicyServer")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)


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


def build_error_response(request_id: str | None, exc: BaseException) -> Dict[str, Any]:
    return {
        "type": "error",
        "request_id": request_id,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


class LingBotPolicyServer:
    def __init__(
        self,
        runner,
        host: str = "0.0.0.0",
        port: int = 8000,
        max_size: int | None = None,
    ):
        self.runner = runner
        self.host = host
        self.port = int(port)
        self.max_size = max_size

    async def handle_client(self, websocket):
        peer = websocket.remote_address
        logger.info("Client connected: %s", peer)

        try:
            async for raw_msg in websocket:
                request_id = None
                t_start = time.time()

                try:
                    request = json.loads(raw_msg)
                    request_id = request.get("request_id", None)
                    msg_type = request.get("type", "infer")

                    if msg_type == "ping":
                        response = {
                            "type": "pong",
                            "request_id": request_id,
                            "time": time.time(),
                        }

                    elif msg_type == "reset":
                        robot_name = request.get("robot_name", None)
                        if hasattr(self.runner, "reset"):
                            self.runner.reset(robot_name)
                        response = {
                            "type": "reset_ok",
                            "request_id": request_id,
                            "robot_name": robot_name,
                        }

                    elif msg_type == "infer":
                        t_decode = time.time()

                        result = self.runner.infer(request)

                        t_end = time.time()

                        response = {
                            "type": "action_chunk",
                            "request_id": request_id,
                            "action_type": result.get(
                                "action_type",
                                "arm_delta_gripper_absolute",
                            ),
                            "actions": result["actions"],
                            "server_timing": {
                                "total_ms": (t_end - t_start) * 1000.0,
                                "decode_to_infer_ms": (t_end - t_decode) * 1000.0,
                            },
                            "extra": result.get("extra", {}),
                        }

                    else:
                        raise ValueError(f"Unsupported message type: {msg_type}")

                    await websocket.send(
                        json.dumps(json_safe(response), ensure_ascii=False)
                    )

                except Exception as exc:
                    logger.exception("Request failed: %s", exc)

                    await websocket.send(
                        json.dumps(
                            json_safe(build_error_response(request_id, exc)),
                            ensure_ascii=False,
                        )
                    )

        except websockets.ConnectionClosed:
            logger.info("Client disconnected: %s", peer)

    async def run_async(self):
        logger.info("Starting LingBot policy server at ws://%s:%d/ws", self.host, self.port)

        async with websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            max_size=self.max_size,
            ping_interval=20,
            ping_timeout=20,
        ):
            await asyncio.Future()

    def serve_forever(self):
        asyncio.run(self.run_async())


def parse_args():
    parser = argparse.ArgumentParser("LingBot-VLA JSON WebSocket policy server")

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)

    parser.add_argument(
        "--repo-dir",
        type=str,
        default="/home/hddData/User/lixiang/lingbotvla_workspace/lingbot-vla",
        help="Official LingBot-VLA repo directory",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=os.environ.get(
            "LINGBOTVLA_MODEL_PATH",
            "/home/hddData/User/lixiang/lingbotvla_workspace/models/lingbot-vla-4b",
        ),
    )

    parser.add_argument(
        "--qwen25-path",
        type=str,
        default=os.environ.get(
            "QWEN25_PATH",
            "/home/hddData/User/lixiang/lingbotvla_workspace/models/Qwen2.5-VL-3B-Instruct",
        ),
    )

    parser.add_argument("--robot-name", type=str, default="my_robot")
    parser.add_argument("--robot-config", type=str, default=None)

    parser.add_argument("--norm-path", type=str, default=None)
    parser.add_argument("--use-length", type=int, default=25)
    parser.add_argument("--num-denoising-step", type=int, default=10)
    parser.add_argument("--use-compile", action="store_true")

    parser.add_argument("--use-bf16", action="store_true", default=True)
    parser.add_argument("--use-fp32", action="store_true")

    parser.add_argument("--dof-of-arm", type=int, default=7)
    parser.add_argument("--action-dim", type=int, default=16)

    parser.add_argument(
        "--return-action-type",
        type=str,
        default="arm_delta_gripper_absolute",
        choices=["arm_delta_gripper_absolute", "absolute_qpos"],
        help=(
            "arm_delta_gripper_absolute: server converts official absolute arm target "
            "to arm delta, gripper remains absolute. This matches local ActionExecutor."
        ),
    )

    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Do not load LingBot-VLA. Return zero actions for communication test.",
    )

    parser.add_argument(
        "--dummy-on-error",
        action="store_true",
        help="If real model load fails, fall back to dummy runner.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.dummy:
        logger.warning("Using DummyModelRunner. No real LingBot-VLA inference.")
        runner = DummyModelRunner(
            action_dim=args.action_dim,
            dof_of_arm=args.dof_of_arm,
            use_length=args.use_length,
            action_type=args.return_action_type,
        )

    else:
        try:
            runner = LingBotVLAModelRunner(
                repo_dir=args.repo_dir,
                model_path=args.model_path,
                qwen25_path=args.qwen25_path,
                robot_name=args.robot_name,
                robot_config_path=args.robot_config,
                norm_path=args.norm_path,
                use_length=args.use_length,
                num_denoising_step=args.num_denoising_step,
                use_compile=args.use_compile,
                use_bf16=args.use_bf16,
                use_fp32=args.use_fp32,
                dof_of_arm=args.dof_of_arm,
                return_action_type=args.return_action_type,
            )
        except Exception:
            logger.exception("Failed to load real LingBot-VLA model.")

            if not args.dummy_on_error:
                raise

            logger.warning("Falling back to DummyModelRunner because --dummy-on-error is set.")
            runner = DummyModelRunner(
                action_dim=args.action_dim,
                dof_of_arm=args.dof_of_arm,
                use_length=args.use_length,
                action_type=args.return_action_type,
            )

    server = LingBotPolicyServer(
        runner=runner,
        host=args.host,
        port=args.port,
        max_size=None,
    )

    server.serve_forever()


if __name__ == "__main__":
    main()
