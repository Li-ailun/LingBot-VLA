#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Local LingBot-VLA ROS2 node.

Camera mode:
- pure python cameras
- pure ROS cameras
- mixed camera mode is intentionally not supported

YAML controls camera source:

cameras:
  camera_top:
    source: "python" or "ros"
  camera_wrist_left:
    source: "python" or "ros"
  camera_wrist_right:
    source: "python" or "ros"

Safety:
- Default mode does NOT send to server.
- Default mode does NOT execute robot commands.
- Use --send to contact policy server.
- Use --execute to publish robot control commands.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml


try:
    from .camera_sources import CameraHub
    from .ros2_bridge import Ros2Bridge
    from .observation_builder import ObservationBuilder
    from .client_ws import PolicyClient
    from .official_policy_client import OfficialPolicyClient
    from .action_executor import ActionExecutor
    from .monitor_server import MonitorServer
except ImportError:
    from camera_sources import CameraHub
    from ros2_bridge import Ros2Bridge
    from observation_builder import ObservationBuilder
    from client_ws import PolicyClient
    from official_policy_client import OfficialPolicyClient
    from action_executor import ActionExecutor
    from monitor_server import MonitorServer


logger = logging.getLogger("LingBotVLALocalNode")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "topics.yaml"


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Empty config file: {path}")

    return cfg


def dof_from_hardware(hardware: str) -> int:
    if hardware == "R1_PRO":
        return 7
    if hardware == "R1_LITE":
        return 6
    raise ValueError(f"Unknown hardware: {hardware}")


def get_camera_mode(cfg: Dict[str, Any]) -> str:
    """
    Return:
        "python" or "ros"

    Reject mixed camera source mode.
    """
    if "python_cameras" in cfg and cfg["python_cameras"]:
        return "python"

    cameras = cfg.get("cameras", {})

    if not cameras:
        raise ValueError("No cameras configured in YAML.")

    sources = []

    for name, cam_cfg in cameras.items():
        source = str(cam_cfg.get("source", "")).lower().strip()

        if source not in ("python", "ros"):
            raise ValueError(
                f"Invalid camera source for {name}: {source}. "
                "Expected 'python' or 'ros'."
            )

        sources.append(source)

    unique_sources = sorted(set(sources))

    if len(unique_sources) != 1:
        raise RuntimeError(
            f"Mixed camera sources are not supported: {unique_sources}. "
            "Please set all cameras to source: 'python' or all cameras to source: 'ros'."
        )

    return unique_sources[0]


def override_if_not_none(value, default):
    return default if value is None else value


def build_runtime_config(args, yaml_cfg: Dict[str, Any]) -> Dict[str, Any]:
    robot_yaml = yaml_cfg.get("robot", {})
    safety_yaml = yaml_cfg.get("safety", {})
    sync_yaml = yaml_cfg.get("sync", {})

    hardware = override_if_not_none(
        args.hardware,
        robot_yaml.get("hardware", "R1_PRO"),
    )

    dof = int(
        override_if_not_none(
            args.dof_of_arm,
            robot_yaml.get("dof_of_arm", dof_from_hardware(hardware)),
        )
    )

    action_dim = int(
        override_if_not_none(
            args.action_dim,
            robot_yaml.get("action_dim", dof * 2 + 2),
        )
    )

    state_dim = int(
        override_if_not_none(
            args.state_dim,
            robot_yaml.get("state_dim", action_dim),
        )
    )

    runtime = {
        "config_path": str(Path(args.config).expanduser().resolve()),
        "server_url": args.server_url,
        "instruction": args.instruction,
        "robot_name": args.robot_name,

        "sync": {
            "time_source": override_if_not_none(
                args.time_source,
                sync_yaml.get("time_source", "receive_time"),
            ),
            "reference_camera": override_if_not_none(
                args.reference_camera,
                sync_yaml.get("reference_camera", "camera_top"),
            ),
            "max_state_time_diff_s": float(
                override_if_not_none(
                    args.max_state_time_diff,
                    sync_yaml.get("max_state_time_diff_s", 0.15),
                )
            ),
        },

        "robot": {
            "hardware": hardware,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "dof_of_arm": dof,
            "action_type": override_if_not_none(
                args.action_type,
                robot_yaml.get("action_type", "arm_delta_gripper_absolute"),
            ),
            "control_frequency": float(
                override_if_not_none(
                    args.control_frequency,
                    robot_yaml.get("control_frequency", 15.0),
                )
            ),
            "action_steps": int(
                override_if_not_none(
                    args.action_steps,
                    robot_yaml.get("action_steps", 5),
                )
            ),
        },

        "safety": {
            "enable": bool(safety_yaml.get("enable", True)),
            "max_joint_delta_per_step": float(
                override_if_not_none(
                    args.max_joint_delta,
                    safety_yaml.get("max_joint_delta_per_step", 0.02),
                )
            ),
            "max_gripper_delta_per_step": float(
                override_if_not_none(
                    args.max_gripper_delta,
                    safety_yaml.get("max_gripper_delta_per_step", 5.0),
                )
            ),
            "low_pass_alpha": float(
                override_if_not_none(
                    args.low_pass_alpha,
                    safety_yaml.get("low_pass_alpha", 0.3),
                )
            ),
            "gripper_deadband": float(
                override_if_not_none(
                    args.gripper_deadband,
                    safety_yaml.get("gripper_deadband", 0.5),
                )
            ),
            "gripper_min_interval_s": float(
                override_if_not_none(
                    args.gripper_min_interval,
                    safety_yaml.get("gripper_min_interval_s", 0.0),
                )
            ),
            "gripper_lower": float(
                override_if_not_none(
                    args.gripper_lower,
                    safety_yaml.get("gripper_lower", 0.0),
                )
            ),
            "gripper_upper": float(
                override_if_not_none(
                    args.gripper_upper,
                    safety_yaml.get("gripper_upper", 100.0),
                )
            ),
        },
    }

    if runtime["robot"]["action_type"] == "delta_qpos":
        runtime["robot"]["action_type"] = "arm_delta_gripper_absolute"

    return runtime


def make_ros_config(runtime_cfg: Dict[str, Any], execute: bool) -> Dict[str, Any]:
    if execute:
        enable_publish = [
            "left_arm",
            "right_arm",
            "left_gripper",
            "right_gripper",
        ]
    else:
        enable_publish = []

    return {
        "robot": {
            "hardware": runtime_cfg["robot"]["hardware"],
            "enable_publish": enable_publish,
        },
        "config_path": runtime_cfg["config_path"],
    }


def make_executor_config(runtime_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "robot": dict(runtime_cfg["robot"]),
        "safety": dict(runtime_cfg["safety"]),
    }


def summarize_obs(obs: Dict[str, Any]) -> str:
    lines = []

    lines.append(f"robot_name: {obs.get('robot_name')}")
    lines.append(f"instruction: {obs.get('instruction')}")
    lines.append(f"timestamp: {obs.get('timestamp')}")
    lines.append(f"state shape: {obs['state'].shape}, dtype={obs['state'].dtype}")

    state = np.asarray(obs["state"], dtype=np.float32)
    lines.append(f"state first values: {np.round(state[: min(8, len(state))], 4)}")

    lines.append("images:")
    for name, img in obs["images"].items():
        source = obs.get("metadata", {}).get("camera_sources", {}).get(name, "?")
        lines.append(
            f"  {name}: shape={img.shape}, dtype={img.dtype}, source={source}"
        )

    meta = obs.get("metadata", {})

    if "max_state_time_diff" in meta:
        lines.append(f"max_state_time_diff: {meta['max_state_time_diff']:.4f}s")

    if "debug_state_keys" in meta:
        lines.append(f"debug_state_keys: {meta['debug_state_keys']}")

    return "\n".join(lines)


def summarize_action_result(result: Dict[str, Any]) -> str:
    actions = result["actions"]

    lines = []
    lines.append(f"request_id: {result.get('request_id')}")
    lines.append(f"action_type: {result.get('action_type')}")
    lines.append(f"actions shape: {actions.shape}")
    lines.append(
        f"first action first values: {np.round(actions[0, : min(8, actions.shape[1])], 4)}"
    )

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Local LingBot-VLA ROS2 node"
    )

    parser.add_argument("--config", type=str, default=str(default_config_path()))
    parser.add_argument("--server-url", type=str, default="ws://127.0.0.1:8000/ws")

    parser.add_argument(
        "--protocol",
        type=str,
        default="json",
        choices=["json", "official"],
        help="json: current JSON/base64 protocol. official: LingBot official msgpack protocol.",
    )
    parser.add_argument("--instruction", type=str, default="do the task")
    parser.add_argument("--robot-name", type=str, default="my_robot")

    parser.add_argument("--hardware", type=str, default=None, choices=["R1_PRO", "R1_LITE"])
    parser.add_argument("--state-dim", type=int, default=None)
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument("--dof-of-arm", type=int, default=None)

    parser.add_argument("--send", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--policy-hz", type=float, default=3.0)

    parser.add_argument("--control-frequency", type=float, default=None)
    parser.add_argument("--action-steps", type=int, default=None)

    parser.add_argument(
        "--action-type",
        type=str,
        default=None,
        choices=[
            "arm_delta_gripper_absolute",
            "absolute_qpos",
            "all_delta_qpos",
            "delta_qpos",
        ],
    )

    parser.add_argument("--max-joint-delta", type=float, default=None)
    parser.add_argument("--max-gripper-delta", type=float, default=None)
    parser.add_argument("--low-pass-alpha", type=float, default=None)
    parser.add_argument("--gripper-deadband", type=float, default=None)
    parser.add_argument("--gripper-min-interval", type=float, default=None)
    parser.add_argument("--gripper-lower", type=float, default=None)
    parser.add_argument("--gripper-upper", type=float, default=None)

    parser.add_argument(
        "--time-source",
        type=str,
        default=None,
        choices=["receive_time", "header_stamp"],
    )
    parser.add_argument("--reference-camera", type=str, default=None)
    parser.add_argument("--max-state-time-diff", type=float, default=None)

    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--stop-on-error", action="store_true")

    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--monitor-host", type=str, default="127.0.0.1")
    parser.add_argument("--monitor-port", type=int, default=9000)
    parser.add_argument("--monitor-hz", type=float, default=5.0)

    return parser.parse_args()


def main():
    args = parse_args()

    yaml_cfg = load_yaml_config(args.config)
    camera_mode = get_camera_mode(yaml_cfg)
    runtime_cfg = build_runtime_config(args, yaml_cfg)

    robot_cfg = runtime_cfg["robot"]
    safety_cfg = runtime_cfg["safety"]
    sync_cfg = runtime_cfg["sync"]

    expected_dim = int(robot_cfg["action_dim"])
    use_recv_time = sync_cfg["time_source"] == "receive_time"

    logger.info("Starting local LingBot-VLA node.")
    logger.info("config=%s", runtime_cfg["config_path"])
    logger.info("camera_mode=%s", camera_mode)
    logger.info("robot=%s", robot_cfg)
    logger.info("safety=%s", safety_cfg)
    logger.info("sync=%s", sync_cfg)
    logger.info("send=%s execute=%s", args.send, args.execute)

    if args.execute and not args.send:
        raise RuntimeError("--execute requires --send.")

    if not args.execute:
        logger.warning("Robot execution is DISABLED. This node will not publish robot control commands.")

    if not args.send:
        logger.warning("Policy server sending is DISABLED. This node will only build and print observations.")

    camera_hub: Optional[CameraHub] = None
    ros_bridge: Optional[Ros2Bridge] = None
    builder: Optional[ObservationBuilder] = None
    client: Optional[PolicyClient] = None
    executor: Optional[ActionExecutor] = None
    monitor: Optional[MonitorServer] = None

    try:
        if camera_mode == "python":
            camera_hub = CameraHub.from_config_file(args.config)
            camera_hub.start()

            camera_ready = camera_hub.wait_until_ready(timeout_s=15.0)
            logger.info("python_camera_ready=%s", camera_ready)

            if not camera_ready:
                raise RuntimeError("Python CameraHub is not ready.")

        else:
            camera_hub = None
            logger.info("Pure ROS camera mode. Python CameraHub is disabled.")

        ros_bridge = Ros2Bridge(
            config=make_ros_config(runtime_cfg, args.execute),
            use_recv_time=use_recv_time,
        )

        ros_ready = ros_bridge.wait_until_ready(timeout_s=15.0)
        logger.info("ros_state_ready=%s", ros_ready)

        if not ros_ready:
            raise RuntimeError(
                "ROS2 feedback is not ready. Check /hdas/feedback_arm_left, "
                "/hdas/feedback_arm_right, /hdas/feedback_gripper_left, "
                "/hdas/feedback_gripper_right."
            )

        if camera_mode == "ros":
            ros_img_ready = ros_bridge.wait_until_images_ready(
                timeout_s=15.0,
                require_all=True,
            )

            logger.info("ros_camera_ready=%s", ros_img_ready)

            if not ros_img_ready:
                raise RuntimeError(
                    "ROS image topics are not ready. Check compressed image topics in YAML."
                )

        builder = ObservationBuilder(
            camera_hub=camera_hub,
            ros_bridge=ros_bridge,
            robot_name=args.robot_name,
            instruction=args.instruction,
            require_all_cameras=True,
            use_camera_time_for_state=True,
            allow_latest_state_fallback=True,
            expected_state_dim=int(robot_cfg["state_dim"]),
            include_debug_state=True,
            reference_camera=sync_cfg.get("reference_camera", "camera_top"),
            max_state_time_diff_s=sync_cfg.get("max_state_time_diff_s", 0.15),
        )

        if args.send:
            if args.protocol == "official":
                client = OfficialPolicyClient(
                    server_url=args.server_url,
                    dof_of_arm=int(robot_cfg["dof_of_arm"]),
                    return_action_type=robot_cfg.get("action_type", "arm_delta_gripper_absolute"),
                )
            else:
                client = PolicyClient(
                    server_url=args.server_url,
                    timeout_s=30.0,
                    jpeg_quality=args.jpeg_quality,
                    expected_action_dim=expected_dim,
                    verbose=False,
                )

        executor = ActionExecutor(
            config=make_executor_config(runtime_cfg),
            ros_bridge=ros_bridge,
        )

        if args.monitor:
            monitor = MonitorServer(
                host=args.monitor_host,
                port=args.monitor_port,
                broadcast_hz=args.monitor_hz,
                jpeg_quality=60,
                max_preview_width=640,
            )
            monitor.start_in_thread()
            logger.info(
                "Monitor server started at ws://%s:%d/monitor",
                args.monitor_host,
                args.monitor_port,
            )

        logger.info("Local node is running.")

        if args.send:
            logger.info("Policy server URL: %s", args.server_url)

        step = 0
        loop_dt = 1.0 / max(args.policy_hz, 1e-6)

        while True:
            tic = time.time()

            obs = builder.build()

            if obs is None:
                time.sleep(0.01)
                continue

            step += 1

            if args.print_every > 0 and step % args.print_every == 0:
                print("=" * 80)
                print(f"valid observation step: {step}")
                print(summarize_obs(obs))

            if monitor is not None:
                for cmd in monitor.get_all_commands():
                    cmd_type = cmd.get("type", "")

                    if cmd_type == "set_instruction":
                        new_instruction = cmd.get("instruction", "")
                        builder.set_instruction(new_instruction)
                        logger.info("Instruction updated from monitor: %s", new_instruction)

                    elif cmd_type in ("stop", "emergency_stop"):
                        logger.warning("Monitor command received: %s. Disable execute.", cmd_type)
                        args.execute = False

                    elif cmd_type == "start_dry_run":
                        logger.info("Monitor command received: start_dry_run")

                if not args.send:
                    monitor.update_from_observation(
                        obs=obs,
                        actions=None,
                        result=None,
                        camera_mode=camera_mode,
                        protocol=args.protocol,
                        policy_connected=False,
                        extra={"step": step},
                    )

            if not args.send:
                if args.max_steps > 0 and step >= args.max_steps:
                    break

                elapsed = time.time() - tic
                time.sleep(max(0.0, loop_dt - elapsed))
                continue

            try:
                assert client is not None
                assert executor is not None

                result = client.infer(obs)

                if args.print_every > 0 and step % args.print_every == 0:
                    print("-" * 80)
                    print(summarize_action_result(result))

                actions = result["actions"]

                if monitor is not None:
                    monitor.update_from_observation(
                        obs=obs,
                        actions=actions,
                        result=result,
                        camera_mode=camera_mode,
                        protocol=args.protocol,
                        policy_connected=True,
                        extra={"step": step},
                    )

                returned_action_type = result.get("action_type", None)
                if returned_action_type is not None:
                    executor.action_type = returned_action_type

                    if executor.action_type == "delta_qpos":
                        executor.action_type = "arm_delta_gripper_absolute"

                if args.execute:
                    executor.execute_action_chunk(
                        action_chunk=actions,
                        current_state=obs["state"],
                        sleep=True,
                    )

                else:
                    processed = executor.process_action_chunk(
                        current_state=obs["state"],
                        action_chunk=actions,
                    )

                    if args.print_every > 0 and step % args.print_every == 0:
                        print("processed action chunk:", processed.shape)
                        print("first processed target:")
                        print(executor.summarize_action(processed[0]))

            except Exception as exc:
                logger.exception("Policy inference / action handling failed: %s", exc)

                if args.stop_on_error:
                    raise

                time.sleep(0.5)

            if args.max_steps > 0 and step >= args.max_steps:
                break

            if not args.execute:
                elapsed = time.time() - tic
                time.sleep(max(0.0, loop_dt - elapsed))

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Stopping local node.")

    finally:
        if monitor is not None:
            monitor.stop()

        if camera_hub is not None:
            camera_hub.stop()

        if ros_bridge is not None:
            ros_bridge.destroy()

        logger.info("Local LingBot-VLA node stopped.")


if __name__ == "__main__":
    main()
