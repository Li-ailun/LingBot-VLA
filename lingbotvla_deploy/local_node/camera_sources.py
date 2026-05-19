#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import time
import threading
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import numpy as np
import yaml


logger = logging.getLogger("CameraSources")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")


def default_topics_yaml_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "topics.yaml"


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Empty YAML config: {path}")
    return cfg


@dataclass
class CameraFrame:
    name: str
    output_name: str
    image: np.ndarray
    timestamp: float
    frame_index: int
    source: str


class LatestFrameBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame: Optional[CameraFrame] = None

    def update(self, frame: CameraFrame) -> None:
        with self._lock:
            self._frame = frame

    def latest(self) -> Optional[CameraFrame]:
        with self._lock:
            return self._frame


class BaseCamera:
    def __init__(self, name: str, output_name: str):
        self.name = name
        self.output_name = output_name
        self.buffer = LatestFrameBuffer()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started = False
        self._frame_index = 0

    def start(self) -> None:
        if self._started:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, name=f"{self.name}_capture_thread", daemon=True)
        self._thread.start()
        self._started = True
        logger.info("Started camera thread: %s -> %s", self.name, self.output_name)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._started = False
        logger.info("Stopped camera thread: %s", self.name)

    def latest(self) -> Optional[CameraFrame]:
        return self.buffer.latest()

    def _capture_loop(self) -> None:
        raise NotImplementedError


class RealSenseRGBCamera(BaseCamera):
    def __init__(
        self,
        name: str,
        output_name: str,
        serial_number: str,
        dim: Tuple[int, int] = (640, 480),
        fps: int = 15,
        warmup_frames: int = 10,
    ):
        super().__init__(name=name, output_name=output_name)
        self.serial_number = serial_number
        self.width = int(dim[0])
        self.height = int(dim[1])
        self.fps = int(fps)
        self.warmup_frames = int(warmup_frames)
        self._pipeline = None
        self._config = None

    def _import_rs(self):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise ImportError("pyrealsense2 is not installed: pip install pyrealsense2") from exc
        return rs

    def _open(self) -> None:
        rs = self._import_rs()
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_device(self.serial_number)
        self._config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)

        logger.info(
            "[RealSense] opening %s serial=%s size=%dx%d fps=%d output=%s",
            self.name, self.serial_number, self.width, self.height, self.fps, self.output_name,
        )

        self._pipeline.start(self._config)
        for _ in range(self.warmup_frames):
            if self._stop_event.is_set():
                break
            self._pipeline.wait_for_frames(timeout_ms=3000)

    def _close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception as exc:
                logger.warning("[RealSense] failed to stop %s: %s", self.name, exc)

    def _capture_loop(self) -> None:
        try:
            self._open()
            while not self._stop_event.is_set():
                try:
                    frames = self._pipeline.wait_for_frames(timeout_ms=3000)
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue

                    image = np.asanyarray(color_frame.get_data())
                    if image.dtype != np.uint8:
                        image = image.astype(np.uint8)
                    if image.ndim != 3 or image.shape[2] != 3:
                        logger.warning("[RealSense] invalid image shape from %s: %s", self.name, image.shape)
                        continue

                    self._frame_index += 1
                    self.buffer.update(
                        CameraFrame(
                            name=self.name,
                            output_name=self.output_name,
                            image=image.copy(),
                            timestamp=time.time(),
                            frame_index=self._frame_index,
                            source="python_realsense",
                        )
                    )
                except Exception as exc:
                    logger.warning("[RealSense] capture error %s: %s", self.name, exc)
                    time.sleep(0.05)
        finally:
            self._close()


class OpenCVRGBCamera(BaseCamera):
    def __init__(
        self,
        name: str,
        output_name: str,
        device_index: int,
        api: str = "CAP_V4L2",
        fourcc: str = "MJPG",
        frame_width: int = 640,
        frame_height: int = 480,
        fps: int = 15,
        resize_to: Optional[Tuple[int, int]] = None,
        warmup_frames: int = 10,
    ):
        super().__init__(name=name, output_name=output_name)
        self.device_index = int(device_index)
        self.api = api
        self.fourcc = fourcc
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.fps = int(fps)
        self.resize_to = tuple(resize_to) if resize_to is not None else None
        self.warmup_frames = int(warmup_frames)
        self._cv2 = None
        self._cap = None

    def _import_cv2(self):
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("opencv-python is not installed: pip install opencv-python") from exc
        return cv2

    def _api_value(self, cv2_module):
        if isinstance(self.api, int):
            return self.api
        if isinstance(self.api, str):
            return getattr(cv2_module, self.api, cv2_module.CAP_ANY)
        return cv2_module.CAP_ANY

    def _open(self) -> None:
        cv2 = self._import_cv2()
        self._cv2 = cv2

        self._cap = cv2.VideoCapture(self.device_index, self._api_value(cv2))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open OpenCV camera {self.name} at device_index={self.device_index}")

        if self.fourcc:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        logger.info(
            "[OpenCV] opening %s device=%d request_size=%dx%d fps=%d resize_to=%s output=%s",
            self.name, self.device_index, self.frame_width, self.frame_height,
            self.fps, self.resize_to, self.output_name,
        )

        for _ in range(self.warmup_frames):
            if self._stop_event.is_set():
                break
            self._cap.read()

    def _close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as exc:
                logger.warning("[OpenCV] failed to release %s: %s", self.name, exc)

    def _capture_loop(self) -> None:
        try:
            self._open()
            cv2 = self._cv2

            while not self._stop_event.is_set():
                ok, frame_bgr = self._cap.read()
                if not ok or frame_bgr is None:
                    logger.warning("[OpenCV] failed to read frame: %s", self.name)
                    time.sleep(0.05)
                    continue

                if self.resize_to is not None:
                    out_w, out_h = self.resize_to
                    frame_bgr = cv2.resize(frame_bgr, (int(out_w), int(out_h)))

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                if frame_rgb.dtype != np.uint8:
                    frame_rgb = frame_rgb.astype(np.uint8)

                self._frame_index += 1
                self.buffer.update(
                    CameraFrame(
                        name=self.name,
                        output_name=self.output_name,
                        image=frame_rgb.copy(),
                        timestamp=time.time(),
                        frame_index=self._frame_index,
                        source="python_opencv",
                    )
                )
                time.sleep(max(0.0, 0.5 / max(self.fps, 1)))
        finally:
            self._close()


class CameraHub:
    def __init__(self, cameras: Dict[str, BaseCamera]):
        self.cameras = cameras

    @classmethod
    def from_default_config(cls) -> "CameraHub":
        return cls.from_config_file(default_topics_yaml_path())

    @classmethod
    def from_config_file(cls, config_path: str | Path) -> "CameraHub":
        cfg = load_yaml_config(config_path)
        return cls.from_config_dict(cfg)

    @classmethod
    def from_config_dict(cls, cfg: Dict[str, Any]) -> "CameraHub":
        cameras: Dict[str, BaseCamera] = {}

        if "cameras" in cfg:
            for logical_name, cam_cfg in cfg.get("cameras", {}).items():
                source = str(cam_cfg.get("source", "")).lower()
                if source != "python":
                    continue

                output_name = cam_cfg.get("output_name", logical_name)
                internal_name = cam_cfg.get("internal_name", output_name)
                py_cfg = cam_cfg.get("python", {})
                cam_type = str(py_cfg.get("type", "")).lower()

                if cam_type == "realsense":
                    cameras[internal_name] = RealSenseRGBCamera(
                        name=internal_name,
                        output_name=output_name,
                        serial_number=str(py_cfg["serial_number"]),
                        dim=(int(py_cfg.get("width", 640)), int(py_cfg.get("height", 480))),
                        fps=int(py_cfg.get("fps", 15)),
                    )
                elif cam_type == "opencv":
                    resize_to = py_cfg.get("resize_to", None)
                    cameras[internal_name] = OpenCVRGBCamera(
                        name=internal_name,
                        output_name=output_name,
                        device_index=int(py_cfg["device_index"]),
                        api=py_cfg.get("api", "CAP_V4L2"),
                        fourcc=py_cfg.get("fourcc", "MJPG"),
                        frame_width=int(py_cfg.get("width", 640)),
                        frame_height=int(py_cfg.get("height", 480)),
                        fps=int(py_cfg.get("fps", 15)),
                        resize_to=tuple(resize_to) if resize_to is not None else None,
                    )
                else:
                    raise ValueError(f"Unknown python camera type for {logical_name}: {cam_type}")

        elif "python_cameras" in cfg:
            for name, c in cfg["python_cameras"].items():
                cam_type = str(c.get("type", "")).lower()
                output_name = {
                    "head_rgb": "camera_top",
                    "left_wrist_rgb": "camera_wrist_left",
                    "right_wrist_rgb": "camera_wrist_right",
                }.get(name, name)

                if cam_type == "realsense":
                    cameras[name] = RealSenseRGBCamera(
                        name=name,
                        output_name=output_name,
                        serial_number=str(c["serial_number"]),
                        dim=(int(c.get("width", 640)), int(c.get("height", 480))),
                        fps=int(c.get("fps", 15)),
                    )
                elif cam_type == "opencv":
                    resize_to = c.get("resize_to", None)
                    cameras[name] = OpenCVRGBCamera(
                        name=name,
                        output_name=output_name,
                        device_index=int(c["device_index"]),
                        api=c.get("api", "CAP_V4L2"),
                        fourcc=c.get("fourcc", "MJPG"),
                        frame_width=int(c.get("width", 640)),
                        frame_height=int(c.get("height", 480)),
                        fps=int(c.get("fps", 15)),
                        resize_to=tuple(resize_to) if resize_to is not None else None,
                    )

        if not cameras:
            logger.info("No python cameras enabled.")
        return cls(cameras=cameras)

    def start(self) -> None:
        for cam in self.cameras.values():
            cam.start()

    def stop(self) -> None:
        for cam in self.cameras.values():
            cam.stop()

    def wait_until_ready(self, timeout_s: float = 10.0, required: Optional[Tuple[str, ...]] = None) -> bool:
        if not self.cameras:
            return True

        required = required or tuple(self.cameras.keys())
        start = time.time()

        while time.time() - start < timeout_s:
            ok = True
            for name in required:
                cam = self.cameras.get(name)
                if cam is None or cam.latest() is None:
                    ok = False
                    break
            if ok:
                return True
            time.sleep(0.05)

        missing = [name for name in required if name not in self.cameras or self.cameras[name].latest() is None]
        logger.warning("CameraHub not ready. Missing: %s", missing)
        return False

    def get_lingbot_images(self, require_all: bool = False) -> Optional[Dict[str, Any]]:
        images: Dict[str, np.ndarray] = {}
        timestamps: Dict[str, float] = {}
        frame_indices: Dict[str, int] = {}

        for _, cam in self.cameras.items():
            frame = cam.latest()
            if frame is None:
                if require_all:
                    return None
                continue

            images[frame.output_name] = frame.image
            timestamps[frame.output_name] = frame.timestamp
            frame_indices[frame.output_name] = frame.frame_index

        if not images:
            return None

        reference_time = timestamps.get("camera_top", max(timestamps.values()))
        return {
            "images": images,
            "timestamps": timestamps,
            "frame_indices": frame_indices,
            "reference_time": reference_time,
            "source": "python",
        }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(default_topics_yaml_path()))
    args = parser.parse_args()

    hub = CameraHub.from_config_file(args.config)
    hub.start()

    try:
        print("camera ready:", hub.wait_until_ready(timeout_s=15.0))
        while True:
            data = hub.get_lingbot_images(require_all=False)
            print("data:", None if data is None else {k: v.shape for k, v in data["images"].items()})
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        hub.stop()


if __name__ == "__main__":
    main()
