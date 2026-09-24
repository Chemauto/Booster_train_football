"""RGB-D + YOLOv8s ball perception -> policy ball_obs (x, y, flag) in yaw frame.

Pipeline (matches the training observation contract, commands.py ball_obs):

    RGB + depth + intrinsics + camera pose
        -> YOLOv8s (COCO default weights, class "sports ball")
        -> bbox depth median -> pinhole deproject -> camera 3D
        -> world -> robot yaw frame
        -> (x, y, flag); undetected => (0, 0, 0) exactly like training

Sources
    MujocoRgbdSource : render from the MuJoCo head_cam (sim2sim closed loop)
    Ros2RgbdSource   : subscribe to RealSense-style ROS2 topics (real robot)
    Ros2RgbdPublisher: republish sim RGB-D on ROS2 (debug / external consumers)

Frames
    OpenCV optical : x right, y down, z forward   (deproject output)
    MuJoCo camera  : x right, y up,   z backward  (cam_xmat columns)
    p_cv = p_mj * (1, -1, -1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# COCO default yolov8s class name for a soccer ball
BALL_CLASS_NAMES = {"sports ball", "soccer ball", "football", "ball"}


def intrinsics_from_fovy(fovy_deg: float, width: int, height: int) -> dict:
    f = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    return {"fx": f, "fy": f, "cx": width / 2.0, "cy": height / 2.0}


def bbox_depth_median(depth: np.ndarray, xyxy, max_range: float = 20.0) -> float | None:
    h, w = depth.shape
    x1, y1 = max(0, int(round(xyxy[0]))), max(0, int(round(xyxy[1])))
    x2, y2 = min(w, int(round(xyxy[2]))), min(h, int(round(xyxy[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    roi = depth[y1:y2, x1:x2]
    roi = roi[np.isfinite(roi) & (roi > 0) & (roi < max_range)]
    if roi.size == 0:
        return None
    return float(np.median(roi))


def deproject(u: float, v: float, z: float, K: dict) -> np.ndarray:
    """Pixel + depth -> OpenCV camera-frame 3D (x right, y down, z fwd)."""
    return np.array([(u - K["cx"]) * z / K["fx"], (v - K["cy"]) * z / K["fy"], z])


@dataclass
class RgbdFrame:
    rgb: np.ndarray          # (H, W, 3) uint8
    depth: np.ndarray        # (H, W) float meters
    K: dict                  # fx, fy, cx, cy
    cam_pos_w: np.ndarray    # (3,) camera origin in world
    cam_R_mj: np.ndarray     # (3, 3) MuJoCo camera axes in world (cols = cam x,y,z)


class RgbdSource:
    def grab(self) -> RgbdFrame | None:
        raise NotImplementedError


class MujocoRgbdSource(RgbdSource):
    """Offscreen RGB-D from a MuJoCo camera attached to the robot head."""

    def __init__(self, model, data, camera_name: str = "head_cam",
                 width: int = 640, height: int = 360):
        import mujoco

        self.model, self.data = model, data
        self.width, self.height = width, height
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.cam_id < 0:
            raise KeyError(f"camera '{camera_name}' not in the scene xml")
        self._mujoco = mujoco
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        fovy = float(model.cam_fovy[self.cam_id])
        self.K = intrinsics_from_fovy(fovy, width, height)
        self.fovy = fovy

    def grab(self) -> RgbdFrame:
        mj = self._mujoco
        r = self._renderer
        r.update_scene(self.data, camera=self.cam_id)
        rgb = r.render()
        r.enable_depth_rendering()
        depth = r.render()
        r.disable_depth_rendering()
        depth = np.where(np.isfinite(depth) & (depth > 0), depth, np.inf).astype(np.float32)
        mj.mj_camlight(self.model, self.data)  # refresh cam_xpos/cam_xmat
        pos = self.data.cam_xpos[self.cam_id].copy()
        R = self.data.cam_xmat[self.cam_id].reshape(3, 3).copy()
        return RgbdFrame(rgb=rgb, depth=depth, K=self.K,
                         cam_pos_w=pos, cam_R_mj=R)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


class Ros2RgbdSource(RgbdSource):
    """Latest-message cache over RealSense-style ROS2 topics.

    color : sensor_msgs/Image  rgb8 (or bgr8)
    depth : sensor_msgs/Image  16UC1 (mm) or 32FC1 (m)
    info  : sensor_msgs/CameraInfo
    pose  : a callable () -> (pos_w(3), R_mj(3,3)) supplying the camera pose in
            the same world/odom frame the robot state uses. On the real robot
            this is FK of the head from IMU + encoders + a yaw/odom estimate.
    """

    def __init__(self, pose_fn, color_topic: str = "/camera/color/image_raw",
                 depth_topic: str = "/camera/depth/image_raw",
                 info_topic: str = "/camera/color/camera_info",
                 depth_scale: float = 0.001):
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import CameraInfo, Image
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Ros2RgbdSource requires ROS2 Python (rclpy + sensor_msgs); "
                "use MujocoRgbdSource in sim2sim"
            ) from e
        self._np = np
        self._pose_fn = pose_fn
        self._depth_scale = depth_scale
        self._last = {"color": None, "depth": None, "info": None}
        if not rclpy.ok():
            rclpy.init()
        self._node = Node("k1_yolo_ball_rgbd")

        def on_color(msg):
            self._last["color"] = self._img_to_np(msg)

        def on_depth(msg):
            self._last["depth"] = self._img_to_np(msg)

        def on_info(msg):
            self._last["info"] = {
                "fx": float(msg.k[0]), "fy": float(msg.k[4]),
                "cx": float(msg.k[2]), "cy": float(msg.k[5]),
            }

        self._node.create_subscription(Image, color_topic, on_color, 10)
        self._node.create_subscription(Image, depth_topic, on_depth, 10)
        self._node.create_subscription(CameraInfo, info_topic, on_info, 10)

    def _img_to_np(self, msg) -> np.ndarray:
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.encoding in ("rgb8", "bgr8"):
            img = arr.reshape(msg.height, msg.width, 3)
            return img[..., ::-1].copy() if msg.encoding == "bgr8" else img.copy()
        if msg.encoding == "16UC1":
            return arr.view(np.uint16).reshape(msg.height, msg.width).astype(np.float32) * self._depth_scale
        if msg.encoding == "32FC1":
            return arr.view(np.float32).reshape(msg.height, msg.width).astype(np.float32)
        raise ValueError(f"unsupported encoding {msg.encoding}")

    def spin_once(self, timeout_s: float = 0.05):
        import rclpy
        from rclpy.spin_once import spin_once  # noqa: F401  (api guard)
        rclpy.spin_once(self._node, timeout_sec=timeout_s)

    def grab(self) -> RgbdFrame | None:
        self.spin_once()
        rgb, depth, K = self._last["color"], self._last["depth"], self._last["info"]
        if rgb is None or depth is None or K is None:
            return None
        pos, R = self._pose_fn()
        return RgbdFrame(rgb=rgb, depth=depth, K=K, cam_pos_w=pos, cam_R_mj=R)


class Ros2RgbdPublisher:
    """Republish an RgbdFrame on ROS2 (sim2sim debug / external consumers)."""

    def __init__(self, color_topic: str = "/camera/color/image_raw",
                 depth_topic: str = "/camera/depth/image_raw",
                 info_topic: str = "/camera/color/camera_info",
                 frame_id: str = "head_realsense_rgb_optical"):
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import CameraInfo, Image
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("Ros2RgbdPublisher requires ROS2 (rclpy)") from e
        self._Image, self._CameraInfo = Image, CameraInfo
        self._frame_id = frame_id
        if not rclpy.ok():
            rclpy.init()
        self._node = Node("k1_yolo_ball_rgbd_pub")
        self._pub_color = self._node.create_publisher(Image, color_topic, 10)
        self._pub_depth = self._node.create_publisher(Image, depth_topic, 10)
        self._pub_info = self._node.create_publisher(CameraInfo, info_topic, 10)

    def publish(self, frame: RgbdFrame):
        import builtin_interfaces.msg
        stamp = self._node.get_clock().now().to_msg()

        c = self._Image()
        c.header.stamp = stamp
        c.header.frame_id = self._frame_id
        c.height, c.width, c.encoding = frame.rgb.shape[0], frame.rgb.shape[1], "rgb8"
        c.step = frame.rgb.shape[1] * 3
        c.data = np.ascontiguousarray(frame.rgb).tobytes()
        self._pub_color.publish(c)

        d = self._Image()
        d.header.stamp = stamp
        d.header.frame_id = self._frame_id
        d.height, d.width, d.encoding = frame.depth.shape[0], frame.depth.shape[1], "32FC1"
        d.step = frame.depth.shape[1] * 4
        d.data = np.ascontiguousarray(frame.depth.astype(np.float32)).tobytes()
        self._pub_depth.publish(d)

        info = self._CameraInfo()
        info.header = c.header
        info.height, info.width = c.height, c.width
        K = frame.K
        info.k = [K["fx"], 0.0, K["cx"], 0.0, K["fy"], K["cy"], 0.0, 0.0, 1.0]
        info.p = [K["fx"], 0.0, K["cx"], 0.0, 0.0, K["fy"], K["cy"], 0.0, 0.0, 0.0, 1.0, 0.0]
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        self._pub_info.publish(info)


class YoloBallDetector:
    """Default COCO yolov8s. 'sports ball' is class 32 and works out of the box."""

    @staticmethod
    def resolve_weights(weights: str = "yolov8s.pt") -> str:
        import os
        if os.path.isfile(weights):
            return weights
        from pathlib import Path
        # .../deploy/tasks/kick_amp/this.py -> parents[3] == repo root
        repo = Path(__file__).resolve().parents[3]
        local = str(repo / "assets" / "models" / weights)
        return local if os.path.isfile(local) else weights

    def __init__(self, weights: str = "yolov8s.pt", conf: float = 0.25, imgsz: int = 640,
                 device: str | None = None):
        import torch
        from ultralytics import YOLO
        self.model = YOLO(self.resolve_weights(weights))
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                try:  # some cards (e.g. sm_120) have no kernel in this torch build
                    self.model.to("cuda")
                    self.model.predict(np.zeros((64, 64, 3), np.uint8),
                                       imgsz=64, verbose=False, device="cuda")
                except Exception:
                    device = "cpu"
        self.device = device
        self.model.to(self.device)
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.names = self.model.names

    def detect(self, rgb: np.ndarray) -> list[dict]:
        """Return [{'xyxy': (x1,y1,x2,y2), 'conf': float, 'name': str}] for ball hits."""
        res = self.model.predict(rgb, conf=self.conf, imgsz=self.imgsz,
                                 verbose=False, device=self.device)[0]
        out = []
        for b in res.boxes:
            name = self.names[int(b.cls)]
            if name.lower() not in BALL_CLASS_NAMES:
                continue
            out.append({
                "xyxy": tuple(float(v) for v in b.xyxy[0].tolist()),
                "conf": float(b.conf),
                "name": name,
            })
        out.sort(key=lambda d: -d["conf"])
        return out


class YoloBallPerception:
    """Drop-in for VirtualBallPerception: emits training's 3-dim ball_obs.

    - refresh_every policy steps (default 2 = 25 Hz, same as training)
    - between refreshes the previous obs is held (training does the same)
    - undetected => (0, 0, 0); do NOT fill in last known position
    - optional delay_steps models transport latency (training uses ~6)
    """

    BUFFER_DEPTH = 20

    def __init__(self, source: RgbdSource, detector: YoloBallDetector | None = None,
                 refresh_every: int = 2, delay_steps: int = 0):
        self.source = source
        self.detector = detector or YoloBallDetector()
        self.refresh_every = max(1, int(refresh_every))
        self.delay_steps = int(np.clip(delay_steps, 0, self.BUFFER_DEPTH - 1))
        self.buffer = np.zeros((self.BUFFER_DEPTH, 3), dtype=np.float32)
        self.step = 0
        self.last_frame: RgbdFrame | None = None
        self.last_det: list[dict] = []
        self.last_xyz_w: np.ndarray | None = None

    def reset(self):
        self.buffer[:] = 0.0
        self.step = 0
        self.last_det = []
        self.last_xyz_w = None

    def update(self, base_pos_xy: np.ndarray, base_yaw: float) -> np.ndarray:
        """Advance one policy step. Returns ball_obs (3,) float32 in yaw frame."""
        self.buffer = np.roll(self.buffer, 1, axis=0)
        if self.step % self.refresh_every == 0:
            self.buffer[0, :] = self._measure(base_pos_xy, base_yaw)
        else:
            self.buffer[0, :] = self.buffer[1, :]
        self.step += 1
        return self.obs()

    def obs(self) -> np.ndarray:
        return self.buffer[self.delay_steps].copy()

    def _measure(self, base_pos_xy: np.ndarray, base_yaw: float) -> np.ndarray:
        frame = self.source.grab()
        self.last_frame = frame
        if frame is None:
            return np.zeros(3, dtype=np.float32)
        dets = self.detector.detect(frame.rgb)
        self.last_det = dets
        if not dets:
            self.last_xyz_w = None
            return np.zeros(3, dtype=np.float32)
        best = dets[0]
        x1, y1, x2, y2 = best["xyxy"]
        u, v = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        z = bbox_depth_median(frame.depth, best["xyxy"])
        if z is None:
            self.last_xyz_w = None
            return np.zeros(3, dtype=np.float32)
        p_cv = deproject(u, v, z, frame.K)
        p_mj = p_cv * np.array([1.0, -1.0, -1.0])
        p_w = frame.cam_R_mj @ p_mj + frame.cam_pos_w
        self.last_xyz_w = p_w
        rel = p_w[:2] - np.asarray(base_pos_xy, dtype=np.float64)
        c, s = math.cos(-base_yaw), math.sin(-base_yaw)
        xy_yaw = np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])
        return np.array([xy_yaw[0], xy_yaw[1], 1.0], dtype=np.float32)


def annotate(rgb: np.ndarray, dets: list[dict], label_extra: str = "") -> np.ndarray:
    import cv2
    vis = rgb.copy()
    for d in dets:
        x1, y1, x2, y2 = (int(round(v)) for v in d["xyxy"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{d['name']} {d['conf']:.2f}{label_extra}",
                    (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return vis
