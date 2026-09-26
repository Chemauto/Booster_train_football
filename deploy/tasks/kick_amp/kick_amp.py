"""K1 kick_amp policy for MuJoCo sim2sim (and, later, the real robot).

This is a 1:1 port of the booster_train kick_amp observation pipeline so the
exported policy.pt (AmpRunner JIT, forward(raw_obs[1,79], stacked[1,50,79])
-> action[1,22]) sees exactly what it saw in training:

  obs (79, in this exact order, all CLEAN -- evaluation runs with obs
       corruption disabled, evaluate_kick_amp.py:243):
    0:3   projected gravity in the base frame          (env_cfg.py:136)
    3:6   base angular velocity (gyro, local frame)    (env_cfg.py:137)
    6:9   ball obs: (x, y) in the robot YAW frame + detection flag
          (env_cfg.py:138 -> commands.py ball_obs)
    9:11  goal center (x, y) in the robot yaw frame    (env_cfg.py:139)
    11:13 world yaw as (cos, sin)                      (env_cfg.py:140)
    13:35 joint pos - default_joint_pos  (sim order)   (env_cfg.py:141)
    35:57 joint vel * 0.1                (sim order)   (env_cfg.py:142)
    57:79 last raw action                (sim order)   (env_cfg.py:143)

  "sim order" = RobotCfg.sim_joint_names, the IsaacLab PhysX BFS order;
  robot.data joint arrays are in REAL order and are remapped through
  real2sim_joint_indexes (same as beyond_mimic).

  The virtual ball perception (commands.py:426-457) is ported verbatim
  below (25 Hz refresh, N(6,1)-step delay clipped to [0,19], per-axis noise
  sigma = 0.149 + 0.124 d, 87x58 deg FOV around the head camera pose,
  detection prob min(0.9, 2.475 - 0.225 d), 1% false positives for a
  near ball behind the head). NOTE both surviving checkpoints were trained
  AND evaluated with perfect_perception=True (ground-truth ball + flag 1),
  so the default here is perception="perfect"; the virtual path exists for
  checkpoints retrained with perceptual noise.

  History stack semantics (runner.py:226-229, export docstring 232-234):
  the SECOND forward input must be a 50-frame history of NORMALIZED obs,
  newest last, zeroed at episode boundaries. The first input is RAW obs --
  EmpiricalNormalization is frozen inside the exported graph.
"""

from __future__ import annotations

import numpy as np
import torch

from booster_deploy.controllers.base_controller import Policy
from booster_deploy.controllers.controller_cfg import PolicyCfg
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.isaaclab.math import (
    quat_apply_inverse, quat_mul, quat_rotate,
)

# -- field geometry (kick_amp mdp/commands.py:111-116) -----------------------
FIELD_HALF_LENGTH = 7.0
FIELD_HALF_WIDTH = 4.5
GOAL_X = 7.0
GOAL_HALF_WIDTH = 1.3
GOAL_HEIGHT = 1.8
BALL_RADIUS = 0.11

NUM_OBS = 79
NUM_ACTIONS = 22

# -- head camera model (commands.py:184-187) ---------------------------------
CAMERA_OFFSET_B = (0.054, 0.0, 0.102)
CAMERA_BODY_TO_OPTICAL_WXYZ = (0.405580, 0.579228, 0.579228, 0.405580)  # 20 deg down, see commands.py
CAMERA_FOV_H = 87.0
CAMERA_FOV_V = 58.0


def yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Heading about world z in [-pi, pi] for (..., 4) wxyz quats.

    Same formula as kick_amp mdp/geometry.py:6-9."""
    w, x, y, z = quat.unbind(dim=-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


def rotate_to_yaw_frame(delta_xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """2D rotate into the robot yaw frame (commands.py:279-285)."""
    cos_y = torch.cos(-yaw)
    sin_y = torch.sin(-yaw)
    return torch.stack(
        (cos_y * delta_xy[0] - sin_y * delta_xy[1],
         sin_y * delta_xy[0] + cos_y * delta_xy[1]),
        dim=-1,
    )


class VirtualBallPerception:
    """Numpy port of the training-side virtual ball perception.

    One instance == one episode; call reset() at episode start, then
    update() once per POLICY step (50 Hz) BEFORE reading obs().
    """

    BUFFER_DEPTH = 20   # commands.py:227

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.buffer = np.zeros((self.BUFFER_DEPTH, 3), dtype=np.float32)
        self.delay_steps = 6
        self.step = 0

    def reset(self):
        # commands.py:604-605: buffer zeroed, delay ~ clamp(randn+6, 0, 19)
        self.buffer[:] = 0.0
        self.delay_steps = int(np.clip(self.rng.normal(6.0, 1.0), 0.0, 19.0))
        self.step = 0

    def update(
        self,
        rel_ball_pos_xy: torch.Tensor,
        cam_pos_w: torch.Tensor,
        cam_quat_w: torch.Tensor,
        ball_pos_w: torch.Tensor,
    ):
        """Advance the perception one policy step (commands.py:426-457)."""
        # camera optical frame: quat_mul(head_quat, body->optical)
        cam_opt_quat = quat_mul(
            cam_quat_w.reshape(1, 4),
            torch.tensor(CAMERA_BODY_TO_OPTICAL_WXYZ).reshape(1, 4),
        )
        ball_to_camera = quat_apply_inverse(
            cam_opt_quat, (ball_pos_w - cam_pos_w).reshape(1, 3)
        ).reshape(3)
        fwd = ball_to_camera[2].clamp(min=1e-8)
        in_fov = float(
            (ball_to_camera[2] > 0)
            & (torch.abs(torch.atan2(ball_to_camera[0], fwd))
               < 0.5 * CAMERA_FOV_H / 180.0 * torch.pi)
            & (torch.abs(torch.atan2(ball_to_camera[1], fwd))
               < 0.5 * CAMERA_FOV_V / 180.0 * torch.pi)
        )
        dist = float(torch.norm(rel_ball_pos_xy))
        # detection prob: 2.475 - 0.225 * max(d, 7) == min(0.9, 2.475 - 0.225 d)
        detect = float(
            self.rng.random() < (2.475 - 0.225 * max(dist, 7.0))
        )
        # 1% false positive for a near ball behind the head
        false_pos = float(
            (1.0 - in_fov)
            * (float(rel_ball_pos_xy[0]) > -2.0)
            * (self.rng.random() < 0.01)
        )
        ball_in_view = in_fov * detect + false_pos

        # 25 Hz refresh: roll, then refresh on even steps (episode parity)
        self.buffer = np.roll(self.buffer, 1, axis=0)
        sigma = 0.149 + 0.124 * dist
        noisy = rel_ball_pos_xy.numpy() + sigma * self.rng.normal(size=2)
        if self.step % 2 == 0:
            self.buffer[0, 0:2] = ball_in_view * noisy
            self.buffer[0, 2] = ball_in_view
        else:
            self.buffer[0, :] = self.buffer[1, :]
        self.step += 1

    def obs(self) -> torch.Tensor:
        return torch.from_numpy(self.buffer[self.delay_steps, :].copy())


def _head_cam_pose_fn(controller):
    """Camera pose from the head link, same math as training (commands.py:353-355)."""
    def pose():
        st = controller
        head_pos = st.head_pos_w.numpy().astype(np.float64).reshape(3)
        head_quat = st.head_quat_w.numpy().astype(np.float64).reshape(4)  # wxyz
        w, x, y, z = head_quat
        Rh = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        pos = head_pos + Rh @ np.asarray(CAMERA_OFFSET_B, dtype=np.float64)
        # MuJoCo camera axes in the head frame (cols = cam x,y,z in head).
        # Matches k1_soccer_14x9.xml head_cam quat: 20 deg pitch down, i.e.
        # cam x = head -y (image right), looks along head +x tilted -z.
        R_cam_in_head = np.array([[0.0, 0.342020, -0.939693],
                                  [-1.0, 0.0, 0.0],
                                  [0.0, 0.939693, 0.342020]])
        return pos, Rh @ R_cam_in_head
    return pose


class KickAmpPolicy(Policy):
    """79-dim obs + 50-frame normalized history stack -> 22 joint targets."""

    def __init__(self, cfg: "KickAmpPolicyCfg", controller):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self._model: torch.jit.ScriptModule = torch.jit.load(
            f"{self.task_path}/{cfg.checkpoint_path}", map_location="cpu"
        ).to(self.device).eval()
        # frozen EmpiricalNormalization, callable on raw (1, 79) obs
        self._obs_norm = self._model.obs_norm

        self.robot = controller.robot
        self.robot.data.to(self.device)

        self.default_real = self.robot.default_joint_pos.to(self.device)
        self.action_scale_real = (
            0.25 * self.robot.effort_limit / self.robot.joint_stiffness
        ).to(self.device)   # real order, = training K1_ACTION_SCALE

        self.stack = torch.zeros(1, cfg.num_stack, NUM_OBS, device=self.device)
        self.last_action = torch.zeros(NUM_ACTIONS, device=self.device)
        self._inference_count = 0
        self._episode_rng = np.random.default_rng(0)
        self.perception = (
            VirtualBallPerception(self._episode_rng)
            if cfg.perception == "virtual" else None
        )
        self.yolo = None
        self._yolo_pending = cfg.perception == "yolo"

    def _ensure_yolo(self):
        """Build the RGB-D + YOLO stack lazily: the mujoco controller only has
        mj_model after BaseController.__init__ constructed this policy."""
        if not self._yolo_pending:
            return
        self._yolo_pending = False
        cfg = self.cfg
        controller = self.controller
        from .yolo_ball_perception import (
            MujocoRgbdSource, YoloBallDetector, YoloBallPerception,
        )
        if getattr(controller, "mj_model", None) is not None:
            src = MujocoRgbdSource(
                controller.mj_model, controller.mj_data,
                width=cfg.yolo_width, height=cfg.yolo_height)
        else:
            from .yolo_ball_perception import Ros2RgbdSource
            src = Ros2RgbdSource(
                pose_fn=cfg.yolo_pose_fn or _head_cam_pose_fn(controller))
        self.yolo = YoloBallPerception(
            src, YoloBallDetector(conf=cfg.yolo_conf),
            refresh_every=cfg.yolo_refresh_every,
            delay_steps=cfg.yolo_delay_steps)

    def set_episode_seed(self, seed: int):
        """Seed the perception RNG for this episode (deterministic reruns)."""
        self._episode_rng = np.random.default_rng(seed)
        if self.perception is not None and hasattr(self.perception, "rng"):
            self.perception.rng = self._episode_rng

    def reset(self) -> None:
        # runner.py:229: zero the whole stack at episode boundaries
        self.stack.zero_()
        self.last_action.zero_()
        self._inference_count = 0
        self._ensure_yolo()
        if self.perception is not None:
            self.perception.reset()
        if self.yolo is not None:
            self.yolo.reset()

    def compute_observation(self) -> torch.Tensor:
        self._ensure_yolo()
        st = self.controller          # soccer state provider (mujoco controller)
        rd = self.robot.data
        dev = self.device

        root_pos_w = rd.root_pos_w.to(dev)
        root_quat_w = rd.root_quat_w.to(dev)

        gravity_b = quat_apply_inverse(
            root_quat_w.reshape(1, 4),
            torch.tensor([0.0, 0.0, -1.0], device=dev).reshape(1, 3),
        ).reshape(3)

        base_yaw = yaw_from_quat(root_quat_w)
        base_xy = root_pos_w[:2]
        rel_ball = rotate_to_yaw_frame(
            st.ball_pos_w[:2].to(dev) - base_xy, base_yaw)
        rel_goal = rotate_to_yaw_frame(
            torch.tensor([GOAL_X, 0.0], device=dev) - base_xy, base_yaw)

        if self.yolo is not None:
            ball_obs = torch.from_numpy(
                self.yolo.update(base_xy.detach().cpu().numpy(),
                                 float(base_yaw))
            ).to(dev)
        elif self.perception is not None:
            head_quat = st.head_quat_w.to(dev).reshape(1, 4)
            # camera optical center = head origin + offset in head frame
            # (training commands.py:353-355)
            cam_pos = st.head_pos_w.to(dev) + quat_rotate(
                head_quat,
                torch.tensor(CAMERA_OFFSET_B, device=dev).reshape(1, 3),
            ).reshape(3)
            self.perception.update(
                rel_ball, cam_pos, head_quat, st.ball_pos_w.to(dev))
            ball_obs = self.perception.obs().to(dev)
        else:
            # perfect_perception (commands.py:590-591): truth + flag 1
            ball_obs = torch.cat(
                (rel_ball, torch.ones(1, device=dev)), dim=-1)

        r2s = rd.real2sim_joint_indexes
        joint_pos = (rd.joint_pos.to(dev) - self.default_real)[r2s]
        joint_vel = rd.joint_vel.to(dev)[r2s] * 0.1

        obs = torch.cat((
            gravity_b,                                  # 3
            rd.root_ang_vel_b.to(dev),                  # 3
            ball_obs,                                   # 3
            rel_goal,                                   # 2
            torch.stack((torch.cos(base_yaw),
                         torch.sin(base_yaw))),         # 2
            joint_pos,                                  # 22
            joint_vel,                                  # 22
            self.last_action,                           # 22
        ))
        assert obs.shape == (NUM_OBS,), obs.shape
        return obs

    def inference(self) -> torch.Tensor:
        obs = self.compute_observation()
        obs1 = obs.reshape(1, NUM_OBS)

        # Stack semantics, exactly as the training rollout (runner.py:341-356,
        # 393): the stack an action reads contains obs_1..obs_n with the
        # CURRENT obs as the newest entry (the previous step's push), EXCEPT
        # at the very first action of an episode where the stack is all zeros
        # (reset zeroes it and the reset obs itself is never pushed).
        if self._inference_count > 0:
            with torch.no_grad():
                self.stack = torch.roll(self.stack, shifts=-1, dims=1)
                self.stack[:, -1, :] = self._obs_norm(obs1)
        self._inference_count += 1

        with torch.no_grad():
            action = self._model(obs1, self.stack).flatten()

        self.last_action = action.clone()

        # decode to REAL-order position targets:
        # target = default + action * scale  (env_cfg.py:181-186, K1_ACTION_SCALE)
        s2r = self.robot.data.sim2real_joint_indexes
        return action[s2r] * self.action_scale_real + self.default_real


@configclass
class KickAmpPolicyCfg(PolicyCfg):
    constructor = KickAmpPolicy
    checkpoint_path: str = "models/kick_amp_it6400_policy.pt"
    # "perfect": ground-truth ball + flag 1 (how both surviving checkpoints
    # were trained and evaluated). "virtual": the noisy/delayed/dropout
    # perception port above, for checkpoints retrained with it.
    perception: str = "perfect"   # "perfect" | "virtual" | "yolo"
    num_stack: int = 50
    yolo_conf: float = 0.15
    yolo_width: int = 640
    yolo_height: int = 360
    yolo_refresh_every: int = 2   # 25 Hz, matches training virtual perception
    yolo_delay_steps: int = 6   # training ball_delay_steps nominal (6 = 120 ms)
    yolo_pose_fn: object = None   # () -> (pos_w, R_mj) for the ROS2 source
