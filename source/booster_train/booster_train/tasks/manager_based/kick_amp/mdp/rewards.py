"""Reward terms for kick_amp (paper 2511.03996 Table 4, ported from t1.py).

The three goal-cluster terms (goal / ball_distance / goal_distance) live in
SoccerStateCommand (reward group 0, fed to critic 0 via extras["rew_groups"]) --
they are NOT RewardManager terms. Everything else is here.
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils.math import quat_rotate, quat_mul, quat_rotate_inverse, quat_apply_inverse, yaw_quat
from .geometry import yaw_from_quat

from .commands import (
    SoccerStateCommand,
    CAMERA_BODY_TO_OPTICAL_WXYZ,
    CAMERA_OFFSET_B,
    task_curriculum,
)

# K1 foot sole corners in the ankle-roll link frame, from Left_Foot.STL bounds:
# x in [-0.066, 0.1195], y in +-0.04, sole z = -0.038. Order matches t1.py:
# 0: +x +y (front-left), 1: +x -y (front-right), 2: -x +y (rear-left), 3: -x -y (rear-right)
FEET_EDGE_POS = [
    [0.1195, 0.04, -0.038],
    [0.1195, -0.04, -0.038],
    [-0.066, 0.04, -0.038],
    [-0.066, -0.04, -0.038],
]


def _cmd(env: ManagerBasedEnv) -> SoccerStateCommand:
    return env.command_manager.get_term("soccer")


def _feet_edge_pos_w(env: ManagerBasedEnv, cmd: SoccerStateCommand) -> torch.Tensor:
    """(N, 2 feet, 4 corners, 3) sole-corner positions, env-local xy / world z.

    Kick rewards compare against the current ball position with origins
    subtracted, so the foot corners use the same coordinates.
    """
    robot = env.scene["robot"]
    feet_state = robot.data.body_state_w[:, cmd.feet_idx, :]  # (N, 2, 13)
    corners = (
        torch.tensor(FEET_EDGE_POS, device=env.device)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(env.num_envs, 2, 4, 3)
    )
    rel = quat_rotate(feet_state[:, :, 3:7].unsqueeze(2).expand(-1, -1, 4, -1).reshape(-1, 4), corners.reshape(-1, 3))
    edge = feet_state[:, :, :3].unsqueeze(2) + rel.reshape(env.num_envs, 2, 4, 3)
    edge[:, :, :, :2] -= env.scene.env_origins[:, None, None, :2]
    return edge


def _feet_yaw(env: ManagerBasedEnv, cmd: SoccerStateCommand) -> torch.Tensor:
    """(N, 2) foot yaw angles."""
    robot = env.scene["robot"]
    quats = robot.data.body_quat_w[:, cmd.feet_idx, :].reshape(-1, 4)
    yaw = yaw_from_quat(quats)
    return yaw.reshape(env.num_envs, 2)


# ------------------------------------------------------------- simple penalties


def survival(env: ManagerBasedEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def termination(env: ManagerBasedEnv) -> torch.Tensor:
    # falls only: time-outs and ball resets are not punished (t1.py)
    return env.termination_manager.terminated.float()


def pos_still(env: ManagerBasedEnv, penalize_pos: float = 0.1, penalize_yaw: float = 1.0,
              penalize_pos_distance: float = 1.0) -> torch.Tensor:
    """1 if the robot barely moved for the last 1 s while the ball is far away.

    Anti-freeze term. Its own 50-step history requirement is the natural
    curriculum: it cannot fire until the robot has survived one full second,
    so no global iteration gate is needed. It also exempts robots within 1 m
    of the ball, preserving the valid "approach then hold balance" solution.
    """
    cmd = _cmd(env)
    buf = cmd.base_pos_buffer
    cur = buf[torch.arange(env.num_envs, device=env.device), cmd.buffer_idx]
    pos_dist = torch.norm(buf[:, :, 0:2] - cur[:, 0:2].unsqueeze(1), dim=-1)
    yaw_diff = buf[:, :, 2] - cur[:, 2].unsqueeze(1)
    yaw_dist = torch.abs(torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff)))
    return (
        (cmd.buffer_count == buf.shape[1])
        & torch.all(pos_dist < penalize_pos, dim=-1)
        & (torch.all(yaw_dist < penalize_yaw, dim=-1) | torch.all(pos_dist < 0.3, dim=-1))
        & (torch.norm(cmd.relative_ball_pos, dim=-1) > penalize_pos_distance)
    ).float() * (1.0 if cmd.cfg.training_phase == "approach" else task_curriculum(env))


# ------------------------------------------------------------- kick shaping


def kick_ball(env: ManagerBasedEnv, dist_thresh: float = 0.2) -> torch.Tensor:
    """Penalize toe-forward pushes of the ball (front two corners near ball)."""
    cmd = _cmd(env)
    edge = _feet_edge_pos_w(env, cmd)[:, :, 0:2, 0:2]  # (N, 2 feet, 2 front corners, xy)
    ball_xy = cmd.ball.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    to_ball = torch.norm(edge - ball_xy[:, None, None, :], dim=-1)  # (N, 2, 2)
    feet_vel = env.scene["robot"].data.body_lin_vel_w[:, cmd.feet_idx, 0:2]
    yaw = _feet_yaw(env, cmd)
    feet_vel_x = torch.cos(yaw) * feet_vel[:, :, 0] + torch.sin(yaw) * feet_vel[:, :, 1]
    return torch.sum(torch.all(to_ball < dist_thresh, dim=-1).float() * feet_vel_x.clip(min=0.0), dim=-1)


def side_kick_ball(env: ManagerBasedEnv, dist_thresh: float = 0.3, max_kick_vel: float = 2.0) -> torch.Tensor:
    """Reward arch (inner-side) sweeping kicks: left foot kicks toward -y, right toward +y."""
    cmd = _cmd(env)
    edge = _feet_edge_pos_w(env, cmd)[:, :, :, 0:2]  # (N, 2, 4, xy)
    ball_xy = cmd.ball.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    to_ball = torch.norm(edge - ball_xy[:, None, None, :], dim=-1)  # (N, 2, 4)
    feet_vel = env.scene["robot"].data.body_lin_vel_w[:, cmd.feet_idx, 0:2]
    yaw = _feet_yaw(env, cmd)
    feet_vel_y = -torch.sin(yaw) * feet_vel[:, :, 0] + torch.cos(yaw) * feet_vel[:, :, 1]
    left = ((to_ball[:, 0, 1] < dist_thresh) & (to_ball[:, 0, 3] < dist_thresh)).float() * (
        -feet_vel_y[:, 0]
    ).clip(min=0.0, max=max_kick_vel)
    right = ((to_ball[:, 1, 0] < dist_thresh) & (to_ball[:, 1, 2] < dist_thresh)).float() * feet_vel_y[:, 1].clip(
        min=0.0, max=max_kick_vel
    )
    return left + right


def face_ball_pitch(env: ManagerBasedEnv) -> torch.Tensor:
    """Squared pitch angle of the ball w.r.t. the head camera optical axis."""
    cmd = _cmd(env)
    return _camera_angles(env, cmd)[:, 1] ** 2


def face_ball_yaw(env: ManagerBasedEnv) -> torch.Tensor:
    cmd = _cmd(env)
    return _camera_angles(env, cmd)[:, 0] ** 2


def _camera_angles(env: ManagerBasedEnv, cmd: SoccerStateCommand) -> torch.Tensor:
    """(yaw, pitch) of the ball direction in the camera optical frame (radians)."""
    robot = env.scene["robot"]
    head = robot.data.body_state_w[:, cmd.head_idx, :]
    cam_quat = quat_mul(
        head[:, 3:7],
        torch.tensor(CAMERA_BODY_TO_OPTICAL_WXYZ, device=env.device).expand(env.num_envs, -1),
    )
    cam_pos_w = head[:, :3] + quat_rotate(
        head[:, 3:7], torch.tensor(CAMERA_OFFSET_B, device=env.device).expand(env.num_envs, -1)
    )
    # RewardManager runs before command caches refresh for this physics step.
    ball_pos_w = cmd.ball.data.root_pos_w
    ball_to_cam = quat_rotate_inverse(cam_quat, ball_pos_w - cam_pos_w)
    z = ball_to_cam[:, 2]
    z = torch.where(torch.abs(z) < 1e-8, torch.sign(z) * 1e-8, z)
    return torch.stack((torch.atan2(ball_to_cam[:, 0], z), torch.atan2(ball_to_cam[:, 1], z)), dim=-1)


# ------------------------------------------------------------- regularization


def root_acc(env: ManagerBasedEnv) -> torch.Tensor:
    """Squared root acceleration (finite difference against the command's cached last vel)."""
    cmd = _cmd(env)
    vel = env.scene["robot"].data.root_state_w[:, 7:13]
    acc = (vel - cmd.last_root_vel) / env.step_dt
    return torch.sum(acc.square(), dim=-1)


def action_rate_legs(env: ManagerBasedEnv) -> torch.Tensor:
    cmd = _cmd(env)
    actions = env.action_manager.action[:, cmd.body_joint_idx]
    last = cmd.last_actions[:, cmd.body_joint_idx]
    return torch.sum((actions - last) ** 2, dim=-1)


def head_action_rate(env: ManagerBasedEnv) -> torch.Tensor:
    cmd = _cmd(env)
    actions = env.action_manager.action[:, cmd.head_joint_idx]
    last = cmd.last_actions[:, cmd.head_joint_idx]
    return torch.sum((actions - last) ** 2, dim=-1)


def dof_pos_limits(env: ManagerBasedEnv, soft_limit_frac: float = 0.9) -> torch.Tensor:
    """Joint-limit violation amount, active from the first iteration.

    Unlike the old collision term, this is continuous and nearly zero at the
    K1 default stance (measured cumulative episode reward about -0.02 versus
    collision's -3.4). Gating it until iter 1000 only created a reward cliff.
    """
    robot = env.scene["robot"]
    # hard URDF limits (soft_joint_pos_limits are already shrunk by
    # soft_joint_pos_limit_factor=0.9 -- using them here would double-apply)
    lo = robot.data.joint_pos_limits[:, :, 0]
    hi = robot.data.joint_pos_limits[:, :, 1]
    span = hi - lo
    lower = lo + 0.5 * (1 - soft_limit_frac) * span
    upper = hi - 0.5 * (1 - soft_limit_frac) * span
    pos = robot.data.joint_pos
    return torch.sum((lower - pos).clip(min=0.0) + (pos - upper).clip(min=0.0), dim=-1)


def collision(env: ManagerBasedEnv, threshold: float = 1.0) -> torch.Tensor:
    """Count non-foot contacts using the sensor's own body ordering.

    Articulation indices cannot index ContactSensor: K1 left foot is sensor
    index 16 but articulation index 21. The former was mislabelled an elbow.
    """
    sensor = env.scene["contact_forces"]
    feet = ("left_ankle_roll_link", "right_ankle_roll_link")
    missing = [n for n in feet if n not in sensor.body_names]
    assert not missing, f"collision: feet {missing} missing from contact sensor bodies"
    penal_ids = [i for i, name in enumerate(sensor.body_names) if name not in feet]
    forces = sensor.data.net_forces_w[:, penal_ids, :]
    return (torch.norm(forces, dim=-1) > threshold).sum(dim=-1).float()


def feet_min_distance(env: ManagerBasedEnv, min_dist: float = 0.1) -> torch.Tensor:
    cmd = _cmd(env)
    edge = _feet_edge_pos_w(env, cmd)[:, :, :, 0:2]  # (N, 2 feet, 4 corners, xy)
    d = torch.cdist(edge[:, 0], edge[:, 1])          # (N, 4, 4) pairwise corner distances
    return (d.reshape(env.num_envs, -1).min(dim=-1).values < min_dist).float()


# ------------------------------------------------------- walk curriculum terms
# Ported from legged_lab velocity/mdp/rewards.py (Booster's G1 walking recipe):
# the positive step incentives that t1.py lacks. Every term scales with
# (1 - c) of the walk-first curriculum (commands.task_curriculum) and is gated
# to zero when the robot is already at the ball (near_ball meters) -- there is
# nothing to walk toward.


def _far_from_ball(env, cmd, near_ball: float) -> torch.Tensor:
    return (torch.norm(cmd.relative_ball_pos, dim=-1) > near_ball).float()


def track_lin_vel_ball(env: ManagerBasedEnv, target_speed: float = 0.5, std: float = 0.5,
                       near_ball: float = 0.5) -> torch.Tensor:
    """legged_lab track_lin_vel_xy_yaw_frame_exp, command = toward the ball."""
    cmd = _cmd(env)
    robot = env.scene["robot"]
    dir_w = cmd.ball_pos_xy - cmd.base_pos_xy                       # world-frame direction
    dist = torch.norm(dir_w, dim=-1)
    far = _far_from_ball(env, cmd, near_ball)
    des_w = dir_w / dist.unsqueeze(-1).clamp(min=1e-6) * (target_speed * far).unsqueeze(-1)
    des_w3 = torch.cat((des_w, torch.zeros_like(des_w[:, :1])), dim=-1)  # quat ops need 3D
    yaw_q = yaw_quat(robot.data.root_quat_w)
    des = quat_apply_inverse(yaw_q, des_w3)                         # into yaw frame
    vel_yaw = quat_apply_inverse(yaw_q, robot.data.root_lin_vel_w[:, :3])
    err = torch.sum(torch.square(des[:, :2] - vel_yaw[:, :2]), dim=-1)
    return torch.exp(-err / std**2) * far * (1.0 - task_curriculum(env))


def feet_air_time_biped(env: ManagerBasedEnv, threshold: float = 0.4,
                        near_ball: float = 0.5) -> torch.Tensor:
    """legged_lab feet_air_time_positive_biped: single-stance step reward."""
    cmd = _cmd(env)
    sensor = env.scene["contact_forces"]
    idx = cmd.feet_sensor_idx                                     # SENSOR order, not articulation
    air_time = sensor.data.current_air_time[:, idx]
    contact_time = sensor.data.current_contact_time[:, idx]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    return reward * _far_from_ball(env, cmd, near_ball) * (1.0 - task_curriculum(env))


def feet_clearance(env: ManagerBasedEnv, target_height: float = 0.10, std: float = 0.05,
                   tanh_mult: float = 2.0, near_ball: float = 0.5) -> torch.Tensor:
    """Reward moving, airborne feet near the clearance target; stance earns zero."""
    cmd = _cmd(env)
    robot = env.scene["robot"]
    z_err = torch.square(robot.data.body_pos_w[:, cmd.feet_idx, 2] - target_height)
    vel = torch.tanh(tanh_mult * torch.norm(robot.data.body_lin_vel_w[:, cmd.feet_idx, :2], dim=2))
    force = env.scene["contact_forces"].data.net_forces_w[:, cmd.feet_sensor_idx].norm(dim=-1)
    swing = (force <= 1.0).float()
    return (torch.sum(torch.exp(-z_err / std**2) * vel * swing, dim=1)
            * _far_from_ball(env, cmd, near_ball) * (1.0 - task_curriculum(env)))


def feet_slide(env: ManagerBasedEnv, near_ball: float = 0.5) -> torch.Tensor:
    """legged_lab feet_slide: penalize horizontal foot velocity while in contact."""
    cmd = _cmd(env)
    sensor = env.scene["contact_forces"]
    idx = cmd.feet_sensor_idx                                     # SENSOR order, not articulation
    contacts = sensor.data.net_forces_w_history[:, :, idx, :].norm(dim=-1).max(dim=1)[0] > 1.0
    robot = env.scene["robot"]
    body_vel = robot.data.body_lin_vel_w[:, cmd.feet_idx, :2]
    return (torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
            * _far_from_ball(env, cmd, near_ball) * (1.0 - task_curriculum(env)))
