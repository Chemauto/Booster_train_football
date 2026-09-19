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
    FIELD_HALF_LENGTH,
    FIELD_HALF_WIDTH,
    GOAL_X,
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
    """Falls and field exits only: a scored goal is a success, not a failure (t1.py).

    Scoring is registered as a DoneTerm so the evaluation can label the episode
    and so the value function treats it as a true terminal state, but it must not
    collect this -1000: the whole objective is to put the ball in the goal, and
    charging 1000 for achieving it would teach the policy to stay away from the
    goal line.
    """
    tm = env.termination_manager
    fired = tm.terminated.float()
    if "goal" in tm.active_terms:
        fired = fired - tm.get_term("goal").float()
    return fired


def pos_still(env: ManagerBasedEnv, penalize_pos: float = 0.1, penalize_yaw: float = 1.0,
              penalize_pos_distance: float = 1.0) -> torch.Tensor:
    """1 if the robot barely moved for the last 1 s while the ball is far away.

    Anti-freeze term. Its own 50-step history requirement is the natural
    curriculum: it cannot fire until the robot has survived one full second,
    so no global iteration gate is needed.

    ``penalize_pos_distance`` is the ball-exemption radius. The config sets it to
    0.3 m, which is contact range, so the legitimate "arrive, then hold balance to
    kick" solution survives; at the original 1.0 m it instead created a parking
    space, where a policy can stop short of the ball and pay nothing.
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


def boundary_distance(env: ManagerBasedEnv, guard: float = 0.3) -> torch.Tensor:
    """Dense penalty for nearing the field edge, ramping quadratically to 1 at the line.

    The only other signal about the boundary is the out-of-field termination,
    which is a 0/1 cliff, and the goal line *is* the field edge
    (GOAL_X == FIELD_HALF_LENGTH == 7.0), so scoring means pushing the ball to
    the boundary the robot must not cross. With no gradient in between, the
    cheapest way to never trip the -1000 termination penalty is to not chase:
    measured at model_3000, the policy parks 0.6 m from the ball and returns
    contact proxies of 1/64. This turns the cliff into a ramp.

    ``guard`` is deliberately small. To push the ball across x = 7 the robot's
    centre stays roughly 0.3-0.7 m from the line, so a wider guard would tax the
    scoring motion itself and recreate the same stall for the opposite reason.
    """
    cmd = _cmd(env)
    d = torch.minimum(
        FIELD_HALF_LENGTH - cmd.base_pos_xy[:, 0].abs(),
        FIELD_HALF_WIDTH - cmd.base_pos_xy[:, 1].abs(),
    )
    # A zero guard would divide by zero; clamping the denominator makes it
    # degenerate to a 0/1 indicator of being on or over the line instead of NaN.
    return ((guard - d).clamp(min=0.0) / max(float(guard), 1e-9)).clamp(max=1.0) ** 2


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

def boundary_outward_speed(env: ManagerBasedEnv, guard: float = 0.6) -> torch.Tensor:
    """Charge only the *outward* component of the base velocity near the field edge.

    The distance-based variant cannot work on this task. GOAL_X equals
    FIELD_HALF_LENGTH, so pushing the ball across the goal line necessarily
    happens within ~0.3 m of the field edge; any guard wide enough to teach
    deceleration therefore taxes the scoring push itself. Measured: switching the
    distance term on took goals 74 -> 37 while it did cut out-of-field 215 -> 44.

    What separates a legitimate scoring approach from running out of bounds is not
    proximity but the direction of travel, so this term charges
    max(0, outward speed) scaled by proximity to the nearest edge. Standing at the
    line pushing the ball in costs nothing; charging at it costs.
    """
    cmd = _cmd(env)
    xy = cmd.base_pos_xy
    vel = env.scene["robot"].data.root_lin_vel_w[:, :2]
    dx = FIELD_HALF_LENGTH - xy[:, 0].abs()
    dy = FIELD_HALF_WIDTH - xy[:, 1].abs()
    nearest_x = dx <= dy
    outward = torch.where(nearest_x, torch.sign(xy[:, 0]) * vel[:, 0],
                          torch.sign(xy[:, 1]) * vel[:, 1])
    proximity = ((guard - torch.minimum(dx, dy)) / max(float(guard), 1e-9)).clamp(min=0.0, max=1.0)
    return outward.clamp(min=0.0) * proximity


def ball_lateral_speed(env: ManagerBasedEnv, dist_thresh: float = 0.3, max_speed: float = 2.0,
                       near_goal: float = 3.0) -> torch.Tensor:
    """Charge the sideways component of the ball's speed while a foot is on it.

    Every remaining out-of-field episode at arm H2 is a ball sent past the goal
    line outside the posts (37 of the last 38), and the goal line is the field
    edge, so a lateral error at the line is both a missed goal and an exit. The
    reward set pays for lateral foot sweeps (side_kick_ball, +20) and nothing
    charges the resulting sideways ball motion.

    MEASURED HARMFUL -- do not enable this without re-deriving it. Charging lateral
    ball speed also charges the sideways taps that STEER the ball, so the policy
    loses the corrections that would keep it on line. Arm H4 (weight -100, gate
    5 m), from a checkpoint scoring 198/256: goals fell to 146, out of field rose
    39 -> 95, and the wide misses rose 40 -> 90 while their contact lateral speed
    only went 1.185 -> 1.021 m/s. At weight -10 the term was inert (logged -0.024
    against side_kick_ball's +0.19). Lateral ball speed is not the same thing as
    aim error. Kept because the measurement and the failure are worth recording.
    """
    cmd = _cmd(env)
    edge = _feet_edge_pos_w(env, cmd)[:, :, :, 0:2]  # (N, 2 feet, 4 corners, xy)
    ball_xy = cmd.ball.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    to_ball = torch.norm(edge - ball_xy[:, None, None, :], dim=-1).amin(dim=(-1, -2))
    lateral = cmd.ball.data.root_lin_vel_w[:, 1].abs().clamp(max=max_speed)
    near = (cmd.ball_pos_xy[:, 0] > GOAL_X - near_goal).float()
    return lateral * (to_ball < dist_thresh).float() * near
