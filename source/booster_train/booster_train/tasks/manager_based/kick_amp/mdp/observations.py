"""Observation terms for kick_amp (K1, 79-dim actor / 93-dim critic).

Actor obs is the POMDP subset of the critic obs: no base linear velocity, ball
seen through the virtual perception system (25 Hz, latency, noise, dropouts).
Sensor noise levels follow T1.yaml.
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils.math import quat_rotate_inverse

from .commands import SoccerStateCommand


def _cmd(env: ManagerBasedEnv) -> SoccerStateCommand:
    return env.command_manager.get_term("soccer")


def ball_obs(env: ManagerBasedEnv) -> torch.Tensor:
    """Virtual-perception ball observation (x, y, detection flag), delayed+noisy."""
    return _cmd(env).ball_obs


def ball_obs_true(env: ManagerBasedEnv) -> torch.Tensor:
    """Ground-truth ball position in the robot yaw frame + in-view flag (critic only)."""
    cmd = _cmd(env)
    return torch.cat((cmd.relative_ball_pos, cmd.ball_in_view.unsqueeze(-1)), dim=-1)


def goal_pos_b(env: ManagerBasedEnv) -> torch.Tensor:
    """Goal center in the robot yaw frame."""
    return _cmd(env).relative_goal_pos


def base_yaw_cos_sin(env: ManagerBasedEnv) -> torch.Tensor:
    yaw = _cmd(env).base_yaw
    return torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)


def joint_vel_scaled(env: ManagerBasedEnv, scale: float = 0.1) -> torch.Tensor:
    return env.scene["robot"].data.joint_vel * scale


def privileged_obs(env: ManagerBasedEnv) -> torch.Tensor:
    """14-dim decoder reconstruction target, maintained by the soccer command."""
    return _cmd(env)._compute_privileged_obs()
