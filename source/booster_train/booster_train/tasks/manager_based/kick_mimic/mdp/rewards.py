"""GOAL reward group for the kick task (ball / kick-target).

Ported from whole_body_tracking/tasks/track_kick_football/mdp/rewards.py
(lines 80-169, the six ball_* functions). The MOTION tracking rewards and the
regularisation group live in beyond_mimic.mdp.rewards and are re-exported
through this package's __init__.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from booster_train.tasks.manager_based.kick_mimic.mdp.commands import KickMotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_target_progress(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Irreversible ball progress towards the target, in meters gained this step.

    Progress is measured against the historical minimum ball-target distance and
    clamped at zero, so a hard shot that rolls past the target is never punished
    (the overshoot-payback failure mode): reaching the target pays once, rolling
    beyond is free.
    """
    command: KickMotionCommand = env.command_manager.get_term(command_name)
    return command._progress_delta.clone()


def foot_ball_contact(
    env: ManagerBasedRLEnv,
    command_name: str,
    contact_dist: float = 0.22,
    paid_steps: int = 3,
    window_steps: int = 100,
) -> torch.Tensor:
    """Sparse bonus for the main foot touching the ball near the reference kick moment.

    Geometric proximity test (main foot link to ball center). Credit is capped at
    ``paid_steps`` payments per episode and only granted inside a window around
    the reference kick step, preventing "stand next to the ball" reward farming.
    """
    command: KickMotionCommand = env.command_manager.get_term(command_name)
    near = torch.norm(command.main_foot_pos_w - command.ball_pos_w, dim=-1) < contact_dist
    in_window = command.time_steps <= command.cfg.kick_step + window_steps
    fresh = near & in_window & (command._contact_paid < paid_steps)
    command._contact_paid += fresh.long()
    return fresh.float()


def ball_speed(
    env: ManagerBasedRLEnv,
    command_name: str,
    ramp_threshold: float = 1.0,
    target: float = 4.5,
    std: float = 1.5,
) -> torch.Tensor:
    """Ball speed shaped into a target band (default 4-5 m/s).

    Below ``ramp_threshold``: 0 (ball untouched). Between threshold and target:
    linear ramp -- any real kick earns something. Above target: Gaussian decay,
    so excessive power earns less than a well-paced shot (the reference swing
    can physically produce 6+ m/s; an unbounded speed reward pushes the policy
    into wild maximal swings that destroy tracking and stability).
    """
    command: KickMotionCommand = env.command_manager.get_term(command_name)
    speed = torch.norm(command.ball_vel_w, dim=-1)
    ramp = ((speed - ramp_threshold) / (target - ramp_threshold)).clamp(0.0, 1.0)
    decay = torch.exp(-(((speed - target) / std) ** 2))
    return torch.where(speed < target, ramp, decay)


def stagnation_penalty(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """1.0 while the robot barely moved its anchor over the trailing window.

    Guards against "stand still / stand next to the ball" reward farming
    (arXiv:2511.03996 uses the same guard). Only active after the window has
    fully elapsed within the episode. Follows the repo convention: penalty
    functions return a positive magnitude, the term weight is negative.
    """
    command: KickMotionCommand = env.command_manager.get_term(command_name)
    return command._stagnant.float()


def ball_kick_alignment(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """One-shot Gaussian kernel on the kick's outgoing direction vs the target.

    The fire event and kernel are computed in the command (which also exposes
    them as the ``kick_accuracy`` metric that the annealing curriculum gates on);
    this term just pays them once per episode. A one-shot payment is essential:
    a per-step kernel rewards slow rolling (more steps above threshold = more
    reward), which inverts the intended 4-5 m/s speed target.
    """
    command: KickMotionCommand = env.command_manager.get_term(command_name)
    return command._alignment_fire * command._alignment_kernel


def ball_approach(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Per-step reduction of the robot-anchor to ball distance (potential-based).

    Keeps the robot walking towards the ball (walking away is penalised). The
    potential is frozen once the ball is moving, so a successful kick is never
    taxed for the distance the ball travels away from the robot.
    """
    command: KickMotionCommand = env.command_manager.get_term(command_name)
    return command._approach_delta.clone()
