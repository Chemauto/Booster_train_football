"""Ball observations for the kick task (actor virtual perception + critic truth).

Ported from whole_body_tracking/tasks/track_kick_football/mdp/observations.py
(lines 86-125, the four ball_* functions). The plain tracking observations
live in beyond_mimic.mdp.observations and are re-exported through this
package's __init__.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from booster_train.tasks.manager_based.kick_mimic.mdp.commands import KickMotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def ball_state_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Ball observation for the actor: [ball_xy(2), unit(ball->target)(2)] in the anchor frame.

    No z channel: the ball always rolls on the ground. Appended (never replacing)
    right after the command term so channel semantics stay aligned for weight transfer.
    """
    command: KickMotionCommand = env.command_manager.get_term(command_name)
    return torch.cat([command.ball_pos_b[:, :2], command.ball_to_target_dir_b], dim=-1)


def ball_velocity_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Ball xy velocity in the anchor frame, critic-only.

    The actor does not see it: the ball is static until the kick, so it carries no
    decision information; for the critic it directly determines return after the kick.
    """
    command: KickMotionCommand = env.command_manager.get_term(command_name)
    return command.ball_vel_b


def ball_state_virtual_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Actor-side ball observation through the virtual perception system.

    [est_pos_xy(2), unit(est_ball->target)(2), visible(1)] in the anchor frame.
    The estimate carries distance-dependent noise, ~25 Hz refresh, latency, and
    stochastic misses (zeros + flag 0). The critic keeps the true ball_state_b.
    """
    command: KickMotionCommand = env.command_manager.get_term(command_name)
    return command.ball_obs_delayed


def ball_history_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Flattened history of perceived ball obs (window * 5), most recent last.

    Gives the actor short-term memory across perception misses.
    """
    command: KickMotionCommand = env.command_manager.get_term(command_name)
    return command.ball_obs_history
