"""Custom event terms for kick_amp (paper 2511.03996).

Standard randomization (friction/restitution/mass/com/PD gains/dof offsets/push)
uses Isaac Lab's built-in mdp.events classes, configured in env_cfg.py. Here we
only add what is task-specific:

  - reset_robot_to_motion_state : reference-state init, robot root+joints sampled
                                  from the AMP dataset, xy random in field
                                  (t1.py _reset_dofs / _reset_root_states)
  - reset_ball_random           : ball random in field, clear of posts and robot
                                  (delegates to the command's ball-reset helper)
  - record_phys_randomization   : stash the applied trunk mass/com offsets into
                                  the command for the privileged/critic obs
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils.math import quat_rotate

from .commands import (
    FIELD_HALF_LENGTH,
    FIELD_HALF_WIDTH,
    GOAL_HALF_WIDTH,
    GOAL_X,
    RESET_MOTION_MAX_FRAC,
    RESET_MOTION_STEPS,
    RESET_STAND_STEPS,
)


def randomize_rigid_body_com_partial(env, env_ids, com_range=(0.05, 0.05, 0.05), asset_cfg=None):
    """CoM randomization that supports partial env resets (beyond_mimic's is full-reset only)."""
    import isaaclab.utils.math as math_utils
    robot = env.scene["robot"]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    cpu_ids = env_ids.cpu()
    coms = robot.root_physx_view.get_coms().clone()  # (N, bodies, 7) on CPU
    body_ids = robot.find_bodies("trunk")[0] if asset_cfg is None else asset_cfg.body_ids
    lo = torch.tensor([-c for c in com_range], device="cpu")
    hi = torch.tensor([c for c in com_range], device="cpu")
    rand = math_utils.sample_uniform(lo, hi, (len(cpu_ids), 1, 3), device="cpu")
    coms[cpu_ids[:, None], torch.as_tensor(body_ids, device="cpu")[None, :], :3] += rand
    robot.root_physx_view.set_coms(coms, cpu_ids)


def _soccer_command(env: ManagerBasedEnv):
    return env.command_manager.get_term("soccer")


def reset_robot_to_motion_state(env: ManagerBasedEnv, env_ids: torch.Tensor, motion_command_name: str = "soccer"):
    """Reference-state initialization: episodes start from the robot's default
    stance, blending linearly into AMP expert motion frames between
    RESET_STAND_STEPS and RESET_MOTION_STEPS, capped at RESET_MOTION_MAX_FRAC
    (see commands.py for why neither endpoint alone trains on this robot, and
    why the blend must not reach 100% motion frames).

    Mirrors t1.py's reset (motion frame + random field pose) for the motion
    part. The AMP discriminator sees the policy's replay distribution either
    way; what the standing phase buys is that the policy is not yet falling in
    82% of episodes while its replay is what the expert is compared against.
    """
    cmd = env.command_manager.get_term("soccer")
    robot = cmd.robot

    steps = env.common_step_counter
    frac = min(max((steps - RESET_STAND_STEPS) / (RESET_MOTION_STEPS - RESET_STAND_STEPS), 0.0), 1.0)
    frac *= RESET_MOTION_MAX_FRAC
    if cmd.cfg.reset_motion_fraction is not None:
        frac = cmd.cfg.reset_motion_fraction
    use_motion = torch.rand(len(env_ids), device=env.device) < frac
    motion_ids = env_ids[use_motion]
    stand_ids = env_ids[~use_motion]

    if len(stand_ids) > 0:
        # default stance at the env origin, zero velocity. default_root_state
        # is origin-LESS (init_state) -- env_origins must be added, same as
        # the motion path below (omitting it stacks every robot at one point)
        root_state = robot.data.default_root_state[stand_ids].clone()
        root_state[:, :2] += env.scene.env_origins[stand_ids, :2]
        root_state[:, 7:] = 0.0
        robot.write_root_state_to_sim(root_state, env_ids=stand_ids)
        robot.write_joint_state_to_sim(
            robot.data.default_joint_pos[stand_ids].clone(),
            robot.data.default_joint_vel[stand_ids].clone(),
            env_ids=stand_ids,
        )

    if len(motion_ids) > 0:
        sample = cmd.motion_dataset.sample_batch(len(motion_ids))
        n = len(motion_ids)
        root_state = robot.data.default_root_state[motion_ids].clone()  # (n, 13)
        # xy random in field
        root_state[:, 0] = torch.empty(n, device=env.device).uniform_(-FIELD_HALF_LENGTH + 0.3, FIELD_HALF_LENGTH - 0.3)
        root_state[:, 1] = torch.empty(n, device=env.device).uniform_(-FIELD_HALF_WIDTH + 0.3, FIELD_HALF_WIDTH - 0.3)
        root_state[:, :2] += env.scene.env_origins[motion_ids, :2]
        # avoid the goal posts
        for post_y in (GOAL_HALF_WIDTH, -GOAL_HALF_WIDTH):
            near = (
                torch.norm(root_state[:, :2] - torch.tensor([GOAL_X, post_y], device=env.device), dim=-1) < 0.5
            ).float().unsqueeze(-1)
            root_state[:, :2] -= near * 0.5
        # height / orientation / velocity from the sampled motion frame
        root_state[:, 2] = sample["base_height"] + 0.01
        root_state[:, 3:7] = sample["base_quat"]
        root_state[:, 7:10] = quat_rotate(sample["base_quat"], sample["base_lin_vel"])
        root_state[:, 10:13] = quat_rotate(sample["base_quat"], sample["base_ang_vel"])
        robot.write_root_state_to_sim(root_state, env_ids=motion_ids)

        joint_pos = robot.data.default_joint_pos[motion_ids].clone()
        joint_vel = robot.data.default_joint_vel[motion_ids].clone()
        # head joints keep defaults; everything else from the motion frame
        # (sample dof arrays are live-robot-order, head-excluded)
        joint_pos[:, cmd.body_joint_idx] = sample["dof_pos"]
        joint_vel[:, cmd.body_joint_idx] = sample["dof_vel"]
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=motion_ids)


def reset_ball_random(env: ManagerBasedEnv, env_ids: torch.Tensor, motion_command_name: str = "soccer"):
    """Ball reset helper -- delegates to the command (keeps post/robot avoidance in one place)."""
    cmd = _soccer_command(env)
    # sync the command's robot-xy cache so the avoidance uses the fresh root state
    cmd.base_pos_xy[env_ids] = env.scene["robot"].data.root_state_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
    cmd._random_ball_reset(env_ids)


def record_phys_randomization(env: ManagerBasedEnv, env_ids: torch.Tensor, motion_command_name: str = "soccer"):
    """Record the trunk mass (normalized) for the privileged obs (critic + decoder)."""
    cmd = _soccer_command(env)
    robot = cmd.robot
    # physx view tensors live on CPU; index on CPU then move
    masses = robot.root_physx_view.get_masses()[env_ids.cpu()].to(env.device)   # (n, num_bodies)
    trunk_idx = robot.find_bodies("trunk")[0][0]
    cmd.base_mass_scaled[env_ids, 0:3] = 0.0            # com delta placeholder
    default_trunk = float(robot.data.default_mass[0, trunk_idx])
    cmd.base_mass_scaled[env_ids, 3] = masses[:, trunk_idx] / max(default_trunk, 1e-3)
