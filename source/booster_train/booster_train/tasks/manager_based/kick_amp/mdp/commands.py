"""Soccer state command for the kick_amp task (paper 2511.03996, ported from t1.py).

One CommandTerm owns everything stateful that the original Isaac Gym env kept in
its step loop:

  - ground-truth ball / goal bookkeeping (relative positions in the robot yaw frame)
  - the virtual perception system: 25 Hz updates, 6+-1 step latency,
    distance-dependent Gaussian noise sigma = 0.124 d + 0.149, FOV 87x58 deg
    w.r.t. the head camera pose, detection probability 2.475 - 0.225 d,
    1% false positives when the ball is behind-but-near
  - ball out-of-bounds / goal detection; on either, the ball alone is reset
    (robot keeps playing) -- replaces t1.py's partial reset
  - random ball teleport (p=1e-3) and kick (p=5e-3, 0.5 m/s)
  - persistent ball rolling-resistance force (re-randomized by the push event)
  - 50-step base position ring buffer for the pos_still reward
  - AMP observation vector (39-dim K1, head+arms excluded) and privileged obs (14-dim)
  - reward group 0 (goal-cluster potentials) via extras["rew_groups"] (N, 2);
    group 1 is left as a placeholder -- the runner fills it with the env's
    reward buffer (all non-goal reward terms) plus the AMP style reward

Isaac Lab computes RewardManager terms before command updates; instantaneous
geometry rewards therefore read physics directly. This command settles goal
potentials before ball interventions, then refreshes observation and force state.
"""

from __future__ import annotations

import os

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_rotate, quat_rotate_inverse, quat_mul

from .motion_lib import AmpMotionDataset
from .geometry import yaw_from_quat

# -- bisection experiment switch (KICK_AMP_MINIMAL=1) ----------------------
# Strips the task to AMP style + survival + termination so one question can be
# answered in isolation: CAN this robot learn to walk at all under these
# observations, actions and actuators? 14 versions of single-bug fixes never
# answered it, because every run debugged the full 13-reward ball+goal system.
# In minimal mode group 0 (ball/goal potentials) is forced to zero here and
# env_cfg zeroes the 11 shaping/penalty weights, so every penalty is inert no
# matter what its gate says. Motion-state resets stay on from iteration 0
# (RESET_STAND_STEPS = 0): reference-state initialization is how AMP normally
# bootstraps a gait, and it is only now safe, with the yaw quaternion fixed.
MINIMAL = os.environ.get("KICK_AMP_MINIMAL", "0") == "1"

# Walk-first curriculum (user design): one scalar c(steps) in [0, 1] drives the
# whole reward handoff. c = 0 (iter 0): gait rewards at full weight, ball/goal
# task off -- learn to walk. c = 1 (iter 4000): gait rewards faded out, the
# paper's task semantics fully on. Rationale: t1.py has NO walking reward (its
# T1 AMP bootstraps gait fast on ideal actuators), but on K1 the style signal
# alone converges to standing (measured on the minimal run: out_of_field = 0,
# policy_pred flat at -2.8, every episode = standing ~12 s then a fall). The
# missing positive step incentive is ported from legged_lab's velocity task
# (rewards.py: track_lin_vel_ball / feet_air_time_biped / feet_clearance /
# feet_slide, each scaled by 1 - c).
# v16b: stretched from 4000 iters. Under the 4000 schedule c hit 0.47 by iter
# 1900 while single_stance was still ~2% -- the gait support was being faded
# out BEFORE walking emerged, defeating the "learn to walk first" intent.
# 16000 keeps gait incentives above 60% strength for the first 6000+ iters
# and above 20% until iter 12000. If stepping appears (single_stance_frac >
# 15%), the remaining fade proceeds as designed.
TASK_RAMP_STEPS = 16000 * 24


def ball_in_goal(ball_pos_xy: torch.Tensor, ball_pos_z: torch.Tensor) -> torch.Tensor:
    """Whole ball past the goal line inside the mouth, below the bar.

    Deliberately the same geometry as the acceptance test in
    scripts/evaluate_kick_amp.py (line = GOAL_X + BALL_RADIUS, ball edge inside
    the posts, ball top below the bar), so that "the episode ended by scoring"
    means the same thing as "the evaluation counted a goal". The paper's
    GOAL_SUCCESS_STEPS (=50 consecutive steps) is a stricter test than anything
    the evaluation applies and in practice never fires -- the robot keeps playing
    after the ball enters and knocks it back out -- so it cannot be the success
    signal it is documented to be.
    """
    return (
        (ball_pos_xy[:, 0] > GOAL_X + BALL_RADIUS)
        & (ball_pos_xy[:, 1].abs() + BALL_RADIUS < GOAL_HALF_WIDTH)
        & (ball_pos_z + BALL_RADIUS < GOAL_HEIGHT)
    )


def task_curriculum(env) -> float:
    """c in [0, 1]: gait rewards scale (1 - c), task rewards scale c."""
    if env.cfg.commands.soccer.training_phase == "approach":
        return 0.0
    return min(max(env.common_step_counter / TASK_RAMP_STEPS, 0.0), 1.0)


def task_policy_weight(env) -> float:
    """Weight normalized soccer advantages without changing gait incentives.

    A floor can promote an already walking policy to soccer training while
    retaining the original gait and stagnation schedules. Zero preserves the
    original experiment; approach-only training always disables this critic.
    """
    cfg = env.cfg.commands.soccer
    floor = float(getattr(cfg, "task_weight_floor", 0.0))
    if not 0.0 <= floor <= 1.0:
        raise ValueError("task_weight_floor must be finite and within [0, 1]")
    if cfg.training_phase == "approach":
        return 0.0
    return max(task_curriculum(env), floor)

FIELD_HALF_LENGTH = 7.0   # x in [-7, 7]  (RoboCup adult-size field, 14 m)
FIELD_HALF_WIDTH = 4.5    # y in [-4.5, 4.5] (9 m)
GOAL_X = 7.0              # goal line
GOAL_HALF_WIDTH = 1.3     # goal mouth |y| < 1.3
GOAL_HEIGHT = 1.8         # crossbar height, same as the acceptance test
BALL_RADIUS = 0.11

GOAL_SUCCESS_STEPS = 50   # ball in goal this many steps => episode success

# Three separate horizons, all in policy steps (horizon_length = 24). They
# were one constant; splitting them is the point of this revision (see
# monitor.log 2026-09-15 17:5x).
#
# All curriculum components are active from iteration 0. The old iter-1000
# cliff bundled three unrelated changes (penalties, goal potential, motion
# resets) and was added to work around two bugs now fixed: a yaw quaternion
# that started 49% of motion resets upside-down, and a collision reward that
# indexed the contact sensor with articulation-order ids, so the right foot's
# normal ground contact (sensor row 16) was billed as a collision on 99.2% of
# standing steps.
WARMUP_STEPS = 0
CURRICULUM_FULL_STEPS = WARMUP_STEPS  # legacy name for config compatibility

# (2) RESET CURRICULUM -- episodes start from the robot's own default stance
# until RESET_STAND_STEPS, then blend linearly to 100% expert motion frames at
# RESET_MOTION_STEPS.
#
# Both endpoints are needed and neither alone works on this robot:
#   - 100% motion frames from step 0: the policy cannot hold a single-support
#     motion frame under K1's actuator models + 2-8 step motor delay, so ~82%
#     of episodes end in a fall (runs 09-13_19-17-17, 09-14_04-17-17, and the
#     v10 attempt at 17-16-37), the replay is dominated by falling transitions
#     and the discriminator separates within ~40 iterations.
#   - 100% standing resets forever: the policy learns to balance (deterministic
#     survival 92-98%) but the reset distribution never overlaps the expert
#     data again, so nothing ever rewards walking. That is the stand-in-place
#     policy in logs/kick_amp_render.mp4.
# Motion-state resets start ramping immediately (iteration 0), matching the
# paper's intent without jumping an untrained policy straight to 100% hard
# states. The ramp reaches its existing 50% cap at iter 3000. The cap remains
# for this single-variable run; remove it only after this corrected curriculum
# demonstrates stable recovery from expert states.
RESET_STAND_STEPS = 0
RESET_MOTION_STEPS = 3000 * 24
# (2b) ... but never all the way to 1.0.
#
# The schedule alone ramps the motion fraction to 1.0 on a fixed step count,
# with no regard for whether the policy can survive the distribution it is
# being handed. It cannot. At frac=1.0 (run 2026-09-16_01-25-31, iters
# 3000-3589) every single completed episode ended in exactly one fall
# (env/Episode_Reward/termination == -1000*dt/max_episode_length_s == -0.3333,
# i.e. exactly one termination per episode) after a mean of ~15 policy steps
# (0.30 s) -- measured from env/Episode_Reward/survival, and corroborated by
# env/Episode_Reward/pos_still being identically 0 (its ring buffer needs 50
# steps, so no env ever survived 1 s). Mean episode length was flat at 14-16
# steps for 560 iterations. When every rollout dies in the same way regardless
# of the action, the return stops depending on the action: the critic collapses
# to one value for every state, advantages degenerate, and the policy stops
# improving -- which is exactly what the flat 0.916-0.923 fall plateau shows.
# (The style term cannot compensate: the discriminator is fully saturated,
# policy_pred ~1e-13, so amp_rew = -log(1-D) ~ 0.)
#
# Capping the fraction keeps a fixed share of episodes initialized from the
# survivable default stance. That restores episodes that live long enough to
# produce a differentiated value function -- which is what makes the advantage
# signal in the hard expert-frame episodes informative too -- while the rest
# still start from expert motion frames, which is the entire point of
# reference-state init.
RESET_MOTION_MAX_FRAC = 0.5

# head camera (RealSense on aahead_pitch_link): offset in the head body frame
# and the body->optical rotation quat (wxyz) -- same convention as t1.py, which
# post-multiplies [0.5, 0.5, 0.5, 0.5].
CAMERA_OFFSET_B = (0.054, 0.0, 0.102)
# Optical frame pitched 20 deg down so ground balls at the 0.8-2.0 m spawn
# band sit inside CAMERA_FOV_V. A level frame misses d<~1.5 m (0.8 m needs
# 46 deg depression vs a 29 deg half-FOV). Matches the deploy head_cam.
CAMERA_BODY_TO_OPTICAL_WXYZ = (0.405580, 0.579228, 0.579228, 0.405580)
CAMERA_FOV_H = 87.0
CAMERA_FOV_V = 58.0


class SoccerStateCommand(CommandTerm):
    """Stateful soccer bookkeeping + virtual ball perception (see module docstring)."""

    cfg: SoccerStateCommandCfg

    def __init__(self, cfg: SoccerStateCommandCfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.robot_asset]
        self.ball: RigidObject = env.scene[cfg.ball_asset]
        self.head_idx = self.robot.find_bodies("aahead_pitch_link")[0][0]
        self.feet_idx = self.robot.find_bodies(("left_ankle_roll_link", "right_ankle_roll_link"))[0]
        # sensor-order foot indices: ContactSensor rows are NOT articulation
        # order (the collision reward learned this the hard way) -- resolve
        # through the sensor itself so the gait rewards index the right rows
        self.feet_sensor_idx = env.scene.sensors["contact_forces"].find_bodies(
            ("left_ankle_roll_link", "right_ankle_roll_link")
        )[0]
        # joint bookkeeping is NAME-based: the live joint order is the PhysX BFS
        # order, not the URDF order, so positional slicing ([ :, 2:]) is wrong
        from .motion_lib import AMP_EXCLUDED_JOINTS, HEAD_JOINTS
        joint_names = list(self.robot.joint_names)
        self.head_joint_mask = torch.tensor(
            [n in HEAD_JOINTS for n in joint_names], dtype=torch.bool, device=self.device
        )
        self.body_joint_idx = (~self.head_joint_mask).nonzero(as_tuple=False).flatten()
        # AMP obs columns: head AND arms dropped, same name-based set the expert
        # side uses (motion_lib.AmpMotionDataset.amp_dof_idx) -- the two must
        # stay column-for-column identical or the discriminator compares
        # different quantities
        self.amp_joint_idx = torch.tensor(
            [i for i, n in enumerate(joint_names) if n not in AMP_EXCLUDED_JOINTS],
            dtype=torch.long, device=self.device,
        )
        self.head_joint_idx = self.head_joint_mask.nonzero(as_tuple=False).flatten()

        # -- virtual perception buffers ---------------------------------------
        # 20-deep ball obs history: (x, y, flag); index 0 = newest
        self.ball_obs_buffer = torch.zeros(self.num_envs, 20, 3, device=self.device)
        self.ball_delay_steps = torch.full((self.num_envs,), 6, dtype=torch.long, device=self.device)
        # robot base ring buffer for pos_still: x, y, yaw
        self.base_pos_buffer = torch.zeros(self.num_envs, 50, 3, device=self.device)
        self.buffer_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.buffer_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # -- cached world state (refreshed every _update_command) -------------
        self.base_pos_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.base_yaw = torch.zeros(self.num_envs, device=self.device)
        self.relative_ball_pos = torch.zeros(self.num_envs, 2, device=self.device)
        self.relative_goal_pos = torch.zeros(self.num_envs, 2, device=self.device)
        self.last_base_pos_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.last_ball_pos_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.ball_pos_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.ball_vel_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.ball_pos_z = torch.full((self.num_envs,), BALL_RADIUS, device=self.device)
        self.camera_pos_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.ball_in_view = torch.zeros(self.num_envs, device=self.device)
        self.goal_cnt = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.goal_scored_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_success_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Strict goal test (acceptance geometry) sampled BEFORE the ball-only
        # teleport below, because that teleport would otherwise erase the goal in
        # the same step it is detected: the ball lands at a random spot and every
        # reader of the live position sees "not in the goal".
        self.ball_in_goal_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_ball_in_goal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # previous-step caches for regularization rewards (t1.py last_* pattern)
        self.last_root_vel = torch.zeros(self.num_envs, 6, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)
        # randomized trunk mass/com offsets (written by the phys-randomization event)
        self.base_mass_scaled = torch.zeros(self.num_envs, 4, device=self.device)
        # rolling-resistance magnitude; applied force vector cached for privileged obs
        self.ball_friction_force = torch.zeros(self.num_envs, device=self.device)
        self.ball_friction_force_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._push_force_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._push_torque = torch.zeros(self.num_envs, 3, device=self.device)
        # trunk body index for the base push disturbance (t1.py pushes the base)
        self.trunk_idx = self.robot.find_bodies("trunk")[0][0]

        # -- AMP expert dataset (reference-state init + expert pairs) ---------
        self.motion_dataset = AmpMotionDataset(cfg.motion_dir, joint_names, device=self.device)

        self.metrics["goal"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return "SoccerState"

    # ------------------------------------------------------------------ utils

    def _rotate_to_yaw_frame(self, delta_xy: torch.Tensor) -> torch.Tensor:
        cos_y = torch.cos(-self.base_yaw)
        sin_y = torch.sin(-self.base_yaw)
        return torch.stack(
            (cos_y * delta_xy[:, 0] - sin_y * delta_xy[:, 1], sin_y * delta_xy[:, 0] + cos_y * delta_xy[:, 1]),
            dim=-1,
        )

    def spawn_bearing(self) -> tuple[float, float]:
        """Angular half-range of the spawn direction, in the robot's heading frame.

        The default cone is +-0.8 rad, which never shows the policy a ball beside
        or behind it. The acceptance protocol places the ball at 0/+-90/180 deg
        from the robot's heading (evaluate_kick_amp.py::scenario_layout), so half
        of every graded cohort tests a geometry the policy was never trained on --
        and that is exactly where the first-episode falls concentrate (34 of 39)
        and where the goal rate is lowest. ``ball_spawn_bearing`` widens it.
        """
        if self.cfg.ball_spawn_bearing is None:
            return (-0.8, 0.8)
        return self.cfg.ball_spawn_bearing

    def _random_ball_reset(self, env_ids: torch.Tensor):
        """Resets balls (only) to a random in-field spot, clear of the posts and the robot."""
        n = len(env_ids)
        xy = torch.empty(n, 2, device=self.device)
        xy[:, 0] = torch.empty(n, device=self.device).uniform_(-FIELD_HALF_LENGTH + 0.3, FIELD_HALF_LENGTH - 0.3)
        xy[:, 1] = torch.empty(n, device=self.device).uniform_(-FIELD_HALF_WIDTH + 0.3, FIELD_HALF_WIDTH - 0.3)
        if self.cfg.ball_spawn_distance is not None:
            # Optional approach curriculum: ball lies ahead in the reset
            # robot's heading frame, outside its support polygon.
            radius = torch.empty(n, device=self.device).uniform_(*self.cfg.ball_spawn_distance)
            heading = yaw_from_quat(self.robot.data.root_quat_w[env_ids])
            heading += torch.empty(n, device=self.device).uniform_(*self.spawn_bearing())

            xy = self.base_pos_xy[env_ids] + radius[:, None] * torch.stack((heading.cos(), heading.sin()), dim=-1)
            xy[:, 0].clamp_(-FIELD_HALF_LENGTH + 0.3, FIELD_HALF_LENGTH - 0.3)
            xy[:, 1].clamp_(-FIELD_HALF_WIDTH + 0.3, FIELD_HALF_WIDTH - 0.3)
        # nudge out of the two goal-mouth posts
        for post_y in (GOAL_HALF_WIDTH, -GOAL_HALF_WIDTH):
            near = (torch.norm(xy - torch.tensor([GOAL_X, post_y], device=self.device), dim=-1) < 0.5).float().unsqueeze(-1)
            xy -= near * 0.5
        # keep away from the robot
        near_robot = (torch.norm(xy - self.base_pos_xy[env_ids], dim=-1) < 0.5).float().unsqueeze(-1)
        xy -= near_robot * 0.5
        root = self.ball.data.default_root_state[env_ids].clone()
        root[:, :2] = xy + self._env.scene.env_origins[env_ids, :2]
        root[:, 2] = BALL_RADIUS + 0.005
        root[:, 7:] = 0.0
        self.ball.write_root_state_to_sim(root, env_ids=env_ids)

    # ---------------------------------------------------------------- command

    def _refresh_ball_state(self):
        """Synchronize ground truth after physics or a ball-only intervention."""
        root = self.ball.data.root_state_w
        self.ball_pos_xy[:] = root[:, :2] - self._env.scene.env_origins[:, :2]
        self.ball_pos_z[:] = root[:, 2]
        self.ball_vel_xy[:] = root[:, 7:9]
        self.relative_ball_pos[:] = self._rotate_to_yaw_frame(self.ball_pos_xy - self.base_pos_xy)

    def _update_command(self):
        env = self._env
        dt = env.step_dt
        env_ids_all = torch.arange(self.num_envs, device=self.device)
        # -- refresh cached world state ---------------------------------------
        base_root = self.robot.data.root_state_w
        self.base_pos_xy[:] = base_root[:, :2] - env.scene.env_origins[:, :2]
        self.base_yaw[:] = yaw_from_quat(base_root[:, 3:7])
        self._refresh_ball_state()
        self.relative_goal_pos[:] = self._rotate_to_yaw_frame(
            torch.tensor([GOAL_X, 0.0], device=self.device).expand(self.num_envs, -1) - self.base_pos_xy
        )

        head_state = self.robot.data.body_state_w[:, self.head_idx, :]
        camera_offset = torch.tensor(CAMERA_OFFSET_B, device=self.device).expand(self.num_envs, -1)
        camera_pos_w = head_state[:, :3] + quat_rotate(head_state[:, 3:7], camera_offset)
        self.camera_pos_xy[:] = camera_pos_w[:, :2] - env.scene.env_origins[:, :2]

        # -- ball out-of-bounds / goal, ball-only reset ------------------------
        ball_out = (
            (self.ball_pos_xy[:, 0] < -FIELD_HALF_LENGTH)
            | ((self.ball_pos_xy[:, 0] > GOAL_X) & (torch.abs(self.ball_pos_xy[:, 1]) > GOAL_HALF_WIDTH))
            | (torch.abs(self.ball_pos_xy[:, 1]) > FIELD_HALF_WIDTH)
        )
        in_goal = (self.ball_pos_xy[:, 0] > GOAL_X) & (torch.abs(self.ball_pos_xy[:, 1]) < GOAL_HALF_WIDTH)
        self.goal_cnt[:] = torch.where(in_goal, self.goal_cnt + 1, torch.zeros_like(self.goal_cnt))
        self.goal_scored_now[:] = in_goal & ~ball_out
        self.goal_success_now[:] = self.goal_cnt >= GOAL_SUCCESS_STEPS
        self.ball_in_goal_now[:] = ball_in_goal(self.ball_pos_xy, self.ball_pos_z)

        # Settle the physical transition before resets or external kicks. In
        # particular, retain the final in-goal reward and any real progress on
        # a teleport step; the teleport itself never contributes a potential.
        s = self.cfg
        ball_dist_rew = (
            torch.norm(self.ball_pos_xy - self.last_base_pos_xy, dim=-1)
            - torch.norm(self.ball_pos_xy - self.base_pos_xy, dim=-1)
        ) * s.ball_distance_scale * dt
        goal_rew = self.goal_scored_now.float() * s.goal_scale * dt
        goal_center = torch.tensor([GOAL_X, 0.0], device=self.device)
        last_in_goal = self.last_ball_in_goal.float()
        cur_in_goal = self.goal_scored_now.float()
        last_pot = torch.norm(self.last_ball_pos_xy - goal_center, dim=-1) * (1.0 - last_in_goal)
        cur_pot = torch.norm(self.ball_pos_xy - goal_center, dim=-1) * (1.0 - cur_in_goal)
        goal_dist_rew = (last_pot - cur_pot) * s.goal_distance_scale * dt
        if MINIMAL or self.cfg.training_phase == "approach":
            # bisection: no task reward at all, AMP style + survival only
            env.extras["rew_groups"] = torch.zeros(self.num_envs, 2, device=self.device)
        else:
            env.extras["rew_groups"] = torch.stack(
                (goal_rew + ball_dist_rew + goal_dist_rew, torch.zeros_like(goal_rew)), dim=-1
            )
        # Apply curriculum AFTER advantage normalization in the runner.
        # Scaling this entire reward group is cancelled by its own std.
        env.extras["task_weight"] = 0.0 if MINIMAL else task_policy_weight(env)
        # a just-reset env computes its potentials from the zeroed last_*
        # caches: that is a +/-tens phantom on the TERMINAL transition
        # (Isaac Lab resets before the command update; t1.py rewards before
        # reset). Mask group 0 on the reset step -- falling must never pay.
        env.extras["rew_groups"][env.episode_length_buf == 0, 0] = 0.0

        reset_ball_ids = (ball_out | self.goal_success_now).nonzero(as_tuple=False).flatten()
        if self.cfg.ball_reset_enabled and len(reset_ball_ids) > 0:
            self._random_ball_reset(reset_ball_ids)
            self.goal_cnt[reset_ball_ids] = 0

        # -- random ball interventions (t1.py _kick_balls / _reset_balls) ------
        teleport = (torch.rand(self.num_envs, device=self.device) < self.cfg.ball_teleport_prob).nonzero(as_tuple=False).flatten()
        if len(teleport) > 0:
            self._random_ball_reset(teleport)
            self.goal_cnt[teleport] = 0
        # A teleport may have stopped a previously moving ball. Refresh before
        # deciding whether the independent random-kick intervention applies.
        self._refresh_ball_state()
        kicked = (torch.rand(self.num_envs, device=self.device) < self.cfg.ball_kick_prob).nonzero(as_tuple=False).flatten()
        if len(kicked) > 0:
            slow = (torch.norm(self.ball_vel_xy[kicked], dim=-1) < 0.1).float().unsqueeze(-1)
            impulse = torch.randn(len(kicked), 2, device=self.device) * self.cfg.ball_kick_vel
            root = self.ball.data.root_state_w[kicked].clone()
            root[:, 7:9] += slow * impulse
            self.ball.write_root_state_to_sim(root, env_ids=kicked)

        # Everything describing the next state must agree with physics,
        # including height, velocity, decoder targets and perfect perception.
        self._refresh_ball_state()

        # -- virtual perception (t1.py _compute_observations) ------------------
        cam_quat = quat_mul(
            head_state[:, 3:7],
            torch.tensor(CAMERA_BODY_TO_OPTICAL_WXYZ, device=self.device).expand(self.num_envs, -1),
        )
        ball_pos_w = torch.stack(
            (self.ball_pos_xy[:, 0] + env.scene.env_origins[:, 0],
             self.ball_pos_xy[:, 1] + env.scene.env_origins[:, 1],
             self.ball_pos_z),
            dim=-1,
        )
        ball_to_camera = quat_rotate_inverse(cam_quat, ball_pos_w - camera_pos_w)
        fwd = ball_to_camera[:, 2].clamp(min=1e-8)
        in_fov = (
            (ball_to_camera[:, 2] > 0)
            & (torch.abs(torch.atan2(ball_to_camera[:, 0], fwd)) < 0.5 * CAMERA_FOV_H / 180.0 * torch.pi)
            & (torch.abs(torch.atan2(ball_to_camera[:, 1], fwd)) < 0.5 * CAMERA_FOV_V / 180.0 * torch.pi)
        ).float()
        dist = torch.norm(self.relative_ball_pos, dim=-1)
        detect = torch.rand(self.num_envs, device=self.device) < (2.475 - 0.225 * dist.clamp(min=7.0))
        self.ball_in_view = in_fov * detect.float() + (1.0 - in_fov) * (
            self.relative_ball_pos[:, 0] > -2.0
        ).float() * (torch.rand(self.num_envs, device=self.device) < 0.01).float()

        # 25 Hz refresh (every other policy step at 50 Hz), latency via delayed read
        self.ball_obs_buffer = torch.roll(self.ball_obs_buffer, 1, dims=1)
        refresh = env.episode_length_buf % 2 == 0
        sigma = 0.149 + 0.124 * dist
        noisy = self.relative_ball_pos + sigma.unsqueeze(-1) * torch.randn_like(self.relative_ball_pos)
        self.ball_obs_buffer[refresh, 0, 0:2] = self.ball_in_view[refresh].unsqueeze(-1) * noisy[refresh]
        self.ball_obs_buffer[refresh, 0, 2] = self.ball_in_view[refresh]
        self.ball_obs_buffer[~refresh, 0, :] = self.ball_obs_buffer[~refresh, 1, :]

        # -- persistent ball rolling resistance (resampled every 5 s, t1.py L518) --
        if env.common_step_counter % int(5.0 / dt) == 0:
            self.ball_friction_force.uniform_(*self.cfg.ball_friction_range)
        speed = torch.norm(self.ball_vel_xy, dim=-1).clip(min=0.1)
        self.ball_friction_force_xy[:] = -self.ball_friction_force.unsqueeze(-1) * self.ball_vel_xy / speed.unsqueeze(-1)
        ball_force = torch.zeros(self.num_envs, 3, device=self.device)
        ball_force[:, :2] = self.ball_friction_force_xy
        # world-frame force: the default local frame of a rolling ball rotates
        # with its spin and partially cancels the resistance direction
        self.ball.set_external_force_and_torque(
            ball_force.unsqueeze(1), torch.zeros(self.num_envs, 1, 3, device=self.device), is_global=True
        )

        # 20 N base push for 1 s every 5 s (t1.yaml push_force / push_duration):
        # t1.py _push_robots pushes the ROBOT TRUNK -- never the 0.43 kg ball
        # (20 N on the ball is ~45 m/s^2 and hurls it across the field)
        push_active = self.cfg.push_enabled and (env.common_step_counter % int(5.0 / dt)) < int(1.0 / dt)
        if push_active and env.common_step_counter % int(5.0 / dt) == 0:
            self._push_force_xy = torch.randn(self.num_envs, 2, device=self.device).clamp(-2, 2) * 10.0
            self._push_torque = torch.randn(self.num_envs, 3, device=self.device).clamp(-1, 1) * 2.0
        robot_force = torch.zeros(self.num_envs, 3, device=self.device)
        robot_torque = torch.zeros(self.num_envs, 3, device=self.device)
        if push_active:
            robot_force[:, :2] = self._push_force_xy
            robot_torque[:] = self._push_torque
        self.robot.set_external_force_and_torque(
            robot_force.unsqueeze(1),
            robot_torque.unsqueeze(1),
            body_ids=[int(self.trunk_idx)],
            is_global=True,
        )

        # -- pos_still ring buffer ---------------------------------------------
        self.base_pos_buffer[env_ids_all, self.buffer_idx, 0:2] = self.base_pos_xy
        self.base_pos_buffer[env_ids_all, self.buffer_idx, 2] = self.base_yaw
        self.buffer_count[:] = torch.clamp(self.buffer_count + 1, max=self.base_pos_buffer.shape[1])
        self.buffer_idx[:] = (self.buffer_idx + 1) % self.base_pos_buffer.shape[1]

        # -- per-step bridge to the AMP runner ----------------------------------
        env.extras["amp_obs"] = self._compute_amp_obs()
        env.extras["privileged_obs"] = self._compute_privileged_obs()
        # Same predicate as the goal termination and the acceptance test: with the
        # 50-step rule the training success rate would read ~0 once the episode
        # ends at the goal, blinding the only training-side progress signal.
        env.extras["success"] = self.ball_in_goal_now.float()

        # -- advance last_* caches ----------------------------------------------
        self.last_base_pos_xy[:] = self.base_pos_xy
        self.last_ball_pos_xy[:] = self.ball_pos_xy
        # The transition event remains available for metrics, while the next
        # potential starts at the post-intervention ball state.
        self.last_ball_in_goal[:] = (
            (self.ball_pos_xy[:, 0] > GOAL_X)
            & (torch.abs(self.ball_pos_xy[:, 1]) < GOAL_HALF_WIDTH)
        )
        self.last_root_vel[:] = env.scene["robot"].data.root_state_w[:, 7:13]
        self.last_actions[:] = env.action_manager.action

        # walk-curriculum diagnostic: fraction of envs currently in single
        # stance, computed from raw forces (NOT from the air-time tracking the
        # reward uses). feet_air_time_biped sat at exactly 0 for 800 iters --
        # this distinguishes "the robot never lifts one foot" (frac == 0) from
        # "the sensor's air-time bookkeeping is broken" (frac > 0 but reward 0).
        _contacts = env.scene.sensors["contact_forces"].data.net_forces_w[:, self.feet_sensor_idx, :].norm(dim=-1) > 1.0
        env.extras.setdefault("log", {})["single_stance_frac"] = (_contacts.sum(dim=-1) == 1).float().mean()
        delta = self.ball_pos_xy - self.base_pos_xy
        distance = delta.norm(dim=-1)
        velocity = self.robot.data.root_lin_vel_w[:, :2]
        log = env.extras["log"]
        log["toward_ball_speed_mps"] = ((delta / distance[:, None].clamp_min(1e-6)) * velocity).sum(-1).mean()
        log["planar_speed_mps"] = velocity.norm(dim=-1).mean()
        log["ball_distance_m"] = distance.mean()
        log["base_height_m"] = self.robot.data.root_pos_w[:, 2].mean()
        log["task_weight"] = env.extras["task_weight"]
        # The gait and stagnation terms scale by the *raw* curriculum, not by the
        # floored policy weight (see task_policy_weight), so log the raw value
        # too. With a non-zero task_weight_floor the two diverge and a matched
        # A/B could not otherwise confirm from the logs that the gait and
        # stagnation schedules were left untouched.
        log["task_curriculum_raw"] = task_curriculum(env)

    @property
    def command(self) -> torch.Tensor:
        """Command tensor placeholder (the delayed ball obs doubles as the command)."""
        return self.ball_obs

    def _update_metrics(self):
        self.metrics["goal"][:] = self.goal_scored_now.float()
        self.metrics["success"][:] = self.ball_in_goal_now.float()

    def _compute_amp_obs(self) -> torch.Tensor:
        """39-dim AMP obs: gravity3 + lin_vel3 + ang_vel3 + dof12 + dof_vel12 + feet_rel6.

        Head AND arms are excluded (motion_lib.AMP_EXCLUDED_JOINTS) -- the arm
        columns were a free giveaway for the discriminator, see the measurement
        in motion_lib. dof columns are the 12 leg joints only.
        """
        robot = self.robot
        root_quat = robot.data.root_quat_w
        gravity_b = quat_rotate_inverse(root_quat, torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1))
        # head joints are excluded NAME-based: the live joint order is PhysX
        # BFS (aahead_yaw at 0, aahead_pitch at 5), never URDF order -- the
        # columns must match the expert dataset built in motion_lib
        # head+arm joints excluded NAME-based (amp_joint_idx): the live joint
        # order is PhysX BFS, never URDF order, and the columns must match the
        # expert dataset built in motion_lib
        dof_pos = robot.data.joint_pos[:, self.amp_joint_idx]
        dof_vel = robot.data.joint_vel[:, self.amp_joint_idx] * 0.1
        feet_pos = robot.data.body_pos_w[:, self.feet_idx, :]
        feet_rel = quat_rotate_inverse(
            root_quat.unsqueeze(1).expand(-1, 2, -1), feet_pos - robot.data.root_pos_w.unsqueeze(1)
        ).flatten(1)
        return torch.cat((gravity_b, robot.data.root_lin_vel_b, robot.data.root_ang_vel_b, dof_pos, dof_vel, feet_rel), dim=-1)

    def _compute_privileged_obs(self) -> torch.Tensor:
        """14-dim decoder reconstruction target (T1.yaml privileged set)."""
        return torch.cat(
            (
                self.base_mass_scaled,               # 4
                self.robot.data.root_lin_vel_b,      # 3
                self.robot.data.root_pos_w[:, 2:3],  # 1
                self.ball_friction_force_xy,         # 2
                self.relative_ball_pos,              # 2
                self.ball_vel_xy,                    # 2
            ),
            dim=-1,
        )

    @property
    def ball_obs(self) -> torch.Tensor:
        """Delayed, noisy ball observation (x, y, flag) drawn from the history buffer."""
        if self.cfg.perfect_perception:
            return torch.cat((self.relative_ball_pos, torch.ones(self.num_envs, 1, device=self.device)), dim=-1)
        env_ids = torch.arange(self.num_envs, device=self.device)
        return self.ball_obs_buffer[env_ids, self.ball_delay_steps, :]

    # ------------------------------------------------------------------- reset

    def _resample_command(self, env_ids: torch.Tensor):
        """Per-episode reset of perception/buffer state (t1.py _reset_idx equivalent).

        CommandTerm's reset path is reset -> _resample -> _resample_command;
        there is no _reset_idx hook (an earlier override of that name was dead code).
        """
        n = len(env_ids)
        self.ball_obs_buffer[env_ids] = 0.0
        self.ball_delay_steps[env_ids] = torch.clamp(torch.randn(n, device=self.device) + 6.0, min=0, max=19).long()
        self.base_pos_buffer[env_ids] = 0.0
        self.buffer_count[env_ids] = 0
        self.buffer_idx[env_ids] = 0
        self.goal_cnt[env_ids] = 0
        self.goal_scored_now[env_ids] = False
        self.goal_success_now[env_ids] = False
        self.ball_in_goal_now[env_ids] = False
        self.last_ball_in_goal[env_ids] = False
        # caches resync on the first _update_command after reset
        self.last_base_pos_xy[env_ids] = 0.0
        self.last_ball_pos_xy[env_ids] = 0.0


@configclass
class SoccerStateCommandCfg(CommandTermCfg):
    class_type = SoccerStateCommand
    robot_asset: str = "robot"
    ball_asset: str = "ball"
    training_phase: str = "soccer"
    task_weight_floor: float = 0.0
    reset_motion_fraction: float | None = None
    motion_dir: str = ""  # AMP dataset root (walk/ + kick/ npz subdirs)

    # goal-critic reward scales (per second; dt applied here like the RewardManager)
    goal_scale: float = 15.0
    ball_distance_scale: float = 50.0
    goal_distance_scale: float = 500.0

    ball_teleport_prob: float = 1.0e-3
    ball_kick_prob: float = 5.0e-3
    ball_kick_vel: float = 0.5
    push_enabled: bool = True
    ball_reset_enabled: bool = True
    ball_friction_range: tuple[float, float] = (0.1, 0.3)
    perfect_perception: bool = False
    ball_spawn_distance: tuple[float, float] | None = None
    # Angular half-range of the spawn direction in the robot's heading frame;
    # None keeps the original +-0.8 rad cone. (radians(-v), radians(v)) spans the
    # full circle at v = 180, matching the acceptance protocol's bearings.
    ball_spawn_bearing: tuple[float, float] | None = None
