"""Kick motion command: BeyondMimic motion tracking + soccer ball state machine.

Ported from whole_body_tracking/tasks/track_kick_football/mdp/commands.py
(KickMotionCommand and KickMotionCommandCfg only, lines 380-733) onto the
booster_train beyond_mimic MotionCommand base class. The base-class hooks it
overrides (_adaptive_sampling, _resample_command, _update_command,
_update_metrics, _set_debug_vis_impl, _debug_vis_callback) are interface-
compatible between the two repos (same BeyondMimic lineage); the call chain
_resample_command -> _adaptive_sampling guarantees the start-time clamp is
applied before the robot state is written from the sampled frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_inv, yaw_quat
from isaaclab.utils.math import sample_uniform

from booster_train.tasks.manager_based.beyond_mimic.mdp.commands import (
    MotionCommand,
    MotionCommandCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class KickMotionCommand(MotionCommand):
    """Motion command extended with ball and kick-target state.

    On top of plain motion tracking (inherited unchanged), every episode:
      1. the start time is clamped to the first ``start_time_fraction`` of the
         reference so the kick always happens inside the episode;
      2. the ball is spawned near the reference kick contact point (world
         offset ``ball_offset`` + uniform noise); the reference motion faces +y;
      3. a target is anchored at ``ball + target_distance`` along an azimuth
         sampled from ``+/- target_azimuth_range`` around the motion heading,
         so the required kick direction changes every episode (identifiability);
      4. the running minimum ball-target distance is tracked so goal rewards
         can pay irreversible progress without penalising overshoot.

    The actor never sees the true ball state: a virtual perception system
    (paper arXiv:2511.03996, parameters measured on a real robot) exposes a
    noisy, low-rate, latency-delayed, sometimes-missing ball estimate plus a
    visibility flag and a 1 s history of that estimate. The critic keeps the
    true state (asymmetric actor-critic).
    """

    cfg: KickMotionCommandCfg

    # ring buffer length for perceived ball observations (history + latency headroom)
    BALL_RING_LENGTH = 64

    def __init__(self, cfg: KickMotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.ball: RigidObject = env.scene[cfg.ball_name]
        self.main_foot_index = self.robot.body_names.index(cfg.main_foot_name)

        self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_azimuth = torch.zeros(self.num_envs, device=self.device)
        self.ball_init_target_dist = torch.full((self.num_envs,), cfg.target_distance, device=self.device)
        self._min_ball_target_dist = torch.full((self.num_envs,), cfg.target_distance, device=self.device)
        self._progress = torch.zeros(self.num_envs, device=self.device)
        self._progress_delta = torch.zeros(self.num_envs, device=self.device)
        self._contact_paid = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # robot->ball approach (potential-based reward state, frozen once the ball moves)
        self._anchor_ball_dist = torch.zeros(self.num_envs, device=self.device)
        self._approach_delta = torch.zeros(self.num_envs, device=self.device)

        # one-shot kick-alignment event flag (pays on the ball's speed rising edge)
        self._alignment_paid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # -- virtual perception state --------------------------------------
        # latest perceived ball obs [est_pos_b(2), dir_b(2), visible(1)]
        self._perceived_ball = torch.zeros(self.num_envs, 5, device=self.device)
        self._perception_countdown = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._perception_delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # ring of perceived obs; index arithmetic in [0, BALL_RING_LENGTH)
        self._ball_ring = torch.zeros(self.num_envs, self.BALL_RING_LENGTH, 5, device=self.device)
        self._ring_step = 0
        self._env_ids = torch.arange(self.num_envs, device=self.device)

        # -- stagnation detection ------------------------------------------
        self._anchor_xy_ring = torch.zeros(self.num_envs, cfg.stagnation_window, 2, device=self.device)
        self._anchor_ring_idx = 0
        self._stagnant = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.metrics["ball_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ball_max_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ball_target_dist"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ball_visible"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["kick_accuracy"] = torch.zeros(self.num_envs, device=self.device)

        # one-shot kick event bookkeeping (reward reads, curriculum gates on it)
        self._alignment_fire = torch.zeros(self.num_envs, device=self.device)
        self._alignment_kernel = torch.zeros(self.num_envs, device=self.device)
        self._kick_accuracy_ema = torch.zeros(self.num_envs, device=self.device)

    # -- ball state (world / anchor frame) -----------------------------------

    @property
    def ball_pos_w(self) -> torch.Tensor:
        return self.ball.data.root_pos_w

    @property
    def ball_vel_w(self) -> torch.Tensor:
        return self.ball.data.root_lin_vel_w

    @property
    def ball_pos_b(self) -> torch.Tensor:
        """Ball position in the robot anchor frame (xy is what matters, ball stays on ground)."""
        return quat_apply(quat_inv(self.robot_anchor_quat_w), self.ball_pos_w - self.robot_anchor_pos_w)

    @property
    def ball_to_target_dir_b(self) -> torch.Tensor:
        """Unit xy direction ball -> target in the robot anchor frame."""
        direction = self.target_pos_w - self.ball_pos_w
        direction[..., 2] = 0.0
        direction = torch.nn.functional.normalize(direction, dim=-1, eps=1e-6)
        return quat_apply(quat_inv(yaw_quat(self.robot_anchor_quat_w)), direction)[..., :2]

    @property
    def ball_vel_b(self) -> torch.Tensor:
        return quat_apply(quat_inv(self.robot_anchor_quat_w), self.ball_vel_w)[..., :2]

    @property
    def main_foot_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.main_foot_index]

    # -- per-episode resampling ----------------------------------------------

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        """Sample start bins restricted to the beginning of the motion.

        The clamp MUST live here, before ``_resample_command`` writes the robot
        state from the sampled frame -- clamping afterwards would write the
        robot at an arbitrary (often post-kick) pose while relabelling time to
        the clamped frame, leaving the ball (spawned at the fixed world offset)
        behind the robot for most episodes.
        """
        super()._adaptive_sampling(env_ids)
        max_start = int(self.cfg.start_time_fraction * (self.motion.time_step_total - 1))
        self.time_steps[env_ids] = torch.clamp(self.time_steps[env_ids], max=max_start)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if len(env_ids) == 0:
            return
        self._spawn_ball_and_target(env_ids)

    def _spawn_ball_and_target(self, env_ids: Sequence[int]):
        n = len(env_ids)
        ball_pos = self._env.scene.env_origins[env_ids] + torch.tensor(self.cfg.ball_offset, device=self.device)
        ball_pos[:, 0] += sample_uniform(*self.cfg.ball_lateral_range, (n,), device=self.device)
        ball_pos[:, 1] += sample_uniform(*self.cfg.ball_forward_range, (n,), device=self.device)
        identity_quat = torch.zeros(n, 4, device=self.device)
        identity_quat[:, 3] = 1.0
        zero_vel = torch.zeros(n, 3, device=self.device)
        self.ball.write_root_state_to_sim(
            torch.cat([ball_pos, identity_quat, zero_vel, zero_vel], dim=-1), env_ids=env_ids
        )

        # target anchored at the ball: azimuth sampled around the motion heading (+y)
        azimuth = sample_uniform(-self.cfg.target_azimuth_range, self.cfg.target_azimuth_range, (n,), device=self.device)
        self.target_azimuth[env_ids] = azimuth
        direction = torch.nn.functional.pad(torch.stack([-torch.sin(azimuth), torch.cos(azimuth)], dim=-1), (0, 1))
        self.target_pos_w[env_ids] = ball_pos + self.cfg.target_distance * direction

        self.ball_init_target_dist[env_ids] = self.cfg.target_distance
        self._min_ball_target_dist[env_ids] = self.cfg.target_distance
        self._progress[env_ids] = 0.0
        self._progress_delta[env_ids] = 0.0
        self._contact_paid[env_ids] = 0
        self._alignment_paid[env_ids] = False
        self._anchor_ball_dist[env_ids] = torch.norm(
            ball_pos[:, :2] - self.robot_anchor_pos_w[env_ids, :2], dim=-1
        )
        self._approach_delta[env_ids] = 0.0
        self.metrics["ball_max_speed"][env_ids] = 0.0

        # fresh perception immediately at episode start; seed the history ring of
        # the RESET envs only -- writing [:] would flatten every other env's
        # history and disable the latency/25 Hz models for them
        self._update_ball_perception(env_ids)
        self._ball_ring[env_ids] = self._perceived_ball[env_ids].unsqueeze(1)
        self._anchor_xy_ring[env_ids] = self.robot_anchor_pos_w[env_ids, None, :2]

    # -- per-step update -------------------------------------------------------

    def _update_command(self):
        super()._update_command()

        # irreversible progress towards the target: pay once on approach, free on overshoot
        ball_target_dist = torch.norm(self.ball_pos_w - self.target_pos_w, dim=-1)
        self._min_ball_target_dist = torch.minimum(self._min_ball_target_dist, ball_target_dist)
        progress = (self.ball_init_target_dist - self._min_ball_target_dist).clamp(min=0.0)
        self._progress_delta = progress - self._progress
        self._progress = progress

        # robot->ball approach delta (potential-based, signed). Frozen once the
        # ball is moving: after a successful kick the ball flying away would
        # otherwise tax the robot for the distance it created (telescoping to
        # roughly -target_distance over the episode).
        ball_moving = torch.norm(self.ball_vel_w[:, :2], dim=-1) > 0.5
        anchor_ball_dist = torch.norm(self.ball_pos_w[:, :2] - self.robot_anchor_pos_w[:, :2], dim=-1)
        self._approach_delta = torch.where(
            ball_moving, torch.zeros_like(anchor_ball_dist), self._anchor_ball_dist - anchor_ball_dist
        )
        self._anchor_ball_dist = torch.where(ball_moving, self._anchor_ball_dist, anchor_ball_dist)

        # virtual perception + history ring + stagnation
        self._update_ball_perception()
        self._ball_ring[:, self._ring_step % self.BALL_RING_LENGTH] = self._perceived_ball
        self._ring_step += 1
        self._update_stagnation()

        # one-shot kick-alignment event: on the ball-speed rising edge, record the
        # angle kernel between the outgoing direction and the ball->target direction.
        # The reward pays it once; the annealing curriculum gates on its EMA.
        ball_speed_xy = torch.norm(self.ball_vel_w[:, :2], dim=-1)
        fire = (ball_speed_xy > 1.0) & ~self._alignment_paid
        self._alignment_paid |= fire
        self._alignment_fire = fire.float()
        self._alignment_kernel = torch.zeros_like(self._alignment_fire)
        if torch.any(fire):
            rows = fire.nonzero(as_tuple=False).flatten()
            vel_xy = self.ball_vel_w[rows, :2]
            outgoing = torch.nn.functional.normalize(vel_xy, dim=-1, eps=1e-6)
            to_target = self.target_pos_w[rows, :2] - self.ball_pos_w[rows, :2]
            to_target = torch.nn.functional.normalize(to_target, dim=-1, eps=1e-6)
            cos = torch.sum(outgoing * to_target, dim=-1).clamp(-1.0, 1.0)
            kernel = torch.exp(-((torch.acos(cos) / self.cfg.alignment_std) ** 2))
            self._alignment_kernel[rows] = kernel
            ema_alpha = 0.02
            self._kick_accuracy_ema[rows] = (
                ema_alpha * kernel + (1 - ema_alpha) * self._kick_accuracy_ema[rows]
            )
        self.metrics["kick_accuracy"][:] = self._kick_accuracy_ema

    def _update_ball_perception(self, refresh_ids: torch.Tensor | None = None):
        """Advance the virtual perception of the ball (actor-side observation).

        Models four characteristics of onboard vision measured in
        arXiv:2511.03996: detection probability, distance-dependent Gaussian
        noise, reduced update frequency (~25 Hz vs 50 Hz control), and latency.
        Between perception refreshes the last estimate is held; on a miss the
        position is zeroed and the visibility flag drops to 0.

        Called every control step without arguments (advance the perception
        clock for all envs), or with explicit ``refresh_ids`` at episode reset
        to force a fresh perception for just those envs without touching the
        countdowns of the others.
        """
        if refresh_ids is None:
            self._perception_countdown -= 1
            fresh_ids = (self._perception_countdown <= 0).nonzero(as_tuple=False).flatten()
        else:
            fresh_ids = refresh_ids
        if fresh_ids.numel() == 0:
            return
        n = fresh_ids.numel()
        cfg = self.cfg

        # next refresh in control steps: N(freq_hz) converted at the control rate
        control_dt = self._env.cfg.decimation * self._env.cfg.sim.dt
        freq_hz = torch.randn(n, device=self.device) * cfg.ball_update_freq_hz[1] + cfg.ball_update_freq_hz[0]
        period = (1.0 / freq_hz / control_dt).round().clamp(min=1)
        # per-env pipeline latency in control steps
        delay_ms = torch.randn(n, device=self.device) * cfg.ball_latency_ms[1] + cfg.ball_latency_ms[0]
        self._perception_delay[fresh_ids] = (delay_ms / 1000.0 / (self._env.cfg.decimation * self._env.cfg.sim.dt)).round().clamp(min=0, max=self.BALL_RING_LENGTH - 2).long()
        self._perception_countdown[fresh_ids] = period.long()

        # detection roll; on miss -> zeros + invisible flag
        detected = torch.rand(n, device=self.device) < cfg.ball_detection_prob

        # distance-dependent Gaussian noise on the perceived position
        distance = torch.norm(self.ball_pos_w[fresh_ids, :2] - self.robot_anchor_pos_w[fresh_ids, :2], dim=-1)
        sigma = cfg.ball_noise_slope * distance + cfg.ball_noise_base
        est_pos_w = self.ball_pos_w[fresh_ids].clone()
        est_pos_w[:, :2] += torch.randn(n, 2, device=self.device) * sigma.unsqueeze(-1)

        # express in the robot anchor frame (position: full rotation, direction: yaw only)
        est_pos_b = quat_apply(quat_inv(self.robot_anchor_quat_w[fresh_ids]), est_pos_w - self.robot_anchor_pos_w[fresh_ids])
        direction = self.target_pos_w[fresh_ids] - est_pos_w
        direction[:, 2] = 0.0
        direction = torch.nn.functional.normalize(direction, dim=-1, eps=1e-6)
        dir_b = quat_apply(quat_inv(yaw_quat(self.robot_anchor_quat_w[fresh_ids])), direction)[:, :2]

        self._perceived_ball[fresh_ids] = 0.0
        visible_rows = detected.nonzero(as_tuple=False).flatten()
        rows = fresh_ids[visible_rows]
        self._perceived_ball[rows, 0:2] = est_pos_b[visible_rows, :2]
        self._perceived_ball[rows, 2:4] = dir_b[visible_rows]
        self._perceived_ball[rows, 4] = 1.0

    def _update_stagnation(self):
        """Flag envs whose anchor barely moved over the trailing window (reward farming guard)."""
        window = self.cfg.stagnation_window
        self._anchor_xy_ring[:, self._anchor_ring_idx] = self.robot_anchor_pos_w[:, :2]
        oldest = self._anchor_xy_ring[:, (self._anchor_ring_idx + 1) % window]
        self._anchor_ring_idx = (self._anchor_ring_idx + 1) % window
        moved = torch.norm(self.robot_anchor_pos_w[:, :2] - oldest, dim=-1)
        self._stagnant = (moved < self.cfg.stagnation_threshold) & (self._env.episode_length_buf > window)

    # -- actor-side (virtual perception) observations --------------------------

    @property
    def ball_obs_delayed(self) -> torch.Tensor:
        """Perceived ball obs [pos(2), dir(2), visible(1)] delayed by the sampled latency."""
        idx = (self._ring_step - 1 - self._perception_delay) % self.BALL_RING_LENGTH
        return self._ball_ring[self._env_ids, idx]

    @property
    def ball_obs_history(self) -> torch.Tensor:
        """Flattened chronological history of perceived ball obs (window x 5)."""
        length = self.cfg.ball_history_length
        idx = (self._ring_step - length + torch.arange(length, device=self.device)) % self.BALL_RING_LENGTH
        return self._ball_ring[:, idx].flatten(1)

    def _update_metrics(self):
        super()._update_metrics()
        speed = torch.norm(self.ball_vel_w, dim=-1)
        self.metrics["ball_speed"] = speed
        self.metrics["ball_max_speed"] = torch.maximum(self.metrics["ball_max_speed"], speed)
        self.metrics["ball_target_dist"] = torch.norm(self.ball_pos_w - self.target_pos_w, dim=-1)
        self.metrics["ball_visible"] = self.ball_obs_delayed[:, 4]

    def _set_debug_vis_impl(self, debug_vis: bool):
        super()._set_debug_vis_impl(debug_vis)
        if debug_vis and not hasattr(self, "target_visualizer"):
            self.target_visualizer = VisualizationMarkers(self.cfg.target_visualizer_cfg)

    def _debug_vis_callback(self, event):
        super()._debug_vis_callback(event)
        if hasattr(self, "target_visualizer"):
            self.target_visualizer.visualize(self.target_pos_w)


@configclass
class KickMotionCommandCfg(MotionCommandCfg):
    """Configuration for the kick motion command."""

    class_type: type = KickMotionCommand

    ball_name: str = "ball"
    main_foot_name: str = "right_ankle_roll_link"

    # ball spawn: world offset from env origin (reference motion faces +y), plus noise
    ball_offset: tuple[float, float, float] = (0.25, 1.2, 0.12)
    ball_forward_range: tuple[float, float] = (-0.1, 0.1)
    ball_lateral_range: tuple[float, float] = (-0.1, 0.1)

    # target: anchored at the ball, azimuth sampled around the motion heading
    target_distance: float = 2.5
    target_azimuth_range: float = 0.2618  # rad (~15 deg), curriculum widens this

    # episode start time clamp so the reference kick step is always reached
    start_time_fraction: float = 0.05

    # reference time step of ball contact (for time-gating the contact reward)
    kick_step: int = 265

    # std (rad) of the one-shot kick-alignment angle kernel (~15 deg)
    alignment_std: float = 0.2618

    # -- virtual perception (arXiv:2511.03996 appendix, measured on a real robot) --
    ball_detection_prob: float = 0.9          # P(detect) per perception refresh
    ball_noise_slope: float = 0.124           # sigma = slope * distance + base  [m]
    ball_noise_base: float = 0.149
    ball_update_freq_hz: tuple[float, float] = (25.36, 1.06)   # (mean, std) of perception rate
    ball_latency_ms: tuple[float, float] = (116.0, 18.0)       # (mean, std) of pipeline latency
    ball_history_length: int = 50             # perceived-obs history fed to the actor (1 s @ 50 Hz)

    # -- stagnation penalty --
    stagnation_window: int = 50               # steps (~1 s) over which motion is checked
    stagnation_threshold: float = 0.05        # anchor xy displacement [m] below which env is stagnant

    target_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/target")
    target_visualizer_cfg.markers["frame"].scale = (0.3, 0.3, 0.3)
