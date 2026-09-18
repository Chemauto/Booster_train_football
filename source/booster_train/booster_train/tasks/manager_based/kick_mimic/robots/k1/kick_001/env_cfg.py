"""Env configs for the K1 kick task (two stages).

Stage 1 (Booster-K1-Kick_001-Track-v0): pure motion tracking of the
retargeted kick motion (full 612-frame right_kick), WoStateEstimation
observations -- identical structure to the mj_dance_002 deployment lineage
(119-dim actor obs, matching booster_deploy).

Stage 2 (Booster-K1-Kick_001-v0): ball + kick target on top, ball channels
appended after `command`. Warm-started from stage 1 with
`train.py --init_policy_path` (channel-aligned weight insertion).

Calibration values (from scripts/g1_to_k1_csv.py two-pass pipeline,
see k1_kick_trimmed.npz analysis):
    kick_step  = 94   (right-foot forward-velocity peak frame, contact-2)
    ball_offset = (0.20, 0.88, 0.12)  (contact-frame foot pos + 0.08 m lead)
"""

from isaaclab.utils import configclass
from booster_assets import BOOSTER_ASSETS_DIR
from booster_train.assets.robots.booster import BOOSTER_K1_CFG as ROBOT_CFG, K1_ACTION_SCALE

from .tracking_env_cfg import TrackingEnvCfg
from .kick_env_cfg import KickEnvCfg

# 14 tracked bodies -- same selection as mj_dance_002 (K1-specific links)
K1_TRACK_BODIES = [
    'trunk',
    'aahead_pitch_link',
    'left_hip_roll_link',
    'left_knee_pitch_link',
    'left_ankle_roll_link',
    'right_hip_roll_link',
    'right_knee_pitch_link',
    'right_ankle_roll_link',
    'left_shoulder_roll_link',
    'left_elbow_pitch_link',
    'left_elbow_yaw_link',
    'right_shoulder_roll_link',
    'right_elbow_pitch_link',
    'right_elbow_yaw_link',
]


def _apply_robot(cfg):
    """Inject the K1 robot + action scale + tracked bodies."""
    cfg.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.actions.joint_pos.scale = K1_ACTION_SCALE
    cfg.commands.motion.anchor_body_name = "trunk"
    cfg.commands.motion.body_names = list(K1_TRACK_BODIES)


##
# Stage 1: pure tracking prior (full motion)
##


@configclass
class KickTrackFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_robot(self)
        self.commands.motion.motion_file = f"{BOOSTER_ASSETS_DIR}/motions/K1/k1_kick_full.npz"


@configclass
class KickTrackFlatWoStateEstimationEnvCfg(KickTrackFlatEnvCfg):
    """Stage-1 as actually trained/deployed: drops anchor_pos + base_lin_vel (119-dim actor)."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class PlayKickTrackFlatWoStateEstimationEnvCfg(KickTrackFlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.play = True
        self.events.push_robot = None


##
# Stage 2: ball + kick target (trimmed motion, K1-calibrated)
##


@configclass
class KickFlatEnvCfg(KickEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_robot(self)
        self.commands.motion.motion_file = f"{BOOSTER_ASSETS_DIR}/motions/K1/k1_kick_trimmed.npz"
        # K1-calibrated contact point (see module docstring)
        self.commands.motion.kick_step = 94
        self.commands.motion.ball_offset = (0.20, 0.88, 0.12)
        # trimmed motion is 6.8 s
        self.episode_length_s = 6.0


@configclass
class KickFlatWoStateEstimationEnvCfg(KickFlatEnvCfg):
    """Stage-2 as trained/deployed: same obs surgery as stage-1's WoSE variant.

    Ball channels stay (they are appended after `command`, before the terms
    that get dropped), so the actor sees command(44) + ball(255) + the
    remaining stage-1 channels -- a strict superset of stage-1's 119.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class PlayKickFlatWoStateEstimationEnvCfg(KickFlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.play = True
        self.events.push_robot = None
