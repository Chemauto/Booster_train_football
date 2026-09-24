"""K1 kick_amp soccer task (sim2sim).

This is the booster_train/deploy copy of booster_deploy/tasks/kick_amp
(same files; the scene path resolves relative to this directory). The
controller framework (booster_deploy package) is imported from the sibling
repo; the entry script bootstraps sys.path. Run:

    python3 deploy/sim2sim_kick_amp.py --episodes 32
    python3 deploy/sim2sim_kick_amp.py --view        # interactive viewer

(Do NOT point the stock booster_deploy scripts/deploy.py --mujoco at this
task: the stock MujocoController's fixed robot-only qpos layout does not fit
the merged soccer scene.)

The task is still registered so the real-robot path (BoosterRobotPortal,
which never touches mjcf) can pick it up later. PD gains / effort limits /
default stance below reproduce the post-651b7a5 training actuator model
(same values as k1_mj2_v3, which verified action_scale = 0.25*effort/
stiffness reproduces training K1_ACTION_SCALE exactly), except hip_roll
effort 76 (E4315) instead of the URDF's stale 43, and the crouch stance
default of kick_amp (env_cfg.py:71-76: hip_pitch -0.19, knee 0.50/0.53,
ankle_pitch -0.18/-0.22, shoulder_roll -/+1.3).

Default checkpoint: models/kick_amp_it6400_policy.pt = the TorchScript export
of booster_train model_6400.pt (run 2026-09-19_13-45-20; acceptance eval
195/256 goals, 0 falls; trained AND evaluated with perfect_perception=True).
"""

from booster_deploy.controllers.controller_cfg import (
    ControllerCfg, MujocoControllerCfg,
)
from booster_deploy.robots.booster import K1_CFG
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task

from .kick_amp import KickAmpPolicyCfg

import os as _os
# scene path resolved from this file's location so the package runs from any copy
_SCENE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "scene", "k1_soccer_14x9.xml")

# training default_joint_pos (kick_amp env_cfg.py:71-76 + booster.py:38-41),
# in RobotCfg.joint_names order: head x2, left arm x4, right arm x4,
# left leg x6, right leg x6.
_CROUCH = [
    0.0, 0.0,                    # head yaw, pitch
    0.0, -1.3, 0.0, 0.0,         # left arm
    0.0, 1.3, 0.0, 0.0,          # right arm
    -0.19, 0.0, 0.0, 0.50, -0.18, 0.0,   # left leg
    -0.19, 0.0, 0.0, 0.53, -0.22, 0.0,   # right leg
]

# E6408/E4315/E4310/E6416 + ankle parallel E4310 / R14 / HT4438 gains, FULL
# PRECISION from the training run's env_cfg.txt (actuator derivation
# armature*(2*pi*f)^2, 2*zeta*armature*2*pi*f) so action_scale = 0.25*effort/
# stiffness reproduces training K1_ACTION_SCALE to <1e-6 relative.
_KP, _KD = 3.9478417602100686, 0.25132741228     # R14/HT4438 (arms + head)
_K1_KICK_STIFFNESS = [
    _KP, _KP,
    _KP, _KP, _KP, _KP,
    _KP, _KP, _KP, _KP,
    30.200989465607023, 21.447961045805584, 17.846013389258083,
    60.401978931214046, 35.692026778516166, 35.692026778516166,
    30.200989465607023, 21.447961045805584, 17.846013389258083,
    60.401978931214046, 35.692026778516166, 35.692026778516166,
]
_K1_KICK_DAMPING = [
    _KD, _KD,
    _KD, _KD, _KD, _KD,
    _KD, _KD, _KD, _KD,
    3.60497756989125, 2.560161764834957, 2.1302109340993156,
    4.806636759855, 4.260421868198631, 4.260421868198631,
    3.60497756989125, 2.560161764834957, 2.1302109340993156,
    4.806636759855, 4.260421868198631, 4.260421868198631,
]
_K1_KICK_EFFORT = [
    6, 6,
    14, 14, 14, 14,
    14, 14, 14, 14,
    68, 76, 38.3, 112, 38.3, 38.3,
    68, 76, 38.3, 112, 38.3, 38.3,
]

# Booster motor torque-speed (T-N) curve parameters per joint (real order):
# tau_max(v) = effort for |v| <= vknee, then linear to 0 at vmax
# (booster_train assets/robots/actuator.py:114-133, tables at 323-396; the
# ankle parallel wrapper passes E4310 ratios (1,1); HT4438's vknee > vmax
# degenerates to box-then-zero, which the shared formula reproduces).
_K1_KICK_VMAX = [
    7.85, 7.85,                    # head HT4438
    33.51, 33.51, 33.51, 33.51,    # left arm R14
    33.51, 33.51, 33.51, 33.51,    # right arm R14
    14.66, 12.57, 17.59, 12.57, 17.59, 17.59,   # left leg E6408/E4315/E4310/E6416/ankle E4310
    14.66, 12.57, 17.59, 12.57, 17.59, 17.59,   # right leg
]
_K1_KICK_VKNEE = [
    10.47, 10.47,
    5.24, 5.24, 5.24, 5.24,
    5.24, 5.24, 5.24, 5.24,
    1.88, 2.62, 7.85, 2.09, 7.85, 7.85,
    1.88, 2.62, 7.85, 2.09, 7.85, 7.85,
]


@configclass
class KickAmpControllerCfg(ControllerCfg):
    actuator_delay_substeps: int = 5     # eval config: fixed 10 ms = 5 x 2 ms
    actuator_vmax: list = _K1_KICK_VMAX          # T-N curve speed limits
    actuator_vknee: list = _K1_KICK_VKNEE        # T-N curve knee speeds
    ball_friction_range: tuple = (0.2, 0.2)   # eval pins rolling resistance
    push_enabled: bool = False           # training disturbance, off for eval
    foot_collision: str = "box"          # "box" | "mesh" (Left/Right_Foot.STL convex hull)

    robot = K1_CFG.replace(     # type: ignore
        mjcf_path=_SCENE,
        joint_stiffness=_K1_KICK_STIFFNESS,
        joint_damping=_K1_KICK_DAMPING,
        effort_limit=_K1_KICK_EFFORT,
        default_joint_pos=_CROUCH,
    )
    enable_velocity_commands = False
    policy: KickAmpPolicyCfg = KickAmpPolicyCfg()
    mujoco = MujocoControllerCfg(
        init_pos=[0.0, 0.0, 0.55],    # kick_amp init_state.pos z
        visualize_reference_ghost=False,
    )


register_task("k1_kick_amp", KickAmpControllerCfg())
