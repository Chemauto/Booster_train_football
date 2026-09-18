from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurriculumTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg

##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import booster_train.tasks.manager_based.kick_mimic.mdp as mdp
from booster_train.tasks.manager_based.kick_mimic.mdp.commands import KickMotionCommand

##
# Scene definition
##

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot and a soccer ball."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    # robots
    robot: ArticulationCfg = MISSING
    # soccer ball (spawned near the reference kick contact point every reset)
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.20, 0.88, 0.12)),
        spawn=sim_utils.SphereCfg(
            radius=0.11,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                linear_damping=0.03,
                angular_damping=0.01,
                max_depenetration_velocity=20,
                max_contact_impulse=3000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.41),
        ),
        collision_group=0,
    )
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0, debug_vis=True
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.KickMotionCommandCfg(
        asset_name="robot",
        ball_name="ball",
        main_foot_name="right_ankle_roll_link",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP.

    The non-ball fields must stay field-for-field identical (names AND order)
    to beyond_mimic's tracking_env_cfg.ObservationsCfg: stage-1 (pure
    tracking) weights are transferred into stage-2 by inserting the ball
    channels after `command` and shifting the rest, so any reordering silently
    corrupts the transferred semantics.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        # append-only extension: ball channels inserted right after `command`,
        # never replacing or reordering existing channels -- transferred stage-1
        # weights stay semantically aligned. The actor sees the VIRTUAL
        # perception (noisy / 25 Hz / latent / sometimes missing + flag) and a
        # 1 s history of it; the critic keeps the true state below.
        ball_state = ObsTerm(func=mdp.ball_state_virtual_b, params={"command_name": "motion"})
        ball_history = ObsTerm(func=mdp.ball_history_b, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.25, n_max=0.25)
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        ball_state = ObsTerm(func=mdp.ball_state_b, params={"command_name": "motion"})
        ball_velocity = ObsTerm(func=mdp.ball_velocity_b, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 0.6),
            "dynamic_friction_range": (0.3, 0.6),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )

    # ball physics domain randomization (real balls differ in friction/bounce/mass)
    ball_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "static_friction_range": (0.3, 0.8),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.3, 0.8),
            "num_buckets": 64,
        },
    )
    ball_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "mass_distribution_params": (0.38, 0.48),
            "distribution": "uniform",
            "operation": "abs",
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    Three globally scalable groups: MOTION (tracking), GOAL (ball), REG
    (regularization). Group scales come from environment variables so the
    two-stage schedule needs no code change:

      stage 1 (tracking prior):  default weights, KICK_GOAL_WEIGHT=0
      stage 2 (vision kick):     KICK_MOTION_WEIGHT=0.2 (or 0), KICK_GOAL_WEIGHT=1

    Blind-kick ceiling must stay low, which requires the MOTION group to fade
    out once goal rewards drive behaviour (handled by the anneal curriculum).
    """

    # --- MOTION group: motion tracking (stage-2: annealed towards 0) ---
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )

    # --- GOAL group: ball / kick target ---
    ball_target_progress = RewTerm(func=mdp.ball_target_progress, weight=50.0, params={"command_name": "motion"})
    foot_ball_contact = RewTerm(
        func=mdp.foot_ball_contact,
        weight=10.0,
        params={"command_name": "motion", "contact_dist": 0.22, "paid_steps": 3, "window_steps": 100},
    )
    ball_kick_alignment = RewTerm(
        func=mdp.ball_kick_alignment,
        weight=10.0,
        params={"command_name": "motion"},
    )
    ball_speed = RewTerm(
        func=mdp.ball_speed,
        weight=0.5,
        params={"command_name": "motion", "ramp_threshold": 1.0, "target": 4.5, "std": 1.5},
    )
    ball_approach = RewTerm(
        func=mdp.ball_approach,
        weight=10.0,
        params={"command_name": "motion"},
    )
    stagnation = RewTerm(func=mdp.stagnation_penalty, weight=-1.0, params={"command_name": "motion"})

    # --- REG group: regularization ---
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_elbow_yaw_link$)(?!right_elbow_yaw_link$)(?!left_ankle_roll_link$)(?!right_ankle_roll_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )

    def __post_init__(self):
        motion_scale = float(os.environ.get("KICK_MOTION_WEIGHT", 1.0))
        goal_scale = float(os.environ.get("KICK_GOAL_WEIGHT", 1.0))
        reg_scale = float(os.environ.get("KICK_REG_WEIGHT", 1.0))
        for term in [
            self.motion_global_anchor_pos,
            self.motion_global_anchor_ori,
            self.motion_body_pos,
            self.motion_body_ori,
            self.motion_body_lin_vel,
            self.motion_body_ang_vel,
        ]:
            term.weight *= motion_scale
        for term in [
            self.ball_target_progress,
            self.foot_ball_contact,
            self.ball_kick_alignment,
            self.ball_speed,
            self.ball_approach,
            self.stagnation,
        ]:
            term.weight *= goal_scale
        for term in [self.action_rate_l2, self.joint_limit, self.undesired_contacts]:
            term.weight *= reg_scale


@configclass
class TerminationsCfg:
    """Termination terms for the MDP (no ball-related terms by design)."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": [
                "left_elbow_yaw_link",
                "right_elbow_yaw_link",
                "left_ankle_roll_link",
                "right_ankle_roll_link",
            ],
        },
    )


def widen_target_azimuth(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    steps_per_iteration: int = 24,
    stages: tuple[tuple[int, float], ...] = ((0, 0.1745), (2000, 0.2618), (6000, 0.7854)),
) -> None:
    """Staged widening of the target azimuth half-range (rad) over training.

    ``stages`` are in PPO iterations (converted via ``steps_per_iteration``,
    the runner's num_steps_per_env). The first stage is +/-10 deg, NOT zero: a
    constant direction stage trains the policy to ignore the direction channel
    (weights stay at random init with no gradient, and the converged network
    routes around them) -- after that, reviving the channel through random
    projections is nearly impossible. Direction must vary from iteration 0;
    the curriculum only widens the range.
    """
    assert all(stages[i][0] < stages[i + 1][0] for i in range(len(stages) - 1)), "stages must ascend"
    command: KickMotionCommand = env.command_manager.get_term("motion")
    iteration = env.common_step_counter // steps_per_iteration
    azimuth = max(azimuth_rad for stage_iteration, azimuth_rad in stages if iteration >= stage_iteration)
    command.cfg.target_azimuth_range = azimuth


def anneal_motion_weight(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    steps_per_iteration: int = 24,
    ladder: tuple[float, ...] = (1.0, 0.5, 0.2, 0.1, 0.0),
    motion_terms: tuple[str, ...] = (
        "motion_global_anchor_pos",
        "motion_global_anchor_ori",
        "motion_body_pos",
        "motion_body_ori",
        "motion_body_lin_vel",
        "motion_body_ang_vel",
    ),
    accuracy_gate: float = 0.22,
    speed_gate: float = 3.0,
    check_interval_iters: int = 100,
    hold_checks: int = 5,
    fallback_iters_per_stage: int = 3000,
) -> None:
    """Metric-gated annealing of the MOTION reward group -- no manual schedule.

    Steps the MOTION weight down the ``ladder`` when EITHER:
      * performance gate: mean kick accuracy (alignment kernel EMA) stays above
        ``accuracy_gate`` AND mean ball max speed above ``speed_gate`` for
        ``hold_checks`` consecutive evaluations (every ``check_interval_iters``
        PPO iterations) -- i.e. the policy has squeezed what the current
        tracking constraint allows; or
      * safety fallback: ``fallback_iters_per_stage`` iterations spent on the
        current stage regardless of metrics, so annealing always progresses.

    Downward only (hysteresis): the weight never climbs back. Initial weights
    are captured on first call and rescaled by the ladder, so this term composes
    with the KICK_MOTION_WEIGHT env var (which should be left at 1.0).
    """
    command: KickMotionCommand = env.command_manager.get_term("motion")
    state = getattr(command, "_anneal_state", None)
    if state is None:
        state = command._anneal_state = {
            "stage": 0,
            "held": 0,
            "last_check_iter": -1,
            "stage_start_iter": None,
            "base_weights": {name: env.reward_manager.get_term_cfg(name).weight for name in motion_terms},
        }
    iteration = env.common_step_counter // steps_per_iteration
    if iteration < state["last_check_iter"] + check_interval_iters:
        return
    state["last_check_iter"] = iteration
    if state["stage_start_iter"] is None:
        state["stage_start_iter"] = iteration
    if state["stage"] >= len(ladder) - 1:
        return

    accuracy = float(command.metrics["kick_accuracy"].mean())
    speed = float(command.metrics["ball_max_speed"].mean())
    if accuracy >= accuracy_gate and speed >= speed_gate:
        state["held"] += 1
    else:
        state["held"] = 0
    elapsed = iteration - state["stage_start_iter"]
    if state["held"] >= hold_checks or elapsed >= fallback_iters_per_stage:
        reason = "metric gate" if state["held"] >= hold_checks else "fallback timer"
        state["stage"] += 1
        state["held"] = 0
        state["stage_start_iter"] = iteration
        scale = ladder[state["stage"]]
        for name in motion_terms:
            env.reward_manager.get_term_cfg(name).weight = state["base_weights"][name] * scale
        print(
            f"[CURRICULUM] MOTION weight -> {scale} at iter {iteration} "
            f"(kick_accuracy={accuracy:.3f}, ball_max_speed={speed:.2f}, trigger: {reason})"
        )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    target_azimuth = CurriculumTerm(func=widen_target_azimuth)
    motion_anneal = CurriculumTerm(func=anneal_motion_weight)


##
# Environment configuration
##


@configclass
class KickEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the motion-tracking soccer-kick environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # shrink physx gpu buffers for small-gpu machines (defaults are sized for ~4096 envs)
        self.sim.physx.gpu_max_rigid_contact_count = 2**20
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**19
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**21
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**19
        self.sim.physx.gpu_collision_stack_size = 2**24
        self.sim.physx.gpu_heap_capacity = 2**24
        self.sim.physx.gpu_temp_buffer_capacity = 2**23
        # viewer settings
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
