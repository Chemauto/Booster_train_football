"""Env config for Booster-K1-KickAMP-v0 (paper 2511.03996, K1 port of t1.py).

Single end-to-end policy: AMP motion priors (walk + kick dataset) + task rewards
(approach ball, push it to the goal, arch-kick shaping) + virtual ball
perception (noise/latency/dropouts) + encoder-decoder POMDP + two-critic PPO.
See mdp/commands.py for the stateful soccer logic.

Field: RoboCup adult-size 14 x 9 m, goal mouth 2.6 m at x=+7.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg, CommandTermCfg, EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GNoise

from booster_assets import BOOSTER_ASSETS_DIR
from booster_train.assets.robots.booster import BOOSTER_K1_CFG as ROBOT_CFG, K1_ACTION_SCALE

import booster_train.tasks.manager_based.kick_amp.mdp as mdp
from booster_train.tasks.manager_based.kick_amp.mdp.commands import (
    FIELD_HALF_LENGTH,
    FIELD_HALF_WIDTH,
    GOAL_X,
    GOAL_HALF_WIDTH,
    MINIMAL,
    SoccerStateCommandCfg,
)

##
# Scene
##


@configclass
class SoccerSceneCfg(InteractiveSceneCfg):
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
    )

    robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.activate_contact_sensors = True
    # default stance = data-derived crouch; the paper's T1
    # default_joint_angles -0.2/0.4/-0.25 equivalent. A straight-leg zero
    # default is not a stable equilibrium under the soft actuator model: the
    # knees buckle and the trunk drops below the fall threshold.
    # Provenance: derived from the walk set, whose means are hip -0.14/-0.22,
    # knee 0.45/0.54, ankle -0.32/-0.26. With motion_dir back on amp_paper the
    # stance matches the data again; it was left unchanged while accad_markers
    # was active (that set sits ~0.1 rad deeper, ankle ~0.2 rad more negative)
    # because the AMP motion reset places the robot at a motion frame anyway and
    # a new spawn pose needs a run to validate.
    robot.init_state.joint_pos.update({
        "left_hip_pitch_joint": -0.19, "right_hip_pitch_joint": -0.19,
        "left_knee_pitch_joint": 0.50, "right_knee_pitch_joint": 0.53,
        "left_ankle_pitch_joint": -0.18, "right_ankle_pitch_joint": -0.22,
    })
    robot.init_state.pos = (0.0, 0.0, 0.55)

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.11,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.25, 0.1)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.43),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.12)),
    )

    # goal: visual-only posts+bar at the far end (v1; ball out/goal is judged
    # geometrically by the command, so no collision volume needed)
    goal_left_post = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GoalLeft",
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.08, 1.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(GOAL_X, GOAL_HALF_WIDTH, 0.9)),
    )
    goal_right_post = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GoalRight",
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.08, 1.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(GOAL_X, -GOAL_HALF_WIDTH, 0.9)),
    )
    goal_crossbar = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GoalBar",
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 2 * GOAL_HALF_WIDTH + 0.08, 0.08),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(GOAL_X, 0.0, 1.8)),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )

    contact_forces = ContactSensorCfg(
        # feet_air_time_biped needs air/contact time tracking and feet_slide
        # needs one step of force history (both legged_lab ports)
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=1, track_air_time=True
    )


##
# Observations: actor 79-dim POMDP, critic 93-dim (true ball + privileged 14)
##


@configclass
class PolicyObsCfg(ObsGroup):
    project_gravity = ObsTerm(func=mdp.projected_gravity, noise=GNoise(std=0.01))
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=GNoise(std=0.1))
    ball_obs = ObsTerm(func=mdp.ball_obs)
    goal_pos = ObsTerm(func=mdp.goal_pos_b, noise=GNoise(std=0.5))
    base_yaw = ObsTerm(func=mdp.base_yaw_cos_sin, noise=GNoise(std=0.1))
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=GNoise(std=0.01))
    joint_vel = ObsTerm(func=mdp.joint_vel_scaled, noise=GNoise(std=0.01), params={"scale": 0.1})
    actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.concatenate_terms = True


@configclass
class CriticObsCfg(ObsGroup):
    project_gravity = ObsTerm(func=mdp.projected_gravity)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    ball_true = ObsTerm(func=mdp.ball_obs_true)
    goal_pos = ObsTerm(func=mdp.goal_pos_b)
    base_yaw = ObsTerm(func=mdp.base_yaw_cos_sin)
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_scaled, params={"scale": 0.1})
    actions = ObsTerm(func=mdp.last_action)
    privileged = ObsTerm(func=mdp.privileged_obs)

    def __post_init__(self):
        self.concatenate_terms = True


@configclass
class ObsCfg:
    """Observation groups: field names are the obs dict keys."""
    policy: PolicyObsCfg = PolicyObsCfg()
    critic_observations: CriticObsCfg = CriticObsCfg()


##
# Actions: delayed joint position targets
##

@configclass
class ActionsCfg:
    # NOTE: no action-level delay term -- the BoosterDelayedPDActuator model in
    # BOOSTER_K1_CFG already simulates 2-8 substeps (10-40 ms) of motor delay,
    # so stacking DelayedJointPositionAction on top would double-count latency (now U(0,20)ms per paper Table 2)
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=K1_ACTION_SCALE,
        use_default_offset=True,
    )


##
# Events
##

# velocity kicks on the robot base every ~2 s (t1.yaml kick_interval_s / kick_lin_vel)
VELOCITY_RANGE = {
    "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1),
    "roll": (-0.02, 0.02), "pitch": (-0.02, 0.02), "yaw": (-0.02, 0.02),
}


@configclass
class EventCfg:
    # startup: static physics randomization buckets
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            # Isaac Lab's range REPLACES the material value (the paper's
            # randomizer ADDS U[0,1] to the robot material >=1.0 and combines
            # with ground friction 0.5); with our ground at 1.0, (0.5, 1.0)
            # reproduces the paper's effective friction range
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.0),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 256,
        },
    )

    # joint zero-offset fixed per env for the whole run (startup): the
    # beyond_mimic implementation ADDS to the current default, so reset mode
    # would random-walk drift across episodes (t1.py samples once per env too)
    add_default_joint_offset = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.05, 0.05),
            "operation": "add",
        },
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.95, 1.05),
            "damping_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    trunk_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    # com offset fixed per env (startup): the randomizer ADDS to the current
    # PhysX com, so reset mode would random-walk drift across episodes
    trunk_com = EventTerm(
        func=mdp.randomize_rigid_body_com_partial,
        mode="startup",
        params={"com_range": (0.05, 0.05, 0.05)},
    )
    ball_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "mass_distribution_params": (0.4, 0.45),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    record_phys = EventTerm(
        func=mdp.record_phys_randomization,
        mode="reset",
        params={},
    )
    robot_to_motion_state = EventTerm(
        func=mdp.reset_robot_to_motion_state,
        mode="reset",
        params={},
    )
    ball_reset = EventTerm(
        func=mdp.reset_ball_random,
        mode="reset",
        params={},
    )

    # interval: random velocity kicks on the robot (opponent collisions)
    kick_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 2.0),
        params={"velocity_range": VELOCITY_RANGE, "asset_cfg": SceneEntityCfg("robot")},
    )


##
# Rewards (group-1 terms; the goal cluster lives in the soccer command)
##


@configclass
class RewardCfg:
    survival = RewTerm(func=mdp.survival, weight=3.0)
    termination = RewTerm(func=mdp.termination, weight=-1000.0)
    # At -100 this costs 2/step after just 1 s: even maximum AMP (0.6)
    # plus survival (0.06) cannot offset it. Dying quickly beats balancing.
    # Keep standing viable while approach/style rewards favor locomotion.
    # paper Table 4 "Stagnation -100"; weight kept, but penalize_pos is NOT the
    # paper's 0.7. That value is T1's: it demands a net displacement of 0.7 m per
    # second, while this robot's measured walking speed is 0.623 m/s
    # (env/planar_speed_mps at arm D). The term was therefore unsatisfiable --
    # permanent -2/step pressure to hurry beyond what the gait can do, which is
    # what a 24% first-episode fall rate at a mean 3.3 s (i.e. during the reset
    # transient, where the rolling training fall rate is only 1%) looks like.
    # 0.45 m/s sits below the achieved speed with margin while still being far
    # above standing still. Same surface-vs-effective trap the earlier note
    # describes, one layer deeper: the parameter is only meaningful relative to
    # the robot that has to satisfy it.
    # penalize_pos_distance is the ball-exemption radius: inside it the
    # anti-freeze penalty does not fire. At 1.0 m it drew a "parking space" --
    # a physics probe of model_3000 found the policy's end state 0.61 m from the
    # ball, stationary (0.009 m/s), i.e. exactly inside the exempt ring, where it
    # pays no freeze penalty and collects survival while never touching the ball
    # (contact proxies 1/64 in that 4-stage run). 0.3 m keeps the legitimate
    # "arrive, then hold balance to kick" solution -- that is contact range --
    # while making a stop 0.6 m short cost the full -2/step again.
    pos_still = RewTerm(
        func=mdp.pos_still,
        weight=-100.0,
        params={"penalize_pos": 0.45, "penalize_yaw": 1.0, "penalize_pos_distance": 0.3},
    )
    # Default 0.0 = off, so every existing run and dataset comparison is
    # unchanged. Turn it on (e.g. -100.0, matching pos_still's scale) to give the
    # field edge a gradient: the out-of-field termination is a 0/1 cliff and the
    # goal line is the field edge, so without this the safe way to push the ball
    # toward the goal is not to push it at all. guard=0.3 keeps the scoring
    # approach itself unpenalized -- see the function's docstring.
    boundary = RewTerm(func=mdp.boundary_distance, weight=0.0, params={"guard": 0.3})
    # Velocity-shaped edge term: charges only outward motion near the line, so the
    # scoring push (which must happen within ~0.3 m of it) stays free. Default 0.
    boundary_outward = RewTerm(func=mdp.boundary_outward_speed, weight=0.0, params={"guard": 0.6})
    # Charge the sideways component of the ball's speed while a foot is on it, near
    # the goal. Default 0 = off; enable with --ball_lateral_weight.
    #
    # Calibrated against a measurement, not a guess. At iteration 6800 the wide
    # misses left the foot at 1.185 m/s of lateral ball speed against 0.764 m/s for
    # the goals (1.55x), and 30 of 40 crossed only ~0.2 m outside the post -- the
    # mechanism is a slightly sideways push, and the misses are marginal. The
    # competing incentive is side_kick_ball (+20, logs +0.19 per step) which pays
    # for lateral sweeps and nothing charged the resulting sideways ball motion;
    # at weight -10 this term logged only -0.024, i.e. it was inert. near_goal=5.0
    # covers the contacts that launch a shot, not just those at the line.
    ball_lateral = RewTerm(func=mdp.ball_lateral_speed, weight=0.0, params={"near_goal": 5.0})

    kick_ball = RewTerm(func=mdp.kick_ball, weight=-20.0)
    side_kick_ball = RewTerm(func=mdp.side_kick_ball, weight=20.0)
    face_ball_pitch = RewTerm(func=mdp.face_ball_pitch, weight=-0.5)
    face_ball_yaw = RewTerm(func=mdp.face_ball_yaw, weight=-0.5)
    root_acc = RewTerm(func=mdp.root_acc, weight=-1.0e-3)
    action_rate = RewTerm(func=mdp.action_rate_legs, weight=-1.0)
    head_action_rate = RewTerm(func=mdp.head_action_rate, weight=-15.0)
    dof_pos_limits = RewTerm(func=mdp.dof_pos_limits, weight=-100.0)
    collision = RewTerm(func=mdp.collision, weight=-100.0)
    feet_min_distance = RewTerm(func=mdp.feet_min_distance, weight=-5.0)

    # -- walk curriculum (legged_lab velocity ports; each scales (1-c) and
    #    gates itself off near the ball, see mdp/rewards.py). Weights are 3x /
    # 2.7x the legged_lab values: our survival term (+0.06/step for standing,
    # inherited from t1.py which has no gait reward) made the ported weights
    # economically irrelevant -- perfect walking earned only +0.045 marginal
    # income vs standing, less than a 0.3%/step fall risk costs. At 3.0/2.0
    # the gait income ~2.4x survival, flipping the local optimum. Measured
    # failure that motivated this: feet_air_time_biped == 0.0000 for 800 iters
    # while fall was a healthy 0.046 (standing local optimum, not instability).
    track_lin_vel_ball = RewTerm(func=mdp.track_lin_vel_ball, weight=3.0)
    feet_air_time_biped = RewTerm(func=mdp.feet_air_time_biped, weight=2.0)
    feet_clearance = RewTerm(func=mdp.feet_clearance, weight=1.0)
    feet_slide = RewTerm(func=mdp.feet_slide, weight=-1.0)


##
# Terminations
##


def robot_out_of_field(env) -> bool:
    """Robot beyond the field border (0.9 m margin, t1.py)."""
    cmd = env.command_manager.get_term("soccer")
    return (
        (cmd.base_pos_xy[:, 0] < -FIELD_HALF_LENGTH - 0.9)
        | (cmd.base_pos_xy[:, 0] > FIELD_HALF_LENGTH + 0.9)
        | (cmd.base_pos_xy[:, 1] < -FIELD_HALF_WIDTH - 0.9)
        | (cmd.base_pos_xy[:, 1] > FIELD_HALF_WIDTH + 0.9)
    )


def ball_scored_goal(env) -> bool:
    """Episode success: the ball is in the goal (same geometry as the evaluation).

    Measured before this existed, at arm G: 137 of 256 accepted goals and 95% of
    them ended with the robot out of field, because the episode kept running
    after the ball crossed and the robot followed it over the line. GOAL_X equals
    FIELD_HALF_LENGTH, so the out-of-field bar and the goal bar cannot both be
    met while a scored episode continues.
    """
    # The command caches this before its ball-only teleport, so a goal is not lost
    # to the reset that the goal itself triggers.
    return env.command_manager.get_term("soccer").ball_in_goal_now


def base_height_low(env) -> bool:
    """Fallen: trunk below 0.35 m (K1 standing ~0.57, deep crouch ~0.43)."""
    return env.scene["robot"].data.root_pos_w[:, 2] < 0.35


def base_vel_high(env) -> bool:
    vel = env.scene["robot"].data.root_state_w[:, 7:13]
    return vel.square().sum(dim=-1) > 50.0


@configclass
class TerminationCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Scoring ends the episode. Deliberately NOT time_out=True: the evaluation
    # reads the cause list to label episodes, and mdp.termination excludes this
    # cause so the success costs no penalty (see the reward's docstring).
    goal = DoneTerm(func=ball_scored_goal)
    out_of_field = DoneTerm(func=robot_out_of_field)
    base_contact = DoneTerm(func=base_height_low)
    high_velocity = DoneTerm(func=base_vel_high)


@configclass
class CommandsCfg:
    # amp_paper_waistfix = the paper's OWN retargeted dataset (code/data/*.csv,
    # 50 fps native) converted to K1 npz by scripts/convert_paper_data.py, with
    # T1's waist yaw folded into K1's hip yaw. 24 clips / 106.3 s, 0 HARD
    # findings.
    # T1's chain is Trunk -> Waist -> legs and K1 has no waist joint, so the
    # earlier build silently dropped that DOF. Its real size is median 5.4 deg
    # (max 24.6 deg), and dropping it left the foot yawed by a median 5.44 deg
    # relative to the trunk when compared against T1's own forward kinematics.
    # Folding it into the hip yaw cuts that to 1.14 deg (p90 5.27) at no cost:
    # the soft-limit charge is unchanged for walk (-0.020/step) and slightly
    # better for kick (-0.017 vs -0.021).
    #
    # Chosen over accad_markers after re-measuring both with threshold-free
    # methods: amp_paper's gait cadence is 1.05 Hz with 83% of clips inside the
    # human 0.7-1.3 Hz band vs accad_markers' 0.70 Hz / 47%, and its soft-limit
    # cost is -0.020/step vs -0.082 (kicks -0.021 vs -0.209). Stance foot slip
    # is a tie (0.049 vs 0.062 m/s on a strict 3 mm contact band). An earlier
    # note here claimed amp_paper was a 0.40 Hz shuffle with 0.229 m/s slip;
    # both were artifacts of a fixed contact-height threshold, retracted in
    # docs/reports/2026-09-18-accad-marker-dataset.md.
    #
    # The one structural argument for accad_markers still stands: amp_paper is
    # T1's joint trajectories copied onto K1 by name with the waist yaw dropped,
    # so K1 strikes T1 poses (audited foot offset up to 7.5 cm, heading 24.6
    # deg). Whether that costs anything is not yet measured; the marker dataset
    # is on disk for that experiment.
    soccer = SoccerStateCommandCfg(
        motion_dir=f"{BOOSTER_ASSETS_DIR}/motions/K1/amp_paper_waistfix",
        resampling_time_range=(1.0e9, 1.0e9),
    )


##
# Full env
##


@configclass
class KickAmpEnvCfg(ManagerBasedRLEnvCfg):
    scene: SoccerSceneCfg = SoccerSceneCfg(num_envs=4096, env_spacing=6.0)
    observations: ObsCfg = ObsCfg()
    critic_observations: CriticObsCfg = CriticObsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardCfg = RewardCfg()
    terminations: TerminationCfg = TerminationCfg()
    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self):
        self.decimation = 4            # 50 Hz policy
        self.episode_length_s = 60.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        if MINIMAL:
            # Bisection run: keep ONLY survival + termination so the remaining
            # learning signal is AMP style vs falling. Every shaping and penalty
            # term goes to zero, which the reward-manager table prints at
            # startup -- that printout is the verification that this took
            # effect. Also removes the pos_still weight risk (-100, about
            # -2/step against survival's +0.06) that would otherwise make
            # falling optimal now that the warm-up gate is gone.
            for _name in (
                "pos_still", "kick_ball", "side_kick_ball", "face_ball_pitch",
                "face_ball_yaw", "root_acc", "action_rate", "head_action_rate",
                "dof_pos_limits", "collision", "feet_min_distance",
                "track_lin_vel_ball", "feet_air_time_biped", "feet_clearance",
                "feet_slide", "boundary", "boundary_outward", "ball_lateral",
            ):
                getattr(self.rewards, _name).weight = 0.0


@configclass
class PlayKickAmpEnvCfg(KickAmpEnvCfg):
    """Play variant: fewer envs, no interventions, deterministic-ish perception."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 2
        self.events.kick_robot = None
        self.episode_length_s = 120.0
