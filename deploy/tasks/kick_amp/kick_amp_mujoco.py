"""MuJoCo sim2sim controller for the K1 kick_amp soccer task.

Subclasses MujocoController but bypasses its fixed qpos layout (base class
assumes a robot-only model: qpos[7:] are joints; the soccer scene adds a ball
freejoint, nq 29 -> 36). Everything here is name-addressed through
jnt_qposadr / jnt_dofadr, so joint order assumptions cannot silently break.

Physics fidelity to training (eval configuration, evaluate_kick_amp.py:237-257):
- PD gains at full training precision + the Booster T-N torque-speed curve
  (tau falls linearly from effort at vknee to 0 at vmax, actuator.py:114-133);
- actuator target delay fixed at 5 substeps (10 ms) by default; the first
  policy step of an episode is applied undelayed (empty DelayBuffer);
- ball rolling resistance as a persistent world-frame force, fixed 0.2 N;
- no pushes, no randomization, clean obs;
- optional 20 N/1 s trunk push every 5 s (training disturbance, default off).

Episode protocol = the acceptance evaluation (scripts/evaluate_kick_amp.py):
robot at the origin, crouch stance, yaw over 4 headings; ball at
U(0.8, 2.0) m over 4 relative bearings; first terminal event ends the
episode: goal (strict geometry) / ball out / fall (trunk < 0.35 m) /
robot out (0.9 m margin) / high base velocity / timeout (default 1500 steps).

Run:  python3 scripts/sim2sim_kick_amp.py --episodes 32
"""

from __future__ import annotations

import json
import sys
import time

import mujoco
import numpy as np
import torch

# mujoco_controller imports booster_assets at module level; make that work in
# environments where the package is not pip-installed (it is in env_isaaclab)
try:
    import booster_assets  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, "/data/rl_robot/BoosterRobotics/booster_assets/src")

from booster_deploy.controllers.base_controller import BaseController
from booster_deploy.controllers.mujoco_controller import MujocoController

from .kick_amp import (
    BALL_RADIUS, FIELD_HALF_LENGTH, FIELD_HALF_WIDTH,
    GOAL_HEIGHT, GOAL_HALF_WIDTH, GOAL_X,
)

try:  # headless metrics runner only needs hashlib
    import hashlib
except ImportError:  # pragma: no cover
    hashlib = None

# training fall / out thresholds (env_cfg.py:384-416)
FALL_HEIGHT = 0.35
OUT_MARGIN = 0.9
HIGH_VEL_SQ = 50.0
BALL_RESET_Z = 0.12          # ball default_root_state z (env_cfg.py:87)
ROOT_SPAWN_Z = 0.55          # init_state.pos z (env_cfg.py:76)


def _resolve_xml_joint_names(model, cfg_joint_names):
    """Map RobotCfg.joint_names -> actual MuJoCo joint names.

    The xml prefix conventions differ from the cfg labels (AAHead_yaw ->
    aahead_yaw_joint, ALeft_Shoulder_Pitch -> aaleft_shoulder_pitch_joint,
    Head_pitch -> aahead_pitch_joint, Left_Hip_Pitch -> left_hip_pitch_joint).
    Matching on lowercase with leading 'a's stripped and '_joint' removed
    handles every case."""
    def norm_cfg(n):
        return n.lower().lstrip("a")

    def norm_xml(n):
        return n[:-len("_joint")].lstrip("a")

    xml_by_norm = {}
    for i in range(model.njnt):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if n.endswith("_joint"):
            xml_by_norm[norm_xml(n)] = n
    resolved = {}
    for n in cfg_joint_names:
        k = norm_cfg(n)
        if k not in xml_by_norm:
            raise KeyError(f"joint '{n}' has no MuJoCo counterpart")
        resolved[n] = xml_by_norm[k]
    used = set(resolved.values())
    n_hinge = sum(
        1 for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    )
    if len(used) != n_hinge:
        raise RuntimeError(
            f"resolved {len(used)} joints but the model has {n_hinge} hinges")
    return resolved


class KickAmpMujocoController(MujocoController):
    """MujocoController on the merged soccer scene (ball + robot)."""

    def __init__(self, cfg):
        # Skip MujocoController.__init__ (fixed robot-only qpos layout);
        # BaseController.__init__ builds robot + policy.
        BaseController.__init__(self, cfg)

        mjcf_path = self._expand_assets_placeholder(cfg.robot.mjcf_path)
        self.mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.mj_model.opt.timestep = cfg.mujoco.physics_dt
        self.decimation = cfg.mujoco.decimation
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        # -- name-addressed index maps --------------------------------------
        jid = lambda n: mujoco.mj_name2id(   # noqa: E731
            self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)
        self._xml_joint = _resolve_xml_joint_names(
            self.mj_model, cfg.robot.joint_names)
        jids = [jid(self._xml_joint[n]) for n in cfg.robot.joint_names]
        self.qposadr = np.array(
            [self.mj_model.jnt_qposadr[i] for i in jids], dtype=int)
        self.dofadr = np.array(
            [self.mj_model.jnt_dofadr[i] for i in jids], dtype=int)
        self.actadr = np.array(
            [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                               self._xml_joint[n])
             for n in cfg.robot.joint_names], dtype=int)
        # (motor names == joint names in the scene xml; verify each actuator
        # really transmits to the matching joint)
        for k, n in enumerate(cfg.robot.joint_names):
            a = self.actadr[k]
            assert self.mj_model.actuator_trnid[a, 0] == jids[k], n

        self.root_jid = jid("world_joint")
        self.ball_jid = jid("ball_joint")
        self.root_qadr = self.mj_model.jnt_qposadr[self.root_jid]
        self.root_vadr = self.mj_model.jnt_dofadr[self.root_jid]
        self.ball_qadr = self.mj_model.jnt_qposadr[self.ball_jid]
        self.ball_vadr = self.mj_model.jnt_dofadr[self.ball_jid]
        self.trunk_bid = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.head_bid = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "aahead_pitch_link")
        self.ball_bid = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "ball")

        # soccer state cache (filled by update_state; read by the policy)
        self.ball_pos_w = torch.zeros(3)
        self.ball_vel_xy = torch.zeros(2)
        self.head_pos_w = torch.zeros(3)
        self.head_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.trunk_z = ROOT_SPAWN_Z

        # actuator target delay in 2 ms physics substeps (training randomizes
        # U(0,20) ms per episode; the evaluation pins 10 ms = 5 substeps here)
        self.delay_substeps = int(getattr(cfg, "actuator_delay_substeps", 5))
        self._prev_targets = self.robot.default_joint_pos.numpy().copy()
        self._first_ctrl_step = True   # empty delay buffer => no delay yet
        self._substep = 0

        # Booster motor T-N torque-speed curve (training _clip_effort,
        # actuator.py:114-133): tau_max(v) = effort for |v| <= vknee, then
        # linear down to 0 at vmax. Falls back to a box clip if unset.
        if getattr(cfg, "actuator_vmax", None) is not None:
            self._tn_vmax = np.asarray(cfg.actuator_vmax, dtype=np.float64)
            self._tn_vknee = np.asarray(cfg.actuator_vknee, dtype=np.float64)
            self._tn_denom = np.maximum(self._tn_vmax - self._tn_vknee, 1e-6)
        else:
            self._tn_vmax = None

        # ball rolling resistance (training: persistent force 0.1-0.3 N,
        # resampled every 5 s; eval pins it at 0.2 N)
        fr = getattr(cfg, "ball_friction_range", (0.2, 0.2))
        self._friction_range = (float(fr[0]), float(fr[1]))
        self._friction_force = 0.2
        self._friction_resample_steps = 250   # 5 s at 50 Hz
        self.push_enabled = bool(getattr(cfg, "push_enabled", False))
        self._push_xy = np.zeros(2)
        self._push_torque = np.zeros(3)
        self._episode_step = 0
        self.foot_collision = "box"
        self._set_foot_collision(getattr(cfg, "foot_collision", "box"))

        self._write_spawn(0.0, np.zeros(2))

    def _set_foot_collision(self, mode: str) -> None:
        if mode not in ("box", "mesh"):
            raise ValueError(f"foot_collision must be 'box' or 'mesh', got {mode!r}")
        if mode == "box":
            names_on, names_off = (
                ("left_foot_box", "right_foot_box"),
                ("left_foot_mesh_col", "right_foot_mesh_col"),
            )
        else:
            names_on, names_off = (
                ("left_foot_mesh_col", "right_foot_mesh_col"),
                ("left_foot_box", "right_foot_box"),
            )
        for name, on in [(n, True) for n in names_on] + [(n, False) for n in names_off]:
            g = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if g < 0:
                raise KeyError(f"geom '{name}' missing from scene xml")
            self.mj_model.geom_contype[g] = 1 if on else 0
            self.mj_model.geom_conaffinity[g] = 1 if on else 0
        self.foot_collision = mode

    def ball_robot_in_contact(self) -> bool:
        m, d = self.mj_model, self.mj_data
        bb = self.ball_bid
        for i in range(d.ncon):
            c = d.contact[i]
            b1 = int(m.geom_bodyid[c.geom1])
            b2 = int(m.geom_bodyid[c.geom2])
            if b1 == bb or b2 == bb:
                other = b2 if b1 == bb else b1
                if other != 0:
                    return True
        return False

    # ------------------------------------------------------------ interface

    def _write_spawn(self, yaw: float, ball_xy: np.ndarray):
        q = self.mj_data.qpos
        q[:] = 0.0
        q[self.root_qadr + 0] = 0.0
        q[self.root_qadr + 2] = ROOT_SPAWN_Z
        q[self.root_qadr + 3] = np.cos(yaw / 2.0)
        q[self.root_qadr + 6] = np.sin(yaw / 2.0)
        q[self.qposadr] = self.robot.default_joint_pos.numpy()
        q[self.ball_qadr + 0:2] = ball_xy
        q[self.ball_qadr + 2] = BALL_RESET_Z
        q[self.ball_qadr + 3] = 1.0
        self.mj_data.qvel[:] = 0.0
        self.mj_data.ctrl[:] = 0.0
        self.mj_data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def reset_episode(self, yaw: float, ball_xy: np.ndarray, seed: int):
        """Full episode reset: robot spawn + policy/perception state."""
        self._write_spawn(yaw, ball_xy)
        self._substep = 0
        self._episode_step = 0
        self._friction_force = float(np.random.uniform(*self._friction_range)) \
            if self._friction_range[0] != self._friction_range[1] \
            else self._friction_range[0]
        self._prev_targets = self.robot.default_joint_pos.numpy().copy()
        self._first_ctrl_step = True
        self.policy.set_episode_seed(seed)
        self.policy.reset()
        self.update_state()

    def update_state(self) -> None:
        q, v = self.mj_data.qpos, self.mj_data.qvel
        d = self.robot.data
        d.joint_pos = torch.from_numpy(
            q[self.qposadr].astype(np.float32)).to(d.device)
        d.joint_vel = torch.from_numpy(
            v[self.dofadr].astype(np.float32)).to(d.device)
        d.feedback_torque = torch.from_numpy(
            self.mj_data.qfrc_actuator[self.dofadr].astype(np.float32)).to(d.device)
        d.root_pos_w = torch.from_numpy(
            q[self.root_qadr:self.root_qadr + 3].astype(np.float32)).to(d.device)
        d.root_quat_w = torch.from_numpy(
            q[self.root_qadr + 3:self.root_qadr + 7].astype(np.float32)).to(d.device)
        # MuJoCo free joint: linear qvel in world, angular qvel in body frame
        d.root_lin_vel_b = torch.from_numpy(
            v[self.root_vadr:self.root_vadr + 3].astype(np.float32)).to(d.device)
        d.root_ang_vel_b = torch.from_numpy(
            v[self.root_vadr + 3:self.root_vadr + 6].astype(np.float32)).to(d.device)

        self.ball_pos_w = torch.from_numpy(
            q[self.ball_qadr:self.ball_qadr + 3].astype(np.float32).copy())
        self.ball_vel_xy = torch.from_numpy(
            v[self.ball_vadr:self.ball_vadr + 2].astype(np.float32).copy())
        self.head_pos_w = torch.from_numpy(
            self.mj_data.xpos[self.head_bid].astype(np.float32).copy())
        self.head_quat_w = torch.from_numpy(
            self.mj_data.xquat[self.head_bid].astype(np.float32).copy())
        self.trunk_z = float(q[self.root_qadr + 2])

    def ctrl_step(self, dof_targets: torch.Tensor):
        targets = dof_targets.detach().cpu().numpy().astype(np.float64)
        prev_targets = self._prev_targets
        kp = self.robot.joint_stiffness.numpy()
        kd = self.robot.joint_damping.numpy()
        lim = self.robot.effort_limit.numpy()

        self._episode_step += 1
        if self._episode_step % self._friction_resample_steps == 0:
            self._friction_force = float(np.random.uniform(*self._friction_range))
        if self.push_enabled and self._episode_step % 250 == 1:
            self._push_xy = np.clip(np.random.randn(2), -2, 2) * 10.0
            self._push_torque = np.clip(np.random.randn(3), -1, 1) * 2.0
        push_active = self.push_enabled and (
            (self._episode_step - 1) % 250) < 50   # 1 s every 5 s

        if not (0 <= self.delay_substeps <= self.decimation):
            raise RuntimeError(
                "actuator_delay_substeps must be within [0, decimation]")

        # IsaacLab's DelayBuffer returns the CURRENT target until it has
        # accumulated enough pushes (delay_buffer.py:162-165 "returns the
        # latest data"): the first policy step of an episode acts undelayed.
        delay = 0 if self._first_ctrl_step else self.delay_substeps

        for i in range(self.decimation):
            self._substep += 1
            # xfrc_applied = [force(3), torque(3)], both world-frame
            vxy = self.mj_data.qvel[self.ball_vadr:self.ball_vadr + 2]
            speed = max(float(np.linalg.norm(vxy)), 0.1)
            self.mj_data.xfrc_applied[self.ball_bid, :] = 0.0
            self.mj_data.xfrc_applied[self.ball_bid, 0:2] = \
                -self._friction_force * vxy / speed
            self.mj_data.xfrc_applied[self.trunk_bid, :] = 0.0
            if push_active:
                self.mj_data.xfrc_applied[self.trunk_bid, 0:2] = self._push_xy
                self.mj_data.xfrc_applied[self.trunk_bid, 3:6] = self._push_torque

            # substep-granular actuator delay: substeps i < delay still apply
            # the previous policy step's target (IsaacLab DelayBuffer with k
            # substeps of delay, k*2 ms here vs k*5 ms in training)
            applied = targets if i >= delay else prev_targets
            q = self.mj_data.qpos[self.qposadr]
            v = self.mj_data.qvel[self.dofadr]
            raw = kp * (applied - q) - kd * v
            if self._tn_vmax is not None:
                # Booster T-N torque-speed clip (training _clip_effort)
                tau_lin = lim * (self._tn_vmax - np.abs(v)) / self._tn_denom
                max_effort = np.clip(tau_lin, 0.0, lim)
                self.mj_data.ctrl[self.actadr] = np.clip(raw, -max_effort, max_effort)
            else:
                self.mj_data.ctrl[self.actadr] = np.clip(raw, -lim, lim)
            mujoco.mj_step(self.mj_model, self.mj_data)

        self._prev_targets = targets
        self._first_ctrl_step = False

    # ---------------------------------------------------------- termination

    def check_terminal(self, step: int, max_steps: int) -> tuple[str | None, dict]:
        """First-terminal-event check, training semantics (env_cfg.py:384-416)."""
        bx, by, bz = (float(x) for x in self.ball_pos_w)
        rx = float(self.robot.data.root_pos_w[0])
        ry = float(self.robot.data.root_pos_w[1])
        # strict goal geometry == acceptance test (commands.py ball_in_goal)
        in_goal = (
            bx > GOAL_X + BALL_RADIUS
            and abs(by) + BALL_RADIUS < GOAL_HALF_WIDTH
            and bz + BALL_RADIUS < GOAL_HEIGHT
        )
        ball_out = (
            bx < -FIELD_HALF_LENGTH
            or (bx > GOAL_X and abs(by) > GOAL_HALF_WIDTH)
            or abs(by) > FIELD_HALF_WIDTH
        )
        v = self.mj_data.qvel[self.root_vadr:self.root_vadr + 6]
        info = {
            "ball_xy": (round(bx, 3), round(by, 3)),
            "ball_speed": round(float(np.linalg.norm(
                self.mj_data.qvel[self.ball_vadr:self.ball_vadr + 6])), 3),
            "robot_xy": (round(rx, 3), round(ry, 3)),
        }
        if in_goal:
            return "goal", info
        if ball_out:
            return "ball_out", info
        if self.trunk_z < FALL_HEIGHT:
            return "fall", info
        if abs(rx) > FIELD_HALF_LENGTH + OUT_MARGIN or \
                abs(ry) > FIELD_HALF_WIDTH + OUT_MARGIN:
            return "robot_out", info
        if float(v @ v) > HIGH_VEL_SQ:
            return "high_velocity", info
        if step >= max_steps:
            return "timeout", info
        return None, info


# ------------------------------------------------------------- episode runner

def scenario_layout(num_episodes: int, seed: int):
    """Port of evaluate_kick_amp.py scenario_layout (scenario 'soccer')."""
    rng = np.random.default_rng(seed)
    ids = np.arange(num_episodes)
    yaw = np.array([0.0, np.pi / 2, np.pi, -np.pi / 2])[ids % 4]
    bearing = np.array([0.0, np.pi / 2, np.pi, -np.pi / 2])[(ids // 4) % 4]
    distance = rng.uniform(0.8, 2.0, num_episodes)
    angle = yaw + bearing
    return yaw, distance[:, None] * np.stack((np.cos(angle), np.sin(angle)), -1)


def run_sim2sim(cfg, episodes: int = 32, seed: int = 123, steps: int = 1500,
                realtime: bool = False, out: str | None = None,
                verbose: bool = True):
    controller = KickAmpMujocoController(cfg)
    controller.start()
    yaws, ball_xys = scenario_layout(episodes, seed)

    results = []
    t0 = time.perf_counter()
    for ep in range(episodes):
        controller.reset_episode(float(yaws[ep]), ball_xys[ep], seed=seed * 1000 + ep)
        min_dist = float("inf")
        touched = False
        ball_contact = False
        first_contact_step = None
        outcome, info = None, {}
        step = 0
        for step in range(steps):
            # first-terminal-event semantics: check on the current state
            outcome, info = controller.check_terminal(step, steps)
            if outcome is not None:
                break
            dist = float(torch.linalg.vector_norm(
                controller.ball_pos_w[:2] - controller.robot.data.root_pos_w[:2]))
            min_dist = min(min_dist, dist)
            touched = touched or dist < 0.30
            if not ball_contact and controller.ball_robot_in_contact():
                ball_contact = True
                first_contact_step = step
            dof_targets = controller.policy_step()
            controller.ctrl_step(dof_targets)
            controller.update_state()
            if realtime:
                time.sleep(cfg.policy_dt)
        else:
            outcome = "timeout"
        is_fall = outcome in ("fall", "high_velocity")
        fall_phase = None
        if is_fall:
            fall_phase = "after_touch" if ball_contact else "before_touch"
        bearing = np.arctan2(ball_xys[ep][1], ball_xys[ep][0]) - yaws[ep]
        results.append({
            "episode": ep,
            "yaw_deg": round(float(np.degrees(yaws[ep])), 1),
            "bearing_deg": round(float(np.degrees((bearing + np.pi) % (2 * np.pi) - np.pi)), 1),
            "ball_start": [round(float(x), 3) for x in ball_xys[ep]],
            "outcome": outcome,
            "steps": step + 1,
            "duration_s": round((step + 1) * cfg.policy_dt, 2),
            "min_ball_dist_m": round(min_dist, 3),
            "touched_ball": bool(touched),
            "ball_contact": bool(ball_contact),
            "first_contact_step": first_contact_step,
            "fall_phase": fall_phase,
            **info,
        })
        if verbose:
            print(f"  ep {ep + 1:2d}/{episodes}  yaw={results[-1]['yaw_deg']:5.1f}deg "
                  f"ball@{results[-1]['ball_start']}  -> {outcome:12s} "
                  f"after {results[-1]['duration_s']:5.1f}s  "
                  f"min_dist={min_dist:.2f}m  touch={'Y' if touched else 'N'}"
                  f"  ball_c={'Y' if ball_contact else 'N'}"
                  + (f"  fall={fall_phase}" if fall_phase else ""))

    counts: dict[str, int] = {}
    for r in results:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    fall_before = sum(1 for r in results if r.get("fall_phase") == "before_touch")
    fall_after = sum(1 for r in results if r.get("fall_phase") == "after_touch")
    summary = {
        "episodes": episodes,
        "seed": seed,
        "steps": steps,
        "perception": cfg.policy.perception,
        "foot_collision": getattr(cfg, "foot_collision", "box"),
        "actuator_delay_substeps": getattr(cfg, "actuator_delay_substeps", 5),
        "checkpoint": cfg.policy.checkpoint_path,
        "outcome_counts": counts,
        "goal_fraction": counts.get("goal", 0) / episodes,
        "out_fraction": counts.get("ball_out", 0) / episodes,
        "fall_fraction": (counts.get("fall", 0) + counts.get("high_velocity", 0)) / episodes,
        "fall_before_touch": fall_before,
        "fall_after_touch": fall_after,
        "touch_fraction": sum(r["touched_ball"] for r in results) / episodes,
        "ball_contact_fraction": sum(r["ball_contact"] for r in results) / episodes,
        "mean_duration_s": float(np.mean([r["duration_s"] for r in results])),
        "wall_time_s": round(time.perf_counter() - t0, 1),
        "per_episode": results,
    }
    if hashlib is not None and out:
        h = hashlib.sha256()
        with open(f"{controller.policy.task_path}/{cfg.policy.checkpoint_path}", "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        summary["checkpoint_sha256"] = h.hexdigest()
    if verbose:
        print("\n== sim2sim summary ==")
        for k, v in summary.items():
            if k != "per_episode":
                print(f"  {k}: {v}")
    if out:
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        if verbose:
            print(f"\nresults written to {out}")
    return summary


def run_viewer(cfg, seed: int = 123):
    """Interactive passive viewer: endless episodes, auto-reset on terminal."""
    import mujoco.viewer

    controller = KickAmpMujocoController(cfg)
    with mujoco.viewer.launch_passive(controller.mj_model, controller.mj_data) as viewer:
        viewer.cam.elevation = -20
        yaws, ball_xys = scenario_layout(16, seed)
        controller.start()
        ep = 0
        while viewer.is_running():
            yaw = float(yaws[ep % 16])
            ball = ball_xys[ep % 16]
            ep += 1
            print(f"episode {ep}: yaw {np.degrees(yaw):.0f} deg, ball at {ball}")
            controller.reset_episode(yaw, ball, seed=seed * 1000 + ep)
            outcome = None
            for step in range(1500):
                if not viewer.is_running():
                    return
                controller.update_state()
                outcome, _ = controller.check_terminal(step, 1500)
                if outcome is not None:
                    break
                dof_targets = controller.policy_step()
                controller.ctrl_step(dof_targets)
                time.sleep(cfg.policy_dt)
                viewer.cam.lookat[:] = controller.mj_data.qpos[
                    controller.root_qadr:controller.root_qadr + 3]
                viewer.sync()
            print(f"episode {ep}: {outcome or 'timeout'} "
                  f"at {(step + 1) * cfg.policy_dt:.1f}s")
