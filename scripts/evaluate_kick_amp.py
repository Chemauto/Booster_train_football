"""Seeded, mean-action K1 evaluation; each initial environment counts once.

Example (Isaac Lab Python):
  python scripts/evaluate_kick_amp.py --headless --checkpoint model_1000.pt \
      --scenario soccer --num_envs 64 --steps 1500 --seed 123 --output eval.json

The statistics and scenario layout import on CPU without starting Isaac Sim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def scenario_layout(num_envs, scenario, seed):
    """Balanced headings/bearings, seeded distances; coordinates are env-local."""
    if num_envs < 1 or scenario not in ("standing", "soccer", "approach"):
        raise ValueError("num_envs must be positive and scenario standing, soccer or approach")
    if scenario == "standing":
        return {"yaw": np.zeros(num_envs), "ball_xy": np.tile([2., 0.], (num_envs, 1))}
    rng = np.random.default_rng(seed)
    ids = np.arange(num_envs)
    yaw = np.array([0., np.pi / 2, np.pi, -np.pi / 2])[ids % 4]
    bearing = np.array([0., np.pi / 2, np.pi, -np.pi / 2])[(ids // 4) % 4]
    distance = rng.uniform(.8, 2., num_envs)
    if scenario == "approach":
        bearing = np.array([-.6, -.3, .3, .6])[(ids // 4) % 4]
        distance = rng.uniform(1., 2., num_envs)
    angle = yaw + bearing
    return {"yaw": yaw, "ball_xy": distance[:, None] * np.stack((np.cos(angle), np.sin(angle)), -1)}


def assess_approach(report):
    """A staged-training gate, not a claim of soccer success."""
    successes = [
        bool(e["survived_first_episode"] and not e.get("ball_discontinuity", False)
             and e["robot_displacement_m"] >= .5
             and e["minimum_robot_ball_distance_m"] <= .55
             and e.get("alternating_support_switches", 0) >= 4)
        for e in report["per_env"]
    ]
    fraction = sum(successes) / report["cohort_size"]
    return {"passed": bool(report["first_episode_survival_fraction"] >= .9 and fraction >= .8),
            "success_count": sum(successes), "success_fraction": fraction,
            "criteria": "survival >=90%; >=80% survive, displace >=0.5m, reach <=0.55m and alternate support >=4 times"}


class CohortMetrics:
    """First-episode statistics using physical snapshots BEFORE automatic reset.

    Contact is a conservative geometric proxy, not pair-specific contact sensing:
    a foot within 0.25 m in 3D followed by >0.2 m ball displacement. Implausible
    speed/jumps permanently invalidate soccer attribution for that environment.
    """

    def __init__(self, robot, ball, feet, dt):
        self.dt = float(dt)
        self.n = len(robot)
        self.initial_robot = np.array(robot, copy=True)
        self.previous_robot = np.array(robot, copy=True)
        self.previous_ball = np.array(ball, copy=True)
        self.previous_feet = np.array(feet, copy=True)
        self.previous_speed = np.zeros(self.n)
        self.active = np.ones(self.n, bool)
        self.fallen = np.zeros(self.n, bool)
        self.elapsed = np.zeros(self.n)
        self.fall_time = np.full(self.n, np.nan)
        self.path = np.zeros(self.n)
        self.displacement = np.zeros(self.n)
        self.initial_distance = np.linalg.norm(robot[:, :2] - ball[:, :2], axis=-1)
        self.minimum_distance = self.initial_distance.copy()
        self.minimum_height = np.array(robot[:, 2], copy=True)
        self.contact_candidate = np.zeros(self.n, bool)
        self.contact_ball = np.zeros_like(ball)
        self.contact_proxy = np.zeros(self.n, bool)
        self.ball_displacement = np.zeros(self.n)
        self.goal_progress = np.zeros(self.n)
        self.goal = np.zeros(self.n, bool)
        self.discontinuous = np.zeros(self.n, bool)
        self.end_reason = np.full(self.n, "horizon_censored", dtype=object)
        self.causes = {}
        self.support_code = np.zeros(self.n, dtype=int)
        self.support_streak = np.zeros(self.n, dtype=int)
        self.last_stable_support = np.zeros(self.n, dtype=int)
        self.support_switches = np.zeros(self.n, dtype=int)

    def update(self, robot, ball, feet, ball_velocity, terminated, timeout, causes, foot_contacts=None):
        terminated, timeout = np.asarray(terminated, bool), np.asarray(timeout, bool)
        active = self.active.copy()
        self.elapsed[active] += self.dt
        if foot_contacts is not None:
            contacts = np.asarray(foot_contacts, bool)
            code = np.where(contacts.sum(axis=1) == 1, contacts[:, 0].astype(int) + 2*contacts[:, 1].astype(int), 0)
            self.support_streak = np.where(active & (code == self.support_code) & (code != 0), self.support_streak + 1, 1)
            self.support_code = code
            stable = active & (code != 0) & (self.support_streak >= int(np.ceil(.06 / self.dt)))
            switched = stable & (self.last_stable_support != 0) & (code != self.last_stable_support)
            self.support_switches[switched] += 1
            self.last_stable_support[stable] = code[stable]
        step_robot = np.linalg.norm(robot[:, :2] - self.previous_robot[:, :2], axis=-1)
        self.path[active] += step_robot[active]
        self.displacement[active] = np.linalg.norm(robot[:, :2] - self.initial_robot[:, :2], axis=-1)[active]
        self.minimum_height[active] = np.minimum(self.minimum_height[active], robot[active, 2])
        speed = np.linalg.norm(ball_velocity, axis=-1)
        step_ball = np.linalg.norm(ball - self.previous_ball, axis=-1)
        plausible = (speed <= 25.) & (step_ball <= np.maximum(.1, 1.5 * (speed + self.previous_speed) * self.dt + .05))
        plausible &= np.isfinite(ball).all(axis=-1) & np.isfinite(speed)
        self.discontinuous |= active & ~plausible
        valid = active & ~self.discontinuous
        distance = np.linalg.norm(robot[:, :2] - ball[:, :2], axis=-1)
        self.minimum_distance[valid] = np.minimum(self.minimum_distance[valid], distance[valid])
        previous_near = np.linalg.norm(self.previous_feet - self.previous_ball[:, None], axis=-1).min(axis=-1) < .25
        current_near = np.linalg.norm(feet - ball[:, None], axis=-1).min(axis=-1) < .25
        new_candidate = valid & ~self.contact_candidate & (previous_near | current_near)
        self.contact_ball[new_candidate] = self.previous_ball[new_candidate]
        self.contact_candidate |= new_candidate
        candidate_disp = np.linalg.norm(ball[:, :2] - self.contact_ball[:, :2], axis=-1)
        self.contact_proxy |= valid & self.contact_candidate & (candidate_disp > .2)
        touched = valid & self.contact_proxy
        self.ball_displacement[touched] = np.maximum(self.ball_displacement[touched], candidate_disp[touched])
        goal_xy = np.array([7., 0.])
        progress = np.linalg.norm(self.contact_ball[:, :2] - goal_xy, axis=-1) - np.linalg.norm(ball[:, :2] - goal_xy, axis=-1)
        self.goal_progress[touched] = progress[touched]
        # Entire 0.11 m radius ball crosses the line inside the mouth, below bar.
        line = 7.11
        crossing = (self.previous_ball[:, 0] <= line) & (ball[:, 0] > line)
        alpha = np.divide(line - self.previous_ball[:, 0], ball[:, 0] - self.previous_ball[:, 0],
                          out=np.zeros(self.n), where=np.abs(ball[:, 0] - self.previous_ball[:, 0]) > 1e-12)
        at_line = self.previous_ball + alpha[:, None] * (ball - self.previous_ball)
        mouth = (np.abs(at_line[:, 1]) + .11 < 1.3) & (at_line[:, 2] + .11 < 1.8) & (at_line[:, 2] >= 0.)
        self.goal |= touched & crossing & mouth
        for name, mask in causes.items():
            mask = np.asarray(mask, bool) & active
            self.causes[name] = self.causes.get(name, 0) + int(mask.sum())
            self.end_reason[mask] = name
        fall = np.asarray(causes.get("base_contact", np.zeros(self.n, bool)), bool) & active
        self.fallen |= fall
        self.fall_time[fall] = self.elapsed[fall]
        self.end_reason[active & terminated & (self.end_reason == "horizon_censored")] = "terminated"
        self.end_reason[active & timeout & ~terminated] = "time_out"
        self.active &= ~(terminated | timeout)
        self.previous_robot = np.array(robot, copy=True)
        self.previous_ball = np.array(ball, copy=True)
        self.previous_feet = np.array(feet, copy=True)
        self.previous_speed = speed

    def report(self):
        per_env = []
        for i in range(self.n):
            per_env.append({
                "env_id": i, "survived_first_episode": bool(self.active[i]),
                "time_to_first_fall_s": float(self.fall_time[i]) if self.fallen[i] else None,
                "episode_duration_s": float(self.elapsed[i]), "end_reason": self.end_reason[i],
                "robot_path_length_m": float(self.path[i]), "robot_displacement_m": float(self.displacement[i]),
                "minimum_trunk_height_m": float(self.minimum_height[i]),
                "initial_robot_ball_distance_m": float(self.initial_distance[i]),
                "minimum_robot_ball_distance_m": float(self.minimum_distance[i]),
                "approach_progress_m": float(self.initial_distance[i] - self.minimum_distance[i]),
                "alternating_support_switches": int(self.support_switches[i]),
                "contact_proxy": bool(self.contact_proxy[i]),
                "ball_displacement_after_contact_m": float(self.ball_displacement[i]),
                "ball_goal_progress_after_contact_m": float(self.goal_progress[i]),
                "validated_goal": bool(self.goal[i]), "ball_discontinuity": bool(self.discontinuous[i]),
            })
        return {
            "cohort_size": self.n, "first_episode_survival_fraction": float(self.active.mean()),
            "fall_count": int(self.fallen.sum()), "fall_fraction": float(self.fallen.mean()),
            "time_to_first_fall_s_mean_observed": float(self.fall_time[self.fallen].mean()) if self.fallen.any() else None,
            "mean_first_episode_duration_s": float(self.elapsed.mean()),
            "contact_proxy_count": int(self.contact_proxy.sum()), "validated_goal_count": int(self.goal.sum()),
            "validated_goal_fraction": float(self.goal.mean()), "ball_discontinuity_count": int(self.discontinuous.sum()),
            "termination_causes": self.causes, "per_env": per_env,
        }


def configure_evaluation(env_cfg, scenario, seed, perfect_perception=False, actuator_delay=2):
    """Disable startup/reset/interval DR before gym.make applies any events."""
    env_cfg.seed = seed
    for name in tuple(vars(env_cfg.events)):
        if not name.startswith("_"):
            setattr(env_cfg.events, name, None)
    env_cfg.observations.policy.enable_corruption = False
    cmd_cfg = env_cfg.commands.soccer
    for name in ("push_enabled", "ball_reset_enabled"):
        if not hasattr(cmd_cfg, name):
            raise RuntimeError(f"Evaluation requires SoccerStateCommandCfg.{name}")
        setattr(cmd_cfg, name, False)
    cmd_cfg.ball_teleport_prob = 0.
    cmd_cfg.ball_kick_prob = 0.
    if not hasattr(cmd_cfg, "ball_friction_range"):
        raise RuntimeError("Evaluation requires configurable ball_friction_range to disable physical DR")
    cmd_cfg.ball_friction_range = (.2, .2)
    cmd_cfg.perfect_perception = perfect_perception
    for actuator in env_cfg.scene.robot.actuators.values():
        if hasattr(actuator, "min_delay"):
            actuator.min_delay = actuator.max_delay = actuator_delay
    from isaaclab.managers import EventTermCfg
    env_cfg.events.evaluation_reset = EventTermCfg(
        func=reset_evaluation_scene, mode="reset", params={"scenario": scenario, "seed": seed})


def reset_evaluation_scene(env, env_ids, scenario, seed):
    """Every reset has the same fixed default stance; only episode one is scored."""
    import torch
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    layout = scenario_layout(env.num_envs, scenario, seed)
    yaw = torch.as_tensor(layout["yaw"], device=env.device, dtype=torch.float32)[env_ids]
    robot, ball = env.scene["robot"], env.scene["ball"]
    root = robot.data.default_root_state[env_ids].clone()
    root[:, :3] += env.scene.env_origins[env_ids]
    root[:, 3:7] = 0.
    root[:, 3], root[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
    root[:, 7:] = 0.
    robot.write_root_state_to_sim(root, env_ids=env_ids)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos[env_ids].clone(),
                                  torch.zeros_like(robot.data.default_joint_vel[env_ids]), env_ids=env_ids)
    state = ball.data.default_root_state[env_ids].clone()
    state[:, :3] += env.scene.env_origins[env_ids]
    state[:, :2] = torch.as_tensor(layout["ball_xy"], device=env.device, dtype=torch.float32)[env_ids] + env.scene.env_origins[env_ids, :2]
    state[:, 7:] = 0.
    ball.write_root_state_to_sim(state, env_ids=env_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scenario", choices=("standing", "soccer", "approach"), default="soccer")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--perfect_perception", action="store_true")
    parser.add_argument("--actuator_delay", type=int, default=2, help="Fixed motor delay in physics steps.")
    parser.add_argument("--output", type=Path, default=Path("kick_amp_evaluation.json"))
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.steps < 1 or args.num_envs < 1:
        parser.error("steps and num_envs must be positive")
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    app = AppLauncher(args).app
    env = None
    try:
        import importlib
        import random
        import gymnasium as gym
        import torch
        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        import booster_train.tasks  # noqa: F401
        from booster_train.rsl_rl.amp.runner import AmpRunner

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        task = "Booster-K1-KickAMP-v0-Play"
        cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
        configure_evaluation(cfg, args.scenario, args.seed, args.perfect_perception, args.actuator_delay)
        cfg.episode_length_s = max(cfg.episode_length_s, args.steps * cfg.sim.dt * cfg.decimation + 1.)
        entry = gym.registry["Booster-K1-KickAMP-v0"].kwargs["rsl_rl_cfg_entry_point"]
        if isinstance(entry, str):
            module, cls = entry.rsplit(":", 1)
            entry = getattr(importlib.import_module(module), cls)()
        elif isinstance(entry, type):
            entry = entry()
        entry.seed = args.seed
        # Evaluation does not train or need a full training-sized replay buffer.
        entry.replay_buffer_size = args.num_envs
        env = gym.make(task, cfg=cfg).unwrapped
        runner = AmpRunner(env, entry, train_mode=False)
        runner.load(str(checkpoint))
        runner.model.eval()
        runner.obs_norm.eval()
        runner.discriminator.eval()
        normalizer_before = {k: v.clone() for k, v in runner.obs_norm.state_dict().items()}
        env.common_step_counter = 0
        obs_dict, _ = env.reset(seed=args.seed)
        cmd = env.command_manager.get_term("soccer")
        cmd.ball_friction_force.fill_(.2)
        cmd._update_command()  # refresh caches after the deterministic reset
        obs = env.observation_manager.compute()["policy"]
        runner.stacked_obs.zero_()

        def physical_snapshot():
            origins = env.scene.env_origins
            robot, ball = env.scene["robot"], env.scene["ball"]
            return tuple(t.detach().cpu().numpy().copy() for t in (
                robot.data.root_pos_w - origins, ball.data.root_pos_w - origins,
                robot.data.body_pos_w[:, cmd.feet_idx] - origins[:, None], ball.data.root_lin_vel_w))

        robot, ball, feet, _ = physical_snapshot()
        metrics = CohortMetrics(robot, ball, feet, env.step_dt)
        tm = env.termination_manager
        original_compute = tm.compute

        def compute_and_capture():
            result = original_compute()
            robot, ball, feet, velocity = physical_snapshot()
            causes = {name: tm.get_term(name).detach().cpu().numpy().copy() for name in tm.active_terms}
            metrics.update(robot, ball, feet, velocity, tm.terminated.detach().cpu().numpy(),
                           tm.time_outs.detach().cpu().numpy(), causes,
                           (env.scene["contact_forces"].data.net_forces_w[:, cmd.feet_sensor_idx].norm(dim=-1) > 1.).detach().cpu().numpy())
            return result

        # Isaac Lab runs termination compute after physics and before any reset.
        tm.compute = compute_and_capture
        success_signal_count = 0
        with torch.inference_mode():
            for step in range(args.steps):
                active_before = metrics.active.copy()
                dist, _ = runner.model.act(runner.obs_norm(obs), runner.stacked_obs)
                observations, _, terminated, timeout, extras = env.step(dist.mean)
                obs = observations["policy"]
                runner._push_obs(runner.obs_norm(obs), terminated | timeout)
                success = extras.get("success")
                if success is not None:
                    success_signal_count += int((success.detach().cpu().numpy().astype(bool) & active_before).sum())
                if step % 100 == 0 or step + 1 == args.steps:
                    print(f"step={step + 1} first_episode_survival={metrics.active.mean():.3f} "
                          f"falls={metrics.fallen.sum()} contact_proxy={metrics.contact_proxy.sum()} "
                          f"validated_goals={metrics.goal.sum()}", flush=True)
        for name, before in normalizer_before.items():
            if not torch.equal(before, runner.obs_norm.state_dict()[name]):
                raise RuntimeError(f"Evaluation mutated observation normalizer: {name}")
        report = metrics.report()
        if args.scenario == "approach":
            report["approach_gate"] = assess_approach(report)
        report.update({
            "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "checkpoint_iteration": runner.tot_iterations, "scenario": args.scenario,
            "seed": args.seed, "steps": args.steps, "step_dt_s": env.step_dt,
            "task_success_signal_steps_untrusted": success_signal_count,
            "initial_layout": {k: v.tolist() for k, v in scenario_layout(args.num_envs, args.scenario, args.seed).items()},
            "protocol": {
                "policy": "mean actions, eval modules, frozen observation normalization",
                "cohort": "first episode only; snapshot before reset; no restart after fall/termination/timeout",
                "randomization": f"all event DR disabled; fixed default stance; actuator delay {args.actuator_delay} physics steps; rolling resistance 0.2 N",
                "perception_noise_disabled": args.perfect_perception,
                "interventions": "robot push, random ball kick, teleport and ball-only resets disabled",
                "contact": "proxy: foot-ball 3D center distance <0.25 m, then ball XY displacement >0.2 m",
                "goal": "prior contact proxy and entire ball crosses x=7 m inside 2.6 m mouth below 1.8 m crossbar",
                "limitations": ["Contact proxy does not prove a pair-specific foot-ball collision.",
                                "Null fall times are censored, not zero; non-fall terminations are reported separately.",
                                "Seeded PhysX results may vary across hardware/software; no bitwise guarantee.",
                                "If perception_noise_disabled is false, virtual camera noise/dropout remains seeded.",
                                "Any implausible ball jump invalidates subsequent soccer attribution."],
            },
        })
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Saved evaluation: {args.output.resolve()}", flush=True)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
