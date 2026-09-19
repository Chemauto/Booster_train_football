"""Train the K1 kick_amp task with the custom AMP runner.

.. code-block:: bash

    # smoke test
    python scripts/rsl_rl/train_kick_amp.py --task Booster-K1-KickAMP-v0 --headless --num_envs 1024 --max_iterations 50
    # full run
    python scripts/rsl_rl/train_kick_amp.py --task Booster-K1-KickAMP-v0 --headless --num_envs 4096
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train kick_amp with the AMP runner.")
parser.add_argument("--task", type=str, default="Booster-K1-KickAMP-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default=None, help="Resume from an AmpRunner checkpoint (.pt).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos after training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=50, help="Interval between video recordings (in steps).")
parser.add_argument("--play", action="store_true", default=False, help="Play a checkpoint instead of training.")
parser.add_argument("--play_steps", type=int, default=1500)
parser.add_argument("--save_interval", type=int, default=None)
parser.add_argument("--experiment_name", type=str, default=None)
parser.add_argument("--nominal_physics", action="store_true", help="Disable physical randomization and disturbances for baseline learning.")
parser.add_argument("--perfect_perception", action="store_true", help="Use true ball coordinates for the initial control benchmark.")
parser.add_argument("--near_ball", action="store_true", help="Reset the ball 0.8-2 m in front of the robot.")
parser.add_argument("--training_phase", choices=("soccer", "approach"), default="soccer")
parser.add_argument("--task_weight_floor", type=float, default=0.0, help="Minimum task policy weight during the curriculum, in [0, 1].")
parser.add_argument("--boundary_weight", type=float, default=0.0,
                    help="Dense field-edge penalty, <= 0 (0 = off). The out-of-field termination is a 0/1 cliff and the goal line is the field edge, so without a gradient the safe way to push the ball goalward is not to push it.")
parser.add_argument("--boundary_outward_weight", type=float, default=0.0,
                    help="Edge penalty charging only the outward base velocity near the line, <= 0 (0 = off). Unlike --boundary_weight this leaves the scoring push free, because pushing the ball over the goal line happens within ~0.3 m of that same line.")
parser.add_argument("--ball_spawn_bearing_deg", type=float, default=None,
                    help="Spawn half-range in degrees around the robot's heading; 180 places the ball anywhere on the circle. Default keeps the configured +-0.8 rad cone.")
parser.add_argument("--ball_lateral_weight", type=float, default=0.0,
                    help="Charge the ball's sideways speed while a foot is on it near the goal, <= 0 (0 = off). Orthogonal to the goalward push, so it cannot tax scoring.")
parser.add_argument("--reset_optimization", action="store_true", help="New reward phase: retain actor, reset critics/PPO state and exploration to sigma=.15.")
parser.add_argument("--motion_dir", type=str, default=None)
parser.add_argument("--completion_file", type=str, default=None, help="Write actual training counters and final checkpoint as JSON after saving.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not math.isfinite(args_cli.task_weight_floor) or not 0.0 <= args_cli.task_weight_floor <= 1.0:
    parser.error("--task_weight_floor must be finite and in [0, 1]")
if not math.isfinite(args_cli.boundary_weight) or args_cli.boundary_weight > 0.0:
    parser.error("--boundary_weight must be finite and <= 0; it is a penalty and 0 disables it")
if not math.isfinite(args_cli.boundary_outward_weight) or args_cli.boundary_outward_weight > 0.0:
    parser.error("--boundary_outward_weight must be finite and <= 0; it is a penalty and 0 disables it")
if not math.isfinite(args_cli.ball_lateral_weight) or args_cli.ball_lateral_weight > 0.0:
    parser.error("--ball_lateral_weight must be finite and <= 0; it is a penalty and 0 disables it")
if args_cli.ball_spawn_bearing_deg is not None and not 0.0 < args_cli.ball_spawn_bearing_deg <= 180.0:
    parser.error("--ball_spawn_bearing_deg must be in (0, 180]")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""

import gymnasium as gym
import os
import torch
import signal

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import booster_train.tasks  # noqa: F401  (registers Booster-* tasks)
from booster_train.rsl_rl.amp.runner import AmpRunner


def learn_with_completion(runner, completion_file=None):
    """Publish completion only after learn has saved its actual final iteration."""
    import json
    from pathlib import Path

    start = int(runner.tot_iterations)
    updates = int(runner.cfg.max_iterations)
    target = start + updates
    runner.learn()
    end = int(runner.tot_iterations)
    checkpoint = Path(runner.log_dir, f"model_{end}.pt").resolve(strict=True)
    result = {
        "status": "completed" if end == target and not runner.stop_requested else "stopped",
        "start_iteration": start, "requested_updates": updates,
        "requested_end_iteration": target, "end_iteration": end,
        "completed_updates": end - start, "checkpoint": str(checkpoint),
    }
    path = Path(completion_file) if completion_file else Path(runner.log_dir, "training_result.json")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    tmp.replace(path)
    print(f"[train] result: {json.dumps(result)}", flush=True)


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    ) if hasattr(args_cli, "disable_fabric") else parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    agent_cfg_cls = gym.registry[args_cli.task].kwargs.get("rsl_rl_cfg_entry_point")
    if isinstance(agent_cfg_cls, str):
        import importlib
        module, cls = agent_cfg_cls.rsplit(":", 1)
        agent_cfg = getattr(importlib.import_module(module), cls)()
    else:
        agent_cfg = agent_cfg_cls()
    if args_cli.num_envs is not None:
        agent_cfg.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.save_interval is not None:
        agent_cfg.save_interval = args_cli.save_interval
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    env_cfg.seed = agent_cfg.seed
    env_cfg.commands.soccer.task_weight_floor = args_cli.task_weight_floor
    env_cfg.rewards.boundary.weight = args_cli.boundary_weight
    env_cfg.rewards.boundary_outward.weight = args_cli.boundary_outward_weight
    env_cfg.rewards.ball_lateral.weight = args_cli.ball_lateral_weight
    if args_cli.ball_spawn_bearing_deg is not None:
        half = math.radians(args_cli.ball_spawn_bearing_deg)
        env_cfg.commands.soccer.ball_spawn_bearing = (-half, half)
    if args_cli.motion_dir:
        env_cfg.commands.soccer.motion_dir = args_cli.motion_dir
    if args_cli.nominal_physics:
        for name in ("physics_material", "add_default_joint_offset", "actuator_gains", "trunk_mass", "trunk_com", "ball_mass", "kick_robot"):
            setattr(env_cfg.events, name, None)
        env_cfg.commands.soccer.push_enabled = False
        env_cfg.commands.soccer.ball_kick_prob = 0.0
        env_cfg.commands.soccer.ball_teleport_prob = 0.0
        env_cfg.commands.soccer.ball_friction_range = (0.2, 0.2)
        for actuator in env_cfg.scene.robot.actuators.values():
            actuator.min_delay = actuator.max_delay = 2
    if args_cli.perfect_perception:
        env_cfg.commands.soccer.perfect_perception = True
        env_cfg.observations.policy.enable_corruption = False
    if args_cli.near_ball:
        env_cfg.commands.soccer.ball_spawn_distance = (0.8, 2.0)
    from booster_train.tasks.manager_based.kick_amp.curriculum import configure_training_phase
    configure_training_phase(env_cfg, agent_cfg, args_cli.training_phase)
    from booster_train.tasks.manager_based.kick_amp.mdp.commands import MINIMAL
    if MINIMAL:
        # A zero task reward must not create a normalized random advantage
        # through the untrained task critic during locomotion ablations.
        agent_cfg.advantage_coef = (0.0, 1.0)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    # sanity: obs group names expected by the runner
    assert "policy" in env.observation_space.spaces, f"obs groups: {list(env.observation_space.spaces)}"
    assert "critic_observations" in env.observation_space.spaces, f"obs groups: {list(env.observation_space.spaces)}"

    runner = AmpRunner(env, agent_cfg, train_mode=not args_cli.play)
    if not args_cli.play:
        import json
        from pathlib import Path
        Path(runner.log_dir, "launch_args.json").write_text(json.dumps(vars(args_cli), indent=2, default=str))
    if args_cli.checkpoint:
        runner.load(args_cli.checkpoint)
    if args_cli.reset_optimization:
        runner.reset_optimization_for_phase()

    if args_cli.play:
        if not args_cli.checkpoint:
            raise ValueError("--play requires --checkpoint; use scripts/evaluate_kick_amp.py for metrics")
        runner.model.eval()
        runner.obs_norm.eval()
        obs, _ = env.reset()
        alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        with torch.inference_mode():
            for _ in range(args_cli.play_steps):
                dist, _ = runner.model.act(runner.obs_norm(obs["policy"]), runner.stacked_obs)
                obs, _, terminated, truncated, _ = env.step(dist.mean)
                alive &= ~terminated
                runner._push_obs(runner.obs_norm(obs["policy"]), terminated | truncated)
        print(f"[play] continuous survival: {alive.float().mean().item():.3f}")
    else:
        def request_stop(signum, frame):
            runner.stop_requested = True
        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        print(f"[train] log dir: {runner.log_dir}")
        learn_with_completion(runner, args_cli.completion_file)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
