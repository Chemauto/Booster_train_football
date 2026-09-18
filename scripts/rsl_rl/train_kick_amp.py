"""Train the K1 kick_amp task with the custom AMP runner.

.. code-block:: bash

    # smoke test
    python scripts/rsl_rl/train_kick_amp.py --task Booster-K1-KickAMP-v0 --headless --num_envs 1024 --max_iterations 50
    # full run
    python scripts/rsl_rl/train_kick_amp.py --task Booster-K1-KickAMP-v0 --headless --num_envs 4096
"""

"""Launch Isaac Sim Simulator first."""

import argparse

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
parser.add_argument("--reset_optimization", action="store_true", help="New reward phase: retain actor, reset critics/PPO state and exploration to sigma=.15.")
parser.add_argument("--motion_dir", type=str, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

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
        runner.learn()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
