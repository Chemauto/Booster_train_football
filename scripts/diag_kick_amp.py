"""Deterministic play diagnostic: run a checkpoint with mean actions (no
exploration noise) on 64 envs and log the physical failure mode -- trunk
height trajectory, termination causes, and which reward terms fire."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import booster_train.tasks  # noqa: F401
from booster_train.rsl_rl.amp.runner import AmpRunner


def main():
    env_cfg = parse_env_cfg("Booster-K1-KickAMP-v0-Play", device=args.device, num_envs=64)
    entry = gym.registry["Booster-K1-KickAMP-v0"].kwargs["rsl_rl_cfg_entry_point"]
    if isinstance(entry, str):
        import importlib
        module, cls = entry.rsplit(":", 1)
        entry = getattr(importlib.import_module(module), cls)()
    entry.num_envs = 64

    env = gym.make("Booster-K1-KickAMP-v0-Play", cfg=env_cfg).unwrapped
    runner = AmpRunner(env, entry, train_mode=False)
    runner.load(args.checkpoint)

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    tm = env.termination_manager
    causes = {"base_contact": 0, "out_of_field": 0, "high_velocity": 0, "time_out": 0}
    heights, alive_frac = [], []
    with torch.no_grad():
        for t in range(args.steps):
            done_prev = tm.dones.clone()
            dist, _ = runner.model.act(runner.obs_norm(obs), runner.stacked_obs)
            obs, _, terminated, time_outs, _ = env.step(dist.mean)  # deterministic
            obs = obs["policy"]
            done = terminated | time_outs
            runner._push_obs(runner.obs_norm(obs), done)  # keep stacked history live
            for name in ("base_contact", "out_of_field", "high_velocity"):
                term_dones = tm.get_term(name)
                causes[name] += int((term_dones if torch.is_tensor(term_dones) else term_dones.dones).sum())
            causes["time_out"] += int(time_outs.sum())
            z = env.scene["robot"].data.root_pos_w[:, 2]
            heights.append(float(z.mean()))
            alive_frac.append(1.0 - float(terminated.float().mean()))
            if t % 50 == 0:
                print(f"[t={t:3d}] 存活率={alive_frac[-1]:.2f} 躯干高 均值={z.mean():.3f} "
                      f"min={z.min():.3f} max={z.max():.3f} | 累计死因 {causes}", flush=True)

    h = torch.tensor(heights)
    print(f"\n=== 诊断汇总 ({args.steps} 步确定性动作) ===")
    print(f"躯干高度: 起{h[0]:.3f} → 终{h[-1]:.3f} (峰{h.max():.3f} 谷{h.min():.3f})")
    print(f"末步存活率: {alive_frac[-1]:.2f} | 全程死因: {causes}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
