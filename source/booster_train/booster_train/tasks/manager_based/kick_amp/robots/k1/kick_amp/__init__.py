"""Registrations for the K1 end-to-end soccer-kick task (paper 2511.03996)."""

import gymnasium as gym

import os

from . import env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Booster-K1-KickAMP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.KickAmpEnvCfg,
        "rsl_rl_cfg_entry_point": "booster_train.tasks.manager_based.kick_amp.robots.k1.kick_amp.ppo_cfg:KickAmpPPORunnerCfg",
    },
)

gym.register(
    id="Booster-K1-KickAMP-v0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.PlayKickAmpEnvCfg,
        "rsl_rl_cfg_entry_point": "booster_train.tasks.manager_based.kick_amp.robots.k1.kick_amp.ppo_cfg:KickAmpPPORunnerCfg",
    },
)
