"""Dump BC-relevant constants from a live IsaacLab instance (run once).

Offline BC must construct observations that are dimension-for-dimension
identical to the deployed policy's. Rather than re-deriving joint order,
default pose, and per-joint action scales from formulas (a classic source of
silent mismatch), this script asks the real env for them and writes a json.

Usage: python scripts/bc_dump_constants.py --headless --device cuda:0
"""

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--output", default="logs/bc_constants_nominal.json")
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
import booster_train.tasks  # noqa: F401  (task registration)
from booster_train.assets.robots.booster import K1_ACTION_SCALE

cfg = parse_env_cfg("Booster-K1-KickAMP-v0-Play", device=args.device, num_envs=1)
from evaluate_kick_amp import configure_evaluation
configure_evaluation(cfg, "standing", seed=42, perfect_perception=True)
env = gym.make("Booster-K1-KickAMP-v0-Play", cfg=cfg).unwrapped
env.reset()
robot = env.scene["robot"]

default_q = robot.data.default_joint_pos[0].cpu().tolist()
# resolved per-joint scale straight from the action term (K1_ACTION_SCALE the
# dict may miss joints; the action manager holds the ground truth actually
# applied at deploy time)
term = next(iter(env.action_manager._terms.values()))
scale_vec = term._scale.cpu().flatten().tolist()
const = {
    "joint_names": list(robot.joint_names),          # PhysX BFS order == obs order
    "default_joint_pos": default_q,                  # crouch legs + shoulder roll
    "action_offset_resolved": term._offset[0].cpu().tolist(),
    "action_scale_resolved": scale_vec,              # aligned to joint_names order
    "obs_dim": int(env.observation_space["policy"].shape[-1]),
    "num_stack": 50,
    "field": {"goal_x": 7.0, "goal_y": 0.0},
}
with open(args.output, "w") as f:
    json.dump(const, f, indent=1)
print("BC_CONSTANTS_WRITTEN", len(const["joint_names"]), "joints, obs_dim", const["obs_dim"], flush=True)
env.close()
app.close()
