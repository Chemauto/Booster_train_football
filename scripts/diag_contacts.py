"""One-env contact audit for kick_amp reward debugging.

Runs the play env at the default reset pose with zero policy actions and reports,
per rigid body, how often net contact force exceeds the collision reward's 1 N
threshold. This identifies persistent legal/self contacts that the current
"all non-foot bodies" penalty mistakes for a collision.
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
import booster_train.tasks  # noqa: F401

cfg = parse_env_cfg("Booster-K1-KickAMP-v0-Play", device=args.device, num_envs=1)
env = gym.make("Booster-K1-KickAMP-v0-Play", cfg=cfg).unwrapped
env.reset()
robot = env.scene["robot"]
sensor = env.scene["contact_forces"]
# the sensor's row order is NOT the articulation order (USD prim traversal,
# see ContactSensor._initialize_impl) -- every tensor here must be sized and
# labelled by the sensor's own body list
body_names = list(sensor.body_names)
steps = 120
hits = torch.zeros(len(body_names), device=env.device)
force_sum = torch.zeros(len(body_names), device=env.device)
force_max = torch.zeros(len(body_names), device=env.device)
zero = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)

# let actuator-delayed default targets and contacts settle first
for _ in range(40):
    env.step(zero)
for _ in range(steps):
    env.step(zero)
    mag = torch.linalg.norm(sensor.data.net_forces_w[0], dim=-1)
    hits += (mag > 1.0)
    force_sum += mag
    force_max = torch.maximum(force_max, mag)

feet = {i for i, name in enumerate(body_names)
        if name in ("left_ankle_roll_link", "right_ankle_roll_link")}
import json
result = []
for i in torch.argsort(force_sum, descending=True).tolist():
    if force_max[i] > 0.1:
        result.append({
            "index": i, "foot": i in feet, "body": body_names[i],
            "hit_pct": 100 * hits[i].item() / steps,
            "mean_N": force_sum[i].item() / steps,
            "max_N": force_max[i].item(),
        })
with open("logs/diag_contacts.json", "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2), flush=True)
env.close()
app.close()
