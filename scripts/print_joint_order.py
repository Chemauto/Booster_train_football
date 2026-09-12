"""Print the actual IsaacLab joint order for BOOSTER_K1_CFG (new URDF asset)."""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, offscreen=True)
simulation_app = app_launcher.app

import torch  # noqa: E402

from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

from booster_train.assets.robots.booster import BOOSTER_K1_CFG  # noqa: E402


class SceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = BOOSTER_K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


sim = SimulationContext(SimulationCfg(device="cuda:0"))
scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.5))
robot: Articulation = scene["robot"]
sim.reset()

print("=" * 60, flush=True)
print("ISAACLAB_JOINT_ORDER:", flush=True)
for i, name in enumerate(robot.joint_names):
    print(f"  {i:2d}. {name}", flush=True)
print("=" * 60, flush=True)

simulation_app.close()
