"""Render a checkpoint policy to mp4: deterministic (mean) actions, a smooth
chase camera tracking robot+ball, 25 fps output.

Requires the training process to be STOPPED first (8 GB GPU cannot host two
Isaac Sim instances -- concurrent runs die with CUSOLVER_INTERNAL_ERROR).
Restart training from the latest checkpoint afterwards.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--out", type=str, default="logs/kick_amp_render.mp4")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--env_index", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.sim.spawners.sensors import PinholeCameraCfg as CameraSpawnCfg  # noqa: E402

import booster_train.tasks  # noqa: F401,E402
from booster_train.rsl_rl.amp.runner import AmpRunner  # noqa: E402

# how far behind / above the tracked target the camera sits, as a function of
# how far apart the robot and the ball are (a fixed rig loses one of them once
# the kick separates them)
BACK_MIN, BACK_PER_M, BACK_MAX = 5.0, 1.1, 13.0
HEIGHT_MIN, HEIGHT_PER_M, HEIGHT_MAX = 2.6, 0.30, 5.5


def main():
    env_cfg = parse_env_cfg("Booster-K1-KickAMP-v0-Play", device=args.device, num_envs=args.num_envs)
    # The prim path MUST be the env_-regex form: a single fixed prim
    # (/World/envs/env_0/...) gives the sensor num_envs==1 while the scene
    # resets env_ids=range(num_envs), so SensorBase._timestamp_last_update[
    # env_ids] indexes out of bounds and trips a PhysX fabric "CUDA
    # device-side assert" at the first reset.
    # The spawn transform here is immediately overridden every frame by
    # set_world_poses_from_view(), which also handles the frame convention.
    env_cfg.scene.track_cam = CameraCfg(
        prim_path="/World/envs/env_.*/track_cam",
        update_period=0,
        offset=CameraCfg.OffsetCfg(pos=(0.0, -6.0, 3.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        spawn=CameraSpawnCfg(focal_length=20.0, clipping_range=(0.05, 80.0)),
        data_types=["rgb"],
        width=960,
        height=540,
    )
    entry = gym.registry["Booster-K1-KickAMP-v0"].kwargs["rsl_rl_cfg_entry_point"]
    if isinstance(entry, str):
        import importlib
        module, cls = entry.rsplit(":", 1)
        entry = getattr(importlib.import_module(module), cls)()

    env = gym.make("Booster-K1-KickAMP-v0-Play", cfg=env_cfg).unwrapped
    entry.num_envs = env.num_envs
    runner = AmpRunner(env, entry, train_mode=False)
    runner.load(args.checkpoint)
    cam = env.scene.sensors["track_cam"]
    robot = env.scene["robot"]
    soccer = env.command_manager.get_term("soccer")
    idx = args.env_index

    def aim():
        """Point env `idx`'s camera at the midpoint of robot and ball, from
        behind (-y) and above. Called AFTER the frame is captured, so the
        capture carries a one-step-old pose (~20 ms, invisible at 25 fps)."""
        r = robot.data.root_pos_w[idx, :2]
        b = soccer.ball_pos_xy[idx]
        sep = float(torch.linalg.norm(r - b))
        back = min(BACK_MAX, BACK_MIN + BACK_PER_M * sep)
        height = min(HEIGHT_MAX, HEIGHT_MIN + HEIGHT_PER_M * sep)
        tgt = 0.5 * (r + b)
        eye = torch.tensor([[float(tgt[0]), float(tgt[1]) - back, height]], device=env.device)
        target = torch.tensor([[float(tgt[0]), float(tgt[1]), 0.45]], device=env.device)
        cam.set_world_poses_from_view(eye, target, env_ids=[idx])

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    aim()
    frames, goals = [], 0
    falls, stand_steps, cur_stand = 0, [], 0
    with torch.no_grad():
        for t in range(args.steps):
            dist, _ = runner.model.act(runner.obs_norm(obs), runner.stacked_obs)
            obs, _, terminated, time_outs, _ = env.step(dist.mean)  # deterministic
            obs = obs["policy"]
            runner._push_obs(runner.obs_norm(obs), terminated | time_outs)
            goals += int(soccer.goal_scored_now[idx].item())
            cur_stand += 1
            if terminated[idx].item():  # fall (not timeout)
                falls += 1
                stand_steps.append(cur_stand)
                cur_stand = 0
            if t % 2 == 0:  # 50 Hz sim -> 25 fps video
                frames.append(cam.data.output["rgb"][idx].cpu().numpy()[..., :3])
            aim()
            if t % 100 == 0:
                r = robot.data.root_pos_w[idx]
                b = soccer.ball_pos_xy[idx]
                print(
                    f"t={t} robot=({r[0]:.2f},{r[1]:.2f}) z={r[2]:.2f} "
                    f"ball=({b[0]:.2f},{b[1]:.2f}) goals={goals} falls={falls}",
                    flush=True,
                )

    if cur_stand > 0:
        stand_steps.append(cur_stand)
    avg_stand = (sum(stand_steps) / len(stand_steps)) if stand_steps else args.steps
    imageio.mimsave(args.out, frames, fps=25, quality=8)
    print(
        f"saved {args.out}: {len(frames)} frames, {len(frames) / 25:.1f} s, goals={goals}, "
        f"falls={falls}, stand_per_episode={[s // 50 for s in stand_steps]}s, "
        f"avg_stand={avg_stand / 50:.1f}s"
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
