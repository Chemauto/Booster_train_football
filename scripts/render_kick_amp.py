"""Render a checkpoint policy to mp4: deterministic (mean) actions, a smooth
chase camera tracking robot+ball, 25 fps output.

The environment is configured by evaluate_kick_amp.configure_evaluation, so the
clip shows the same physics, perception and ball layout the acceptance test
measures. Without it the Play defaults leave perception noise, actuation delay
randomisation and DR on, and a checkpoint trained with --nominal_physics
--perfect_perception scores nothing (measured: 0 goals, 14 terminations in 1500
steps) -- a video of a distribution the policy was never trained on.

Requires the training process to be STOPPED first (8 GB GPU cannot host two
Isaac Sim instances -- concurrent runs die with CUSOLVER_INTERNAL_ERROR).
Restart training from the latest checkpoint afterwards.
"""

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--out", type=str, default="logs/kick_amp_render.mp4")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--env_index", type=int, default=0)
# The graded protocol places the ball 0.8-2 m away at 0/+-90/180 deg from the
# robot's heading; the Play default scatters it anywhere on the field, which
# makes for a slow video that shows nothing about the cases being measured.
parser.add_argument("--scenario", choices=("standing", "soccer", "approach"), default="soccer",
                    help="Acceptance-protocol layout (evaluate_kick_amp.scenario_layout).")
parser.add_argument("--seed", type=int, default=123,
                    help="Acceptance cohort seed; the layout places the ball at 0/+-90/180 deg.")
parser.add_argument("--imperfect_perception", action="store_true",
                    help="Keep the perception noise/delay the policy was trained against (default off).")
parser.add_argument("--actuator_delay", type=int, default=2)
parser.add_argument("--env_spacing", type=float, default=40.0,
                    help="Metres between environments. The chase camera sees ~13 m, so the "
                         "default grid spacing puts a dozen other robots and goals in frame and "
                         "the subject cannot be picked out.")
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
    # Same environment as the acceptance test: no DR events, perception and
    # actuation matching training, and the protocol's ball layout on reset.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "evaluate_kick_amp", Path(__file__).resolve().with_name("evaluate_kick_amp.py"))
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    evaluation.configure_evaluation(env_cfg, args.scenario, args.seed,
                                    not args.imperfect_perception, args.actuator_delay)
    # Render-only deviation, documented: the acceptance test disables the
    # ball-only reset so that a robot chasing a ball out of play is measured as
    # out of field. That makes a clip of a miss run to the edge of the arena and
    # stay there (measured: 700 steps of walking after the ball at local x=11.5).
    # Play continues here, and the ball comes back inside the graded geometry.
    env_cfg.commands.soccer.ball_reset_enabled = True
    env_cfg.commands.soccer.ball_spawn_distance = (0.8, 2.0)
    env_cfg.commands.soccer.ball_spawn_bearing = (-math.pi, math.pi)
    if args.env_spacing > 0:
        env_cfg.scene.env_spacing = args.env_spacing
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
        # World frame for both: soccer.ball_pos_xy is field-local (env origins
        # subtracted) while root_pos_w is world, so mixing them aims the camera at
        # a point that drifts by the whole env offset -- invisible with one env
        # (origin 0) and badly wrong at 40 m spacing.
        r = robot.data.root_pos_w[idx, :2]
        b = soccer.ball.data.root_pos_w[idx, :2]
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
    oof = 0
    term_mgr = env.termination_manager
    with torch.no_grad():
        for t in range(args.steps):
            dist, _ = runner.model.act(runner.obs_norm(obs), runner.stacked_obs)
            obs, _, terminated, time_outs, _ = env.step(dist.mean)  # deterministic
            obs = obs["policy"]
            runner._push_obs(runner.obs_norm(obs), terminated | time_outs)
            goals += int(bool(soccer.ball_in_goal_now[idx].item()))
            cur_stand += 1
            # Count the cause, never the raw `terminated` flag: the goal
            # termination is a non-timeout DoneTerm, so the flag reports every
            # scored episode as a fall (measured: falls == goals in all four
            # clips of the first batch).
            if bool(term_mgr.get_term("base_contact")[idx].item()):
                falls += 1
                stand_steps.append(cur_stand)
                cur_stand = 0
            oof += int(bool(term_mgr.get_term("out_of_field")[idx].item()))
            if t % 2 == 0:  # 50 Hz sim -> 25 fps video
                frames.append(cam.data.output["rgb"][idx].cpu().numpy()[..., :3])
            aim()
            if t % 100 == 0:
                # Field coordinates: root_pos_w is world, and with several envs the
                # origin offset makes world x meaningless (env 8 sits ~20 m away).
                r = robot.data.root_pos_w[idx][:2] - env.scene.env_origins[idx, :2]
                b = soccer.ball_pos_xy[idx]
                print(
                    f"t={t} robot=({float(r[0]):.2f},{float(r[1]):.2f}) "
                    f"ball=({float(b[0]):.2f},{float(b[1]):.2f}) goals={goals} falls={falls} oof={oof}",
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
