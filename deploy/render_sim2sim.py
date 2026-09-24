#!/usr/bin/env python3
"""Offscreen MuJoCo video for the kick_amp sim2sim task.

    python3 deploy/render_sim2sim.py --episodes 2 --foot-collision box \
        --checkpoint kick_amp_it7200_policy.pt --out /tmp/sim2sim_box

Writes <out>_ep{N}_{outcome}.mp4 and <out>_ep{N}_key{step}.png.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

_DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
_BOOSTER_DEPLOY = "/data/rl_robot/BoosterRobotics/booster_deploy"
sys.path.insert(0, _DEPLOY_DIR)
sys.path.insert(1, _BOOSTER_DEPLOY)
try:
    import booster_assets  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(2, "/data/rl_robot/BoosterRobotics/booster_assets/src")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from tasks.kick_amp import KickAmpControllerCfg  # noqa: E402
from tasks.kick_amp.kick_amp_mujoco import (  # noqa: E402
    KickAmpMujocoController,
    scenario_layout,
)


def follow_cam(model, data, look, dist=3.2, height=1.6, azimuth=90.0):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = look
    cam.distance = dist
    cam.elevation = -np.degrees(np.arctan2(height, dist)) * 0.6
    cam.azimuth = azimuth
    return cam


def record_episode(controller, yaw, ball_xy, seed, frames_every=2, max_steps=1500):
    controller.reset_episode(float(yaw), ball_xy, seed=seed)
    renderer = mujoco.Renderer(controller.mj_model, height=540, width=960)
    frames = []
    keys = {}
    outcome, info = None, {}
    step = 0
    for step in range(max_steps):
        outcome, info = controller.check_terminal(step, max_steps)
        if outcome is not None:
            break
        dof_targets = controller.policy_step()
        controller.ctrl_step(dof_targets)
        controller.update_state()
        if step % frames_every == 0:
            root = controller.robot.data.root_pos_w.numpy()
            ball = controller.ball_pos_w.numpy()
            mid = 0.5 * (root[:2] + ball[:2])
            look = np.array([mid[0], mid[1], 0.25])
            cam = follow_cam(controller.mj_model, controller.mj_data, look)
            renderer.update_scene(controller.mj_data, camera=cam)
            img = renderer.render()
            frames.append(img)
            if step in (0, max_steps // 4, max_steps // 2) or (
                outcome is not None
            ):
                keys[step] = img
    else:
        outcome = "timeout"
    renderer.close()
    return frames, outcome, info, step + 1, keys


def write_mp4(frames, path, fps=50):
    if not frames:
        return
    h, w, _ = frames[0].shape
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-preset", "fast", path,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
    print("wrote", path, f"{len(frames)} frames")


def write_png(arr, path):
    h, w, _ = arr.shape
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-i", "-", "-frames:v", "1", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p.stdin.write(arr.tobytes())
    p.stdin.close()
    p.wait()
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=128)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--foot-collision", choices=("box", "mesh"), default="box")
    ap.add_argument("--checkpoint", default="kick_amp_it7200_policy.pt")
    ap.add_argument("--every", type=int, default=2,
                    help="record every N policy steps (2 -> 25 fps of sim)")
    ap.add_argument("--out", default="/tmp/sim2sim")
    args = ap.parse_args()

    cfg = KickAmpControllerCfg()
    cfg.policy.checkpoint_path = f"models/{args.checkpoint}"
    cfg.foot_collision = args.foot_collision
    controller = KickAmpMujocoController(cfg)
    controller.start()
    yaws, ball_xys = scenario_layout(args.episodes, args.seed)

    for ep in range(args.episodes):
        frames, outcome, info, nsteps, keys = record_episode(
            controller, yaws[ep], ball_xys[ep],
            seed=args.seed * 1000 + ep,
            frames_every=args.every, max_steps=args.steps)
        base = f"{args.out}_{args.foot_collision}_ep{ep + 1:02d}_{outcome}"
        write_mp4(frames, f"{base}.mp4", fps=int(50 / args.every))
        for k, img in keys.items():
            write_png(img, f"{base}_key{k:04d}.png")
        print(f"ep{ep + 1}: {outcome} after {nsteps} steps  "
              f"ball={info.get('ball_xy')}  -> {base}.mp4")


if __name__ == "__main__":
    main()
