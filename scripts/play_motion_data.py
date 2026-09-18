"""Play (visualize) the AMP motion dataset offline, no Isaac Sim needed.

Renders a side view (X-Z) + top view (X-Y) skeleton animation of a single npz
clip to mp4, so the dataset can be eyeballed while training occupies the GPU.
Skeleton is drawn straight from body_pos_w world coordinates (no FK needed).

Usage:
  python scripts/play_motion_data.py --clip walk/walk1_subject1.npz \
      --t0 5.0 --dur 6.0 --out logs/motion_walk.mp4
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

# default = the dataset the training now consumes (env_cfg motion_dir); pass
# --root .../amp_paper for the paper-native set or .../amp for the original
# G1-chain clips.
ROOT = "/data/rl_robot/BoosterRobotics/booster_assets/motions/K1/accad_markers"

# body-chain segments (name pairs) in the URDF link parent->child order
# (K1_22dof.urdf). Every link here is a distinct PhysX body in body_pos_w.
SKELETON = [
    # legs: trunk -> hip_pitch -> hip_roll -> hip_yaw -> knee -> ankle -> foot
    ("trunk", "left_hip_pitch_link"), ("left_hip_pitch_link", "left_hip_roll_link"),
    ("left_hip_roll_link", "left_hip_yaw_link"), ("left_hip_yaw_link", "left_knee_pitch_link"),
    ("left_knee_pitch_link", "left_ankle_pitch_link"), ("left_ankle_pitch_link", "left_ankle_roll_link"),
    ("trunk", "right_hip_pitch_link"), ("right_hip_pitch_link", "right_hip_roll_link"),
    ("right_hip_roll_link", "right_hip_yaw_link"), ("right_hip_yaw_link", "right_knee_pitch_link"),
    ("right_knee_pitch_link", "right_ankle_pitch_link"), ("right_ankle_pitch_link", "right_ankle_roll_link"),
    # arms: trunk -> shoulder_pitch -> shoulder_roll -> elbow_pitch -> elbow_yaw
    ("trunk", "aaleft_shoulder_pitch_link"), ("aaleft_shoulder_pitch_link", "left_shoulder_roll_link"),
    ("left_shoulder_roll_link", "left_elbow_pitch_link"), ("left_elbow_pitch_link", "left_elbow_yaw_link"),
    ("trunk", "aaright_shoulder_pitch_link"), ("aaright_shoulder_pitch_link", "right_shoulder_roll_link"),
    ("right_shoulder_roll_link", "right_elbow_pitch_link"), ("right_elbow_pitch_link", "right_elbow_yaw_link"),
    # head: trunk -> yaw -> pitch
    ("trunk", "aahead_yaw_link"), ("aahead_yaw_link", "aahead_pitch_link"),
    # pelvis crossbar (visual anchor)
    ("left_hip_pitch_link", "right_hip_pitch_link"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=str, default="walk/walk1_subject1.npz")
    ap.add_argument("--root", type=str, default=ROOT)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=8.0)
    ap.add_argument("--out", type=str, default="logs/motion_play.mp4")
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()

    path = os.path.join(args.root, args.clip)
    d = np.load(path)
    bp = d["body_pos_w"]  # (T, 23, 3)
    bn = [str(n) for n in d["body_names"]]
    idx = {n: i for i, n in enumerate(bn)}
    T = len(bp)
    fps_src = int(d["fps"])
    dt = 1.0 / fps_src

    # resolve skeleton to body indices
    segs = []
    for a, b in SKELETON:
        if a in idx and b in idx:
            segs.append((idx[a], idx[b]))

    # frame selection
    t0 = max(0, int(args.t0 * fps_src))
    t1 = min(T, t0 + int(args.dur * fps_src))
    stride = max(1, fps_src // args.fps)  # 50 -> 25
    frames = range(t0, t1, stride)

    # center on the trunk's path so the figure does not walk off-screen
    trunk = idx["trunk"]
    center = bp[frames, trunk, :2].mean(axis=0)

    fig, (ax_side, ax_top) = plt.subplots(1, 2, figsize=(12, 5))
    title = f"{args.clip}  ({fps_src} Hz)  trunk h = {bp[frames, trunk, 2].mean():.3f} m"

    def draw(frame_i):
        ax_side.clear(); ax_top.clear()
        p = bp[frame_i]  # (23,3)
        # side view: X horizontal (walk dir), Z vertical
        for a, b in segs:
            ax_side.plot([p[a, 0], p[b, 0]], [p[a, 2], p[b, 2]], "-o", ms=3, color="tab:blue")
            ax_top.plot([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]], "-o", ms=3, color="tab:red")
        # feet trail (last ~0.8 s)
        trail = max(0, frame_i - int(0.8 * fps_src))
        lf = idx["left_ankle_roll_link"]; rf = idx["right_ankle_roll_link"]
        for foot, c in ((lf, "tab:green"), (rf, "tab:orange")):
            xs = bp[trail:frame_i:stride, foot, 0]; zs = bp[trail:frame_i:stride, foot, 2]
            ax_side.plot(xs, zs, "-", color=c, alpha=0.5, lw=1.5)
            ax_side.plot(p[foot, 0], p[foot, 2], "o", color=c, ms=6)
        ax_side.set_title("side (X-Z)")
        ax_side.set_xlabel("x (m)"); ax_side.set_ylabel("z (m)")
        ax_side.set_xlim(center[0] - 1.2, center[0] + 1.2)
        ax_side.set_ylim(0.0, 1.1)
        ax_side.set_aspect("equal")
        ax_top.set_title("top (X-Y)")
        ax_top.set_xlabel("x (m)"); ax_top.set_ylabel("y (m)")
        ax_top.set_xlim(center[0] - 1.2, center[0] + 1.2)
        ax_top.set_ylim(center[1] - 1.2, center[1] + 1.2)
        ax_top.set_aspect("equal")
        fig.suptitle(title + f"   t={frame_i*dt:.1f}s")

    writer = FFMpegWriter(fps=args.fps, bitrate=2000)
    n = len(frames)
    with writer.saving(fig, args.out, dpi=110):
        for i, fi in enumerate(frames):
            draw(fi)
            writer.grab_frame()
    plt.close(fig)
    print(f"saved {args.out}: {n} frames @ {args.fps} fps = {n/args.fps:.1f}s")


if __name__ == "__main__":
    main()
