"""Build the K1 AMP motion dataset (walk + kick) for the kick_amp task (paper 2511.03996).

Sources (all G1 29-dof, retargeted to K1 22-dof via the g1_to_k1_csv mapping):
  - LAFAN1 walk  : /data/rl_robot/LAFAN1_Retargeting_Dataset/g1/walk*.csv
                   headerless csv: root_pos(3) + root_quat_xyzw(4) + 29 joints @ 30 fps
  - Soccer kicks : /data/rl_robot/HumanoidSoccer/motions/**/*.npz
                   npz: fps/joint_pos(T,29)/body_pos_w(T,30,3)/body_quat_w(T,30,4 wxyz) @ 50 fps

Output: K1 csvs (same format as g1_to_k1_csv.py output, root_xyz + quat_xyzw + 22 joints,
header row) under --output_dir/{walk,kick}/, one csv per clip, ready for csv_to_npz.py.

Usage:
    python scripts/amp_data_build.py --output_dir /tmp/amp_k1_csv
"""

import argparse
import os
from pathlib import Path

import numpy as np

from booster_assets.motions import K1_JOINT_NAMES as K1_JOINT_NAMES_ORDER
from g1_to_k1_csv import G1_JOINT_NAMES, npz_joint_names, retarget_g1_to_k1

LAFAN_DIR = "/data/rl_robot/LAFAN1_Retargeting_Dataset/g1"
SOCCER_DIRS = [
    "/data/rl_robot/HumanoidSoccer/motions/soccer-standard",
    "/data/rl_robot/HumanoidSoccer/motions/soccer-stylized",
]
# walk3_subject5 contains a fall/crouch segment (root_z dips to 0.34, feet 7cm below
# ground) -- excluded after numeric QC of the generated npz.
EXCLUDE = {"walk3_subject5.csv"}


def motion_source_files(directory, pattern):
    """Discover a required source, failing explicitly on missing or empty paths."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Motion source directory does not exist: {directory}")
    files = sorted(path for path in directory.glob(pattern) if path.is_file())
    if not files:
        raise ValueError(f"No motion files matching {pattern} in {directory}")
    return files


def write_k1_csv(path: str, root_pos: np.ndarray, root_quat_xyzw: np.ndarray, k1_pos: np.ndarray):
    header = ["root_x", "root_y", "root_z", "qx", "qy", "qz", "qw"] + list(K1_JOINT_NAMES_ORDER)
    out = np.concatenate([root_pos, root_quat_xyzw, k1_pos], axis=1)
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in out:
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")


def convert_lafan_walk(output_dir: str, z_shift: float, max_clip_sec: float):
    files = motion_source_files(LAFAN_DIR, "walk*.csv")
    print(f"[walk] {len(files)} LAFAN1 clips")
    for src in files:
        if os.path.basename(src) in EXCLUDE:
            print(f"  {os.path.basename(src)}: skipped (EXCLUDE)")
            continue
        raw = np.loadtxt(src, delimiter=",")  # (T, 36): pos3 + quat_xyzw4 + 29 joints
        assert raw.shape[1] == 3 + 4 + len(G1_JOINT_NAMES), f"unexpected cols in {src}: {raw.shape[1]}"
        max_frames = int(max_clip_sec * 30)  # LAFAN1 csvs are 30 fps
        if raw.shape[0] > max_frames:
            raw = raw[:max_frames]
        root_pos = raw[:, :3].copy()
        root_pos[:, 2] += z_shift
        root_quat_xyzw = raw[:, 3:7]
        k1_pos = retarget_g1_to_k1(raw[:, 7:])
        name = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(output_dir, "walk", f"{name}.csv")
        write_k1_csv(dst, root_pos, root_quat_xyzw, k1_pos)
        print(f"  {name}: {raw.shape[0]} frames @30fps -> {dst}")


def convert_soccer_kicks(output_dir: str, z_shift: float, legacy_layout=None):
    files = []
    for d in SOCCER_DIRS:
        files.extend(motion_source_files(d, "*.npz"))
    print(f"[kick] {len(files)} soccer clips")
    for src in files:
        data = np.load(src, allow_pickle=True)
        root_pos = data["body_pos_w"][:, 0, :].astype(np.float64)
        root_pos[:, 2] += z_shift
        root_quat_xyzw = data["body_quat_w"][:, 0, :].astype(np.float64)[:, [1, 2, 3, 0]]  # wxyz -> xyzw
        k1_pos = retarget_g1_to_k1(data["joint_pos"].astype(np.float64), npz_joint_names(data, legacy_layout))
        name = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(output_dir, "kick", f"{name}.csv")
        write_k1_csv(dst, root_pos, root_quat_xyzw, k1_pos)
        print(f"  {name}: {data['joint_pos'].shape[0]} frames @50fps ({data['kick_leg']}) -> {dst}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--z_shift", type=float, default=-0.22,
                        help="coarse root-z offset, G1 pelvis ~0.79 vs K1 trunk ~0.57")
    parser.add_argument("--max_walk_sec", type=float, default=30.0,
                        help="trim each walk clip to this many seconds (FK conversion cost)")
    parser.add_argument("--legacy_layout", choices=["humanoid_soccer_physx"],
                        help="Explicit layout for audited HumanoidSoccer NPZ files")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, "walk"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "kick"), exist_ok=True)
    convert_lafan_walk(args.output_dir, args.z_shift, args.max_walk_sec)
    convert_soccer_kicks(args.output_dir, args.z_shift, args.legacy_layout)


if __name__ == "__main__":
    main()
