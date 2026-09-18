"""Convert the paper's native retargeted CSV data (code/data) to AMP npz (2026-09-18).

The paper's repo ships `/data/rl_robot/BoosterRobotics/code/data/*.csv`:
root pose+vel (7+6 cols, quat xyzw) and 13 T1 joint angles + velocities --
already the product of the paper's optimization-based retargeting. The T1 leg
joints map to K1 1:1 BY NAME (hip pitch/roll/yaw + knee_pitch + ankle
pitch/roll); T1's waist yaw is approximately folded into both hip yaws;
K1's head+arms take
the K1 neutral pose (arms frozen at -+1.3, matching amp_corrected_v2).

Why this beats the LAFAN1->G1->K1 chain: these clips ARE the paper's dataset
(12 walk clips = exactly the claimed 76.28 s of omnidirectional walking,
incl. turns and side-steps our LAFAN set lacks; 12 official kick clips,
right = mirrored left), natively 50 fps, produced by the authors' own
retargeting. No third-party G1 approximation in the loop.

FK, sole grounding and velocity reconstruction reuse build_amp_motion_npz.
Output: <out>/walk/*.npz and <out>/kick/*.npz in the motion_lib layout.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_amp_motion_npz import ASSETS, Kinematics, sha256  # noqa: E402

CODE_DATA = Path("/data/rl_robot/BoosterRobotics/code/data")

# K1 hard joint limits, verbatim from K1_22dof.urdf. T1's ranges are wider in
# places (ankle_roll +-0.44 vs K1 +-0.345; ankle_pitch +0.35 vs +0.345), so
# raw T1 angles can violate K1 hardware. Positions are projected into these
# limits BEFORE FK so the stored angles and body positions stay self-consistent
# (the policy can physically reach every stored pose). Known cost: projecting
# perturbs foot trajectories a few cm on the offending frames -- unavoidable
# without a morphology-aware IK retarget.
K1_JOINT_LIMITS = {
    "aahead_yaw_joint": (-1.012, 1.012),
    "aahead_pitch_joint": (-0.314, 0.794),
    "aaleft_shoulder_pitch_joint": (-2.932, 1.196),
    "left_shoulder_roll_joint": (-1.642, 1.629),
    "left_elbow_pitch_joint": (-1.885, 1.885),
    "left_elbow_yaw_joint": (-2.242, 0.816),
    "aaright_shoulder_pitch_joint": (-2.932, 1.196),
    "right_shoulder_roll_joint": (-1.642, 1.629),
    "right_elbow_pitch_joint": (-1.885, 1.885),
    "right_elbow_yaw_joint": (-0.816, 2.242),
    "left_hip_pitch_joint": (-2.958, 2.226),
    "left_hip_roll_joint": (-0.375, 1.536),
    "left_hip_yaw_joint": (-1.012, 1.012),
    "left_knee_pitch_joint": (0.0, 2.321),
    "left_ankle_pitch_joint": (-0.87, 0.345),
    "left_ankle_roll_joint": (-0.345, 0.345),
    "right_hip_pitch_joint": (-2.958, 2.226),
    "right_hip_roll_joint": (-1.536, 0.375),
    "right_hip_yaw_joint": (-1.012, 1.012),
    "right_knee_pitch_joint": (0.0, 2.321),
    "right_ankle_pitch_joint": (-0.87, 0.345),
    "right_ankle_roll_joint": (-0.345, 0.345),
}

# K1 motor velocity limits (rad/s), verbatim from K1_22dof.urdf <limit velocity>
K1_VEL_LIMITS = {
    "aahead_yaw_joint": 7.85, "aahead_pitch_joint": 7.85,
    "aaleft_shoulder_pitch_joint": 33.51, "aaright_shoulder_pitch_joint": 33.51,
    "left_shoulder_roll_joint": 33.51, "right_shoulder_roll_joint": 33.51,
    "left_elbow_pitch_joint": 33.51, "right_elbow_pitch_joint": 33.51,
    "left_elbow_yaw_joint": 33.51, "right_elbow_yaw_joint": 33.51,
    "left_hip_pitch_joint": 14.66, "right_hip_pitch_joint": 14.66,
    "left_hip_roll_joint": 12.57, "right_hip_roll_joint": 12.57,
    "left_hip_yaw_joint": 17.59, "right_hip_yaw_joint": 17.59,
    "left_knee_pitch_joint": 12.57, "right_knee_pitch_joint": 12.57,
    "left_ankle_pitch_joint": 17.59, "right_ankle_pitch_joint": 17.59,
    "left_ankle_roll_joint": 17.59, "right_ankle_roll_joint": 17.59,
}

# T1 csv column name -> K1 joint name (legs are 1:1 by name)
T1_TO_K1 = {
    "q_left_hip_pitch": "left_hip_pitch_joint",
    "q_left_hip_roll": "left_hip_roll_joint",
    "q_left_hip_yaw": "left_hip_yaw_joint",
    "q_left_knee_pitch": "left_knee_pitch_joint",
    "q_left_ankle_pitch": "left_ankle_pitch_joint",
    "q_left_ankle_roll": "left_ankle_roll_joint",
    "q_right_hip_pitch": "right_hip_pitch_joint",
    "q_right_hip_roll": "right_hip_roll_joint",
    "q_right_hip_yaw": "right_hip_yaw_joint",
    "q_right_knee_pitch": "right_knee_pitch_joint",
    "q_right_ankle_pitch": "right_ankle_pitch_joint",
    "q_right_ankle_roll": "right_ankle_roll_joint",
}


def read_t1_csv(path: Path):
    with open(path) as fh:
        cols = next(csv.reader(fh))
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    idx = {c: i for i, c in enumerate(cols)}
    root = raw[:, 0:3]
    quat_wxyz = raw[:, [idx["qw"], idx["qx"], idx["qy"], idx["qz"]]]  # csv is xyzw
    # q_waist is included: T1's chain is Trunk -> Waist -> legs and K1 has no
    # waist, so main() folds it into K1's hip yaw (see the comment there).
    joints = {c: raw[:, idx[c]] for c in cols if c.startswith("q_")}
    return root, quat_wxyz, joints, len(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ASSETS / "motions/K1/amp_paper"))
    ap.add_argument("--model", default=str(ASSETS / "robots/K1/K1_22dof.xml"))
    ap.add_argument("--schema_npz", default=str(next((ASSETS / "motions/K1/amp_corrected_v2/walk").glob("*.npz"))))
    args = ap.parse_args()

    # neutral for the non-leg joints = booster.py K1 nominal init_state
    # (shoulder_roll -+1.3, everything else 0). NOT logs/bc_constants.json --
    # that file holds a randomized env sample (head_yaw 0.045, shoulder_roll
    # -+1.335, elbow_yaw ~0.02-1.0 off nominal) which an audit found baked
    # into every clip of the first amp_paper build.
    with np.load(args.schema_npz, allow_pickle=False) as schema:
        k1_names = [str(n) for n in schema["joint_names"]]  # live PhysX order, the truth
    neutral = {n: 0.0 for n in k1_names}
    neutral["left_shoulder_roll_joint"] = -1.3
    neutral["right_shoulder_roll_joint"] = 1.3
    # head+arms at neutral; legs filled from T1 below
    fill = np.zeros(len(k1_names))
    for i, n in enumerate(k1_names):
        if n not in T1_TO_K1.values():
            fill[i] = neutral[n]

    with np.load(args.schema_npz, allow_pickle=False) as schema:
        fk = Kinematics(args.model, schema["joint_names"].tolist(), schema["body_names"].tolist())
    # k1_names came from the same schema, so fk.joint_names == k1_names: no
    # permutation needed (kept as an assertion, not a silent assumption)
    assert list(fk.joint_names) == k1_names, "schema joint order drift"

    csvs = sorted(CODE_DATA.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"no csv under {CODE_DATA}")
    args.out = Path(args.out)
    if args.out.exists():
        raise FileExistsError(f"{args.out} exists")

    for f in csvs:
        kind = "kick" if "kick" in f.stem else "walk"
        root, quat, jt, T = read_t1_csv(f)
        joints = np.tile(fill, (T, 1))  # (T, 22) neutral
        for t1_col, k1_name in T1_TO_K1.items():
            joints[:, k1_names.index(k1_name)] = jt[t1_col]
        # T1 rotates the legs upstream of hip pitch/roll. Moving that yaw to
        # K1's downstream hip-yaw joint is an approximation: rotations do not
        # commute when hip pitch/roll are nonzero, and the link lengths differ.
        # Retain this measured improvement over discarding waist motion, but
        # do not claim exact pose preservation or dynamic feasibility.
        waist = jt["q_waist"]
        for hip in ("left_hip_yaw_joint", "right_hip_yaw_joint"):
            i = k1_names.index(hip)
            joints[:, i] = joints[:, i] + waist
        # project T1 angles into K1 hardware limits BEFORE FK: T1's ankle_roll
        # range (+-0.44) exceeds K1's (+-0.345) and kicks edge past ankle_pitch
        # +0.345, so unprojected frames are poses the policy cannot reach. The
        # clamp slightly perturbs foot trajectories on those frames -- the
        # honest alternative is a morphology-aware IK retarget (future work).
        for i, n in enumerate(k1_names):
            lo, hi = K1_JOINT_LIMITS[n]
            joints[:, i] = joints[:, i].clip(lo, hi)
        # velocity-limit projection: a frame-to-frame jump faster than the
        # motor limit is physically unreachable (kick_5's knee hits 13.1 vs
        # 12.57 rad/s on one frame). Replace the landing angle with the local
        # midpoint until the pair obeys the limit -- standard motion cleanup,
        # touches only the offending frames.
        dt = 0.02
        for i, n in enumerate(k1_names):
            vlim = K1_VEL_LIMITS[n] * dt
            for _ in range(4):
                bad = np.where(np.abs(np.diff(joints[:, i])) > vlim)[0]
                if len(bad) == 0:
                    break
                for k in bad:
                    j = min(k + 2, len(joints) - 1)
                    joints[k + 1, i] = 0.5 * (joints[k, i] + joints[j, i])
        metadata = {
            "source": str(f), "source_sha256": sha256(f),
            "source_fps": 50.0, "time_scale": 1.0,
            "generator_sha256": sha256(__file__), "model_sha256": sha256(args.model),
            "origin": "paper repo code/data (T1-optimized retargeting by the authors)",
            "mapping": "T1 legs by name; waist yaw approximately added to both hip yaws; K1 nominal head/arms; hardware projection before FK",
        }
        result, stats = fk.build(root, quat, joints, 50, anchor_walk=(kind == "walk"), metadata=metadata)
        dst = args.out / kind / f"{f.stem}.npz"
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, **result)
        print(f"{kind}/{f.stem}: {stats['frames']} fr, slip {stats['support_speed_before_mps']:.3f}"
              f"->{stats['support_speed_after_mps']:.3f} m/s", flush=True)
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
