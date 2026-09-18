"""Retarget a Unitree G1 (29-dof) motion npz to a Booster K1 (22-dof) CSV.

The output CSV is consumed by scripts/csv_to_npz.py:
    root_xyz(3) + root_quat_xyzw(4) + 22 joint angles in K1_JOINT_NAMES order,
    with a header row (csv_to_npz auto-detects and skips it).

Joint mapping strategy (plan: expressive-swimming-balloon, step 1):
    legs 12   : copied 1:1 (G1 *_knee_joint -> K1 *_knee_pitch_joint renamed)
    arms/head : K1 neutral pose; G1 and K1 upper-body zero poses differ
    head x2   : set to 0 (G1 has no head joints)
    waist x3  : dropped (K1 has no waist)
Root: xy and quaternion copied as-is (quat wxyz -> xyzw); z shifted by
--z_shift (pass 1, coarse) or --z_profile (pass 2, per-frame calibration
derived from foot-link heights of the pass-1 npz).

Usage:
    # pass 1 (coarse z)
    python scripts/g1_to_k1_csv.py --input_npz right_kick.npz \
        --z_shift -0.22 --output_csv /tmp/k1_kick_pass1.csv
    # pass 2 (per-frame z profile produced by calibrate from pass-1 npz)
    python scripts/g1_to_k1_csv.py --input_npz right_kick.npz \
        --z_profile z_profile.npy --output_csv k1_kick_full.csv
"""

import argparse

import numpy as np

try:
    from booster_assets.motions import K1_JOINT_NAMES
except ImportError:
    raise SystemExit(
        "booster_assets not importable; run inside the env_isaaclab environment"
    )

# G1 CSV order ONLY (not the PhysX order in legacy HumanoidSoccer NPZ), from
# whole_body_tracking/scripts/csv_to_npz.py (run_simulator joint_names=...)
G1_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Legacy HumanoidSoccer PhysX order, verified against G1 FK (max error 1.24e-7 m).
# Never infer this from a 29-column shape: callers must explicitly identify legacy data.
G1_PHYSX_JOINT_NAMES = ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint', 'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint']

K1_NEUTRAL = {"left_shoulder_roll_joint": -1.3, "right_shoulder_roll_joint": 1.3}
G1_TO_K1_MAP = {
    name: ((name.replace("knee_pitch", "knee"), 1.0)
           if any(part in name for part in ("hip_", "knee_", "ankle_")) else None)
    for name in K1_JOINT_NAMES
}


def npz_joint_names(data, legacy_layout=None):
    """Resolve named NPZ columns, failing closed for unspecified legacy layouts."""
    if "joint_names" in data:
        names = [str(name) for name in data["joint_names"]]
    elif legacy_layout == "humanoid_soccer_physx":
        names = list(G1_PHYSX_JOINT_NAMES)
    else:
        raise ValueError("NPZ lacks joint_names; supply the verified legacy layout humanoid_soccer_physx")
    if len(names) != data["joint_pos"].shape[1] or len(set(names)) != len(names):
        raise ValueError("joint_names must be unique and match joint_pos columns")
    if set(names) != set(G1_JOINT_NAMES):
        raise ValueError("Source joint names do not describe the supported G1 29-dof skeleton")
    return names


def retarget_g1_to_k1(g1_pos, source_joint_names=G1_JOINT_NAMES):
    """Approximate leg-angle transfer, named columns and a legal K1 upper pose.

    The default is specifically the LAFAN CSV layout. NPZ callers must resolve
    their metadata with npz_joint_names before calling this function.
    """
    names = npz_joint_names({"joint_names": source_joint_names, "joint_pos": g1_pos})
    if g1_pos.ndim != 2 or not np.isfinite(g1_pos).all():
        raise ValueError("joint_pos must be a finite two-dimensional array")
    index = {name: i for i, name in enumerate(names)}
    k1 = np.zeros((len(g1_pos), len(K1_JOINT_NAMES)), dtype=np.float64)
    for k, name in enumerate(K1_JOINT_NAMES):
        entry = G1_TO_K1_MAP[name]
        k1[:, k] = K1_NEUTRAL.get(name, 0.0) if entry is None else entry[1] * g1_pos[:, index[entry[0]]]
        k1[:, k] = np.clip(k1[:, k], *K1_JOINT_LIMITS[name])
    return k1


# K1_22dof.urdf revolute limits (lower, upper), extracted from the asset.
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
    "left_ankle_pitch_joint": (-0.870, 0.345),
    "left_ankle_roll_joint": (-0.345, 0.345),
    "right_hip_pitch_joint": (-2.958, 2.226),
    "right_hip_roll_joint": (-1.536, 0.375),
    "right_hip_yaw_joint": (-1.012, 1.012),
    "right_knee_pitch_joint": (0.0, 2.321),
    "right_ankle_pitch_joint": (-0.870, 0.345),
    "right_ankle_roll_joint": (-0.345, 0.345),
}


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Moving average with edge replication (no artificial endpoint impulses)."""
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    out = x.copy()
    for j in range(x.shape[1]):
        out[:, j] = np.convolve(np.pad(x[:, j], (window // 2, (window - 1) // 2), mode="edge"), kernel, mode="valid")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_npz", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--z_shift", type=float, default=-0.22,
                        help="coarse root-z offset (pass 1), G1 pelvis ~0.79 vs K1 trunk 0.57")
    parser.add_argument("--z_profile", type=str, default=None,
                        help="npy file of per-frame root-z offsets (pass 2), overrides --z_shift")
    parser.add_argument("--smooth_window", type=int, default=0,
                        help="moving-average window on joint angles (0 = off)")
    parser.add_argument("--legacy_layout", choices=["humanoid_soccer_physx"],
                        help="Explicit layout for audited legacy NPZ without joint_names")
    args = parser.parse_args()

    data = np.load(args.input_npz, allow_pickle=True)
    g1_pos = data["joint_pos"].astype(np.float64)          # (T, 29)
    root_pos = data["body_pos_w"][:, 0, :].astype(np.float64)   # (T, 3) pelvis
    root_quat_wxyz = data["body_quat_w"][:, 0, :].astype(np.float64)  # (T, 4)
    num_frames = g1_pos.shape[0]
    print(f"[INFO] loaded {args.input_npz}: {num_frames} frames, "
          f"{g1_pos.shape[1]} dof, fps={float(data['fps'].item())}")

    source_names = npz_joint_names(data, args.legacy_layout)
    k1_pos = retarget_g1_to_k1(g1_pos, source_names)

    if args.smooth_window > 1:
        k1_pos = smooth(k1_pos, args.smooth_window)

    # clip to K1 limits, report how much was clipped
    clip_frac = []
    for k, name in enumerate(K1_JOINT_NAMES):
        lo, hi = K1_JOINT_LIMITS[name]
        before = k1_pos[:, k].copy()
        k1_pos[:, k] = np.clip(before, lo, hi)
        frac = np.mean((before != k1_pos[:, k]))
        if frac > 0.01:
            clip_frac.append((name, float(frac)))
    if clip_frac:
        print("[WARN] joints clipped >1% of frames (motion may lose amplitude):")
        for name, frac in clip_frac:
            print(f"    {name}: {frac:.1%}")

    # ---- root ----
    root_out = root_pos.copy()
    if args.z_profile is not None:
        profile = np.load(args.z_profile)
        assert profile.shape[0] == num_frames, \
            f"z_profile length {profile.shape[0]} != frames {num_frames}"
        # the profile is a per-frame correction measured relative to the
        # z_shift-based pass-1 replay, so both are applied together
        root_out[:, 2] += args.z_shift + profile
        print(f"[INFO] applied z shift {args.z_shift:+.3f} m + per-frame profile "
              f"(mean {profile.mean():+.3f} m, min {profile.min():+.3f}, max {profile.max():+.3f})")
    else:
        root_out[:, 2] += args.z_shift
        print(f"[INFO] applied constant z shift {args.z_shift:+.3f} m")

    quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]  # wxyz -> xyzw

    # ---- write CSV (with header; csv_to_npz auto-skips it) ----
    header = ["root_x", "root_y", "root_z", "qx", "qy", "qz", "qw"] + list(K1_JOINT_NAMES)
    out = np.concatenate([root_out, quat_xyzw, k1_pos], axis=1)
    with open(args.output_csv, "w") as f:
        f.write(",".join(header) + "\n")
        for row in out:
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")
    print(f"[INFO] wrote {args.output_csv}: {out.shape[0]} frames x {out.shape[1]} cols")


if __name__ == "__main__":
    main()
