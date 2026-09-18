"""Which AMP obs dimension does the discriminator separate on? (2026-09-16)

26 runs have ended with the discriminator saturated (policy ~ -5.9 / expert
~ +5.0) within ~40 iterations. "The distributions are disjoint" describes that
but does not explain it. This script finds the responsible DIMENSIONS.

Method, no IsaacSim needed:
  runner.py updates amp_normalizer with one policy minibatch AND one equally
  sized expert minibatch per inner step, so its running mean converges to
      mix_mean = (policy_mean + expert_mean) / 2
  The expert side is computable exactly from the motion npz files (same code
  path as motion_lib). Therefore
      policy_mean = 2 * mix_mean - expert_mean
  and the per-dimension separation in the discriminator's own (whitened) input
  space is
      z = |policy_mean - expert_mean| / mix_std
  NOTE on scale: for a 50/50 mixture mix_var = (v_pol+v_exp)/2 + (gap/2)^2, so
  z SATURATES AT 2 -- z ~ 2 already means the two clusters barely overlap. The
  unbounded measure is Cohen's d = |gap| / sqrt((v_pol+v_exp)/2), reported as
  'd': d > 3 means a single-feature threshold classifies policy vs expert
  almost perfectly, which is all the discriminator needs. No amount of gait
  learning closes such a gap if its cause is kinematic (retarget offsets,
  default-stance mismatch) rather than dynamic.
"""

import argparse
import ast
import glob
import os

import numpy as np
import torch

MOTION_LIB = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "source", "booster_train", "booster_train",
    "tasks", "manager_based", "kick_amp", "mdp", "motion_lib.py"))


def _excluded_from_source() -> tuple[str, ...]:
    """Read the AMP-excluded joint names straight out of motion_lib.py.

    A local copy rots silently: this script still assumed the pre-v14 55-dim
    layout (head-excluded only) after training had moved to 39. motion_lib
    cannot be imported here (it pulls in isaaclab), so parse the literals.
    """
    tree = ast.parse(open(MOTION_LIB).read())
    vals = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                vals[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return tuple(vals.get("HEAD_JOINTS", ())) + tuple(vals.get("ARM_JOINTS", ()))


HEAD_JOINTS = ("aahead_yaw_joint", "aahead_pitch_joint")
FEET_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def quat_rotate_inverse_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """wxyz quaternion, same math as isaaclab.utils.math.quat_rotate_inverse."""
    w = q[:, 0:1]
    xyz = q[:, 1:4]
    t = np.cross(xyz, v) * 2.0
    return v - w * t + np.cross(xyz, t)


def expert_amp_obs(motion_dir: str, excluded: tuple[str, ...]) -> tuple[np.ndarray, list[str]]:
    """Rebuild the expert AMP obs exactly as motion_lib does, for the given
    excluded-joint set (v14: head+arms -> 39 dims; pre-v14: head only -> 55)."""
    files = sorted(glob.glob(os.path.join(motion_dir, "**", "*.npz"), recursive=True))
    if not files:
        raise FileNotFoundError(motion_dir)
    joint_names = [str(n) for n in np.load(files[0])["joint_names"]]
    body_dof = [i for i, n in enumerate(joint_names) if n not in excluded]
    out = []
    for f in files:
        d = np.load(f)
        jp = d["joint_pos"].astype(np.float64)
        jv = d["joint_vel"].astype(np.float64)
        bp = d["body_pos_w"].astype(np.float64)
        bq = d["body_quat_w"].astype(np.float64)
        bl = d["body_lin_vel_w"].astype(np.float64)
        ba = d["body_ang_vel_w"].astype(np.float64)
        rq = bq[:, 0, :]
        T = len(jp)
        grav = quat_rotate_inverse_np(rq, np.tile([0.0, 0.0, -1.0], (T, 1)))
        lin_b = quat_rotate_inverse_np(rq, bl[:, 0, :])
        ang_b = quat_rotate_inverse_np(rq, ba[:, 0, :])
        body_names = [str(n) for n in d["body_names"]]
        fidx = [body_names.index(n) for n in FEET_BODIES]
        feet = np.concatenate(
            [quat_rotate_inverse_np(rq, bp[:, i, :] - bp[:, 0, :]) for i in fidx], axis=1
        )
        obs = np.concatenate([grav, lin_b, ang_b, jp[:, body_dof], jv[:, body_dof] * 0.1, feet], axis=1)
        assert obs.shape[1] == 9 + 2 * len(body_dof) + 6, obs.shape
        out.append(obs)
    return np.concatenate(out), [joint_names[i] for i in body_dof]


def labels(dof_names: list[str]) -> list[str]:
    n = ["grav_x", "grav_y", "grav_z", "linv_x", "linv_y", "linv_z", "angv_x", "angv_y", "angv_z"]
    n += [f"q:{j}" for j in dof_names]
    n += [f"qd:{j}" for j in dof_names]
    n += ["Lfoot_x", "Lfoot_y", "Lfoot_z", "Rfoot_x", "Rfoot_y", "Rfoot_z"]
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--motion_dir", required=True)
    ap.add_argument("--top", type=int, default=18)
    ap.add_argument("--include_arms", action="store_true",
                    help="analyse a pre-v14 (55-dim) checkpoint: exclude only the head")
    args = ap.parse_args()
    excluded = HEAD_JOINTS if args.include_arms else _excluded_from_source()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    mix_mean = ck["amp_mean"].double().numpy()
    mix_var = ck["amp_var"].double().numpy()
    mix_std = np.sqrt(np.maximum(mix_var, 1e-12))

    exp_obs, dof_names = expert_amp_obs(args.motion_dir, excluded)
    if len(mix_mean) != exp_obs.shape[1]:
        raise SystemExit(
            f"checkpoint amp_mean is {len(mix_mean)}-dim but the expert rebuild is "
            f"{exp_obs.shape[1]}-dim -- pass --include_arms for pre-v14 checkpoints"
        )
    exp_mean = exp_obs.mean(0)
    exp_std = exp_obs.std(0)
    pol_mean = 2.0 * mix_mean - exp_mean          # from the 50/50 update scheme
    # var identity for a 50/50 mixture:
    #   mix_var = (pol_var + exp_var)/2 + ((pol_mean - exp_mean)/2)^2
    gap = pol_mean - exp_mean
    pol_var = 2.0 * (mix_var - (gap / 2.0) ** 2) - exp_obs.var(0)
    pol_std = np.sqrt(np.maximum(pol_var, 0.0))

    z = np.abs(gap) / mix_std                     # in whitened space; caps at 2
    # unbounded separability: pooled-sd effect size (rank by this)
    pooled = np.sqrt(np.maximum((pol_var + exp_obs.var(0)) / 2.0, 1e-12))
    dprime = np.abs(gap) / pooled
    name = labels(dof_names)
    order = np.argsort(-dprime)

    print(f"checkpoint: {args.checkpoint}  (iter {ck['iter']})")
    print(f"expert: {len(exp_obs)} frames from {args.motion_dir}\n")
    print(f"{'dim':>4} {'name':<26} {'expert(mu+-sd)':>18} {'policy(mu+-sd)':>18} {'d':>7} {'z':>6} {'overlap':>8}")
    print("-" * 100)
    for i in order[: args.top]:
        # crude overlap: fraction of a normal pair within +-2sd of each other
        lo = max(exp_mean[i] - 2 * exp_std[i], pol_mean[i] - 2 * pol_std[i])
        hi = min(exp_mean[i] + 2 * exp_std[i], pol_mean[i] + 2 * pol_std[i])
        ov = max(0.0, hi - lo) / max(1e-9, (4 * max(exp_std[i], pol_std[i])))
        print(f"{i:>4} {name[i]:<26} {exp_mean[i]:>9.3f}+-{exp_std[i]:<7.3f} "
              f"{pol_mean[i]:>9.3f}+-{pol_std[i]:<7.3f} {dprime[i]:>7.2f} {z[i]:>6.2f} {ov:>7.0%}")

    nq = len(dof_names)
    ndim = exp_obs.shape[1]
    print(f"\nd>3 (single-feature separable): {int((dprime > 3).sum())}/{ndim} dims")
    print(f"d>2: {int((dprime > 2).sum())}   d>1: {int((dprime > 1).sum())}")
    print(f"block max d -- gravity/rootvel: {dprime[:9].max():.2f}  "
          f"q: {dprime[9:9+nq].max():.2f}  qd: {dprime[9+nq:9+2*nq].max():.2f}  "
          f"feet: {dprime[-6:].max():.2f}")
    print("\ntop separable dims are the features the discriminator can use for free;")
    print("if they are joint ANGLES with tiny expert sd, the cause is a pose/retarget")
    print("offset and no gait learning can ever close it.")


if __name__ == "__main__":
    main()
