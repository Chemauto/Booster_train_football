"""Retime AMP expert motion clips to a slower time base (2026-09-16 root-cause fix).

Diagnosis that motivated this: the AMP discriminator has been fully saturated
since iter ~40 (policy_pred -5.9 / expert_pred +5.0, amp_loss -1.91 = both
tanh terms pegged, grad-pen ~0), so the style reward is the constant
1+tanh(0.4*-5.9) ~= 0.018 -- a constant contributes zero advantage, meaning
AMP has taught the policy nothing for the entire run. The saturation source is
distribution disjointness: the policy's quasi-static distribution does not
overlap the expert's velocity content, and on disjoint supports any
discriminator (tanh or BCE) saturates.

Fix: play the expert clips slower. fps stays 50 (motion_lib hard-checks it --
the expert pair dt must equal the policy dt), and every time series is
resampled onto a stretched time base by interpolation, so the clip simply
plays slower in real time. Velocities scale by TIME_SCALE (v = dx/dt with dt
stretched by 1/TIME_SCALE); joint position trajectories and gait topology are
preserved. The expert distribution moves into reach of the soft-actuated K1
policy, the discriminator score starts varying inside the policy
distribution, and the style reward becomes informative again.

Success signal for the retrained run: policy_pred climbing off the -5.9 floor
within the first few hundred iterations (it never moved in 2400 iters before).

Source data is never modified; retimed copies are written to the destination
directory preserving the walk/ kick/ structure. Non-timeseries fields (fps,
joint_names, body_names, kick_leg, ...) are copied verbatim -- joint order is
identity-preserved.
"""

import argparse
import glob
import os

import numpy as np

# every per-frame array in the csv_to_npz layout; anything else is metadata
TIME_SERIES = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
               "body_lin_vel_w", "body_ang_vel_w")
# derivatives of position: interpolated AND rescaled by the time scale
VEL_SERIES = ("joint_vel", "body_lin_vel_w", "body_ang_vel_w")


def _make_sign_continuous(q_flat: np.ndarray) -> np.ndarray:
    """Flip quaternion rows that dot negative with the previous frame (per body).

    wxyz quaternions q and -q encode the same rotation; without this, linear
    interpolation between antipodal neighbors passes through zero.
    q_flat: (T, B, 4), modified copy returned.
    """
    q = q_flat.copy()
    for t in range(1, q.shape[0]):
        dots = np.sum(q[t] * q[t - 1], axis=-1)
        q[t, dots < 0.0] *= -1.0
    return q


def retime_file(src: str, dst: str, scale: float) -> tuple[int, int]:
    d = np.load(src)
    out = {k: d[k] for k in d.files if k not in TIME_SERIES}
    T = int(d["joint_pos"].shape[0])
    T_new = int(round((T - 1) / scale)) + 1
    src_t = np.arange(T, dtype=np.float64)
    dst_t = np.arange(T_new, dtype=np.float64) * scale

    for name in TIME_SERIES:
        a = d[name]
        shape = a.shape
        flat = a.reshape(T, -1).astype(np.float64)
        if name == "body_quat_w":
            flat = _make_sign_continuous(flat.reshape(T, -1, 4)).reshape(T, -1)
        ret = np.empty((T_new, flat.shape[1]), dtype=np.float64)
        for c in range(flat.shape[1]):
            ret[:, c] = np.interp(dst_t, src_t, flat[:, c])
        if name == "body_quat_w":
            # normalize PER BODY quaternion: ret is flattened (T_new, B*4)
            q = ret.reshape(T_new, -1, 4)
            q /= np.maximum(np.linalg.norm(q, axis=2, keepdims=True), 1e-12)
            ret = q.reshape(T_new, -1)
        if name in VEL_SERIES:
            ret *= scale
        out[name] = ret.reshape((T_new,) + shape[1:]).astype(np.float32)

    np.savez(dst, **out)
    return T, T_new


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--src", required=True, help="source amp root dir (walk/ kick/ ...)")
    parser.add_argument("--dst", required=True, help="destination dir (created; never overwritten from source)")
    parser.add_argument("--time_scale", type=float, default=0.6,
                        help="playback speed factor, (0, 1) slows the clips down (default 0.6)")
    args = parser.parse_args()
    if not (0.0 < args.time_scale < 1.0):
        parser.error("--time_scale must be in (0, 1)")

    files = sorted(glob.glob(os.path.join(args.src, "**", "*.npz"), recursive=True))
    if not files:
        parser.error(f"no npz under {args.src}")

    print(f"retiming {len(files)} clips x{args.time_scale}: {args.src} -> {args.dst}")
    for f in files:
        rel = os.path.relpath(f, args.src)
        dst = os.path.join(args.dst, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        src_d, dst_d = np.load(f), None
        T, T_new = retime_file(f, dst, args.time_scale)
        dst_d = np.load(dst)
        # per-file self-check: fps/names verbatim, quat unit norm, velocity scaled
        assert int(round(float(dst_d["fps"].item() if hasattr(dst_d["fps"], "item") else dst_d["fps"]))) == 50, f"{rel}: fps changed"
        assert list(map(str, dst_d["joint_names"])) == list(map(str, src_d["joint_names"])), f"{rel}: joint order changed"
        assert list(map(str, dst_d["body_names"])) == list(map(str, src_d["body_names"])), f"{rel}: body order changed"
        qn = np.abs(np.linalg.norm(dst_d["body_quat_w"], axis=-1) - 1.0).max()
        assert qn < 1e-5, f"{rel}: quat norm dev {qn}"
        sv = np.percentile(np.abs(src_d["joint_vel"]), 98)
        dv = np.percentile(np.abs(dst_d["joint_vel"]), 98)
        # the *= scale on velocity series is the one step whose silent loss
        # would corrupt every downstream (s,t+1) pair; assert it, not just log
        assert abs(dv / sv - args.time_scale) < 0.15 * args.time_scale, f"{rel}: vel ratio {dv/sv:.3f}"
        rs_src = np.linalg.norm(src_d["body_lin_vel_w"][:, 0, :], axis=-1).mean()
        rs_dst = np.linalg.norm(dst_d["body_lin_vel_w"][:, 0, :], axis=-1).mean()
        print(f"  {rel}: {T}->{T_new} fr | joint_vel p98 {sv:.2f}->{dv:.2f} "
              f"({dv/sv:.2f}x) | root spd {rs_src:.2f}->{rs_dst:.2f} m/s")
    print("done")


if __name__ == "__main__":
    main()
