"""Offline kinematic initialization for AMP; closed-loop evaluation is required.

Next-frame joint positions are position-target heuristics, not measured expert
motor commands. A low validation error does not establish dynamically feasible
walking. Observation transforms must match the online runner exactly.
"""

import argparse
import copy
import glob
import json
import os

import numpy as np
import torch


OBS_DIM, NUM_STACK, N_ACT = 79, 50, 22


def qrot_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """wxyz quaternion, isaaclab quat_rotate_inverse math."""
    w, xyz = q[:, 0:1], q[:, 1:4]
    t = np.cross(xyz, v) * 2.0
    return v - w * t + np.cross(xyz, t)


def yaw_of(q: np.ndarray) -> np.ndarray:
    return np.arctan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                      1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def build_clip(path: str, const: dict) -> tuple[torch.Tensor, torch.Tensor]:
    d = np.load(path)
    names = [str(n) for n in d["joint_names"]]
    assert names == const["joint_names"], f"{path}: joint order mismatch"
    q, qd = d["joint_pos"].astype(np.float64), d["joint_vel"].astype(np.float64)
    rq, rp = d["body_quat_w"][:, 0, :].astype(np.float64), d["body_pos_w"][:, 0, :].astype(np.float64)
    ang_w = d["body_ang_vel_w"][:, 0, :].astype(np.float64)
    default = np.array(const["default_joint_pos"])
    scale = np.array(const["action_scale_resolved"])
    offset = np.array(const["action_offset_resolved"])

    T = len(q)
    obs = np.zeros((T, OBS_DIM), dtype=np.float64)
    obs[:, 0:3] = qrot_inv(rq, np.tile([0.0, 0.0, -1.0], (T, 1)))          # projected gravity
    obs[:, 3:6] = qrot_inv(rq, ang_w)                                       # base ang vel (body)
    obs[:, 6:9] = 0.0                                                       # ball: unseen
    yaw = yaw_of(rq)
    gx, gy = const["field"]["goal_x"], const["field"]["goal_y"]
    dx, dy = gx - rp[:, 0], gy - rp[:, 1]
    c, s = np.cos(-yaw), np.sin(-yaw)
    obs[:, 9] = c * dx - s * dy                                             # goal in yaw frame
    obs[:, 10] = s * dx + c * dy
    obs[:, 11], obs[:, 12] = np.cos(yaw), np.sin(yaw)
    obs[:, 13:35] = q - default                                             # joint_pos_rel
    obs[:, 35:57] = qd * 0.1                                                # joint_vel scaled
    a_prev = (q - offset) / scale                                          # last issued target ~ tracks q[t]
    obs[1:, 57:79] = a_prev[1:]
    target = (q[1:] - offset) / scale                                      # a_t tracks q[t+1]
    return torch.tensor(obs[:-1]), torch.tensor(target)


def normalize_observations(obs, state):
    """Use the same implementation and epsilon as AmpRunner."""
    from rsl_rl.networks.normalization import EmpiricalNormalization
    norm = EmpiricalNormalization(OBS_DIM).eval()
    norm.load_state_dict(state)
    return norm(obs.float())


def main():
    # Load this pure-torch module without importing the simulator task registry.
    import importlib.util
    from pathlib import Path
    source = Path(__file__).resolve().parents[1] / "source/booster_train/booster_train/rsl_rl/amp/modules.py"
    spec = importlib.util.spec_from_file_location("amp_models", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ActorCriticAMP = module.ActorCriticAMP
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/data/rl_robot/BoosterRobotics/booster_assets/motions/K1/amp_paper")
    ap.add_argument("--init_checkpoint", default="logs/rsl_rl/k1_kick_amp_repaired/2026-09-17_09-28-14/model_3721.pt")
    ap.add_argument("--out", default="logs/bc/model_bc_init.pt")
    ap.add_argument("--constants", default="logs/bc_constants_nominal.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_clips", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    const = json.load(open(args.constants))
    ck = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
    model = ActorCriticAMP(OBS_DIM, 93, 14, N_ACT, num_stack=NUM_STACK)
    model.load_state_dict(ck["model"])

    # Keep the checkpoint normalization frozen for offline fitting.
    om = ck["obs_norm"]

    files = sorted(glob.glob(os.path.join(args.data_dir, "**", "*.npz"), recursive=True))
    print(f"{len(files)} clips")
    train_set, val_set = [], []
    for i, f in enumerate(files):
        obs, tgt = build_clip(f, const)
        obs_n = normalize_observations(obs, om)                      # normalize
        T = len(obs_n)
        # windows: stacked[t] = zeros then normalized obs[max(0,t-49)..t]
        tgt_idx = np.arange(NUM_STACK, T)                                           # need full-ish stack
        stacked = torch.zeros(len(tgt_idx), NUM_STACK, OBS_DIM, dtype=torch.float32)
        for k, t in enumerate(tgt_idx):
            lo = max(0, t - NUM_STACK + 1)
            stacked[k, NUM_STACK - (t - lo + 1):] = obs_n[lo:t + 1]
        X = (obs_n[tgt_idx].float(), stacked.float())
        Y = tgt[tgt_idx].float()
        (val_set if i < args.val_clips else train_set).append((X, Y))

    def _cat(sets):
        x0 = torch.cat([s[0][0] for s in sets]); x1 = torch.cat([s[0][1] for s in sets])
        return x0, x1, torch.cat([s[1] for s in sets])
    X0, X1, Y = _cat(train_set)
    V0, V1, VY = _cat(val_set)
    print(f"train {len(Y)} frames | val {len(VY)} frames")

    for p in model.critics.parameters():
        p.requires_grad_(False)
    params = list(model.actor.parameters()) + list(model.encoder.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scale_t = torch.tensor(const["action_scale_resolved"])

    def actor_mean(o, s):
        emb = model.encoder(s.flatten(start_dim=-2))
        return model.actor(torch.cat((o, emb), dim=-1))

    best, best_state = 1e9, None
    for ep in range(args.epochs):
        perm = torch.randperm(len(Y))
        tot = 0.0
        for i in range(0, len(Y), 2048):
            idx = perm[i:i + 2048]
            pred = actor_mean(X0[idx], X1[idx])
            loss = torch.nn.functional.mse_loss(pred, Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        with torch.no_grad():
            vp = actor_mean(V0, V1)
            vloss = torch.nn.functional.mse_loss(vp, VY).item()
            rad = float(((vp - VY).abs() * scale_t).mean())                   # mean joint cmd error, radians
        if vloss < best:
            best, best_state = vloss, copy.deepcopy(model.state_dict())
        if ep % 20 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep:3d} train {tot/len(Y):.5f} val {vloss:.5f} val_mean_rad {rad:.4f}", flush=True)

    model.load_state_dict(best_state)
    out_ck = dict(ck)
    out_ck["model"] = model.state_dict()
    # The actor/encoder changed offline: do not reuse their old Adam moments.
    out_ck.pop("optimizer", None)
    with torch.no_grad():
        rad = float(((actor_mean(V0, V1) - VY).abs() * scale_t).mean())
    out_ck["bc_metadata"] = {"seed": args.seed, "constants": args.constants,
                             "action_labels": "next-frame kinematic targets, not inverse dynamics"}
    torch.save(out_ck, args.out)
    json.dump({"val_loss": best, "val_mean_action_err_rad": rad, "clips": len(files),
               "train_frames": int(len(Y)), "init": args.init_checkpoint},
              open(args.out.replace(".pt", "_metrics.json"), "w"), indent=1)
    print(f"saved {args.out} | val_loss {best:.5f} | mean action err {rad:.4f} rad")


if __name__ == "__main__":
    main()
