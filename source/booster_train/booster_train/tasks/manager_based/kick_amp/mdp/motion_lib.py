"""AMP expert motion library for kick_amp.

Loads every npz under a root dir (walk/, kick/ subdirs produced by
scripts/amp_data_build.py + csv_to_npz.py batch mode) onto the device and serves:

  - sample_batch(env_ids)      : reference-state initialization states
                                 (t1.py Motion.sample_batch: random frame +
                                 random yaw re-orientation)
  - feed_forward_generator(..) : minibatches of expert (s_t, s_{t+1}) AMP pairs
                                 (55-dim, same layout as SoccerStateCommand._compute_amp_obs)

npz layout (from csv_to_npz.py): fps / joint_pos(T,J) / joint_vel / body_pos_w(T,B,3)
/ body_quat_w(T,B,4) / body_lin_vel_w / body_ang_vel_w / joint_names / body_names.

IMPORTANT: the npz joint order is the PhysX articulation order (BFS, e.g.
aahead_pitch sits at index 5), NOT the URDF declaration order. Everything here
permutes joint columns into the caller-supplied joint_names (the live robot's
order) at load time. All downstream slicing is name-based, never positional.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch
from isaaclab.utils.math import quat_rotate_inverse, quat_mul

# The AMP obs excludes the head AND the arms. The arm columns were a free
# discriminator feature rather than a gait signal: K1's default stance holds
# shoulder_roll at -+1.3 rad (arms out) while g1_to_k1_csv.py copies the human
# shoulder_roll 1:1 with no offset, so the expert clips sit near -+0.2 with the
# OPPOSITE sign. Measured on run 13-27-46/model_1000 (scripts/diag_amp_gap.py):
# right_shoulder_roll expert -0.236+-0.353 vs policy +1.331+-0.099, Cohen's
# d = 6.05 at 0% overlap; left_shoulder_roll d = 5.13. Two dimensions on which
# a threshold gives a perfect verdict without ever looking at the legs -- which
# is why the discriminator saturated within ~40 iterations in all 26 runs, why
# the style reward was therefore a constant (zero gradient, no gait ever
# taught), and why the 0.6x retime changed nothing (it rescales velocities and
# never touches joint angles). Dropping the 8 arm columns leaves max
# separability at d = 0.77 (hip_yaw): overlapping distributions, i.e. a
# discriminator that has to look at the gait to win.
AMP_OBS_DIM = 39  # 3 gravity + 3 lin_vel + 3 ang_vel + 12 dof_pos + 12 dof_vel + 6 feet

HEAD_JOINTS = ("aahead_yaw_joint", "aahead_pitch_joint")
# the 'aa' prefixes exist to force the PhysX BFS ordering
ARM_JOINTS = (
    "aaleft_shoulder_pitch_joint", "aaright_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_elbow_pitch_joint", "right_elbow_pitch_joint",
    "left_elbow_yaw_joint", "right_elbow_yaw_joint",
)
# AMP obs only. Resets and the action-rate penalty keep the head-only exclusion
# (body_dof_idx / body_joint_idx), so arms are still reset from motion frames
# and still pay for jitter -- this changes what the discriminator SEES, nothing
# about what the robot does.
AMP_EXCLUDED_JOINTS = HEAD_JOINTS + ARM_JOINTS
FEET_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


class AmpMotionDataset:
    def __init__(self, root_dir: str, joint_names: list[str], device: str = "cpu"):
        self.device = device
        self.joint_names = list(joint_names)
        self.head_mask = torch.tensor(
            [n in HEAD_JOINTS for n in self.joint_names], dtype=torch.bool, device=device
        )
        # reset path (sample_batch -> events.ref_state_init): head-excluded only,
        # so motion-frame resets still place the arms
        self.body_dof_idx = (~self.head_mask).nonzero(as_tuple=False).flatten()
        # AMP obs path: head AND arms excluded (see AMP_EXCLUDED_JOINTS above)
        self.amp_dof_idx = torch.tensor(
            [i for i, n in enumerate(self.joint_names) if n not in AMP_EXCLUDED_JOINTS],
            dtype=torch.long, device=device,
        )

        files = sorted(glob.glob(os.path.join(root_dir, "**", "*.npz"), recursive=True))
        if not files:
            raise FileNotFoundError(f"no motion npz under {root_dir}")

        amp_obs, heights, quats, lin_vels, ang_vels, dofs, dof_vels, valid = [], [], [], [], [], [], [], []
        start = 0
        for f in files:
            d = np.load(f)
            # expert pair dt must equal the policy dt (0.02 s): a mismatched
            # clip silently corrupts the (s_t, s_{t+1}) velocity terms
            if int(round(float(d["fps"]))) != 50:
                raise ValueError(f"{f}: fps {float(d['fps'])} != 50 (policy dt 0.02 s)")
            file_joints = [str(n) for n in d["joint_names"]]
            # permute file columns into the live robot's joint order
            perm = [file_joints.index(n) for n in self.joint_names]
            joint_pos = torch.from_numpy(d["joint_pos"][:, perm]).float().to(device)          # (T, J)
            joint_vel = torch.from_numpy(d["joint_vel"][:, perm]).float().to(device)
            body_pos = torch.from_numpy(d["body_pos_w"]).float().to(device)          # (T, B, 3)
            body_quat = torch.from_numpy(d["body_quat_w"]).float().to(device)        # (T, B, 4) wxyz
            body_lin = torch.from_numpy(d["body_lin_vel_w"]).float().to(device)
            body_ang = torch.from_numpy(d["body_ang_vel_w"]).float().to(device)

            root_quat = body_quat[:, 0, :]
            gravity = quat_rotate_inverse(root_quat, torch.tensor([0.0, 0.0, -1.0], device=device).expand(len(root_quat), -1))
            lin_b = quat_rotate_inverse(root_quat, body_lin[:, 0, :])
            ang_b = quat_rotate_inverse(root_quat, body_ang[:, 0, :])
            feet_idx = [list(map(str, d["body_names"])).index(n) for n in FEET_BODIES]
            feet_rel = quat_rotate_inverse(
                root_quat.unsqueeze(1).expand(-1, 2, -1), body_pos[:, feet_idx, :] - body_pos[:, 0:1, :]
            ).flatten(1)
            # AMP excludes head + arms -- name-based column selection
            dof_amp = joint_pos[:, self.amp_dof_idx]
            dof_vel_amp = joint_vel[:, self.amp_dof_idx]
            obs = torch.cat((gravity, lin_b, ang_b, dof_amp, dof_vel_amp * 0.1, feet_rel), dim=-1)
            assert obs.shape[1] == AMP_OBS_DIM, f"{f}: amp obs dim {obs.shape[1]} != {AMP_OBS_DIM}"

            # reset path keeps head-excluded columns
            dof_body = joint_pos[:, self.body_dof_idx]
            dof_vel_body = joint_vel[:, self.body_dof_idx]

            T = len(joint_pos)
            amp_obs.append(obs)
            heights.append(body_pos[:, 0, 2])
            quats.append(root_quat)
            lin_vels.append(body_lin[:, 0, :])   # world frame; sample_batch rotates it
            ang_vels.append(body_ang[:, 0, :])
            dofs.append(dof_body)                # head-excluded, robot order
            dof_vels.append(dof_vel_body)
            valid.append(torch.arange(start, start + T - 1, device=device))  # pairs need t+1
            start += T

        self.files = [os.path.basename(f) for f in files]
        self.amp_obs = torch.cat(amp_obs)          # (N, 55)
        self.heights = torch.cat(heights)          # (N,)
        self.quats = torch.cat(quats)              # (N, 4)
        self.lin_vels = torch.cat(lin_vels)        # (N, 3)
        self.ang_vels = torch.cat(ang_vels)        # (N, 3)
        self.dofs = torch.cat(dofs)                # (N, J-2) robot order, head-excluded
        self.dof_vels = torch.cat(dof_vels)
        self.valid_idxs = torch.cat(valid)         # frames whose t+1 exists
        print(f"[AmpMotionDataset] {len(files)} clips, {len(self.amp_obs)} frames "
              f"({len(self.amp_obs) / 50.0:.1f}s @50fps), joint order: live robot")

    def sample_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Reference-state init: random frame, random yaw re-orientation (t1.py).

        Velocities are returned in the original body frame; the caller rotates
        them with the returned base_quat when writing world-frame root states.
        dof_pos/dof_vel cover the NON-head joints in live-robot order; the caller
        writes them via name-based indices.
        """
        idx = self.valid_idxs[torch.randint(0, len(self.valid_idxs), (batch_size,), device=self.device)]
        yaw = torch.rand(batch_size, device=self.device) * 2 * torch.pi
        # wxyz, NOT xyzw: this was (0, 0, sin, cos) -- i.e. w=0, a 180-degree
        # flip about an axis in the y-z plane rather than a yaw. Measured on the
        # real clips: 49% of motion resets came out upside down (body up-axis
        # world z < 0) and 74% severely toppled, mean up_z +0.03 instead of
        # +0.99. That, not actuator weakness, is why frac=1.0 made every
        # episode end in exactly one fall after ~15 steps (free fall from 0.5 m
        # to the 0.35 m termination height takes ~8.7 policy steps), and it
        # would have handed the discriminator a fresh non-gait giveaway via
        # gravity_b the moment motion resets begin at RESET_STAND_STEPS.
        yaw_quat = torch.stack(
            (torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)), dim=1
        )
        return {
            "base_height": self.heights[idx],
            "base_quat": quat_mul(yaw_quat, self.quats[idx]),
            "base_lin_vel": quat_rotate_inverse(self.quats[idx], self.lin_vels[idx]),
            "base_ang_vel": quat_rotate_inverse(self.quats[idx], self.ang_vels[idx]),
            "dof_pos": self.dofs[idx],
            "dof_vel": self.dof_vels[idx],
        }

    def feed_forward_generator(self, num_mini_batches: int, mini_batch_size: int):
        """Yields (s_t, s_{t+1}) expert pairs, shuffled per minibatch."""
        for _ in range(num_mini_batches):
            idx = self.valid_idxs[torch.randint(0, len(self.valid_idxs), (mini_batch_size,), device=self.device)]
            yield self.amp_obs[idx], self.amp_obs[idx + 1]
