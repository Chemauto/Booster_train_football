"""Offline cloning must use the same coordinates and transforms as deployment."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from rsl_rl.networks.normalization import EmpiricalNormalization

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bc_pretrain as bc


class BehaviorCloningContracts(unittest.TestCase):
    def test_heading_is_world_z_rotation(self):
        angle = np.array([0., np.pi / 2, -np.pi / 2, np.pi])
        q = np.zeros((4, 4))
        q[:, 0], q[:, 3] = np.cos(angle / 2), np.sin(angle / 2)
        np.testing.assert_allclose(bc.yaw_of(q), angle, atol=1e-7)

    def test_offline_normalization_matches_runner_including_small_variance(self):
        norm = EmpiricalNormalization(79).eval()
        norm._mean.fill_(0.2)
        norm._var.fill_(1e-6)
        norm._std.fill_(1e-3)
        obs = torch.ones(3, 79)
        torch.testing.assert_close(bc.normalize_observations(obs, norm.state_dict()), norm(obs))

    def test_previous_action_is_previous_label_with_distinct_action_offset(self):
        n, frames = 22, 4
        q = np.arange(frames)[:, None] * np.ones((1, n)) * .1
        quat = np.zeros((frames, 1, 4)); quat[:, :, 0] = 1.
        const = {"joint_names": [str(i) for i in range(n)],
                 "default_joint_pos": [0.] * n, "action_offset_resolved": [.03] * n,
                 "action_scale_resolved": [.5] * n, "field": {"goal_x": 7., "goal_y": 0.}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "motion.npz"
            np.savez(path, joint_names=const["joint_names"], joint_pos=q,
                     joint_vel=np.zeros_like(q), body_quat_w=quat,
                     body_pos_w=np.zeros((frames, 1, 3)), body_ang_vel_w=np.zeros((frames, 1, 3)))
            obs, target = bc.build_clip(str(path), const)
        torch.testing.assert_close(obs[1:, 57:79], target[:-1])
        np.testing.assert_allclose(target.numpy(), (q[1:] - .03) / .5)
        np.testing.assert_allclose(obs[:, 13:35].numpy(), q[:-1])


if __name__ == "__main__":
    unittest.main()
