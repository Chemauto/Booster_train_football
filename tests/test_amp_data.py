"""CPU-only regression tests for G1 column layouts and K1 motion consistency."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import amp_data_build as converter
import g1_to_k1_csv as mapping


class SourceDiscoveryTests(unittest.TestCase):
    def test_stylized_source_uses_actual_soccer_directory(self):
        self.assertEqual(Path(converter.SOCCER_DIRS[1]).name, 'soccer-stylized')

    def test_missing_source_directory_raises_instead_of_returning_empty(self):
        with tempfile.TemporaryDirectory() as temp:
            missing = Path(temp) / 'missing-soccer'
            with self.assertRaisesRegex(FileNotFoundError, 'missing-soccer'):
                converter.motion_source_files(missing, '*.npz')

    def test_empty_source_directory_raises(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'No motion files'):
                converter.motion_source_files(temp, '*.npz')

    def test_source_discovery_is_sorted_and_excludes_nonmatching_files(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for name in ['b.npz', 'a.npz', 'notes.txt']:
                (directory / name).touch()
            self.assertEqual([p.name for p in converter.motion_source_files(directory, '*.npz')], ['a.npz', 'b.npz'])


class JointMappingTests(unittest.TestCase):
    def test_named_columns_are_order_independent(self):
        names = list(mapping.G1_JOINT_NAMES)
        values = np.zeros((3, len(names)))
        values[:, names.index('left_hip_pitch_joint')] = 0.12
        values[:, names.index('right_hip_pitch_joint')] = -0.34
        values[:, names.index('left_knee_joint')] = 0.56
        expected = converter.retarget_g1_to_k1(values, source_joint_names=names)
        perm = np.random.default_rng(9).permutation(len(names))
        actual = converter.retarget_g1_to_k1(values[:, perm], source_joint_names=np.array(names)[perm])
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual[0, converter.K1_JOINT_NAMES_ORDER.index('left_knee_pitch_joint')], 0.56)

    def test_upper_body_uses_k1_neutral_pose(self):
        values = np.zeros((3, 29))
        actual = converter.retarget_g1_to_k1(values)
        self.assertEqual(actual[0, converter.K1_JOINT_NAMES_ORDER.index('left_shoulder_roll_joint')], -1.3)
        self.assertEqual(actual[0, converter.K1_JOINT_NAMES_ORDER.index('right_shoulder_roll_joint')], 1.3)

    def test_unnamed_npz_requires_explicit_verified_layout(self):
        with self.assertRaisesRegex(ValueError, 'layout'):
            mapping.npz_joint_names({'joint_pos': np.zeros((3, 29))})

    def test_legacy_physx_column_is_correct(self):
        names = mapping.npz_joint_names({'joint_pos': np.zeros((3, 29))}, legacy_layout='humanoid_soccer_physx')
        self.assertEqual(names[1], 'right_hip_pitch_joint')
        self.assertEqual(names[9], 'left_knee_joint')

    def test_smoothing_does_not_zero_pad_edges(self):
        x = np.full((5, 2), 0.7)
        np.testing.assert_allclose(mapping.smooth(x, 3), x)


class MotionBuildTests(unittest.TestCase):
    def test_grounding_and_support_anchor_remove_translational_skating(self):
        import build_amp_motion_npz as builder
        root = np.zeros((20, 3)); root[:, 0] = np.arange(20) * 0.01
        body = np.repeat(root[:, None, :], 3, axis=1)
        body[:, 1:, 2] = 0.1
        quat = np.zeros((20, 3, 4)); quat[:, :, 0] = 1
        names = ['trunk', 'left_ankle_roll_link', 'right_ankle_roll_link']
        corrected, stats = builder.ground_and_anchor(root, body, quat, names, 50, True)
        shifted = body + (corrected-root)[:, None, :]
        corners = builder.sole_corners(shifted, quat, names)
        np.testing.assert_allclose(corners[..., 2].min(axis=(1, 2)), 0.005, atol=1e-10)
        np.testing.assert_allclose(np.diff(shifted[:, 1:, 0], axis=0), 0, atol=1e-10)
        self.assertGreater(stats['support_speed_before_mps'], 0.49)
        self.assertLess(stats['support_speed_after_mps'], 1e-10)

    def test_world_angular_velocity_and_quaternion_sign_invariance(self):
        from scipy.spatial.transform import Rotation
        import build_amp_motion_npz as builder
        q = Rotation.from_euler('z', np.arange(20) * 0.02).as_quat()[:, [3, 0, 1, 2]]
        q[::2] *= -1
        v = builder.angular_velocity(q[:, None, :], 50)
        np.testing.assert_allclose(v[:, 0], np.tile([0, 0, 1], (20, 1)), atol=1e-10)


if __name__ == '__main__':
    unittest.main()
