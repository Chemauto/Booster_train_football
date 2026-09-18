"""The dataset gate rejects corrupt/impossible trajectories without loading Isaac."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT.parent / 'booster_assets/robots/K1/K1_22dof.urdf'
SCRIPT = ROOT / 'scripts/validate_motion_dataset.py'


class MotionDatasetGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        joints = [j for j in ET.parse(URDF).getroot().findall('joint')
                  if j.get('type') in ('revolute', 'prismatic')]
        self.names = [j.get('name') for j in joints]
        bodies = ['trunk'] + [j.find('child').get('link') for j in joints]
        q = np.array([(float(j.find('limit').get('lower')) +
                       float(j.find('limit').get('upper'))) / 2 for j in joints])
        quat = np.zeros((100, len(bodies), 4), dtype=np.float32)
        quat[..., 0] = 1
        self.data = dict(fps=np.array([50]), joint_names=np.array(self.names),
                         body_names=np.array(bodies), joint_pos=np.tile(q, (100, 1)).astype(np.float32),
                         joint_vel=np.zeros((100, len(joints)), dtype=np.float32),
                         body_pos_w=np.zeros((100, len(bodies), 3), dtype=np.float32),
                         body_quat_w=quat, body_lin_vel_w=np.zeros((100, len(bodies), 3)),
                         body_ang_vel_w=np.zeros((100, len(bodies), 3)))

    def run_gate(self, *, corrupt=False, extra=(), empty=False):
        if not empty:
            if corrupt:
                (self.directory / 'clip.npz').write_bytes(b'not a zip')
            else:
                np.savez(self.directory / 'clip.npz', **self.data)
        report = self.directory / 'report.json'
        result = subprocess.run([sys.executable, str(SCRIPT), '--data_dir', str(self.directory),
                                 '--report', str(report), *extra], capture_output=True, text=True)
        self.assertTrue(report.exists(), result.stderr)
        return result.returncode, json.loads(report.read_text())

    def assert_hard(self, fragment, **kwargs):
        code, report = self.run_gate(**kwargs)
        self.assertEqual(code, 1)
        self.assertTrue(any(fragment in issue for issue in report['hard']), report)
        if report['clips']:
            self.assertIn('clip.npz', report['per_clip'])

    def test_valid_schema_passes(self):
        code, report = self.run_gate()
        self.assertEqual(code, 0, report)
        self.assertEqual(report['per_clip']['clip.npz']['frames'], 100)

    def test_every_motion_array_must_be_finite(self):
        for key in ('joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w',
                    'body_lin_vel_w', 'body_ang_vel_w'):
            with self.subTest(key=key):
                old = self.data[key].copy()
                self.data[key].flat[0] = np.nan
                self.assert_hard(f'{key}: non-finite')
                self.data[key] = old

    def test_zero_quaternion_is_hard_failure(self):
        self.data['body_quat_w'][0, 0] = 0
        self.assert_hard('quaternion norm')

    def test_directional_asymmetric_joint_limit(self):
        self.data['joint_pos'][0, self.names.index('right_hip_roll_joint')] = .5
        self.assert_hard('right_hip_roll_joint: position')

    def test_positive_left_hip_roll_is_valid(self):
        self.data['joint_pos'][:, self.names.index('left_hip_roll_joint')] = .5
        self.assertEqual(self.run_gate()[0], 0)

    def test_velocity_uses_precise_urdf_limit(self):
        self.data['joint_vel'][0, self.names.index('left_knee_pitch_joint')] = 12.58
        self.assert_hard('left_knee_pitch_joint: velocity')

    def test_head_velocity_is_checked(self):
        self.data['joint_vel'][0, self.names.index('aahead_yaw_joint')] = 8
        self.assert_hard('aahead_yaw_joint: velocity')

    def test_float32_boundary_noise_is_allowed(self):
        self.data['joint_pos'][0, self.names.index('right_hip_roll_joint')] = .375 + 5e-7
        self.data['joint_vel'][0, self.names.index('left_knee_pitch_joint')] = 12.57 + 5e-5
        self.assertEqual(self.run_gate()[0], 0)

    def test_fractional_fps_is_rejected(self):
        self.data['fps'] = np.array([50.1])
        self.assert_hard('fps')

    def test_missing_field_still_writes_report(self):
        del self.data['joint_pos']
        self.assert_hard('missing field joint_pos')

    def test_damaged_archive_still_writes_report(self):
        self.assert_hard('cannot read', corrupt=True)

    def test_empty_directory_still_writes_report(self):
        self.assert_hard('no npz', empty=True)

    def test_wrong_shape_is_rejected(self):
        self.data['body_pos_w'] = self.data['body_pos_w'][:-1]
        self.assert_hard('body_pos_w: shape')

    def test_duplicate_names_are_rejected(self):
        self.data['joint_names'][1] = self.data['joint_names'][0]
        self.assert_hard('joint_names: duplicate')

    def test_unknown_joint_is_rejected(self):
        self.data['joint_names'][0] = 'unknown_joint'
        self.assert_hard('joint_names: expected')

    def test_missing_support_body_is_rejected(self):
        self.data['body_names'][0] = 'not_trunk'
        self.assert_hard('body_names: missing')

    def test_override_urdf_controls_limit(self):
        root = ET.parse(URDF)
        next(j for j in root.getroot().findall('joint')
             if j.get('name') == 'right_hip_roll_joint').find('limit').set('upper', '0.6')
        custom = self.directory / 'custom.urdf'
        root.write(custom)
        self.data['joint_pos'][0, self.names.index('right_hip_roll_joint')] = .5
        self.assertEqual(self.run_gate(extra=('--urdf', str(custom)))[0], 0)

    def test_near_limits_and_support_findings_remain_soft(self):
        self.data['joint_pos'][:, self.names.index('left_ankle_roll_joint')] = .34
        self.data['body_pos_w'][:, 0, 0] = .4
        code, report = self.run_gate()
        self.assertEqual(code, 0, report)
        self.assertTrue(any('soft position' in x for x in report['soft']), report)

    def test_ankle_origin_height_is_not_mistaken_for_foot_flight(self):
        for name in ('left_ankle_roll_link', 'right_ankle_roll_link'):
            index = self.data['body_names'].tolist().index(name)
            self.data['body_pos_w'][:, index, 2] = .06
        code, report = self.run_gate()
        self.assertEqual(code, 0, report)
        self.assertFalse(any('possible flight' in x for x in report['soft']), report['soft'])

    def test_raised_soles_are_reported(self):
        for name in ('left_ankle_roll_link', 'right_ankle_roll_link'):
            index = self.data['body_names'].tolist().index(name)
            self.data['body_pos_w'][:, index, 2] = .12
        _, report = self.run_gate()
        self.assertTrue(any('possible flight' in x for x in report['soft']))


if __name__ == '__main__':
    unittest.main()
