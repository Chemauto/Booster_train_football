"""Marker orientations already use the K1 body convention; offsets must preserve it."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import accad_markers_to_k1 as retarget


class MarkerOrientationTests(unittest.TestCase):
    def test_ik_uses_training_joint_limits_and_foot_geometry(self):
        model = mujoco.MjModel.from_xml_path(str(retarget.gmr_params.ROBOT_XML_DICT['booster_k1']))
        training = mujoco.MjModel.from_xml_path(str(retarget.TRAIN_MODEL))
        retarget.align_ik_model(model)
        data, target = mujoco.MjData(model), mujoco.MjData(training)
        data.qpos[:7] = target.qpos[:7] = [0, 0, .6, 1, 0, 0, 0]
        for source, destination in retarget.JOINT_MAP.items():
            source_id, target_id = model.joint(source).id, training.joint(destination).id
            np.testing.assert_allclose(model.jnt_range[source_id], retarget.K1_JOINT_LIMITS[destination])
            angle = .4 if 'knee' in destination else (.12 if 'hip_pitch' in destination else 0.)
            data.qpos[model.jnt_qposadr[source_id]] = angle
            target.qpos[training.jnt_qposadr[target_id]] = angle
        mujoco.mj_kinematics(model, data)
        mujoco.mj_kinematics(training, target)
        for side in ('left', 'right'):
            np.testing.assert_allclose(data.xpos[model.body(side + '_foot_link').id],
                                       target.xpos[training.body(side + '_ankle_roll_link').id], atol=1e-10)

    def test_ik_offsets_preserve_nontrivial_marker_orientation(self):
        config = json.loads(retarget.build_config(retarget.TASK_SETS['foot_arms']).read_text())
        measured = Rotation.from_euler('xyz', [.2, -.15, 1.1])
        for table in ('ik_match_table1', 'ik_match_table2'):
            for body, entry in config[table].items():
                with self.subTest(table=table, body=body):
                    actual = measured * Rotation.from_quat(entry[4], scalar_first=True)
                    self.assertLess((actual.inv() * measured).magnitude(), 1.e-10)

    def test_marker_anatomical_axes_map_forward_and_up(self):
        expected = Rotation.from_euler('z', 1.2)
        frame = retarget.orientation_from_axes(expected.apply([1,0,0]), expected.apply([0,0,1]), [1,0,0], [0,0,1])
        np.testing.assert_allclose(frame.as_matrix(), expected.as_matrix(), atol=1e-12)

if __name__ == '__main__':
    unittest.main()
