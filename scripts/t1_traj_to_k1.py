"""T1 forward-kinematics helper, plus an IK retarget that was evaluated and rejected.

Status: the IK path at the bottom of this file is NOT used. The paper's data
turned out to need only a small correction, which now lives in
`scripts/convert_paper_data.py` (T1's waist yaw folded into K1's hip yaw). What is
still useful here is `t1_source_poses()`: forward kinematics of the paper's T1
trajectories, i.e. ground truth to verify that conversion against.

Corrections to an earlier version of this file's reasoning
----------------------------------------------------------
It claimed T1's waist yaw reaches ~40 deg median / 87 deg max and that dropping it
yaws every foot by a median 44.8 deg. Both numbers were wrong, for two separate
reasons that are worth recording:

  * the joints start at CSV column 13, after the root pose and its velocities.
    Reading `raw[:, 7:]` feeds angular velocities in as joint angles.
  * `q_left_hip_pitch`[2:] maps to 'left_hip_pitch', not T1's 'Left_Hip_Pitch', so
    `mj_name2id` returned -1 for every joint and each write silently landed on
    qpos[-1], leaving the whole leg at its rest pose.

With the column offset and an explicit name map the real waist yaw is median
5.4 deg, max 24.6 deg, and the true cost of dropping it is a median 5.44 deg
error in the foot's orientation relative to the trunk. Folding it into the hip
yaw cuts that to 1.14 deg (docs/reports/2026-09-18-accad-marker-dataset.md).

Why the IK retarget was rejected: T1's thigh:shank is 0.48 against K1's 0.78
(0.135/0.280 m vs 0.1915/0.2452 m). Matching T1's foot poses exactly therefore
forces K1 into a different knee configuration, and an early (buggy-input) run
produced a permanently straight knee - posture is what an AMP prior teaches, so
preserving the joint angles matters more than preserving the absolute foot pose.
"""
import argparse
import csv
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import accad_markers_to_k1 as base  # noqa: E402
import general_motion_retargeting.params as gmr_params  # noqa: E402
from general_motion_retargeting import GeneralMotionRetargeting as GMR  # noqa: E402

from build_amp_motion_npz import Kinematics, sha256  # noqa: E402

T1_XML = Path('/data/rl_robot/BoosterRobotics/code/resources/T1.xml')
# The CSVs put the 13 joint columns *after* the root pose and its velocities:
# x,y,z,qx,qy,qz,qw,v_x,v_y,v_z,w_x,w_y,w_z,q_waist,q_left_hip_pitch,...
# so the joints start at column 13. Reading raw[:, 7:] silently feeds angular
# velocities in as joint angles - a bug that invalidated an earlier version of
# this analysis, so the offset is asserted against the header below.
T1_JOINT_COLUMN = 13
T1_JOINTS = [
    'q_waist', 'q_left_hip_pitch', 'q_left_hip_roll', 'q_left_hip_yaw',
    'q_left_knee_pitch', 'q_left_ankle_pitch', 'q_left_ankle_roll',
    'q_right_hip_pitch', 'q_right_hip_roll', 'q_right_hip_yaw',
    'q_right_knee_pitch', 'q_right_ankle_pitch', 'q_right_ankle_roll',
]
# Derived names like 'q_left_hip_pitch'[2:].title() are NOT T1's joint names
# ('Left_Hip_Pitch', and the waist is 'Waist' not 'q_waist'), and a wrong name
# makes mj_name2id return -1, which silently writes to qpos[-1] and leaves the
# whole leg at rest. So map explicitly.
T1_TO_K1_NAME = {
    'q_waist': 'Waist',
    'q_left_hip_pitch': 'Left_Hip_Pitch', 'q_left_hip_roll': 'Left_Hip_Roll',
    'q_left_hip_yaw': 'Left_Hip_Yaw', 'q_left_knee_pitch': 'Left_Knee_Pitch',
    'q_left_ankle_pitch': 'Left_Ankle_Pitch', 'q_left_ankle_roll': 'Left_Ankle_Roll',
    'q_right_hip_pitch': 'Right_Hip_Pitch', 'q_right_hip_roll': 'Right_Hip_Roll',
    'q_right_hip_yaw': 'Right_Hip_Yaw', 'q_right_knee_pitch': 'Right_Knee_Pitch',
    'q_right_ankle_pitch': 'Right_Ankle_Pitch', 'q_right_ankle_roll': 'Right_Ankle_Roll',
}
FEET = (('left', 'left_foot_link', 'Hip_Yaw_Left', 'Ankle_Cross_Left'),
        ('right', 'right_foot_link', 'Hip_Yaw_Right', 'Ankle_Cross_Right'))


def t1_source_poses(model, data, csv_path):
    """Forward-kinematics T1 over a clip.

    Returns the trunk and foot world poses per frame plus the hip-to-ankle span
    (the quantity the retarget scale is set from).
    """
    with open(csv_path) as fh:
        header = next(csv.reader(fh))
    assert header[T1_JOINT_COLUMN] == 'q_waist', f'unexpected csv layout: {header[:16]}'
    raw = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    root, quat_xyzw = raw[:, :3], raw[:, [6, 3, 4, 5]]
    q = raw[:, T1_JOINT_COLUMN:T1_JOINT_COLUMN + len(T1_JOINTS)]
    jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    for csv_name, t1_name in T1_TO_K1_NAME.items():
        if jid(t1_name) < 0:
            raise ValueError(f'T1 model has no joint {t1_name!r} (from {csv_name!r})')
    trunk, feet, legs = [], {'left': [], 'right': []}, []
    for t in range(len(root)):
        data.qpos[:] = 0
        data.qpos[:3], data.qpos[3:7] = root[t], quat_xyzw[t]
        for name, value in zip(T1_JOINTS, q[t]):
            data.qpos[model.jnt_qposadr[jid(T1_TO_K1_NAME[name])]] = value
        mujoco.mj_kinematics(model, data)
        trunk.append((data.xpos[bid('Trunk')].copy(), data.xquat[bid('Trunk')].copy()))
        for side, foot_body, hip_body, ankle_body in FEET:
            i = bid(foot_body)
            feet[side].append((data.xpos[i].copy(), data.xquat[i].copy()))
            legs.append(np.linalg.norm(data.xpos[bid(hip_body)] - data.xpos[bid(ankle_body)]))
    return trunk, feet, float(np.median(legs))


def retarget(csv_path, target_fps, solver='quadprog', verbose=False):
    model = mujoco.MjModel.from_xml_path(str(T1_XML))
    data = mujoco.MjData(model)
    trunk, feet, src_leg = t1_source_poses(model, data, csv_path)
    frames = len(trunk)

    src_fps = 50.0                      # the paper's CSVs are 50 fps native
    step = max(1, int(round(src_fps / target_fps)))
    fps = src_fps / step

    # Scale so T1's hip-to-ankle matches K1's everywhere (leg_ratio = 1): the two
    # robots' legs differ by only ~5%, so this is nearly the identity and keeps
    # every foot target inside K1's reach. GMR scales positions by
    # scale_table * height / assumption = 0.6 * height / 1.8.
    scale = base.k1_leg_length() / src_leg
    height = 1.8 * scale / 0.6

    gmr_params.IK_CONFIG_DICT.setdefault('marker', {})['booster_k1'] = \
        base.build_config(base.TASK_SETS['foot'])
    retargeter = GMR(src_human='marker', tgt_robot='booster_k1',
                     actual_human_height=height, solver=solver, verbose=verbose)
    keys = set(retargeter.human_body_to_task1) | set(retargeter.human_body_to_task2)
    expected = {'pelvis', 'left_foot', 'right_foot'}
    if keys != expected:
        raise ValueError(f'the foot task set should need {expected}, got {keys}')

    frames_out = [
        {'pelvis': (trunk[t][0], trunk[t][1]),
         'left_foot': (feet['left'][t][0], feet['left'][t][1]),
         'right_foot': (feet['right'][t][0], feet['right'][t][1])}
        for t in range(frames)]

    retargeter.update_targets(frames_out[0])
    position, quat = retargeter.scaled_human_data['pelvis']
    retargeter.configuration.data.qpos[:3] = position
    retargeter.configuration.data.qpos[3:7] = quat

    qpos, errors = [], []
    for frame in frames_out:
        qpos.append(retargeter.retarget(frame).copy())
        errors.append(retargeter.error1())
    meta = {'source': str(csv_path), 'source_sha256': sha256(csv_path),
            'source_fps': src_fps, 'time_scale': 1.0,
            'retarget_method': 'T1 own FK (trunk + foot world poses) -> GMR mink IK on K1',
            'upper_body': 'K1 nominal',
            'waist': 'T1 waist expressed through K1 hip/ankle by the IK',
            't1_hip_to_ankle_m': src_leg, 'k1_leg_length_m': base.k1_leg_length(),
            'applied_position_scale': scale, 'ground_clearance_m': 0.005}
    return np.stack(qpos), fps, meta, np.asarray(errors)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--target_fps', type=float, default=50.0)
    ap.add_argument('--anchor_walk', action='store_true')
    ap.add_argument('--solver', default='quadprog')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    qpos, fps, meta, errors = retarget(args.csv, args.target_fps, args.solver, args.verbose)
    columns = base.gmr_qpos_columns()

    with np.load(base.SCHEMA_NPZ, allow_pickle=False) as schema:
        train_joints = [str(x) for x in schema['joint_names']]
        train_bodies = [str(x) for x in schema['body_names']]
    idx = {n: i for i, n in enumerate(train_joints)}
    joints = np.zeros((len(qpos), len(train_joints)))
    for gmr_joint, train_joint in base.JOINT_MAP.items():
        joints[:, idx[train_joint]] = qpos[:, columns[gmr_joint]]
    for name, value in base.NOMINAL_ARMS.items():
        joints[:, idx[name]] = value
    joints, cleanup = base.clean_joints(joints, train_joints, fps)
    meta.update(cleanup)

    result, stats = Kinematics(base.TRAIN_MODEL, train_joints, train_bodies).build(
        qpos[:, :3], qpos[:, 3:7], joints, fps, args.anchor_walk, meta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **result)
    print('ik mean=%.5f | %d fr | scale=%.4f (T1 leg %.4f -> K1 %.4f) | clipped %d | vlim %d -> %s'
          % (errors.mean(), stats['frames'], meta['applied_position_scale'],
             meta['t1_hip_to_ankle_m'], meta['k1_leg_length_m'],
             cleanup['joint_limit_clipped_samples'], cleanup['velocity_limited_samples'], args.out))


if __name__ == '__main__':
    main()
