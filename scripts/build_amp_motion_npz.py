"""Build auditable K1 AMP NPZ files using CPU MuJoCo forward kinematics.

Joint-angle transfer is an approximation, not full morphology-aware IK. Grounding
uses the four sole corners used by the reward. Walk root translation is optionally
fitted to the low support foot; this changes travel distance, never clip timing.
All velocities are recomputed after correction. Output must be a new directory.

Example:
  python scripts/build_amp_motion_npz.py --output_dir /path/amp_corrected \
      --legacy_layout humanoid_soccer_physx
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from amp_data_build import EXCLUDE, LAFAN_DIR, SOCCER_DIRS, motion_source_files
from g1_to_k1_csv import G1_JOINT_NAMES, K1_JOINT_NAMES, K1_JOINT_LIMITS, G1_TO_K1_MAP, npz_joint_names, retarget_g1_to_k1

ASSETS = Path(__file__).resolve().parents[1] / 'assets'
if not (ASSETS / 'robots').is_dir():
    ASSETS = Path('/data/rl_robot/BoosterRobotics/booster_assets')
SOLE = np.array([[0.1195, 0.04, -0.038], [0.1195, -0.04, -0.038],
                 [-0.066, 0.04, -0.038], [-0.066, -0.04, -0.038]])
FEET = ['left_ankle_roll_link', 'right_ankle_roll_link']


def sole_corners(body_pos, body_quat, body_names):
    ids = [list(body_names).index(name) for name in FEET]
    q = body_quat[:, ids]
    rot = Rotation.from_quat(q[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(*q.shape[:2], 3, 3)
    return body_pos[:, ids, None, :] + np.einsum('tfij,cj->tfci', rot, SOLE)


def ground_and_anchor(root, body_pos, body_quat, body_names, fps, anchor_walk, clearance=0.005):
    corners = sole_corners(body_pos, body_quat, body_names)
    height = corners[..., 2].min(axis=2)
    centers = corners.mean(axis=2)
    # Smooth weights avoid hard left/right switching. A lifted foot rapidly loses
    # weight; both grounded feet constrain root translation in double support.
    weights = np.exp(-(height - height.min(axis=1, keepdims=True)) / 0.008)
    weights /= weights.sum(axis=1, keepdims=True)
    interval_weights = np.sqrt(weights[:-1] * weights[1:])
    interval_weights /= interval_weights.sum(axis=1, keepdims=True)
    delta = np.diff(centers[..., :2], axis=0)
    before = np.sum(np.linalg.norm(delta * fps, axis=-1) * interval_weights, axis=1)
    result = root.copy()
    if anchor_walk:
        correction_steps = -np.sum(delta * interval_weights[..., None], axis=1)
        result[1:, :2] += np.cumsum(correction_steps, axis=0)
    result[:, 2] += clearance - height.min(axis=1)
    after_delta = delta + np.diff(result[:, :2] - root[:, :2], axis=0)[:, None, :]
    after = np.sum(np.linalg.norm(after_delta * fps, axis=-1) * interval_weights, axis=1)
    return result, {
        'support_speed_before_mps': float(before.mean()),
        'support_speed_after_mps': float(after.mean()),
        'support_speed_after_p95_mps': float(np.percentile(after, 95)),
        'root_xy_speed_before_mps': float(np.linalg.norm(np.diff(root[:, :2], axis=0) * fps, axis=-1).mean()),
        'root_xy_speed_after_mps': float(np.linalg.norm(np.diff(result[:, :2], axis=0) * fps, axis=-1).mean()),
        'root_xy_correction_endpoint_m': float(np.linalg.norm(result[-1, :2] - root[-1, :2])),
        'ground_z_correction_minmax_m': [float(x) for x in (np.min(result[:, 2]-root[:, 2]), np.max(result[:, 2]-root[:, 2]))],
    }


def angular_velocity(quat_wxyz, fps):
    """Central SO(3) finite differences, expressed in world coordinates."""
    frames, bodies = quat_wxyz.shape[:2]
    before = np.maximum(np.arange(frames) - 1, 0)
    after = np.minimum(np.arange(frames) + 1, frames - 1)
    q = quat_wxyz[..., [1, 2, 3, 0]]
    delta = (Rotation.from_quat(q[after].reshape(-1, 4)) *
             Rotation.from_quat(q[before].reshape(-1, 4)).inv()).as_rotvec().reshape(frames, bodies, 3)
    return delta * (fps / (after - before))[:, None, None]


def resample(root, quat_wxyz, joints, input_fps, output_fps=50):
    if len(root) < 3 or input_fps <= 0:
        raise ValueError('A motion requires at least three frames and positive fps')
    old_t = np.arange(len(root)) / input_fps
    new_t = np.arange(int(np.floor(old_t[-1] * output_fps + 1e-6)) + 1) / output_fps
    root_out = np.stack([np.interp(new_t, old_t, x) for x in root.T], axis=1)
    joint_out = np.stack([np.interp(new_t, old_t, x) for x in joints.T], axis=1)
    quat_out = Slerp(old_t, Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]]))(new_t).as_quat()[:, [3, 0, 1, 2]]
    return root_out, quat_out, joint_out


class Kinematics:
    def __init__(self, model_path, joint_names, body_names):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.joint_names = list(joint_names)
        self.body_names = list(body_names)
        self.qpos_ids = [self.model.jnt_qposadr[self.model.joint(n).id] for n in self.joint_names]
        self.body_ids = [self.model.body(n).id for n in self.body_names]

    def forward(self, root, quat, joints):
        pos = np.empty((len(root), len(self.body_names), 3))
        rot = np.empty((len(root), len(self.body_names), 4))
        for t in range(len(root)):
            self.data.qpos[:3] = root[t]
            self.data.qpos[3:7] = quat[t]
            self.data.qpos[self.qpos_ids] = joints[t]
            mujoco.mj_kinematics(self.model, self.data)
            pos[t] = self.data.xpos[self.body_ids]
            rot[t] = self.data.xquat[self.body_ids]
        return pos, rot

    def build(self, root, quat, joints, input_fps, anchor_walk, metadata):
        root, quat, joints = resample(root, quat, joints, input_fps)
        pos, rot = self.forward(root, quat, joints)
        corrected_root, stats = ground_and_anchor(root, pos, rot, self.body_names, 50, anchor_walk)
        pos, rot = self.forward(corrected_root, quat, joints)
        corners = sole_corners(pos, rot, self.body_names)
        # Compute derivatives from the actual float32 positions saved on disk.
        pos = pos.astype(np.float32)
        joints = joints.astype(np.float32)
        result = {
            'fps': np.array([50], dtype=np.int64),
            'joint_pos': joints, 'joint_vel': np.gradient(joints, 0.02, axis=0),
            'body_pos_w': pos, 'body_quat_w': rot.astype(np.float32),
            'body_lin_vel_w': np.gradient(pos, 0.02, axis=0),
            'body_ang_vel_w': angular_velocity(rot, 50).astype(np.float32),
            'joint_names': np.array(self.joint_names), 'body_names': np.array(self.body_names),
            'metadata_json': np.array(json.dumps(metadata, sort_keys=True)),
        }
        stats.update(frames=len(root), seconds=(len(root)-1)/50,
                     sole_minmax_m=[float(corners[..., 2].min()), float(corners[..., 2].max())],
                     min_sole_height_error_m=float(np.max(np.abs(corners[..., 2].min(axis=(1, 2))-0.005))),
                     root_z_minmax_m=[float(pos[:, 0, 2].min()), float(pos[:, 0, 2].max())],
                     joint_velocity_abs_p99_radps=float(np.percentile(np.abs(result['joint_vel']), 99)),
                     joint_velocity_abs_max_radps=float(np.max(np.abs(result['joint_vel']))))
        if not all(np.isfinite(result[k]).all() for k in ['joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w']):
            raise ValueError('Non-finite generated motion')
        return result, stats


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--legacy_layout', choices=['humanoid_soccer_physx'])
    parser.add_argument('--model', type=Path, default=ASSETS/'robots/K1/K1_22dof.xml')
    parser.add_argument('--schema_npz', type=Path, default=next((ASSETS/'motions/K1/amp/walk').glob('*.npz')))
    parser.add_argument('--max_walk_sec', type=float, default=30.0)
    parser.add_argument('--keep_walk_root_xy', action='store_true')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite existing dataset: {args.output_dir}')
    with np.load(args.schema_npz, allow_pickle=False) as schema:
        fk = Kinematics(args.model, schema['joint_names'].tolist(), schema['body_names'].tolist())
    order = [list(K1_JOINT_NAMES).index(n) for n in fk.joint_names]
    sources = [('walk', p) for p in motion_source_files(LAFAN_DIR, 'walk*.csv') if p.name not in EXCLUDE]
    sources += [('kick', p) for d in SOCCER_DIRS for p in motion_source_files(d, '*.npz')]
    if not sources or not any(kind == 'kick' for kind, _ in sources):
        raise ValueError('Missing motion sources')
    args.output_dir.mkdir(parents=True)
    report = {'method': 'named leg-angle transfer + sole grounding + optional walk support translation; no IK',
              'fps': 50, 'time_scale': 1.0, 'model_sha256': sha256(args.model),
              'excluded_sources': sorted(EXCLUDE), 'clips': []}
    for kind, src in sources:
        extra = {}
        if kind == 'walk':
            raw = np.loadtxt(src, delimiter=',')[:int(args.max_walk_sec*30)]
            root, quat, g1 = raw[:, :3], raw[:, [6, 3, 4, 5]], raw[:, 7:]
            names, fps = G1_JOINT_NAMES, 30
        else:
            with np.load(src, allow_pickle=False) as data:
                names = npz_joint_names(data, args.legacy_layout)
                root_id = list(data['body_names']).index('pelvis') if 'body_names' in data else 0
                root, quat, g1 = data['body_pos_w'][:, root_id].copy(), data['body_quat_w'][:, root_id].copy(), data['joint_pos'].copy()
                fps = float(data['fps'].item())
                if 'kick_leg' in data:
                    extra['kick_leg'] = data['kick_leg'].copy()
        metadata = {'source': str(src), 'source_sha256': sha256(src), 'source_fps': fps,
                    'source_joint_names': list(names), 'time_scale': 1.0, 'upper_body': 'K1 neutral',
                    'ground_clearance_m': 0.005, 'walk_support_translation': kind == 'walk' and not args.keep_walk_root_xy,
                    'retarget_method': report['method'], 'model_sha256': report['model_sha256']}
        joints = retarget_g1_to_k1(g1, names)[:, order]
        result, stats = fk.build(root, quat, joints, fps, metadata['walk_support_translation'], metadata)
        result.update(extra)
        clipped = {}
        for name, entry in G1_TO_K1_MAP.items():
            if entry:
                raw_values = g1[:, list(names).index(entry[0])] * entry[1]
                lo, hi = K1_JOINT_LIMITS[name]
                fraction = float(np.mean((raw_values < lo) | (raw_values > hi)))
                if fraction:
                    clipped[name] = fraction
        stats.update(source=str(src), output=f'{kind}/{src.stem}.npz', clipped_fraction_by_joint=clipped)
        dst = args.output_dir / stats['output']
        dst.parent.mkdir(exist_ok=True)
        np.savez_compressed(dst, **result)
        report['clips'].append(stats)
        print(f'{kind}/{src.stem}: {stats["frames"]} frames, support slip {stats["support_speed_before_mps"]:.3f} -> {stats["support_speed_after_mps"]:.3f} m/s', flush=True)
    report['total_frames'] = sum(x['frames'] for x in report['clips'])
    (args.output_dir/'build_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Wrote {len(report["clips"])} clips / {report["total_frames"]} frames to {args.output_dir}')


if __name__ == '__main__':
    main()
