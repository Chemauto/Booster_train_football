"""Check AMP motion schema and K1 URDF limits before training.

A pass establishes numerical/schema validity, not dynamic feasibility. Contact
and soft-limit diagnostics are warnings: they cannot replace a physics rollout.
Exit code 0 = no hard failures, 1 = hard failures (including unreadable input).
"""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

DEFAULT_URDF = Path(__file__).resolve().parents[2] / 'booster_assets/robots/K1/K1_22dof.urdf'
FEET = ('left_ankle_roll_link', 'right_ankle_roll_link')
FPS = 50
MIN_CLIP_S = 1.0
POSITION_TOL = 1e-6
VELOCITY_TOL = 1e-4
QUATERNION_TOL = 1e-4
ARRAY_WIDTHS = {'body_pos_w': 3, 'body_quat_w': 4, 'body_lin_vel_w': 3, 'body_ang_vel_w': 3}
REQUIRED = ('fps', 'joint_names', 'body_names', 'joint_pos', 'joint_vel', *ARRAY_WIDTHS)


def load_urdf(path):
    """Read directional position limits and exact velocities for every motor."""
    root = ET.parse(path).getroot()
    limits = {}
    expected_bodies = {'trunk'}
    for joint in root.findall('joint'):
        if joint.get('type') not in ('revolute', 'prismatic'):
            continue
        limit = joint.find('limit')
        values = tuple(float(limit.get(key)) for key in ('lower', 'upper', 'velocity'))
        if not np.isfinite(values).all() or values[0] >= values[1] or values[2] <= 0:
            raise ValueError(f"invalid limits for {joint.get('name')}")
        limits[joint.get('name')] = values
        expected_bodies.add(joint.find('child').get('link'))
    if not limits:
        raise ValueError('URDF has no bounded movable joints')
    return limits, expected_bodies, {link.get('name') for link in root.findall('link')}


def read_names(array, field, issues):
    if array.ndim != 1 or array.dtype.kind not in ('U', 'S'):
        issues.append(f'HARD {field}: expected one-dimensional string array')
        return None
    names = array.astype(str).tolist()
    if len(names) != len(set(names)):
        issues.append(f'HARD {field}: duplicate names')
    if any(not name for name in names):
        issues.append(f'HARD {field}: empty name')
    return names


def check_clip(path, limits, expected_bodies, all_bodies, reference_names=None):
    issues = []
    result = {'frames': None, 'seconds': None, 'issues': issues}
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = [key for key in REQUIRED if key not in archive.files]
            issues.extend(f'HARD missing field {key}' for key in missing)
            if missing:
                return result
            data = {key: archive[key] for key in REQUIRED}
    except Exception as exc:
        issues.append(f'HARD cannot read npz: {type(exc).__name__}: {exc}')
        return result

    names = read_names(data['joint_names'], 'joint_names', issues)
    bodies = read_names(data['body_names'], 'body_names', issues)
    if names is not None:
        if set(names) != set(limits):
            issues.append(f'HARD joint_names: expected URDF joint set; missing={sorted(set(limits)-set(names))}, '
                          f'unknown={sorted(set(names)-set(limits))}')
        if reference_names is not None and names != reference_names:
            issues.append('HARD joint order differs from reference schema')
    if bodies is not None:
        if expected_bodies - set(bodies):
            issues.append(f'HARD body_names: missing URDF motion bodies {sorted(expected_bodies-set(bodies))}')
        if set(bodies) - all_bodies:
            issues.append(f'HARD body_names: unknown URDF bodies {sorted(set(bodies)-all_bodies)}')

    fps = data['fps']
    if fps.size != 1 or fps.dtype.kind not in 'fiu' or not np.isfinite(fps).all() or float(fps.flat[0]) != FPS:
        issues.append(f'HARD fps: expected exactly {FPS}, got {fps.tolist()}')

    positions = data['joint_pos']
    if positions.ndim != 2:
        issues.append(f'HARD joint_pos: shape {positions.shape}, expected (frames, joints)')
    else:
        frames = positions.shape[0]
        result.update(frames=frames, seconds=round(frames / FPS, 4))
        if frames < MIN_CLIP_S * FPS:
            issues.append(f'HARD too short: {frames} frames ({frames / FPS:.2f}s)')
        if names is not None and bodies is not None:
            expected = {'joint_pos': (frames, len(names)), 'joint_vel': (frames, len(names))}
            expected.update({key: (frames, len(bodies), width) for key, width in ARRAY_WIDTHS.items()})
            for key, shape in expected.items():
                if data[key].shape != shape:
                    issues.append(f'HARD {key}: shape {data[key].shape}, expected {shape}')

    for key in ('joint_pos', 'joint_vel', *ARRAY_WIDTHS):
        value = data[key]
        if value.dtype.kind not in 'fiu':
            issues.append(f'HARD {key}: expected real numeric array')
        elif not np.isfinite(value).all():
            issues.append(f'HARD {key}: non-finite values')
    if issues:
        return result

    norm_error = float(np.max(np.abs(np.linalg.norm(data['body_quat_w'], axis=-1) - 1)))
    result['quaternion_norm_max_error'] = norm_error
    if norm_error > QUATERNION_TOL:
        issues.append(f'HARD quaternion norm: max deviation {norm_error:.6g} exceeds {QUATERNION_TOL}')
        return result

    joint_diagnostics = {}
    soft_penalty = np.zeros(result['frames'])
    for index, name in enumerate(names):
        lower, upper, velocity = limits[name]
        q = data['joint_pos'][:, index].astype(np.float64)
        speed = np.abs(data['joint_vel'][:, index].astype(np.float64))
        bad_pos = (q < lower - POSITION_TOL) | (q > upper + POSITION_TOL)
        bad_vel = speed > velocity + VELOCITY_TOL
        if bad_pos.any():
            issues.append(f'HARD {name}: position outside [{lower}, {upper}] rad on {bad_pos.mean():.3%} '
                          f'of frames (range [{q.min():.6g}, {q.max():.6g}])')
        if bad_vel.any():
            issues.append(f'HARD {name}: velocity exceeds {velocity} rad/s on {bad_vel.mean():.3%} '
                          f'of frames (max {speed.max():.6g})')
        midpoint, half_span = (lower + upper) / 2, (upper - lower) * .9 / 2
        soft_distance = np.maximum(midpoint - half_span - q, 0) + np.maximum(q - midpoint - half_span, 0)
        soft_penalty += soft_distance
        if (soft_distance > POSITION_TOL).any():
            issues.append(f'SOFT {name}: outside 0.9 soft position limits on '
                          f'{(soft_distance > POSITION_TOL).mean():.1%} of frames')
        joint_diagnostics[name] = {'position_min': float(q.min()), 'position_max': float(q.max()),
                                   'velocity_peak': float(speed.max()), 'position_violation_fraction': float(bad_pos.mean()),
                                   'velocity_violation_fraction': float(bad_vel.mean())}
    result['joint_limits'] = joint_diagnostics
    result['soft_position_limit_penalty_mean_rad'] = float(soft_penalty.mean())
    result['soft_position_limit_penalty_max_rad'] = float(soft_penalty.max())

    # This geometric proxy uses link origins, not sole contact or a computed CoM.
    # Flight can be intentional; dynamic balance cannot be inferred from this gate.
    bp = data['body_pos_w']
    trunk, left, right = [bodies.index(name) for name in ('trunk', *FEET)]
    segment = bp[:, right, :2] - bp[:, left, :2]
    offset = bp[:, trunk, :2] - (bp[:, left, :2] + bp[:, right, :2]) / 2
    span = np.linalg.norm(segment, axis=-1)
    cross = segment[:, 0] * offset[:, 1] - segment[:, 1] * offset[:, 0]
    distance = np.where(span > 1e-6, np.abs(cross) / np.maximum(span, 1e-6), np.linalg.norm(offset, axis=-1))
    from build_amp_motion_npz import sole_corners
    sole_height = sole_corners(bp, data['body_quat_w'], bodies)[..., 2].min(axis=2)
    off_support = float((distance > .15).mean())
    both_raised = float((sole_height.min(axis=1) > .05).mean())
    result['support_diagnostics'] = {'trunk_off_feet_line_fraction': off_support,
                                     'both_soles_above_5cm_fraction': both_raised,
                                     'sole_height_min_m': float(sole_height.min()),
                                     'note': 'Training sole corners; geometric proxies only, no contact forces or CoM dynamics checked.'}
    if off_support > .02:
        issues.append(f'SOFT trunk off feet line >15cm on {off_support:.1%} of frames (geometric proxy)')
    if both_raised > .02:
        issues.append(f'SOFT both soles above 5cm on {both_raised:.1%} of frames (possible flight)')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--report', default='logs/dataset_gate_report.json')
    parser.add_argument('--ref_joint_names', default='', help='optional npz whose joint order must match exactly')
    parser.add_argument('--urdf', default=str(DEFAULT_URDF), help='URDF supplying expected joints and motor limits')
    args = parser.parse_args()
    files = sorted(Path(args.data_dir).rglob('*.npz'))
    report = {'data_dir': args.data_dir, 'urdf': args.urdf, 'clips': len(files), 'hard': [], 'soft': [], 'per_clip': {},
              'tolerances': {'position_rad': POSITION_TOL, 'velocity_rad_s': VELOCITY_TOL,
                             'quaternion_norm': QUATERNION_TOL},
              'scope': 'Schema and numerical limits only; passing does not establish dynamic feasibility.'}
    if not files:
        report['hard'].append('HARD no npz found')
    try:
        limits, expected_bodies, all_bodies = load_urdf(args.urdf)
        reference_names = None
        if args.ref_joint_names:
            with np.load(args.ref_joint_names, allow_pickle=False) as reference:
                reference_issues = []
                reference_names = read_names(reference['joint_names'], 'joint_names', reference_issues)
                if reference_issues or set(reference_names) != set(limits):
                    raise ValueError(f'invalid reference joint names: {reference_issues}')
    except Exception as exc:
        report['hard'].append(f'HARD cannot load validation schema: {type(exc).__name__}: {exc}')
    else:
        for path in files:
            relative = str(path.relative_to(args.data_dir))
            result = check_clip(path, limits, expected_bodies, all_bodies, reference_names)
            report['per_clip'][relative] = result
            for severity in ('hard', 'soft'):
                report[severity].extend(f'{relative}: {issue}' for issue in result['issues'] if issue.startswith(severity.upper()))

    durations = [clip['seconds'] for clip in report['per_clip'].values() if clip['seconds'] is not None]
    report['total_seconds'] = round(sum(durations), 4)
    report['clips_under_1s'] = sum(duration < MIN_CLIP_S for duration in durations)
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f"clips={report['clips']} total={report['total_seconds']}s")
    for severity, count in (('hard', 20), ('soft', 10)):
        print(f"{severity.upper()} findings: {len(report[severity])}")
        for message in report[severity][:count]:
            print('  ' + message)
    print('A numerical pass does not prove dynamic feasibility; inspect motion and run physics evaluations.')
    return 1 if report['hard'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
