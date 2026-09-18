"""Retarget ACCAD (AMASS ``*_stageii.npz``) mocap onto K1 using GMR's IK.

Why markers instead of SMPL-X
-----------------------------
The ACCAD stage-II files carry the raw 41-marker Vicon trajectories alongside the
SMPL-X parameters. GMR's own ``smplx`` path needs both the ``smplx`` package and
the licence-gated SMPL-X body model, neither of which exists on this machine;
the markers are also the actual measurement rather than a model fit, and they
place every joint we care about (knee, ankle, heel, toe, shoulder, elbow, wrist)
directly.

Frame convention
----------------
Every K1 body shares a single rest-frame orientation (forward ``+X``, left
``+Y``, up ``+Z``; verified: no ``quat``/``euler`` attribute anywhere in
``K1_serial.xml``), so for each body we map two anatomical axes measured from
markers onto the two axes that define that body locally, and hand the resulting
rotation to the IK unchanged. ``marker_to_k1.json`` therefore carries identity
``rot_offset`` for every body, unlike ``smplx_to_k1.json`` whose uniform 120 deg
offset converts SMPL-X's Y-up convention into the robot's Z-up one.

Downstream the joint trajectory is written in the *training* model's joint order
(``booster_assets/robots/K1/K1_22dof.xml``) and handed to the shared
``Kinematics.build`` so grounding, 50 fps resampling and velocity computation are
identical to the rest of the dataset.

Example:
  python scripts/accad_markers_to_k1.py \
      --clip /data/rl_robot/accad/ACCAD/Female1Walking_c3d/B3_-_walk1_stageii.npz \
      --out /tmp/marker_probe/B3_walk1.npz --no_arms
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

GMR_ROOT = Path('/data/rl_robot/GMR')
if str(GMR_ROOT) not in sys.path:
    sys.path.insert(0, str(GMR_ROOT))

import general_motion_retargeting.params as gmr_params  # noqa: E402
from general_motion_retargeting import GeneralMotionRetargeting as GMR  # noqa: E402

from build_amp_motion_npz import Kinematics, sha256  # noqa: E402
from convert_paper_data import K1_VEL_LIMITS  # noqa: E402
from g1_to_k1_csv import K1_JOINT_LIMITS  # noqa: E402

MARKER_CONFIG = GMR_ROOT / 'general_motion_retargeting/ik_configs/marker_to_k1.json'
TRAIN_MODEL = Path('/data/rl_robot/BoosterRobotics/booster_assets/robots/K1/K1_22dof.xml')
SCHEMA_NPZ = Path('/data/rl_robot/BoosterRobotics/booster_assets/motions/K1/amp/walk/walk1_subject1.npz')

# GMR's K1 model (used for IK) -> the training model (used for the dataset).
JOINT_MAP = {
    'AAHead_yaw': 'aahead_yaw_joint',
    'Head_pitch': 'aahead_pitch_joint',
    'ALeft_Shoulder_Pitch': 'aaleft_shoulder_pitch_joint',
    'Left_Shoulder_Roll': 'left_shoulder_roll_joint',
    'Left_Elbow_Pitch': 'left_elbow_pitch_joint',
    'Left_Elbow_Yaw': 'left_elbow_yaw_joint',
    'ARight_Shoulder_Pitch': 'aaright_shoulder_pitch_joint',
    'Right_Shoulder_Roll': 'right_shoulder_roll_joint',
    'Right_Elbow_Pitch': 'right_elbow_pitch_joint',
    'Right_Elbow_Yaw': 'right_elbow_yaw_joint',
    'Left_Hip_Pitch': 'left_hip_pitch_joint',
    'Left_Hip_Roll': 'left_hip_roll_joint',
    'Left_Hip_Yaw': 'left_hip_yaw_joint',
    'Left_Knee_Pitch': 'left_knee_pitch_joint',
    'Left_Ankle_Pitch': 'left_ankle_pitch_joint',
    'Left_Ankle_Roll': 'left_ankle_roll_joint',
    'Right_Hip_Pitch': 'right_hip_pitch_joint',
    'Right_Hip_Roll': 'right_hip_roll_joint',
    'Right_Hip_Yaw': 'right_hip_yaw_joint',
    'Right_Knee_Pitch': 'right_knee_pitch_joint',
    'Right_Ankle_Pitch': 'right_ankle_pitch_joint',
    'Right_Ankle_Roll': 'right_ankle_roll_joint',
}

# (e1, e2) local axes per robot body, both in the shared K1 rest frame. e1 maps
# the one unambiguous anatomical direction of the body; e2 resolves the twist.
BASIS = {
    'Trunk': ((0, 1, 0), (0, 0, 1)),           # left, up
    'Left_Hip_Yaw': ((0, 0, -1), (0, 1, 0)),   # distal (hip->knee), flexion axis
    'Left_Shank': ((0, 0, -1), (0, 1, 0)),
    'Right_Hip_Yaw': ((0, 0, -1), (0, 1, 0)),
    'Right_Shank': ((0, 0, -1), (0, 1, 0)),
    'left_foot_link': ((1, 0, 0), (0, 0, 1)),  # forward, up
    'right_foot_link': ((1, 0, 0), (0, 0, 1)),
    'Left_Arm_3': ((0, 1, 0), (0, 0, 1)),      # distal (elbow->wrist), bend axis
    'left_hand_link': ((0, 1, 0), (0, 0, 1)),
    'Right_Arm_3': ((0, -1, 0), (0, 0, 1)),
    'right_hand_link': ((0, -1, 0), (0, 0, 1)),
}

# K1's hip-to-ankle span, used to pick the human->robot scale. Read from the
# model rather than hardcoded: note this is hip-*pitch* to ankle, not the
# hip-*yaw* link (0.3625), which is 17% shorter and silently makes every foot
# target unreachable.
K1_LEG_LENGTH = None
K1_LEG_JOINTS = ('Left_Hip_Pitch', 'Left_Ankle_Cross')

# Markers the retargeter actually reads, and the only ones whose loss is fatal.
# Shoulder/elbow/hand/finger markers and the metatarsals are deliberately absent:
# the arms are driven by wrist position alone, and LMT5 is missing in ~60% of
# ACCAD frames. Listing unused markers here rejects clips that would retarget
# perfectly well.
REQUIRED_MARKERS = (
    'LFWT', 'RFWT', 'LBWT', 'RBWT',
    'LKNE', 'LANK', 'LHEE', 'LTOE', 'RKNE', 'RANK', 'RHEE', 'RTOE',
    'LIWR', 'LOWR', 'RIWR', 'ROWR',
)


def unit(v, fallback=None):
    n = np.linalg.norm(v)
    if n < 1e-9:
        if fallback is None:
            raise ValueError('degenerate direction')
        return np.asarray(fallback, dtype=float)
    return np.asarray(v, dtype=float) / n


def orthonormal_basis(e1, e2):
    e1 = unit(e1)
    e2 = np.asarray(e2, dtype=float)
    e2 = unit(e2 - e1 * np.dot(e2, e1))
    return np.column_stack([e1, e2, np.cross(e1, e2)])


def orientation_from_axes(d1, d2, e1, e2, fallback=None):
    """Rotation taking the local basis (e1, e2) onto the measured (d1, d2)."""
    try:
        d1 = unit(d1)
    except ValueError:
        return fallback
    d2 = np.asarray(d2, dtype=float)
    d2 = d2 - d1 * np.dot(d2, d1)
    if np.linalg.norm(d2) < 1e-9:
        return fallback
    d2 = unit(d2)
    measured = np.column_stack([d1, d2, np.cross(d1, d2)])
    return Rotation.from_matrix(measured @ orthonormal_basis(e1, e2).T)


# Marker pairs whose separation is near-constant in a real skeleton, used to
# detect teleporting markers. Only genuinely rigid pairs belong here: a pair that
# straddles a joint (pelvis-to-thigh, C7-to-shoulder, clavicle-to-shoulder) has a
# distance that legitimately changes with the pose, so including it flags healthy
# markers as corrupt - measured on the roundhouse kicks, where a 90 deg hip
# flexion made (LFWT, LTHI) swing far enough to condemn the pelvis markers in
# 25% of frames.
BONES = [
    # pelvis: the four waist markers form a rigid cluster
    ('LFWT', 'RFWT'), ('LBWT', 'RBWT'), ('LFWT', 'LBWT'), ('RFWT', 'RBWT'),
    # legs
    ('LTHI', 'LKNE'), ('LKNE', 'LSHN'), ('LSHN', 'LANK'),
    ('LANK', 'LHEE'), ('LANK', 'LTOE'), ('LHEE', 'LTOE'), ('LTOE', 'LMT5'),
    ('RTHI', 'RKNE'), ('RKNE', 'RSHN'), ('RSHN', 'RANK'),
    ('RANK', 'RHEE'), ('RANK', 'RTOE'), ('RHEE', 'RTOE'), ('RTOE', 'RMT5'),
    # arms: shoulder-to-upper-arm and everything distal of the elbow
    ('LSHO', 'LUPA'), ('LSHO', 'LELB'), ('LELB', 'LFRM'),
    ('LELB', 'LIWR'), ('LELB', 'LOWR'), ('LIWR', 'LOWR'), ('LIWR', 'LFIN'),
    ('RSHO', 'RUPA'), ('RSHO', 'RELB'), ('RELB', 'RFRM'),
    ('RELB', 'RIWR'), ('RELB', 'ROWR'), ('RIWR', 'ROWR'), ('RIWR', 'RFIN'),
    # head
    ('LFHD', 'RFHD'), ('LFHD', 'LBHD'), ('RFHD', 'RBHD'), ('LBHD', 'RBHD'),
]


def repair_markers(markers, labels, fps, max_speed=30.0, max_missing=0.2,
                   rounds=3, verbose=False):
    """Flag and interpolate unusable marker samples.

    Three failure modes show up in ACCAD: markers recorded as exact zeros
    (unlabelled, clamped at the floor, or occluded by a prop - several clips have
    the subject carrying a box), single-frame teleports, and samples that break a
    constant bone length. All three are treated as missing and linearly
    interpolated in time, and the tests are repeated because filling a frame can
    expose the next outlier.

    ``max_speed`` is deliberately loose: it exists to catch teleports, which show
    up as metres in a single 1/120 s frame, not to second-guess fast motion. A
    martial-arts kick legitimately swings a foot at well over 5 m/s, so a tight
    threshold flags real kicks as corrupt.

    Markers missing more than ``max_missing`` of the clip are declared unusable
    and excluded from every bone test, along with every frame that was
    originally missing. Without that, a marker like LMT5 (absent in ~60% of
    ACCAD frames) is interpolated into a piecewise-linear fiction, which then
    breaks the bone-length test against its healthy neighbours and cascades
    until the whole foot is flagged.
    """
    out = markers.copy()
    index = {name: i for i, name in enumerate(labels)}
    frames = np.arange(len(out))
    step = max(1.0 / fps, 1e-6)

    missing0 = np.all(np.abs(out) < 1e-9, axis=2)
    fraction = missing0.mean(axis=0)
    unusable = {labels[j] for j in range(len(labels)) if fraction[j] > max_missing}
    if verbose and unusable:
        print('[repair] unusable (>%.0f%% missing): %s'
              % (100 * max_missing, ', '.join(sorted(unusable))))
    bones = [(index[a], index[b], a, b) for a, b in BONES
             if a in index and b in index and a not in unusable and b not in unusable]

    flagged = missing0.copy()
    for _ in range(rounds):
        # Teleports: a marker cannot move faster than max_speed.
        for j in range(out.shape[1]):
            if labels[j] in unusable:
                continue
            velocity = np.linalg.norm(np.gradient(out[:, j], axis=0), axis=1) / step
            flagged[:, j] |= velocity > max_speed
        # Bone-length violations, judged only on samples we actually trust.
        for a, b, _, _ in bones:
            trusted = ~flagged[:, a] & ~flagged[:, b] & ~missing0[:, a] & ~missing0[:, b]
            if trusted.sum() < 10:
                continue
            length = np.linalg.norm(out[:, a] - out[:, b], axis=1)
            median = np.median(length[trusted])
            mad = np.median(np.abs(length[trusted] - median))
            tol = max(0.025, 6.0 * mad)
            bad = (np.abs(length - median) > tol) & ~missing0[:, a] & ~missing0[:, b]
            flagged[bad, a] = True
            flagged[bad, b] = True
        # Interpolate everything flagged.
        for j in range(out.shape[1]):
            bad = flagged[:, j]
            if not bad.any():
                continue
            good = ~bad
            if good.sum() < 2:
                raise ValueError(f'marker {labels[j]} has no usable samples')
            for axis in range(3):
                out[bad, j, axis] = np.interp(frames[bad], frames[good], out[good, j, axis])
    if verbose:
        worst = np.argsort(-flagged.sum(axis=0))[:8]
        print('[repair] most-repaired markers: '
              + ', '.join(f'{labels[j]}={flagged[:, j].sum()}/{len(out)}' for j in worst
                          if flagged[:, j].any()))
    return out, flagged


def measured_height(markers, labels, up=np.array([0.0, 0.0, 1.0])):
    """Standing height from the head markers plus a skull cap allowance."""
    index = {name: i for i, name in enumerate(labels)}
    heads = [index[n] for n in ('LFHD', 'RFHD', 'LBHD', 'RBHD') if n in index]
    if not heads:
        return None
    top = markers[:, heads, 2].max(axis=1)
    ground = markers[:, [index[n] for n in ('LTOE', 'RTOE', 'LHEE', 'RHEE')], 2].min()
    return float(np.median(top - ground) + 0.13)


def load_markers(path, verbose=False):
    """Return (positions[N,41,3] metres Z-up, labels, fps, source metadata)."""
    with np.load(path, allow_pickle=True) as data:
        markers = np.asarray(data['markers'], dtype=float)
        labels = [str(x) for x in data['labels']]
        fps = float(data['mocap_frame_rate'].item())
        betas = np.asarray(data['betas'], dtype=float).reshape(-1)
    if markers.shape[0] < 3:
        raise ValueError(f'{path}: too few frames')
    if np.isnan(markers).any():
        raise ValueError(f'{path}: NaN marker samples')
    missing = [n for n in REQUIRED_MARKERS if n not in labels]
    if missing:
        raise ValueError(f'{path}: marker set lacks {missing} (labels: {sorted(labels)})')
    repaired, flagged = repair_markers(markers, labels, fps, verbose=verbose)
    unusable = [n for n in REQUIRED_MARKERS if n in set(labels)
                and flagged[:, labels.index(n)].mean() > 0.2]
    if unusable:
        raise ValueError(f'{path}: markers needed for retargeting are unusable: {unusable}')
    meta = {'betas0': float(betas[0]), 'height_from_betas': 1.66 + 0.1 * float(betas[0]),
            'height_from_markers': measured_height(repaired, labels),
            'repaired_fraction': float(flagged.mean()),
            'unusable_markers': sorted({labels[j] for j in range(len(labels))
                                        if flagged[:, j].mean() > 0.2})}
    return repaired, labels, fps, meta


def marker_geometry(markers, labels):
    """Anatomical frames and joint centres for every frame, from markers alone."""
    m = {name: markers[:, i] for i, name in enumerate(labels)}

    def g(*names):
        return np.mean([m[x] for x in names], axis=0)

    def axial(raw, fallback):
        return np.stack([unit(v, fallback=fallback) for v in raw])

    geom = {}
    geom['pelvis_origin'] = g('LFWT', 'RFWT', 'LBWT', 'RBWT')
    # Waist markers: front/back and left/right pairs define the pelvis frame.
    width = np.linalg.norm(g('LFWT', 'LBWT') - g('RFWT', 'RBWT'), axis=1)
    left = axial(g('LFWT', 'LBWT') - g('RFWT', 'RBWT'), (0, 1, 0))
    forward = axial(g('LFWT', 'RFWT') - g('LBWT', 'RBWT'), (1, 0, 0))
    up = np.stack([unit(np.cross(f, l), fallback=(0, 0, 1)) for f, l in zip(forward, left)])
    geom['left'] = np.stack([unit(np.cross(u, f), fallback=(0, 1, 0)) for u, f in zip(up, forward)])
    geom['up'] = up

    # Hip joint centres sit inboard and below the ASIS markers.
    hip_offset, drop = 0.30 * width[:, None], 0.15 * width[:, None]
    geom['hip'] = {
        'left': g('LFWT', 'LBWT') - geom['left'] * hip_offset - up * drop,
        'right': g('RFWT', 'RBWT') + geom['left'] * hip_offset - up * drop,
    }

    geom['knee'] = {s: g(f'{s[0].upper()}KNE') for s in ('left', 'right')}
    geom['ankle'] = {s: g(f'{s[0].upper()}ANK') for s in ('left', 'right')}
    geom['heel'] = {s: g(f'{s[0].upper()}HEE') for s in ('left', 'right')}
    geom['toe'] = {s: g(f'{s[0].upper()}TOE') for s in ('left', 'right')}
    # The wrist markers are the arms' only input, and a fast kick makes them
    # noisy; the IK chases that noise and the arm joints spike past their motor
    # limits (measured: 32 rad/s at the shoulder). Smoothing the target, not the
    # resulting joints, keeps the leg pipeline untouched.
    geom['arm_wrist'] = {
        s: savgol_filter(g(f'{s[0].upper()}IWR', f'{s[0].upper()}OWR'), window_length=15,
                         polyorder=2, axis=0)
        for s in ('left', 'right')}

    # Knee flexion axis. Both knees are hinges about the body's left axis, which
    # also fixes the sign.
    geom['knee_axis'] = {}
    for side in ('left', 'right'):
        thigh = geom['knee'][side] - geom['hip'][side]
        shank = geom['ankle'][side] - geom['knee'][side]
        n = np.cross(thigh, shank)
        # A straight leg spans no plane; the foot's long axis lies in the same
        # sagittal plane, so it stands in.
        fallback = np.cross(thigh, geom['toe'][side] - geom['heel'][side])
        weak = (np.linalg.norm(n, axis=1, keepdims=True) <
                0.05 * np.maximum(np.linalg.norm(thigh, axis=1, keepdims=True) *
                                  np.linalg.norm(shank, axis=1, keepdims=True), 1e-9))
        n = np.where(weak, fallback, n)
        sign = np.where(np.einsum('ij,ij->i', n, geom['left']) < 0, -1.0, 1.0)
        geom['knee_axis'][side] = (
            np.stack([unit(v, fallback=l) for v, l in zip(n, geom['left'])]) * sign[:, None])

    # Foot axes. Heel and toe markers sit at the same height as the ankle when
    # the foot is flat, so heel->toe lies in the sole plane and carries the
    # foot's pitch: measured over ACCAD stance frames its pitch is ~0.4 deg.
    #
    # For "up" there is no usable third axis: the 5th-metatarsal markers are
    # missing in ~60% of ACCAD frames, and the ankle sits mostly *medially* of
    # the heel->toe line (only ~1.7 cm of its offset is vertical), so its
    # perpendicular component is dominated by that sideways displacement. We
    # therefore take up = world-up projected off the forward axis, i.e. assume
    # no inversion/eversion. Consequence: the retargeted ankle roll stays near
    # zero.
    geom['foot_fwd'] = {}
    geom['foot_up'] = {}
    for side in ('left', 'right'):
        long_axis = axial(geom['toe'][side] - geom['heel'][side], (1, 0, 0))
        ups = []
        for f in long_axis:
            ups.append(unit(np.array([0.0, 0.0, 1.0]) - f * f[2], fallback=(0, 0, 1)))
        geom['foot_fwd'][side] = long_axis
        geom['foot_up'][side] = np.stack(ups)

    geom['frames'] = len(markers)
    return geom


def leg_length(geom):
    """Median hip-to-ankle span in the source motion, in metres."""
    return float(np.median(np.concatenate([
        np.linalg.norm(geom['ankle'][s] - geom['hip'][s], axis=1) for s in ('left', 'right')])))


def build_frames(geom, human_keys):
    """Yield GMR ``human_data`` dicts, restricted to ``human_keys``.

    GMR's ``offset_human_data`` requires every key in the dict to appear in the
    IK config's tables, so the task set and the emitted keys must stay in step.

    A direction is undefined on some frames (a leg momentarily straight, an arm
    collapsed); rather than dropping the key, the last defined orientation is
    carried forward, which keeps the key set constant and the targets smooth.
    """
    previous = {}

    def orient(key, d1, d2, e1, e2):
        result = orientation_from_axes(d1, d2, e1, e2)
        if result is None:
            result = previous.get(key, Rotation.identity())
        quat = result.as_quat(scalar_first=True)
        previous[key] = result
        return quat

    for t in range(geom['frames']):
        data = {'pelvis': (geom['pelvis_origin'][t], orient(
            'pelvis', geom['left'][t], geom['up'][t], *BASIS['Trunk']))}
        for side in ('left', 'right'):
            key = side.capitalize()
            hip, knee, ankle = geom['hip'][side][t], geom['knee'][side][t], geom['ankle'][side][t]
            data[f'{side}_hip'] = (hip, orient(
                f'{side}_hip', knee - hip, geom['knee_axis'][side][t],
                *BASIS[f'{key}_Hip_Yaw']))
            data[f'{side}_knee'] = (knee, orient(
                f'{side}_knee', ankle - knee, geom['knee_axis'][side][t],
                *BASIS[f'{key}_Shank']))
            data[f'{side}_foot'] = (ankle, orient(
                f'{side}_foot', geom['foot_fwd'][side][t], geom['foot_up'][side][t],
                *BASIS[f'{side}_foot_link']))
            data[f'{side}_wrist'] = (geom['arm_wrist'][side][t], np.array([1.0, 0.0, 0.0, 0.0]))
        yield {k: v for k, v in data.items() if k in human_keys}


def k1_leg_length():
    """K1's hip-pitch to ankle span, measured on the model the IK solves against."""
    global K1_LEG_LENGTH
    if K1_LEG_LENGTH is None:
        model = mujoco.MjModel.from_xml_path(str(gmr_params.ROBOT_XML_DICT['booster_k1']))
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        hip, ankle = (model.body(n).id for n in K1_LEG_JOINTS)
        K1_LEG_LENGTH = float(data.xpos[hip][2] - data.xpos[ankle][2])
    return K1_LEG_LENGTH


def human_height_for_legs(human_leg, leg_ratio, scale_table_leg=0.6, assumption=1.8):
    """Height to hand GMR so the scaled human leg spans ``leg_ratio`` x K1's leg.

    The IK config scales every target position by ``scale_table * height /
    assumption``. GMR's own g1 configuration lands the scaled human leg near the
    robot's, so the robot walks with a slight crouch instead of straining at full
    extension; this makes that ratio explicit and measurable rather than
    something inherited from a betas heuristic.
    """
    return assumption * (leg_ratio * k1_leg_length()) / (scale_table_leg * human_leg)


# Which robot bodies each task set constrains.
TASK_SETS = {
    # Foot-driven: each leg's 6 DOF follow the foot's 6 DOF targets. Adding the
    # thigh/shank orientations on top over-constrains the leg, and the solver
    # then settles on a compromise (measured: 8.8 cm foot position error, the
    # knee held near straight, hip yaw spinning in the resulting null space).
    'foot': ('Trunk', 'left_foot_link', 'right_foot_link'),
    # Foot-driven legs plus the arm orientations (see the arm mapping notes).
    'foot_arms': ('Trunk', 'left_foot_link', 'right_foot_link',
                  'left_hand_link', 'right_hand_link'),
    # Adds every limb orientation; kept for comparison.
    'full': tuple(BASIS),
}

# Nominal arm pose, byte-identical to amp_paper's, which holds the arms still.
# Driving the arms from markers is a net loss: they are excluded from the AMP
# observation (AMP_EXCLUDED_JOINTS), so swing buys the discriminator nothing,
# while K1's arm is ~15% shorter than a human's, so the wrist target is always
# out of reach and the shoulder roll swings past the soft-limit band
# [-1.478, 1.465] on 72% of frames. The soft-limit reward then charges the
# expert poses -0.85/step (AMP's whole per-step scale is 0.6, survival 0.06),
# i.e. the task reward fights the reference the AMP term pulls toward.
NOMINAL_ARMS = {
    'aaleft_shoulder_pitch_joint': 0.0, 'left_shoulder_roll_joint': -1.300,
    'left_elbow_pitch_joint': 0.0, 'left_elbow_yaw_joint': 0.0,
    'aaright_shoulder_pitch_joint': 0.0, 'right_shoulder_roll_joint': 1.300,
    'right_elbow_pitch_joint': 0.0, 'right_elbow_yaw_joint': 0.0,
}

# Arms: target the wrist POSITION only, with the rotation left free. K1's arm
# cannot bend at the elbow at all - Left_Elbow_Pitch's axis (0,1,0) is the arm's
# own axis and its child offset is also (0,1,0), so that joint is a pure twist -
# which leaves the shoulder's two DOF to place the wrist on a sphere. A wrist
# position pins the arm's direction exactly; adding an orientation target
# over-constrains it (measured: ik error 0.03 -> 1.79).
ARM_WRISTS = {'left_hand_link': 'left_wrist', 'right_hand_link': 'right_wrist'}
# The human arm is ~15% longer than K1's, so the scaled wrist target is always a
# few cm out of reach. At weight 100 that fight costs the legs 3x their ankle
# accuracy (0.009 -> 0.027 m) and adds 0.5 rad of knee crouch, while the arm
# direction is achieved just as well at 20.
ARM_POS_WEIGHT = 20

# The root pose must come from the human's scaled pelvis, so weight it to beat
# the feet. With the stock weights (Trunk rot 5-10 against feet 50-100) the body
# never turns to face the walk; the legs then have to absorb a yaw they cannot
# reach, and the IK lands in a mirrored branch.
ROOT_WEIGHTS = {'Trunk': {'ik_match_table1': (100, 100), 'ik_match_table2': (100, 100)}}


def build_config(tasks):
    """Trim ``marker_to_k1.json`` down to ``tasks`` and return the config path."""
    config = json.loads(MARKER_CONFIG.read_text())
    for table in ('ik_match_table1', 'ik_match_table2'):
        for body in list(config[table]):
            if body not in tasks:
                config[table].pop(body)
        for body, weights in ROOT_WEIGHTS.items():
            if body in config[table]:
                config[table][body][1], config[table][body][2] = weights[table]
        for body, human in ARM_WRISTS.items():
            if body in config[table]:
                config[table][body] = [human, ARM_POS_WEIGHT, 0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
                config['human_scale_table'][human] = 0.6
        # build_frames already emits robot-convention wxyz rotations. GMR
        # reads offsets with scalar_first=True: xyzw's [0,0,0,1] would turn
        # every target 180 degrees, forcing the legs against their limits.
        for entry in config[table].values():
            entry[4] = [1.0, 0.0, 0.0, 0.0]
    path = Path(tempfile.gettempdir()) / f'marker_to_k1_{"_".join(sorted(tasks))}.json'
    path.write_text(json.dumps(config))
    return path


def retarget(clip, target_fps, tasks, leg_ratio, solver='quadprog', verbose=False):
    markers, labels, src_fps, meta = load_markers(clip, verbose=verbose)
    step = max(1, int(round(src_fps / target_fps)))
    geom = marker_geometry(markers[::step], labels)
    fps = src_fps / step
    human_leg = leg_length(geom)
    height = human_height_for_legs(human_leg, leg_ratio)
    meta.update(leg_length_m=human_leg, leg_ratio=leg_ratio, tasks=tasks,
                gmr_height_argument_m=height,
                applied_position_scale=0.6 * height / 1.8)

    gmr_params.IK_CONFIG_DICT.setdefault('marker', {})['booster_k1'] = build_config(tasks)
    retargeter = GMR(src_human='marker', tgt_robot='booster_k1',
                     actual_human_height=height, solver=solver, verbose=verbose)
    # The bundled GMR asset is an older K1: hip origins differ by 15 mm and
    # several arm/hip ranges are wider. Enforce the training model DURING IK,
    # rather than solving an incompatible pose and hiding it by clipping.
    import mink
    align_ik_model(retargeter.model)
    retargeter.ik_limits = [mink.ConfigurationLimit(retargeter.model)]
    human_keys = set(retargeter.human_body_to_task1) | set(retargeter.human_body_to_task2)

    # Seed the root with the human's first pelvis pose. Starting from the model
    # default (facing +X) makes the first solve rotate the whole body from the
    # wrong side, which is where the mirrored-branch solution came from.
    frames = list(build_frames(geom, human_keys))
    retargeter.update_targets(frames[0])
    position, quat = retargeter.scaled_human_data[retargeter.human_root_name]
    retargeter.configuration.data.qpos[:3] = position
    retargeter.configuration.data.qpos[3:7] = quat

    qpos, errors = [], []
    for frame in frames:
        qpos.append(retargeter.retarget(frame).copy())
        errors.append(retargeter.error1())
    return np.stack(qpos), fps, meta, np.asarray(errors)


def align_ik_model(model):
    """Use training kinematic transforms and hardware limits for named motors.

    Inertias and collision shapes are not used by the kinematic solve; final
    grounding continues to use the training model's actual foot geometry.
    """
    from validate_motion_dataset import load_urdf, DEFAULT_URDF
    training = mujoco.MjModel.from_xml_path(str(TRAIN_MODEL))
    limits, _, _ = load_urdf(DEFAULT_URDF)
    for source, destination in JOINT_MAP.items():
        source_id, target_id = model.joint(source).id, training.joint(destination).id
        if not np.allclose(model.jnt_axis[source_id], training.jnt_axis[target_id]):
            raise ValueError(f'Joint axis mismatch: {source} -> {destination}')
        body_id, target_body = model.jnt_bodyid[source_id], training.jnt_bodyid[target_id]
        model.body_pos[body_id] = training.body_pos[target_body]
        model.body_quat[body_id] = training.body_quat[target_body]
        model.jnt_pos[source_id] = training.jnt_pos[target_id]
        model.jnt_range[source_id] = limits[destination][:2]
        model.jnt_limited[source_id] = True


def gmr_qpos_columns():
    model = mujoco.MjModel.from_xml_path(str(gmr_params.ROBOT_XML_DICT['booster_k1']))
    return {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): model.jnt_qposadr[i]
            for i in range(model.njnt)}


def clean_joints(joints, joint_names, fps=50.0):
    """Smooth, then project onto K1's hardware and motor velocity limits.

    Three separate problems, in this order:
      1. The IK tracks noisy marker targets, so it amplifies knee motion ~2.3x
         against the source (human knee peaks at 6.5 rad/s, the retarget at 15).
         Walking is well under 5 Hz, so a 5-tap quadratic Savitzky-Golay pass
         removes that without flattening the swing phase.
      2. The IK honours ``K1_serial.xml``'s ranges, which are wider than the
         URDF's for the ankle, so clip to the hardware limits.
      3. Even then a joint can jump further in one 20 ms step than its motor can
         drive; reuse the midpoint rule from ``convert_paper_data.py`` so every
         dataset in the repo is cleaned the same way.
    """
    smoothed = savgol_filter(joints, window_length=5, polyorder=2, axis=0)

    clipped = 0
    for i, name in enumerate(joint_names):
        lo, hi = K1_JOINT_LIMITS[name]
        clipped += int(((smoothed[:, i] < lo) | (smoothed[:, i] > hi)).sum())
        smoothed[:, i] = smoothed[:, i].clip(lo, hi)

    # Rate limit: a forward pass then a backward pass, each clamping the step to
    # what the motor can drive. Unlike the midpoint rule in
    # convert_paper_data.py this converges on sustained runs and provably leaves
    # every step within the limit.
    limited = 0
    for i, name in enumerate(joint_names):
        limit = K1_VEL_LIMITS[name] / fps
        column = smoothed[:, i]
        for k in range(len(column) - 1):
            if abs(column[k + 1] - column[k]) > limit:
                limited += 1
                column[k + 1] = column[k] + np.sign(column[k + 1] - column[k]) * limit
        for k in range(len(column) - 2, -1, -1):
            if abs(column[k] - column[k + 1]) > limit:
                column[k] = column[k + 1] + np.sign(column[k] - column[k + 1]) * limit
    return smoothed, {'joint_limit_clipped_samples': clipped,
                      'velocity_limited_samples': limited}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--target_fps', type=float, default=50.0)
    parser.add_argument('--solver', default='quadprog',
                        help='qpsolvers backend; daqp 0.7.2 (pinned by IsaacLab) is too old for '
                             'the installed qpsolvers, which passes primal_start')
    parser.add_argument('--arms', choices=('nominal', 'markers'), default='nominal',
                        help="nominal = hold amp_paper's fixed arm pose (default; zero "
                             "soft-limit cost). markers = drive the arms from the wrist "
                             "markers, which crosses the shoulder soft limit on ~72%% of frames")
    parser.add_argument('--tasks', choices=sorted(TASK_SETS), default=None)
    parser.add_argument('--leg_ratio', type=float, default=1.00,
                        help='scaled human hip-to-ankle as a fraction of K1 leg length; '
                             '1.0 walks at full extension, lower crouches more')
    parser.add_argument('--anchor_walk', action='store_true',
                        help='fit root translation to the support foot (walk clips), as '
                             'build_amp_motion_npz does for the other walk sources')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    tasks = TASK_SETS[args.tasks] if args.tasks else TASK_SETS['foot_arms' if args.arms == 'markers' else 'foot']
    qpos, fps, meta, errors = retarget(args.clip, args.target_fps, tasks,
                                      args.leg_ratio, solver=args.solver, verbose=args.verbose)
    columns = gmr_qpos_columns()

    with np.load(SCHEMA_NPZ, allow_pickle=False) as schema:
        train_joints = [str(x) for x in schema['joint_names']]
        train_bodies = [str(x) for x in schema['body_names']]
    train_idx = {name: i for i, name in enumerate(train_joints)}

    joints = np.zeros((len(qpos), len(train_joints)))
    for gmr_joint, train_joint in JOINT_MAP.items():
        joints[:, train_idx[train_joint]] = qpos[:, columns[gmr_joint]]
    if args.arms == 'nominal':
        for name, value in NOMINAL_ARMS.items():
            joints[:, train_idx[name]] = value
    joints, cleanup = clean_joints(joints, train_joints, fps)
    meta.update(cleanup)

    fk = Kinematics(TRAIN_MODEL, train_joints, train_bodies)
    metadata = {
        'source': str(args.clip), 'source_sha256': sha256(args.clip),
        'generator_sha256': sha256(__file__),
        'training_model_sha256': sha256(TRAIN_MODEL),
        'ik_config_sha256': sha256(build_config(tasks)),
        'ik_model_alignment': 'training joint transforms and URDF hard limits',
        'source_fps': float(fps), 'time_scale': 1.0,
        'retarget_method': 'ACCAD 41-marker anatomical axes -> GMR mink IK (marker_to_k1.json)',
        'upper_body': 'markers (wrist position)' if args.arms == 'markers' else 'K1 nominal',
        'ground_clearance_m': 0.005, **meta,
    }
    print('source:', json.dumps(meta, indent=2))
    print(f'ik_error mean={errors.mean():.4f} p95={np.percentile(errors, 95):.4f}')

    result, stats = fk.build(qpos[:, :3], qpos[:, 3:7], joints, fps, args.anchor_walk, metadata)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **result)
    for key in ('frames', 'seconds', 'support_speed_before_mps', 'support_speed_after_mps',
                'joint_velocity_abs_max_radps', 'root_z_minmax_m', 'sole_minmax_m'):
        if key in stats:
            print(f'  {key}: {stats[key]}')
    print('written:', args.out)


if __name__ == '__main__':
    main()
