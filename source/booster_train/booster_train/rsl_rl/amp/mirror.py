"""Mirror (left-right) symmetry operators for the K1 kick_amp task.

Builds linear mirroring matrices for the actor observation and action vectors.
All matrices are built FROM the live robot joint names (the PhysX articulation
order, which differs from the URDF declaration order!) -- callers must pass
env.scene["robot"].joint_names.

Obs layout (79-dim, must match kick_amp env_cfg PolicyObsCfg term order):

    [0:3]   projected gravity      (y flips)
    [3:6]   base ang vel           (x, z flip)
    [6:9]   ball obs (x, y, flag)  (y flips)
    [9:11]  goal pos (x, y)        (y flips)
    [11:13] cos/sin base yaw       (sin flips)
    [13:13+J] joint pos
    [.. +J]  joint vel
    [.. +J]  action      (J = number of joints, live order)

Joint mirroring swaps left/right joints and negates roll/yaw-type joints.
"""

from __future__ import annotations

import torch

# joints whose angle flips sign under left-right mirroring (roll/yaw axes).
# NOTE: K1 joint names carry an inconsistent 'aa' prefix on some joints
# (aaleft_shoulder_pitch vs left_shoulder_roll) -- always match by suffix.
NEGATIVE_TOKENS = ("_roll_joint", "_yaw_joint")

HEAD_JOINTS = ("aahead_yaw_joint", "aahead_pitch_joint")


def _mirror_partner(name: str) -> str:
    # strip any 'aa' prefix to find the left/right token, then swap it
    core = name[2:] if name.startswith(("aaleft_", "aaright_")) else name
    if core.startswith("left_"):
        partner = "aa" + core.replace("left_", "right_", 1) if name.startswith("aaleft_") else core.replace("left_", "right_", 1)
    elif core.startswith("right_"):
        partner = "aa" + core.replace("right_", "left_", 1) if name.startswith("aaright_") else core.replace("right_", "left_", 1)
    else:
        return name  # head joints map to themselves
    return partner


def joint_mirror_matrix(joint_names: list[str], device) -> torch.Tensor:
    """(J, J) matrix M with mirrored_joint = M @ joint (column convention)."""
    n = len(joint_names)
    index = {name: i for i, name in enumerate(joint_names)}
    for name in joint_names:
        partner = _mirror_partner(name)
        assert partner in index, f"mirror partner {partner} not found for {name}"
    mat = torch.zeros(n, n, device=device)
    for i, name in enumerate(joint_names):
        j = index[_mirror_partner(name)]
        sign = -1.0 if name.endswith(NEGATIVE_TOKENS) else 1.0
        mat[j, i] = sign
    return mat


def obs_mirror_matrix(num_obs: int, joint_names: list[str], device) -> torch.Tensor:
    """(num_obs, num_obs) mirroring for the actor obs layout above."""
    num_joints = len(joint_names)
    mat = torch.zeros(num_obs, num_obs, device=device)
    p = 0
    # gravity: (g_x, g_y, g_z) -> (g_x, -g_y, g_z)
    mat[p + 0, p + 0] = 1.0
    mat[p + 2, p + 2] = 1.0
    mat[p + 1, p + 1] = -1.0
    p += 3
    # ang vel: (w_x, w_y, w_z) -> (-w_x, w_y, -w_z)
    mat[p + 0, p + 0] = -1.0
    mat[p + 1, p + 1] = 1.0
    mat[p + 2, p + 2] = -1.0
    p += 3
    # ball obs: (x, y, flag) -> (x, -y, flag)
    mat[p + 0, p + 0] = 1.0
    mat[p + 1, p + 1] = -1.0
    mat[p + 2, p + 2] = 1.0
    p += 3
    # goal pos: (x, y) -> (x, -y)
    mat[p + 0, p + 0] = 1.0
    mat[p + 1, p + 1] = -1.0
    p += 2
    # yaw: (cos, sin) -> (cos, -sin)
    mat[p + 0, p + 0] = 1.0
    mat[p + 1, p + 1] = -1.0
    p += 2
    # joint blocks: pos, vel, action
    jmat = joint_mirror_matrix(joint_names, device)
    for _ in range(3):
        mat[p:p + num_joints, p:p + num_joints] = jmat
        p += num_joints
    assert p == num_obs, f"obs layout mismatch: built {p}, expected {num_obs}"
    return mat


def mirror(obs: torch.Tensor, joint_names: list[str], device) -> torch.Tensor:
    key = (obs.shape[-1], tuple(joint_names), str(device))
    if key not in _OBS_MAT_CACHE:
        _OBS_MAT_CACHE[key] = obs_mirror_matrix(obs.shape[-1], joint_names, device)
    return obs @ _OBS_MAT_CACHE[key].T


def mirror_act(act: torch.Tensor, joint_names: list[str], device) -> torch.Tensor:
    key = (tuple(joint_names), str(device))
    if key not in _JOINT_MAT_CACHE:
        _JOINT_MAT_CACHE[key] = joint_mirror_matrix(joint_names, device)
    return act @ _JOINT_MAT_CACHE[key].T


_OBS_MAT_CACHE: dict = {}
_JOINT_MAT_CACHE: dict = {}
