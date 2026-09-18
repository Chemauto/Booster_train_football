"""Geometry shared by observations and rewards (Isaac Lab wxyz convention)."""

import torch


def yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Heading about world z, in [-pi, pi], for (..., 4) wxyz quaternions."""
    w, x, y, z = quat.unbind(dim=-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
