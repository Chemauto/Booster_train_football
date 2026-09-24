# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

import sys
from pathlib import Path as _Path

# Prefer the in-repo assets/ vendored booster_assets shim over the external
# booster_assets package. Must run before .tasks (which imports booster_assets).
_vendored = _Path(__file__).resolve().parents[3] / "assets"
if _vendored.is_dir():
    _p = str(_vendored)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Register Gym environments.
from .tasks import *

