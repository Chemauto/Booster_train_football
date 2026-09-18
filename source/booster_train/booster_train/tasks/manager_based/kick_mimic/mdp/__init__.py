# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the functions that are specific to the kick environment.

Re-exports everything from beyond_mimic.mdp (tracking rewards/observations,
events, terminations) so env cfgs can reference a single `mdp` namespace and
call the identical function objects the stage-1 task uses."""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from booster_train.tasks.manager_based.beyond_mimic.mdp import *  # noqa: F401, F403

from .commands import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
