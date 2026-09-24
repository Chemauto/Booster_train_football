#!/usr/bin/env python3
"""MuJoCo sim2sim for the K1 kick_amp soccer policy (booster_train/deploy copy).

Usage (from anywhere; conda base python works):
    python3 deploy/sim2sim_kick_amp.py                      # 32-episode batch
    python3 deploy/sim2sim_kick_amp.py --view               # interactive viewer
    python3 deploy/sim2sim_kick_amp.py --checkpoint kick_amp_it7200_policy.pt

This is a relocation of booster_deploy/scripts/sim2sim_kick_amp.py: the task
package lives at deploy/tasks/kick_amp (copied verbatim, scene path resolves
from the package location), and the booster_deploy controller framework is
imported from the sibling repository via sys.path.
"""

import argparse
import os
import sys

_DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
_BOOSTER_DEPLOY = "/data/rl_robot/BoosterRobotics/booster_deploy"

# 1) this deploy/ dir provides the tasks.kick_amp package
sys.path.insert(0, _DEPLOY_DIR)
# 2) the sibling booster_deploy repo provides the controller framework
#    (base_controller / mujoco_controller / robots / utils) and registers the
#    original task set; 3) booster_assets may not be pip-installed in this env
sys.path.insert(1, _BOOSTER_DEPLOY)
try:
    import booster_assets  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(2, os.path.join(_DEPLOY_DIR, "..", "assets"))

from tasks.kick_amp import KickAmpControllerCfg            # noqa: E402
from tasks.kick_amp.kick_amp_mujoco import (               # noqa: E402
    run_sim2sim, run_viewer,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=1500,
                        help="max policy steps per episode (1500 = 30 s)")
    parser.add_argument("--perception", choices=("perfect", "virtual"),
                        default="perfect")
    parser.add_argument("--delay-substeps", type=int, default=5,
                        help="actuator target delay in 2 ms substeps "
                             "(default 5 = the evaluation's fixed 10 ms)")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--foot-collision", choices=("box", "mesh"), default="box",
                        help="foot collision geometry A/B: flat box (default) "
                             "or Left/Right_Foot.STL convex hull")
    parser.add_argument("--realtime", action="store_true",
                        help="sleep to wall-clock pace (headless demo)")
    parser.add_argument("--view", action="store_true",
                        help="interactive MuJoCo viewer instead of headless")
    parser.add_argument("--out", type=str, default=None,
                        help="write the summary JSON here")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="override the policy file name under "
                             "tasks/kick_amp/models/")
    args = parser.parse_args()
    if args.episodes < 1 or args.steps < 1:
        parser.error("--episodes and --steps must be >= 1")

    cfg = KickAmpControllerCfg()
    cfg.policy.perception = args.perception
    cfg.actuator_delay_substeps = args.delay_substeps
    cfg.push_enabled = args.push
    cfg.foot_collision = args.foot_collision
    if args.checkpoint:
        cfg.policy.checkpoint_path = f"models/{args.checkpoint}"

    if args.view:
        run_viewer(cfg, seed=args.seed)
    else:
        base = os.path.basename(cfg.policy.checkpoint_path)
        base = base[:-3] if base.endswith(".pt") else base   # avoid .pt.json
        out = args.out or os.path.join(
            "/tmp", f"kick_amp_sim2sim_e{args.episodes}_seed{args.seed}_"
                    f"{args.perception}_{args.foot_collision}_{base}.json")
        run_sim2sim(cfg, episodes=args.episodes, seed=args.seed,
                    steps=args.steps, realtime=args.realtime, out=out)


if __name__ == "__main__":
    main()
