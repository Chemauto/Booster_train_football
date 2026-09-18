from isaaclab.utils import configclass
from booster_train.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class TrackPPORunnerCfg(BasePPORunnerCfg):
    """Stage 1: pure tracking prior on the full kick motion."""

    max_iterations = 30000
    save_interval = 500
    experiment_name = "k1_kick_001_track"


@configclass
class KickPPORunnerCfg(BasePPORunnerCfg):
    """Stage 2: ball + kick target, warm-started via --init_policy_path."""

    max_iterations = 30000
    save_interval = 500
    experiment_name = "k1_kick_001"
