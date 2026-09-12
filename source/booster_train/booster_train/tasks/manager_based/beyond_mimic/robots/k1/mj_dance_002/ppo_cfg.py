from isaaclab.utils import configclass
from booster_train.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class PPORunnerCfg(BasePPORunnerCfg):
    max_iterations = 30000
    save_interval = 500  # 机器今日两次突然断电，缩短存档间隔减少进度损失
    experiment_name = "k1_mj_dance_002"
