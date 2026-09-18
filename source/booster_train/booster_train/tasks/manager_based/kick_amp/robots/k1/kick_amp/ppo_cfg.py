"""Runner cfg for the AMP kick task (custom AmpRunner, not the stock rsl_rl PPO)."""

from isaaclab.utils import configclass

from booster_train.rsl_rl.amp.runner import AmpRunnerCfg
from booster_train.tasks.manager_based.kick_amp.mdp.commands import WARMUP_STEPS


@configclass
class KickAmpPPORunnerCfg(AmpRunnerCfg):
    num_envs = 4096
    experiment_name = "k1_kick_amp"
    max_iterations = 20000
    save_interval = 200

    # policy dt = 0.005 * 4 = 0.02 s; 1 s of history at 50 Hz = 50 frames
    num_stack = 50
    horizon_length = 24
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1.0e-3
    discriminator_lr = 1.0e-4
    gamma = 0.995
    lam = 0.95
    bound_coef = 100.0
    reconstruction_coef = 1.0
    # paper Table 1 "Entropy coefficient 0.01"; T1.yaml (the runnable source)
    # encodes it as -0.01 and the loss adds coef*entropy, so the negative sign
    # is the standard entropy BONUS under gradient descent (not a penalty)
    entropy_coef = -0.01
    symmetric_coef = 10.0
    amp_coef = 1.0
    # back to the paper's 0.3 (Table 4 "AMP style"). The 1.0 seed-protection
    # value predates the amp_paper dataset fix: with the paper's own data the
    # reward no longer needs amplification, and 0.3 is what the paper tuned.
    amp_reward_coef = 0.3
    # paper Table 1 "Gradient penalty coefficient 50" (T1.yaml's 5.0 predates
    # the paper's final tuning, same as its 8192 envs vs the paper's 16384)
    # paper Table 1 effective value = 50 = the loss's internal 10* (runner.py,
    # same as the original repo) x this 5.0 (T1.yaml). Setting the config to
    # 50 yields an effective 500, 10x the paper -- the exact mistake an
    # independent Codex audit caught in my Table-1 alignment pass.
    grad_pen_coef = 5.0
    # paper Table 1. The adaptive-lr rule now decides on the iteration-MEAN kl
    # with a 2x raise threshold, so 0.01 no longer pins lr at the floor (that
    # was the old last-minibatch decision bug, fixed in runner.py)
    desired_kl = 0.01
    advantage_coef = (2.0, 1.0)
    # not used by the runner anymore (it gates nothing); kept so the value
    # stays in the checkpoint for older tools that read it
    curriculum_full_steps = WARMUP_STEPS
