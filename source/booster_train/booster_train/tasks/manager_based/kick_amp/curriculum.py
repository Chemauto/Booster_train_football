"""Explicit training phases; progression requires external physical evaluation."""
from pathlib import Path


def configure_training_phase(env_cfg, agent_cfg, phase):
    if phase not in ("soccer", "approach"):
        raise ValueError(f"Unknown training phase: {phase}")
    cmd = env_cfg.commands.soccer
    cmd.training_phase = phase
    if phase == "soccer":
        return
    # A walk prior supports this phase without mixing in stationary kicks.
    directory = Path(cmd.motion_dir)
    if directory.name != "walk":
        directory = directory / "walk"
    if not directory.is_dir() or not list(directory.glob("*.npz")):
        raise FileNotFoundError(f"Approach phase requires walk motion data: {directory}")
    cmd.motion_dir = str(directory)
    cmd.reset_motion_fraction = 0.0
    cmd.ball_spawn_distance = (1.0, 2.0)
    cmd.goal_scale = cmd.goal_distance_scale = cmd.ball_distance_scale = 0.0
    for name in ("kick_ball", "side_kick_ball", "face_ball_pitch", "face_ball_yaw"):
        getattr(env_cfg.rewards, name).weight = 0.0
    env_cfg.rewards.survival.weight = 1.0
    env_cfg.rewards.termination.weight = -100.0
    env_cfg.rewards.pos_still.weight = -1.0
    env_cfg.rewards.track_lin_vel_ball.weight = 8.0
    env_cfg.rewards.track_lin_vel_ball.params = {"target_speed": .4, "std": .25, "near_ball": .5}
    agent_cfg.advantage_coef = (0.0, 1.0)
    agent_cfg.amp_reward_coef = .3
