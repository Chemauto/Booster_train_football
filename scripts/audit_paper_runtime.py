"""Controlled runtime probes for soccer reward and reset semantics; no training."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument('--output', default='logs/audit_paper_current/runtime.json')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
import json
import gymnasium as gym
import torch
import isaaclab_tasks
import booster_train.tasks
from isaaclab_tasks.utils import parse_env_cfg
from evaluate_kick_amp import configure_evaluation

env = None
try:
    cfg = parse_env_cfg('Booster-K1-KickAMP-v0-Play', device=args.device, num_envs=4)
    configure_evaluation(cfg, 'standing', 123, perfect_perception=True)
    cfg.commands.soccer.ball_reset_enabled = True
    env = gym.make('Booster-K1-KickAMP-v0-Play', cfg=cfg).unwrapped
    env.reset()
    cmd = env.command_manager.get_term('soccer')
    env.common_step_counter = 8000 * 24
    env.episode_length_buf.fill_(100)
    ball = env.scene['ball']
    state = ball.data.root_state_w.clone()
    state[:, :2] = env.scene.env_origins[:, :2] + torch.tensor([7.3, 0.], device=env.device)
    state[:, 2] = .115
    state[:, 7:] = 0.
    ball.write_root_state_to_sim(state)
    cmd.last_base_pos_xy[:] = env.scene['robot'].data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    cmd.last_ball_pos_xy[:] = torch.tensor([7.3, 0.], device=env.device)
    cmd.last_ball_in_goal.fill_(True)
    cmd.goal_cnt.fill_(49)
    cmd._update_command()
    actual_ball = ball.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    expected_relative = cmd._rotate_to_yaw_frame(actual_ball - cmd.base_pos_xy)
    result = {'probe': '50th consecutive in-goal step, stationary robot, automatic ball reset enabled',
              'dataset': cfg.commands.soccer.motion_dir,
              'task_reward_at_goal_completion': env.extras['rew_groups'][:,0].tolist(),
              'success_flag': env.extras['success'].tolist(),
              'new_ball_xy': actual_ball.tolist(),
              'actor_ball_xy': cmd.ball_obs[:, :2].tolist(),
              'actual_relative_ball_xy': expected_relative.tolist(),
              'observation_error_m': (cmd.ball_obs[:, :2] - expected_relative).norm(dim=-1).tolist(),
              'goal_flag_after_reset': cmd.goal_scored_now.tolist()}
    result['actuators'] = {
        name: {'joints': actuator.joint_names,
               'effort_limit': actuator.effort_limit[0].tolist(),
               'effort_limit_sim': actuator.effort_limit_sim[0].tolist(),
               'velocity_limit': actuator.velocity_limit[0].tolist(),
               'stiffness': actuator.stiffness[0].tolist()}
        for name, actuator in env.scene['robot'].actuators.items()
    }
    result['action_scale'] = env.action_manager.get_term('joint_pos')._scale[0].tolist()
    result['joint_names'] = env.scene['robot'].joint_names
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print('AUDIT_RESULT '+json.dumps(result),flush=True)
finally:
    if env is not None:
        env.close()
    app.close()
