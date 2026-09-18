"""CPU transition regression tests; execute real command/reward ASTs without Isaac."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch

MDP = Path(__file__).resolve().parents[1] / 'source/booster_train/booster_train/tasks/manager_based/kick_amp/mdp'


def load(path, names, **scope):
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes, type_ignores=[])
    scope = {'torch': torch, **scope}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), scope)
    return scope


def rotate(q, v):
    uv = torch.cross(q[..., 1:], v, dim=-1)
    return v + 2 * (q[..., :1] * uv + torch.cross(q[..., 1:], uv, dim=-1))


def inverse(q, v):
    q = q.clone(); q[..., 1:] *= -1
    return rotate(q, v)


def multiply(a, b):
    return torch.cat((a[..., :1]*b[..., :1] - (a[..., 1:]*b[..., 1:]).sum(-1, keepdim=True),
                      a[..., :1]*b[..., 1:] + b[..., :1]*a[..., 1:] + torch.cross(a[..., 1:], b[..., 1:], dim=-1)), -1)


class Scene(dict):
    pass


class Ball:
    def __init__(self, state):
        self.data = NS(root_state_w=state, root_pos_w=state[:, :3])
        self.force = None

    def write_root_state_to_sim(self, state, env_ids=None):
        self.data.root_state_w[slice(None) if env_ids is None else env_ids] = state

    def set_external_force_and_torque(self, force, torque, **kwargs):
        self.force = force.clone()


def command():
    geo = load(MDP/'geometry.py', ['yaw_from_quat'])
    cls = load(MDP/'commands.py', ['SoccerStateCommand'], CommandTerm=object,
               MINIMAL=False, FIELD_HALF_LENGTH=7., FIELD_HALF_WIDTH=4.5, GOAL_X=7.,
               GOAL_HALF_WIDTH=1.3, BALL_RADIUS=.11, GOAL_SUCCESS_STEPS=50,
               CAMERA_OFFSET_B=(0.,0.,0.), CAMERA_BODY_TO_OPTICAL_WXYZ=(1.,0.,0.,0.),
               CAMERA_FOV_H=87., CAMERA_FOV_V=58., task_curriculum=lambda _:1.,
               quat_rotate=rotate, quat_rotate_inverse=inverse, quat_mul=multiply,
               yaw_from_quat=geo['yaw_from_quat'])['SoccerStateCommand']
    c = cls.__new__(cls); c.num_envs = 1; c.device = 'cpu'
    c.cfg = NS(ball_reset_enabled=True, ball_teleport_prob=0., ball_kick_prob=0., ball_kick_vel=.5,
               ball_friction_range=(.2,.2), push_enabled=False, perfect_perception=True,
               training_phase='soccer', ball_distance_scale=50., goal_distance_scale=500., goal_scale=15.)
    root = torch.zeros(1,13); root[:,3] = 1
    robot = NS(data=NS(root_state_w=root, root_quat_w=root[:,3:7], root_pos_w=root[:,:3],
                      root_lin_vel_w=root[:,7:10], root_lin_vel_b=root[:,7:10],
                      body_state_w=root[:,None,:].repeat(1,3,1)),
               set_external_force_and_torque=lambda *args, **kwargs:None)
    ball_state = root.clone(); ball_state[0,0] = 2.; ball_state[0,2] = .115
    ball = Ball(ball_state)
    scene = Scene(robot=robot, ball=ball)
    scene.env_origins = torch.zeros(1,3)
    scene.sensors = {'contact_forces': NS(data=NS(net_forces_w=torch.zeros(1,2,3)))}
    env = NS(scene=scene, step_dt=.02, common_step_counter=1, episode_length_buf=torch.tensor([100]),
             extras={}, action_manager=NS(action=torch.zeros(1,2)))
    c._env = env; c.robot = robot; c.ball = ball; c.head_idx = 0; c.trunk_idx = 0; c.feet_sensor_idx = [0,1]
    for name in ['base_pos_xy', 'relative_ball_pos', 'relative_goal_pos', 'last_base_pos_xy', 'last_ball_pos_xy',
                 'ball_pos_xy', 'ball_vel_xy', 'camera_pos_xy', 'ball_friction_force_xy', '_push_force_xy']:
        setattr(c, name, torch.zeros(1,2))
    c.base_yaw = torch.zeros(1); c.ball_pos_z = torch.zeros(1); c.ball_in_view = torch.zeros(1)
    c.ball_friction_force = torch.tensor([.2]); c._push_torque = torch.zeros(1,3)
    c.ball_obs_buffer = torch.zeros(1,20,3); c.ball_delay_steps = torch.tensor([6])
    c.base_pos_buffer = torch.zeros(1,50,3); c.buffer_count = torch.zeros(1,dtype=torch.long); c.buffer_idx = c.buffer_count.clone()
    c.goal_cnt = c.buffer_count.clone(); c.goal_scored_now = torch.zeros(1,dtype=torch.bool)
    c.goal_success_now = torch.zeros(1,dtype=torch.bool); c.last_ball_in_goal = c.goal_scored_now.clone()
    c.last_root_vel = torch.zeros(1,6); c.last_actions = torch.zeros(1,2); c.base_mass_scaled = torch.zeros(1,4)
    c.metrics = {'goal': torch.zeros(1), 'success': torch.zeros(1)}
    c._compute_amp_obs = lambda:torch.zeros(1,39)
    c.last_ball_pos_xy[:] = ball_state[:,:2]
    def reset(ids):
        state = ball.data.root_state_w[ids].clone()
        state[:,:3] = torch.tensor([1.,1.,.115]); state[:,7:] = 0.
        ball.write_root_state_to_sim(state, env_ids=ids)
    c._random_ball_reset = reset
    return c, env


class SoccerTiming(unittest.TestCase):
    def test_goal_completion_keeps_reward_and_success_before_reset(self):
        c,e = command(); c.ball.data.root_state_w[0,0] = 7.3
        c.last_ball_pos_xy[0,0] = 7.3; c.last_ball_in_goal[:] = True; c.goal_cnt[:] = 49
        c._update_command()
        self.assertAlmostEqual(float(e.extras['rew_groups'][0,0]), .3, places=5)
        self.assertEqual(float(e.extras['success'][0]), 1.)
        self.assertEqual(int(c.goal_cnt[0]), 0)
        self.assertFalse(bool(c.last_ball_in_goal[0]))
        c._update_command()
        self.assertAlmostEqual(float(e.extras['rew_groups'][0,0]), 0., places=5)
        self.assertEqual(float(e.extras['success'][0]), 0.)

    def test_teleport_preserves_real_progress_but_not_teleport_distance(self):
        c,e = command(); c.cfg.ball_teleport_prob = 1.
        c.ball.data.root_state_w[0,0] = 3. # moved 1 m toward goal from previous 2 m
        c._update_command()
        self.assertAlmostEqual(float(e.extras['rew_groups'][0,0]), 10., places=5)
        c.cfg.ball_teleport_prob = 0.; c._update_command()
        self.assertAlmostEqual(float(e.extras['rew_groups'][0,0]), 0., places=5)

    def test_reset_refreshes_truth_velocity_friction_and_perfect_observation(self):
        c,e = command(); c.cfg.ball_teleport_prob = 1.
        c.ball.data.root_state_w[0,2] = 1.2; c.ball.data.root_state_w[0,7:9] = torch.tensor([1.,2.])
        c._update_command()
        torch.testing.assert_close(c.ball_obs[:,:2], torch.tensor([[1.,1.]]))
        torch.testing.assert_close(c.ball_pos_z, torch.tensor([.115]))
        torch.testing.assert_close(c.ball_vel_xy, torch.zeros(1,2))
        torch.testing.assert_close(c.ball.force, torch.zeros(1,1,3))
        c.cfg.perfect_perception = False
        torch.testing.assert_close(c._compute_privileged_obs()[:,10:12], torch.tensor([[1.,1.]]))
        torch.testing.assert_close(c._compute_privileged_obs()[:,12:14], torch.zeros(1,2))

    def test_teleport_keeps_intentional_perception_latency(self):
        c,e = command(); c.cfg.ball_teleport_prob = 1.; c.cfg.perfect_perception = False
        c.ball_obs_buffer[:] = torch.tensor([9.,8.,1.])
        c._update_command()
        torch.testing.assert_close(c.ball_obs, torch.tensor([[9.,8.,1.]]))
        torch.testing.assert_close(c.relative_ball_pos, torch.tensor([[1.,1.]]))

    def test_newest_perception_sample_is_post_reset_but_delayed_read_stays_old(self):
        c,e = command(); c.cfg.ball_teleport_prob = 1.; c.cfg.perfect_perception = False
        c.ball_obs_buffer[:] = torch.tensor([9.,8.,1.])
        with patch('torch.rand', side_effect=lambda *shape, **kwargs:torch.zeros(*shape)), \
             patch('torch.randn_like', side_effect=torch.zeros_like):
            c._update_command()
        torch.testing.assert_close(c.ball_obs_buffer[:,0], torch.tensor([[1.,1.,1.]]))
        torch.testing.assert_close(c.ball_obs, torch.tensor([[9.,8.,1.]]))

    def test_out_of_field_reset_retains_real_transition_penalty(self):
        c,e = command(); c.last_ball_pos_xy[:] = torch.tensor([[7.,2.]])
        c.ball.data.root_state_w[0,:2] = torch.tensor([8.,2.])
        c._update_command()
        self.assertAlmostEqual(float(e.extras['rew_groups'][0,0]), (2.-5.**.5)*10., places=5)
        self.assertEqual(float(e.extras['success'][0]), 0.)
        c._update_command()
        self.assertAlmostEqual(float(e.extras['rew_groups'][0,0]), 0., places=5)

    def test_random_kick_refreshes_velocity_before_resistance(self):
        c,e = command(); c.cfg.ball_kick_prob = 1.; torch.manual_seed(10)
        c._update_command()
        self.assertGreater(float(c.ball.data.root_state_w[:,7:9].norm()), 0.)
        torch.testing.assert_close(c.ball_vel_xy, c.ball.data.root_state_w[:,7:9])
        self.assertLess(float((c.ball.force[:,0,:2] * c.ball_vel_xy).sum()), 0.)

    def test_episode_reset_still_masks_task_reward(self):
        c,e = command(); e.episode_length_buf[:] = 0
        c.last_ball_pos_xy[:] = 0.; c._update_command()
        self.assertEqual(float(e.extras['rew_groups'][0,0]), 0.)

    def test_kick_geometry_uses_current_physics_ball_with_nonzero_origin(self):
        c,e = command(); e.device='cpu'; e.num_envs=1; c.feet_idx=[0,1]
        e.scene.env_origins[0,:2] = torch.tensor([10.,20.])
        c.ball.data.root_state_w[0,:2] = torch.tensor([10.,20.]); c.ball_pos_xy[:] = 100.
        e.scene['robot'].data.body_lin_vel_w = torch.tensor([[[1.,-1.,0.],[1.,1.,0.]]])
        scope = load(MDP/'rewards.py', ['kick_ball', 'side_kick_ball'], _cmd=lambda _:c,
                     _feet_edge_pos_w=lambda *_:torch.zeros(1,2,4,3), _feet_yaw=lambda *_:torch.zeros(1,2))
        self.assertAlmostEqual(float(scope['kick_ball'](e)), 2.)
        self.assertAlmostEqual(float(scope['side_kick_ball'](e)), 2.)

    def test_camera_geometry_uses_current_physics_height_and_xy(self):
        c,e = command(); e.device='cpu'; e.num_envs=1
        c.ball_pos_xy[:] = torch.tensor([[10.,0.]]); c.ball_pos_z[:] = 10.
        c.ball.data.root_state_w[0,:3] = torch.tensor([1.,0.,1.])
        fn = load(MDP/'rewards.py', ['_camera_angles'], quat_mul=multiply, quat_rotate=rotate,
                  quat_rotate_inverse=inverse, CAMERA_OFFSET_B=(0.,0.,0.),
                  CAMERA_BODY_TO_OPTICAL_WXYZ=(1.,0.,0.,0.))['_camera_angles']
        actual = fn(e,c)
        # Deliberately distinguish cached direction from physical direction.
        c.ball.data.root_state_w[0,0] = 0.
        torch.testing.assert_close(fn(e,c), torch.zeros(1,2))
        self.assertGreater(float(actual[0,0]), .7)


if __name__ == '__main__': unittest.main()
