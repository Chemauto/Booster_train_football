"""The field edge needs a gradient, or chasing the ball is never worth the risk.

The out-of-field termination is a 0/1 cliff (weight -1000) and the goal line is
the field edge (GOAL_X == FIELD_HALF_LENGTH), so pushing the ball toward the goal
walks the robot straight at a hard boundary with no intermediate signal. Measured
consequence at model_3000: the policy parks 0.6 m from the ball, contact proxies
1/64. boundary_distance turns that cliff into a ramp without taxing the scoring
motion itself.
"""
from types import SimpleNamespace as NS
import ast
import argparse
import contextlib
import io
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from test_kick_amp_contracts import functions, MDP

FIELD_HALF_LENGTH, FIELD_HALF_WIDTH = 7.0, 4.5
ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / 'scripts/rsl_rl/train_kick_amp.py'


class BoundaryDistance(unittest.TestCase):
    def fn(self, guard=0.3):
        scope = functions(MDP / 'rewards.py', ['boundary_distance'],
                          _cmd=lambda env: env.cmd,
                          FIELD_HALF_LENGTH=FIELD_HALF_LENGTH,
                          FIELD_HALF_WIDTH=FIELD_HALF_WIDTH)
        self.assertIn('boundary_distance', scope, 'Missing dense boundary shaping term')

        def call(xy):
            env = NS(cmd=NS(base_pos_xy=torch.tensor(xy, dtype=torch.float32)))
            return scope['boundary_distance'](env, guard=guard)
        return call

    def test_free_in_the_interior_and_saturated_on_either_edge(self):
        f = self.fn()
        torch.testing.assert_close(f([[0., 0.]]), torch.zeros(1))          # centre
        torch.testing.assert_close(f([[7., 0.]]), torch.ones(1))           # x edge
        torch.testing.assert_close(f([[0., -4.5]]), torch.ones(1))         # y edge, both signs
        torch.testing.assert_close(f([[-7., 0.]]), torch.ones(1))

    def test_scoring_approach_is_not_penalized(self):
        """Pushing the ball across x=7 holds the robot centre ~0.3-0.7 m inside."""
        f = self.fn()
        torch.testing.assert_close(f([[6.7, 0.]]), torch.zeros(1))
        torch.testing.assert_close(f([[6.5, 0.]]), torch.zeros(1))
        torch.testing.assert_close(f([[6.3, 1.2]]), torch.zeros(1))

    def test_ramp_is_monotone_bounded_and_graded_inside_the_guard(self):
        f = self.fn()
        values = [float(f([[FIELD_HALF_LENGTH - d, 0.]])) for d in (0.3, 0.2, 0.1, 0.0)]
        self.assertEqual(values, sorted(values))
        self.assertTrue(all(0.0 <= v <= 1.0 for v in values))
        self.assertGreater(values[-1] - values[0], 0.5, 'no usable gradient at the edge')

    def test_guard_width_is_configurable_and_zero_guard_is_safe(self):
        wide = self.fn(guard=1.0)
        self.assertGreater(float(wide([[6.5, 0.]])), 0.0, 'guard=1.0 should reach 0.5 m out')
        zero = self.fn(guard=0.0)
        self.assertEqual(float(zero([[7., 0.]])), 0.0, 'guard=0 must not divide by zero')
        self.assertEqual(float(zero([[0., 0.]])), 0.0)




class BoundaryArguments(unittest.TestCase):
    """The penalty must be off by default and reach the reward config when asked."""

    flag = '--boundary_weight'
    reward = 'boundary'

    def parse_arguments(self, value=None):
        tree = ast.parse(TRAIN.read_text())
        start = next(i for i, n in enumerate(tree.body)
                     if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'parser')
        end = next(i for i, n in enumerate(tree.body)
                   if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'app_launcher')
        scope = {'argparse': argparse, 'math': math, 'AppLauncher': NS(add_app_launcher_args=lambda _: None)}
        argv = ['train'] + ([] if value is None else [self.flag, value])
        with patch.object(sys, 'argv', argv):
            exec(compile(ast.Module(body=tree.body[start:end], type_ignores=[]), str(TRAIN), 'exec'), scope)
        return scope['args_cli']

    attribute = 'boundary_weight'

    def test_defaults_to_off_and_accepts_penalties(self):
        self.assertEqual(getattr(self.parse_arguments(), self.attribute), 0.)
        for value in ('-1', '-100', '-2.5'):
            self.assertEqual(getattr(self.parse_arguments(value), self.attribute), float(value))

    def test_rejects_positive_and_nonfinite_before_the_simulator(self):
        """A positive weight would reward leaving the field."""
        for value in ('1', '0.5', 'nan', 'inf'):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as exc:
                self.parse_arguments(value)
            self.assertEqual(exc.exception.code, 2)

    def test_reaches_the_reward_configuration(self):
        target = f'env_cfg.rewards.{self.reward}.weight'
        tree = ast.parse(TRAIN.read_text())
        nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(
            ast.unparse(t) == target for t in n.targets)]
        self.assertEqual(len(nodes), 1)
        env_cfg = NS(rewards=NS(**{self.reward: NS(weight=0.)}))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TRAIN), 'exec'),
             {'env_cfg': env_cfg, 'args_cli': NS(**{self.attribute: -100.})})
        self.assertEqual(getattr(env_cfg.rewards, self.reward).weight, -100.)

    def test_stages_runner_forwards_the_flag(self):
        """The staged runner must pass it, or an arm would silently run with 0."""
        src = (ROOT / 'scripts/run_kick_amp_stages.py').read_text()
        self.assertIn(f'"{self.flag}", str(args.{self.attribute})', src)
        self.assertIn(f'"{self.attribute}": args.{self.attribute}', src)


class ParkingSpace(unittest.TestCase):
    """The anti-freeze term must not leave a stop-short haven next to the ball.

    Evidence: a physics probe of model_3000 found the policy's end state 0.61 m
    from the ball and stationary. With the exemption at 1.0 m that position pays
    no freeze penalty, so "stop just short of the ball" is free -- contact
    proxies collapsed 55 -> 15 -> 1 across four stages while falls went to 0.
    """

    def configured_params(self):
        src = (ROOT / 'source/booster_train/booster_train/tasks/manager_based/kick_amp/'
                      'robots/k1/kick_amp/env_cfg.py').read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'RewTerm':
                func = next((kw.value for kw in node.keywords if kw.arg == 'func'), None)
                if isinstance(func, ast.Attribute) and func.attr == 'pos_still':
                    params = next(kw.value for kw in node.keywords if kw.arg == 'params')
                    return ast.literal_eval(params)
        self.fail('pos_still reward not found in env_cfg')

    def test_ball_exemption_is_contact_range_not_a_parking_ring(self):
        params = self.configured_params()
        self.assertLessEqual(params['penalize_pos_distance'], 0.35,
                             'a wide ball exemption reopens the stop-short haven')
        self.assertGreater(params['penalize_pos_distance'], 0.0)
        # Not the paper's 0.7: that demands 0.7 m/s of net displacement while this
        # robot walks at 0.623 m/s, making the anti-freeze term unsatisfiable and
        # the policy hurry itself into falls during the reset transient.
        self.assertLess(params['penalize_pos'], 0.623, 'must be reachable at the achieved walking speed')
        self.assertGreaterEqual(params['penalize_pos'], 0.3, 'must still detect standing still')

    def test_function_still_honours_its_default_when_called_bare(self):
        tree = ast.parse((MDP / 'rewards.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'pos_still')
        defaults = {a.arg: d.value for a, d in zip(fn.args.args[-3:], fn.args.defaults)}
        self.assertEqual(defaults['penalize_pos_distance'], 1.0,
                         'the config is the deliberate override; the default stays documented')

if __name__ == '__main__':
    unittest.main()

class BoundaryOutwardArguments(BoundaryArguments):
    """Same contract for the velocity-shaped term."""
    flag = '--boundary_outward_weight'
    reward = 'boundary_outward'
    attribute = 'boundary_outward_weight'


class BoundaryOutwardSpeed(unittest.TestCase):
    """Charging outward motion, not proximity, is what lets the scoring push through.

    The distance-shaped term cannot separate the two: to push the ball across
    x = 7 the robot must stand within ~0.3 m of that same line, so any guard wide
    enough to teach deceleration taxes the goal. Measured with the distance term
    on: goals 74 -> 37 while out-of-field fell 215 -> 44.
    """

    GUARD = 0.6

    def fn(self, guard=GUARD):
        scope = functions(MDP / 'rewards.py', ['boundary_outward_speed'],
                          _cmd=lambda env: env.cmd,
                          FIELD_HALF_LENGTH=FIELD_HALF_LENGTH,
                          FIELD_HALF_WIDTH=FIELD_HALF_WIDTH)
        self.assertIn('boundary_outward_speed', scope, 'Missing outward-speed edge term')

        def call(xy, vel):
            env = NS(cmd=NS(base_pos_xy=torch.tensor(xy, dtype=torch.float32)),
                     scene={'robot': NS(data=NS(root_lin_vel_w=torch.tensor(vel, dtype=torch.float32)))})
            return scope['boundary_outward_speed'](env, guard=guard)
        return call

    def test_the_interior_is_free_however_fast_the_robot_moves(self):
        f = self.fn()
        torch.testing.assert_close(f([[0., 0.]], [[5., 0.]]), torch.zeros(1))
        torch.testing.assert_close(f([[2., 1.]], [[0., -9.]]), torch.zeros(1))

    def test_holding_position_at_the_line_to_push_the_ball_in_is_free(self):
        """The whole point: scoring happens at the line, and must not be taxed."""
        f = self.fn()
        torch.testing.assert_close(f([[6.6, 0.]], [[0., 0.]]), torch.zeros(1))       # standing
        torch.testing.assert_close(f([[6.6, 0.]], [[-0.4, 0.]]), torch.zeros(1))     # stepping back in
        torch.testing.assert_close(f([[6.6, 0.]], [[0., 1.2]]), torch.zeros(1))      # sliding along the line

    def test_running_at_the_edge_is_charged_and_grows_with_speed(self):
        f = self.fn()
        slow, fast = (float(f([[6.6, 0.]], [[v, 0.]])) for v in (0.3, 1.5))
        self.assertGreater(slow, 0.0, 'outward motion at the edge must be charged')
        self.assertGreater(fast, slow)
        self.assertLessEqual(fast, 1.5)

    def test_all_four_edges_and_both_signs_are_covered(self):
        f = self.fn()
        cases = [(( 6.6, 0.), ( 1., 0.)), ((-6.6, 0.), (-1., 0.)),
                 ((0.,  4.2), (0.,  1.)), ((0., -4.2), (0., -1.))]
        for xy, vel in cases:
            with self.subTest(xy=xy, vel=vel):
                self.assertGreater(float(f([xy], [vel])), 0.0)
                inward = tuple(-v for v in vel)
                self.assertEqual(float(f([xy], [inward])), 0.0)

    def test_proximity_ramps_the_charge_to_zero_at_the_guard(self):
        f = self.fn()
        near, far = float(f([[6.9, 0.]], [[0.5, 0.]])), float(f([[7.0 - self.GUARD, 0.]], [[0.5, 0.]]))
        self.assertGreater(near, far)
        # float32: 7.0 - 6.4 lands ~1e-7 short of the guard, so allow dust here.
        self.assertLess(far, 1e-4, 'outside the guard the edge is irrelevant')
        self.assertGreater(near, 1e3 * max(far, 1e-9))

    def test_no_nan_anywhere_including_degenerate_positions(self):
        f = self.fn()
        xy = torch.tensor([[0., 0.], [7., 4.5], [-7., -4.5], [6.9, 0.]])
        vel = torch.tensor([[0., 0.], [3., 3.], [-3., -3.], [1., 1.]])
        out = f(xy, vel)
        self.assertEqual(out.shape, (4,))
        self.assertFalse(bool(torch.isnan(out).any()))
        self.assertTrue(bool(((out >= 0.0) & (out <= 10.0)).all()))
        zero_guard = self.fn(guard=0.0)
        self.assertEqual(float(zero_guard([[7., 0.]], [[1., 0.]])), 0.0)



class BallLateralSpeed(unittest.TestCase):
    """Charging the sideways ball speed is what the boundary terms could not do.

    Every remaining out-of-field episode at arm H2 is a ball sent past the goal
    line outside the posts (37 of 38 at iteration 6400). This term charges only
    the lateral component, so a hard goalward push stays free -- the reason it
    cannot recreate the "never approach the line" solution that made the boundary
    penalties unusable despite GOAL_X == FIELD_HALF_LENGTH.
    """

    def scope(self):
        s = functions(MDP / 'rewards.py', ['ball_lateral_speed'],
                      _cmd=lambda env: env.cmd,
                      GOAL_X=FIELD_HALF_LENGTH,
                      _feet_edge_pos_w=lambda env, cmd: env.edge)
        self.assertIn('ball_lateral_speed', s, 'Missing lateral ball-speed term')
        return s['ball_lateral_speed']

    def call(self, fn, ball_xy, ball_vy, edge_dist=0.1):
        n = len(ball_xy)
        cmd = NS(ball_pos_xy=torch.tensor(ball_xy, dtype=torch.float32),
                 ball=NS(data=NS(root_pos_w=torch.cat(
                     (torch.tensor(ball_xy, dtype=torch.float32), torch.zeros(n, 1)), -1),
                     root_lin_vel_w=torch.cat(
                         (torch.zeros(n, 1), torch.tensor(ball_vy, dtype=torch.float32).unsqueeze(-1),
                          torch.zeros(n, 1)), -1))))
        env = NS(cmd=cmd, num_envs=n, scene=NS(env_origins=torch.zeros(n, 3)))
        # foot corners: 2 feet, 4 corners each, all at edge_dist on the +x side of the ball
        env.edge = torch.tensor(ball_xy).reshape(n, 1, 1, 2).repeat(1, 2, 4, 1)
        env.edge[..., 0] += edge_dist
        return fn(env)

    def test_sideways_motion_costs_and_goalward_motion_is_free(self):
        fn = self.scope()
        near_goal = [[6.0, 0.0], [6.0, 0.0]]
        out = self.call(fn, near_goal, [0.0, 0.0])
        torch.testing.assert_close(out, torch.zeros(2))            # no lateral speed
        torch.testing.assert_close(self.call(fn, [[6.0, 0.0]], [1.5]), torch.tensor([1.5]))
        torch.testing.assert_close(self.call(fn, [[6.0, 0.0]], [-1.5]), torch.tensor([1.5]),
                                   msg='the sign of the sideways motion must not matter')

    def test_midfield_dribbling_is_untouched(self):
        """Only the shot is gated, so the rest of the task keeps its incentives."""
        fn = self.scope()
        torch.testing.assert_close(self.call(fn, [[1.0, 0.0]], [1.5]), torch.tensor([0.0]))

    def test_no_foot_on_the_ball_means_no_charge(self):
        fn = self.scope()
        torch.testing.assert_close(self.call(fn, [[6.0, 0.0]], [1.5], edge_dist=0.5), torch.tensor([0.0]))

    def test_speed_is_capped_so_one_bad_frame_cannot_dominate(self):
        fn = self.scope()
        self.assertEqual(float(self.call(fn, [[6.0, 0.0]], [99.0])), 2.0)


class BallLateralArguments(BoundaryArguments):
    flag = '--ball_lateral_weight'
    reward = 'ball_lateral'
    attribute = 'ball_lateral_weight'
