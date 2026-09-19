"""Training must not sample a narrower geometry than the protocol it is graded on.

`--near_ball` places the ball 0.8-2 m ahead of the robot within +-0.8 rad of its
heading (commands.py::_random_ball_reset). The acceptance protocol places it
0.8-2 m away at 0/+-90/180 deg from that heading
(evaluate_kick_amp.py::scenario_layout), so half of every graded cohort tests a
geometry the policy never trained on. Measured consequence at iteration 4800:
34 of 39 first-episode falls sat in the "ball behind" branch, where the goal rate
was lowest, while the rolling training fall rate stayed at 0.6%.

These tests drive the real reset sampler rather than restating its formula: the
method is lifted out of the class and bound back onto a stub, so spawn_bearing()
is exercised as written.
"""
from types import SimpleNamespace as NS
import argparse
import ast
import contextlib
import io
import math
from unittest.mock import patch
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_kick_amp_contracts import functions, MDP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = MDP / 'commands.py'
STAGES = ROOT / 'scripts/run_kick_amp_stages.py'
TRAIN = ROOT / 'scripts/rsl_rl/train_kick_amp.py'
CLASS = 'SoccerStateCommand'


def module_constants(path, names):
    """Read module-level numeric constants by name."""
    tree = ast.parse(path.read_text())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                found[target.id] = ast.literal_eval(node.value)
    missing = set(names) - set(found)
    assert not missing, f'missing constants in {path.name}: {sorted(missing)}'
    return found


def lift_methods(path, class_name, method_names, **scope):
    """Bind the named methods of a class onto a fresh stub class.

    Lifting the real FunctionDef nodes keeps the method bodies under test; the
    alternative (re-implementing spawn_bearing in the test) would pass even if
    the production code ignored the new field.
    """
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in method_names]
    assert len(nodes) == len(method_names), 'method not found'
    module = ast.Module(body=[
        ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes,
        type_ignores=[])
    namespace = dict(torch=torch, **scope)
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    stub = type('Stub', (), {})
    for name in method_names:
        setattr(stub, name, namespace[name])
    return stub


class SpawnSampler(unittest.TestCase):
    RADIUS = (0.8, 2.0)
    N = 512

    def sample(self, bearing, seed=0):
        constants = module_constants(COMMANDS, ('FIELD_HALF_LENGTH', 'FIELD_HALF_WIDTH',
                                                'GOAL_X', 'GOAL_HALF_WIDTH', 'BALL_RADIUS'))
        geometry = functions(MDP / 'geometry.py', ['yaw_from_quat'])
        stub = lift_methods(COMMANDS, CLASS, ['spawn_bearing', '_random_ball_reset'],
                            yaw_from_quat=geometry['yaw_from_quat'], **constants)
        torch.manual_seed(seed)
        n = self.N
        captured = {}

        cmd = stub()
        cmd.device = 'cpu'
        cmd.cfg = NS(ball_spawn_distance=self.RADIUS, ball_spawn_bearing=bearing)
        cmd.robot = NS(data=NS(root_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1)))
        cmd.base_pos_xy = torch.zeros(n, 2)
        cmd.ball = NS(data=NS(default_root_state=torch.zeros(n, 13)))
        cmd.ball.write_root_state_to_sim = lambda root, env_ids=None: captured.update(root=root)
        cmd._env = NS(scene=NS(env_origins=torch.zeros(n, 3)))

        cmd._random_ball_reset(torch.arange(n))
        root = captured['root'][:, :2]
        assert torch.isfinite(root).all(), 'reset wrote non-finite coordinates'
        bearing_rad = torch.atan2(root[:, 1], root[:, 0])
        return bearing_rad, torch.norm(root, dim=-1)

    def test_default_cone_never_puts_the_ball_beside_or_behind(self):
        """This is the behaviour the widening exists to change."""
        angles, radii = self.sample(None)
        self.assertLessEqual(float(angles.abs().max()), 0.81)
        self.assertFalse(bool((angles.abs() > math.pi / 2).any()))
        self.assertGreaterEqual(float(radii.min()), 0.75)
        self.assertLessEqual(float(radii.max()), 2.05)

    def test_full_circle_trains_on_the_graded_bearings(self):
        angles, radii = self.sample((-math.pi, math.pi))
        behind = (angles.abs() > math.pi / 2).float().mean()
        # The protocol is half behind/beside; a uniform circle must sample that
        # often, and all four quadrants must be reachable at all.
        self.assertGreater(float(behind), 0.35, 'the "ball behind" branch is still untrained')
        self.assertLess(float(behind), 0.65)
        quadrants = {int(math.floor((float(a) + math.pi) / (math.pi / 2))) for a in angles}
        self.assertEqual(quadrants, {0, 1, 2, 3}, 'a quadrant is never sampled')
        self.assertGreaterEqual(float(radii.min()), 0.75)
        self.assertLessEqual(float(radii.max()), 2.05)

    def test_default_cone_is_unchanged_by_the_new_field(self):
        """The config default must keep every earlier run reproducible."""
        constants = module_constants(COMMANDS, ('FIELD_HALF_LENGTH', 'FIELD_HALF_WIDTH',
                                                'GOAL_X', 'GOAL_HALF_WIDTH', 'BALL_RADIUS'))
        geometry = functions(MDP / 'geometry.py', ['yaw_from_quat'])
        stub = lift_methods(COMMANDS, CLASS, ['spawn_bearing', '_random_ball_reset'],
                            yaw_from_quat=geometry['yaw_from_quat'], **constants)
        self.assertEqual(stub.spawn_bearing(NS(cfg=NS(ball_spawn_bearing=None))), (-0.8, 0.8))
        self.assertEqual(stub.spawn_bearing(NS(cfg=NS(ball_spawn_bearing=(-math.pi, math.pi)))),
                         (-math.pi, math.pi))

    def test_config_default_is_the_narrow_cone(self):
        tree = ast.parse(COMMANDS.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name.endswith('Cfg'))
        fields = {n.target.id: n.value for n in cls.body if isinstance(n, ast.AnnAssign)}
        self.assertIn('ball_spawn_bearing', fields)
        self.assertIsInstance(fields['ball_spawn_bearing'], ast.Constant)
        self.assertIsNone(fields['ball_spawn_bearing'].value)


class SpawnArguments(unittest.TestCase):
    """The flag must be validated, reach the config, and survive the runner."""

    def parse(self, value=None):
        tree = ast.parse(TRAIN.read_text())
        start = next(i for i, n in enumerate(tree.body)
                     if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'parser')
        end = next(i for i, n in enumerate(tree.body)
                   if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'app_launcher')
        scope = {'argparse': argparse, 'math': math,
                 'AppLauncher': NS(add_app_launcher_args=lambda _: None)}
        argv = ['train'] + ([] if value is None else ['--ball_spawn_bearing_deg', value])
        with patch.object(sys, 'argv', argv):
            exec(compile(ast.Module(body=tree.body[start:end], type_ignores=[]), str(TRAIN), 'exec'), scope)
        return scope['args_cli']

    def test_defaults_to_the_configured_cone(self):
        self.assertIsNone(self.parse().ball_spawn_bearing_deg)

    def test_accepts_a_full_circle_and_rejects_nonsense(self):
        for value in ('180', '90', '45.5'):
            self.assertEqual(self.parse(value).ball_spawn_bearing_deg, float(value))
        for value in ('0', '-180', '181', 'nan', 'inf'):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as exc:
                self.parse(value)
            self.assertEqual(exc.exception.code, 2)

    def test_reaches_the_command_configuration_through_radians(self):
        """Exec the whole `if` block so the degrees-to-radians conversion is under test."""
        tree = ast.parse(TRAIN.read_text())
        blocks = [n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and 'ball_spawn_bearing_deg' in ast.unparse(n.test)
                  and 'env_cfg.commands.soccer.ball_spawn_bearing' in ast.unparse(n)]
        self.assertEqual(len(blocks), 1, 'expected exactly one config block for the flag')
        env_cfg = NS(commands=NS(soccer=NS(ball_spawn_bearing=None)))
        exec(compile(ast.Module(body=blocks[0].body, type_ignores=[]), str(TRAIN), 'exec'),
             {'env_cfg': env_cfg, 'args_cli': NS(ball_spawn_bearing_deg=180.), 'math': math})
        low, high = env_cfg.commands.soccer.ball_spawn_bearing
        self.assertAlmostEqual(low, -math.pi)
        self.assertAlmostEqual(high, math.pi)
        # and a partial widening lands where asked
        env_cfg2 = NS(commands=NS(soccer=NS(ball_spawn_bearing=None)))
        exec(compile(ast.Module(body=blocks[0].body, type_ignores=[]), str(TRAIN), 'exec'),
             {'env_cfg': env_cfg2, 'args_cli': NS(ball_spawn_bearing_deg=90.), 'math': math})
        self.assertAlmostEqual(env_cfg2.commands.soccer.ball_spawn_bearing[1], math.pi / 2)

    def test_stages_runner_forwards_the_flag(self):
        src = STAGES.read_text()
        self.assertIn('"--ball_spawn_bearing_deg", str(args.ball_spawn_bearing_deg)', src)
        self.assertIn('"ball_spawn_bearing_deg": args.ball_spawn_bearing_deg', src)


if __name__ == '__main__':
    unittest.main()
