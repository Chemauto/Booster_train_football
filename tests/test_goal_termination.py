"""Scoring must end the episode, and must not be charged the failure penalty.

GOAL_X equals FIELD_HALF_LENGTH, so pushing the ball in takes the robot to the
field edge. Measured at iteration 4800: 137 accepted goals, of which 125 (95%)
ended with the robot out of field, and 225 of 256 episodes ended out of field at
all. The acceptance bars "goals >= 30%" and "out of field <= 10%" therefore
cannot both hold while a scored episode keeps running.

The paper's GOAL_SUCCESS_STEPS (=50 consecutive steps in the goal) is documented
as "=> episode success" but only resets the ball; nothing ends the episode, and
the 50-step window effectively never completes because the robot keeps playing
and knocks the ball back out. Two things are needed and are tested here:

  1. a goal DoneTerm whose geometry is the acceptance test's geometry, and
  2. an exception in the `termination` reward, because that term charges -1000
     and fires for every non-timeout DoneTerm -- including this one.
"""
from types import SimpleNamespace as NS
import ast
import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_kick_amp_contracts import functions, MDP  # noqa: E402

COMMANDS = MDP / 'commands.py'
REWARDS = MDP / 'rewards.py'
EVAL = Path(__file__).resolve().parents[1] / 'scripts/evaluate_kick_amp.py'
ENV_CFG = Path(__file__).resolve().parents[1] / (
    'source/booster_train/booster_train/tasks/manager_based/kick_amp/robots/k1/kick_amp/env_cfg.py')


def constants(path, names):
    tree = ast.parse(path.read_text())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                found[target.id] = ast.literal_eval(node.value)
    assert set(names) <= set(found), f'missing {sorted(set(names) - set(found))} in {path.name}'
    return found


class AcceptanceGeometryMatch(unittest.TestCase):
    """Training success and the acceptance test must mean the same thing."""

    def goal_geometry(self):
        return constants(COMMANDS, ('GOAL_X', 'GOAL_HALF_WIDTH', 'GOAL_HEIGHT', 'BALL_RADIUS'))

    def test_declared_constants_match_the_evaluation(self):
        tree = ast.parse(EVAL.read_text())
        found = dict((n.targets[0].id, ast.literal_eval(n.value)) for n in tree.body
                     if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                     and n.targets[0].id in ('GOAL_LINE_M', 'GOAL_HALF_MOUTH_M'))
        self.assertIn('GOAL_LINE_M', found, 'the evaluation moved its goal test; re-derive this contract')
        self.assertIn('GOAL_HALF_MOUTH_M', found)
        g = self.goal_geometry()
        self.assertAlmostEqual(found['GOAL_LINE_M'], g['GOAL_X'] + g['BALL_RADIUS'], places=6)
        self.assertAlmostEqual(found['GOAL_HALF_MOUTH_M'], g['GOAL_HALF_WIDTH'], places=6)

    def test_crossbar_height_matches_the_evaluation_literal(self):
        """The evaluation hardcodes 1.8; keep both in step."""
        source = EVAL.read_text()
        self.assertIn('at_line[:, 2] + .11 < 1.8', source)
        self.assertAlmostEqual(self.goal_geometry()['GOAL_HEIGHT'], 1.8)

    def test_ball_in_goal_geometry(self):
        g = self.goal_geometry()
        scope = functions(COMMANDS, ['ball_in_goal'], **g)
        fn = scope['ball_in_goal']
        inside = torch.tensor([[g['GOAL_X'] + g['BALL_RADIUS'] + 1e-3, 0.0]])
        shallow = torch.tensor([[g['GOAL_X'] + 1e-3, 0.0]])
        wide = torch.tensor([[g['GOAL_X'] + 0.5, g['GOAL_HALF_WIDTH']]])
        low = torch.tensor([g['BALL_RADIUS']])
        high = torch.tensor([g['GOAL_HEIGHT']])
        self.assertTrue(bool(fn(inside, low)[0]), 'ball past the line, in the mouth, on the ground')
        self.assertFalse(bool(fn(shallow, low)[0]), 'the whole ball must be past the line')
        self.assertFalse(bool(fn(wide, low)[0]), 'outside the posts is not a goal')
        self.assertFalse(bool(fn(inside, high)[0]), 'above the bar is not a goal')
        self.assertEqual(fn(inside, low).dtype, torch.bool)


class GoalTermination(unittest.TestCase):
    def termination_cfg(self):
        tree = ast.parse(ENV_CFG.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TerminationCfg')
        fields = {}
        for node in cls.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                fields[node.targets[0].id] = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                fields[node.target.id] = node.value
        return fields

    def test_goal_is_registered_and_not_a_time_out(self):
        """time_out=True would hide the cause from the evaluation's cause list."""
        fields = self.termination_cfg()
        self.assertIn('goal', fields, 'nothing ends the episode when the ball goes in')
        call = fields['goal']
        self.assertIsInstance(call, ast.Call)
        self.assertEqual(call.func.id, 'DoneTerm')
        keywords = {kw.arg: kw.value for kw in call.keywords}
        self.assertNotIn('time_out', keywords, 'a goal must be a real terminal state, and the '
                                               'evaluation labels episodes from the cause list')
        self.assertEqual(ast.unparse(keywords['func']), 'ball_scored_goal')

    def test_the_terminal_state_function_uses_the_pre_reset_cache(self):
        """Reading the live ball position loses a goal to its own ball reset.

        Detecting a goal triggers a ball-only teleport inside the same command
        update, so any reader sampling afterwards sees the ball at a random spot.
        The command therefore caches the strict test before that teleport, and the
        termination must read the cache.
        """
        tree = ast.parse(ENV_CFG.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'ball_scored_goal')
        body = ast.unparse(fn)
        self.assertIn('ball_in_goal_now', body)
        self.assertNotIn('ball_pos_xy', body, 'the live position is post-teleport by then')

    def test_the_command_caches_the_goal_before_resetting_the_ball(self):
        source = COMMANDS.read_text()
        cache = source.index('self.ball_in_goal_now[:] = ball_in_goal(')
        teleport = source.index('self._random_ball_reset(reset_ball_ids)')
        self.assertLess(cache, teleport,
                        'the goal must be sampled before the reset it triggers erases it')


class TerminationRewardExcludesGoals(unittest.TestCase):
    """-1000 for scoring would teach the policy to avoid the goal line."""

    def fn(self):
        scope = functions(REWARDS, ['termination'])
        return scope['termination']

    def manager(self, active_terms, terminated, goal):
        return NS(active_terms=list(active_terms),
                  terminated=torch.tensor(terminated, dtype=torch.bool),
                  get_term=lambda name: torch.tensor(goal, dtype=torch.bool))

    def test_a_scored_goal_costs_nothing(self):
        env = NS(termination_manager=self.manager(['goal', 'out_of_field'], [1., 1., 0.], [1., 0., 0.]))
        torch.testing.assert_close(self.fn()(env), torch.tensor([0., 1., 0.]))

    def test_falls_and_field_exits_are_still_charged(self):
        env = NS(termination_manager=self.manager(['goal', 'base_contact'], [0., 1., 1.], [0., 0., 0.]))
        torch.testing.assert_close(self.fn()(env), torch.tensor([0., 1., 1.]))

    def test_missing_goal_term_is_not_an_error(self):
        """Earlier runs and the approach phase have no goal termination."""
        env = NS(termination_manager=self.manager(['base_contact'], [1., 0.], [0., 0.]))
        torch.testing.assert_close(self.fn()(env), torch.tensor([1., 0.]))


class SuccessMetric(unittest.TestCase):
    def test_success_uses_the_same_predicate_as_the_goal(self):
        """Otherwise the training success rate reads ~0 once goals end episodes."""
        body = COMMANDS.read_text()
        # The cache is fed by the acceptance geometry, and both readers use the
        # cache (the live position is post-teleport by then).
        self.assertIn('self.ball_in_goal_now[:] = ball_in_goal(', body)
        self.assertIn('env.extras["success"] = self.ball_in_goal_now.float()', body)
        self.assertIn('self.metrics["success"][:] = self.ball_in_goal_now.float()', body)
        self.assertNotIn('env.extras["success"] = self.goal_success_now', body)


class EvaluationReporting(unittest.TestCase):
    """A scored episode must not be counted as "did not survive"."""

    def metrics(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('evaluate_kick_amp', EVAL)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_goal_cause_is_recorded_and_survival_view_kept_separate(self):
        module = self.metrics()
        n = 4
        import numpy as np
        robot = np.tile(np.array([0., 0., .57]), (n, 1))
        m = module.CohortMetrics(robot, np.tile(np.array([1., 0., .11]), (n, 1)), np.zeros((n, 2, 3)), .02)
        causes = {'goal': np.array([True, False, False, False]),
                  'out_of_field': np.array([False, True, False, False])}
        m.update(robot, np.tile(np.array([1., 0., .11]), (n, 1)), np.zeros((n, 2, 3)), np.zeros((n, 3)),
                 np.array([True, True, False, False]), np.array([False, False, False, False]), causes)
        report = m.report()
        self.assertEqual(report['goal_terminated_count'], 1)
        self.assertEqual(report['termination_causes']['goal'], 1)
        self.assertEqual(report['per_env'][0]['ended_by_goal'], True)
        self.assertEqual(report['per_env'][1]['ended_by_goal'], False)
        self.assertEqual(report['per_env'][1]['end_reason'], 'out_of_field')
        # env 0 ended by scoring, envs 2-3 are still playing: three of four are
        # "fine" under the success-aware view, but only two survived to the end.
        self.assertAlmostEqual(report['first_episode_success_or_survival_fraction'], 0.75)
        self.assertAlmostEqual(report['first_episode_survival_fraction'], 0.5)


if __name__ == '__main__':
    unittest.main()
