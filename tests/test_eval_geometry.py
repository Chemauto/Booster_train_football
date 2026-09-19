"""Out-of-field is a count; these fields turn it into a physical location.

The acceptance metric counts a goal as a one-step ball crossing of x = 7.11
inside the posts, so a 215/256 out-of-field figure cannot distinguish "the ball
went wide", "the ball never reached the line" or "the ball left the side". The
cohort now records where the ball and the robot finished, and the answers are
not what the counters suggested: the F arm's out-of-field episodes had the ball
past the goal line (final x median 8.54) with the robot following it out (7.92).

These tests pin the two things that make the fields trustworthy: the classifier,
and the fact that the snapshots are taken at the LAST ACTIVE step rather than
after the automatic reset teleports the scene back to the start state.
"""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / 'scripts/evaluate_kick_amp.py'


def load_eval():
    spec = importlib.util.spec_from_file_location('evaluate_kick_amp', EVAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BallEndZone(unittest.TestCase):
    def setUp(self):
        self.module = load_eval()

    def test_goal_mouth_requires_crossing_the_line_inside_the_posts(self):
        zone = self.module.ball_end_zone
        self.assertEqual(zone(7.5, 0.0), 'goal_mouth')
        self.assertEqual(zone(7.5, 1.29), 'goal_mouth')
        self.assertEqual(zone(7.5, -1.29), 'goal_mouth')

    def test_wide_and_side_and_on_field_are_distinguished(self):
        zone = self.module.ball_end_zone
        self.assertEqual(zone(7.5, 1.31), 'past_line_wide')
        self.assertEqual(zone(7.5, -4.9), 'past_line_wide', 'past the goal line, not around the side')
        self.assertEqual(zone(0.0, 5.0), 'side_out')
        self.assertEqual(zone(6.9, 0.0), 'on_field')
        self.assertEqual(zone(-8.0, 0.0), 'far_end_out')
        self.assertEqual(zone(-6.9, 4.4), 'on_field')

    def test_classifier_is_total_so_counts_always_reconstruct_the_cohort(self):
        zones = self.module.BALL_END_ZONES
        for x in (8.0, 7.0, 0.0, -7.0, -8.0):
            for y in (5.0, 4.4, 1.3, 0.0, -1.3, -4.4, -5.0):
                with self.subTest(x=x, y=y):
                    self.assertIn(self.module.ball_end_zone(x, y), zones)


class SnapshotSemantics(unittest.TestCase):
    """The recorded position must be the last live state, not the reset state."""

    def setUp(self):
        self.module = load_eval()
        self.dt = 0.02

    def metrics(self, robot, ball, feet):
        return self.module.CohortMetrics(robot, ball, feet, self.dt)

    def test_records_the_last_active_step_not_the_post_reset_teleport(self):
        robot = np.array([[0., 0., .57], [0., 0., .57]])
        ball = np.array([[1., 0., .11], [1., 0., .11]])
        feet = np.zeros((2, 2, 3))
        m = self.metrics(robot, ball, feet)

        # env 0 walks to the goal line and leaves the field; env 1 keeps playing.
        ball[0] = [8.2, 0.4, .11]
        robot[0] = [7.95, 0.3, .57]
        m.update(robot, ball, feet, np.zeros((2, 3)), np.array([True, False]), np.array([False, False]),
                 {'out_of_field': np.array([True, False])})
        # the environment resets: the scene teleports home
        ball[0] = [1., 0., .11]
        robot[0] = [0., 0., .57]
        m.update(robot, ball, feet, np.zeros((2, 3)), np.array([False, False]), np.array([False, False]), {})

        np.testing.assert_allclose(m.final_ball[0, :2], [8.2, 0.4])
        np.testing.assert_allclose(m.final_robot[0, :2], [7.95, 0.3])
        self.assertEqual(self.module.ball_end_zone(*m.final_ball[0, :2]), 'goal_mouth')

    def test_max_ball_x_keeps_the_furthest_reach_even_if_the_ball_comes_back(self):
        robot = np.zeros((1, 3), dtype=float); robot[:, 2] = .57
        ball = np.array([[0., 0., .11]])
        m = self.metrics(robot, ball, np.zeros((1, 2, 3)))
        for x in (2.0, 7.4, 3.0):
            ball[0, 0] = x
            m.update(robot, ball, np.zeros((1, 2, 3)), np.zeros((1, 3)),
                     np.array([False]), np.array([False]), {})
        self.assertEqual(m.max_ball_x[0], 7.4, 'the line was reached even though the ball returned')
        self.assertEqual(self.module.ball_end_zone(*m.final_ball[0, :2]), 'on_field')

    def test_report_counts_reconstruct_the_cohort(self):
        robot = np.zeros((4, 3), dtype=float); robot[:, 2] = .57
        ball = np.array([[8.2, 0., .11], [8.2, 3., .11], [0., 5., .11], [3., 0., .11]])
        m = self.metrics(robot, ball, np.zeros((4, 2, 3)))
        m.update(robot, ball, np.zeros((4, 2, 3)), np.zeros((4, 3)),
                 np.array([False] * 4), np.array([False] * 4), {})
        report = m.report()
        counts = report['ball_end_zone_counts']
        self.assertEqual(sum(counts.values()), report['cohort_size'])
        self.assertEqual(counts['goal_mouth'], 1)
        self.assertEqual(counts['past_line_wide'], 1)
        self.assertEqual(counts['side_out'], 1)
        self.assertEqual(counts['on_field'], 1)
        self.assertEqual(report['ball_reached_goal_line_count'], 2)
        self.assertEqual(report['per_env'][0]['ball_end_zone'], 'goal_mouth')
        self.assertEqual(len(report['per_env'][0]['final_ball_pos_m']), 2)


if __name__ == '__main__':
    unittest.main()


class BallLateralDiagnostics(unittest.TestCase):
    """The wide-miss mechanism must be measurable, not assumed.

    Two hypotheses for a shot that crosses the line outside the posts: the ball
    leaves the foot sideways (charge the contact) or it is pushed straight and
    drifts wide (charging the contact cannot help). These fields separate them.
    """

    def setUp(self):
        module = load_eval()
        self.module = module
        n = 2
        self.robot = np.tile(np.array([0., 0., .57]), (n, 1))
        self.ball = np.tile(np.array([1., 0., .11]), (n, 1))
        self.feet = np.zeros((n, 2, 3))
        self.ball = self.ball.copy()
        self.m = module.CohortMetrics(self.robot, self.ball, self.feet, .02)

    def step(self, ball, vel, feet=None):
        feet = self.feet if feet is None else feet
        self.m.update(self.robot, ball, feet, vel, np.array([False] * len(ball)),
                      np.array([False] * len(ball)), {})

    def test_contact_lateral_speed_only_counts_frames_with_a_foot_on_the_ball(self):
        ball = self.ball.copy()
        far_feet = np.full((2, 2, 3), 5.0)
        self.step(ball, np.array([[0., 1.5, 0.], [0., 0.9, 0.]]), far_feet)
        np.testing.assert_allclose(self.m.contact_lateral_speed, 0., atol=1e-9)
        near_feet = np.zeros((2, 2, 3))
        near_feet[:, :, 0] = 1.0
        self.step(ball, np.array([[0., 1.5, 0.], [0., 0.9, 0.]]), near_feet)
        np.testing.assert_allclose(self.m.contact_lateral_speed, [1.5, .9])

    def test_crossing_lateral_speed_and_y_are_recorded_at_the_line(self):
        ball = self.ball.copy()
        self.step(ball, np.zeros((2, 3)))
        ball[0] = [7.4, .8, .11]          # crosses the line inside the mouth region
        ball[1] = [8.0, 3.0, .11]         # crosses outside the posts
        self.step(ball, np.array([[1.0, .35, 0.], [1.0, 2.4, 0.]]))
        np.testing.assert_allclose(self.m.crossing_lateral_speed, [.35, 2.4])
        # y interpolated to the exact line crossing (x = 7.11), not the final y:
        # that is how wide the shot was when it mattered.
        alpha = (7.11 - 1.0) / 6.4
        np.testing.assert_allclose(self.m.crossing_y[0], alpha * .8, rtol=1e-6)
        np.testing.assert_allclose(self.m.crossing_y[1], (7.11 - 1.0) / 7.0 * 3.0, rtol=1e-6)
        report = self.m.report()
        self.assertAlmostEqual(report['per_env'][0]['crossing_lateral_speed_mps'], .35)
        self.assertAlmostEqual(report['per_env'][0]['crossing_y_m'], alpha * .8, places=6)

    def test_never_crossing_is_reported_as_absent_rather_than_zero(self):
        ball = self.ball.copy()
        self.step(ball, np.zeros((2, 3)))
        report = self.m.report()
        self.assertEqual(report['per_env'][0]['crossing_y_m'], None)
        self.assertEqual(report['per_env'][0]['crossing_lateral_speed_mps'], -1.0)
