"""CPU-only contracts for first-episode evaluation (never imports Isaac Sim)."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_kick_amp.py"


class EvaluationContracts(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.exists(), "deterministic evaluation script is missing")
        spec = importlib.util.spec_from_file_location("kick_evaluation", SCRIPT)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def make_metrics(self, n=2):
        robot = np.tile([0., 0., .55], (n, 1))
        ball = np.tile([1., 0., .11], (n, 1))
        feet = np.tile([[[0., .1, .1], [0., -.1, .1]]], (n, 1, 1))
        return self.module.CohortMetrics(robot, ball, feet, dt=.1), robot, ball, feet

    def step(self, metrics, robot, ball, feet, terminated=None, timeout=None, fall=None):
        n = len(robot)
        metrics.update(robot, ball, feet, np.tile([3., 0., 0.], (n, 1)),
                       np.zeros(n, bool) if terminated is None else terminated,
                       np.zeros(n, bool) if timeout is None else timeout,
                       {"base_contact": np.zeros(n, bool) if fall is None else fall})

    def test_first_fall_is_cumulative_and_reset_cannot_resurrect_cohort(self):
        m, robot, ball, feet = self.make_metrics()
        self.step(m, robot, ball, feet, terminated=[True, False], fall=[True, False])
        robot[0, 0] = 100.  # auto-reset must never enter travel statistics
        self.step(m, robot, ball, feet)
        report = m.report()
        self.assertEqual(report["first_episode_survival_fraction"], .5)
        self.assertEqual(report["fall_count"], 1)
        self.assertEqual(report["per_env"][0]["time_to_first_fall_s"], .1)
        self.assertEqual(report["per_env"][0]["robot_path_length_m"], 0.)
        self.assertEqual(report["per_env"][0]["episode_duration_s"], .1)

    def test_timeout_is_censored_not_a_fall(self):
        m, robot, ball, feet = self.make_metrics()
        self.step(m, robot, ball, feet, timeout=[True, False])
        self.assertEqual(m.report()["fall_count"], 0)
        self.assertIsNone(m.report()["per_env"][0]["time_to_first_fall_s"])
        self.assertEqual(m.report()["per_env"][0]["end_reason"], "time_out")

    def test_ball_teleport_does_not_count_as_touch_or_goal(self):
        m, robot, ball, feet = self.make_metrics(1)
        feet[:] = ball[:, None, :]
        ball[:, 0] = 7.3
        self.step(m, robot, ball, feet)
        report = m.report()
        self.assertEqual(report["validated_goal_count"], 0)
        self.assertEqual(report["contact_proxy_count"], 0)
        self.assertEqual(report["ball_discontinuity_count"], 1)

    def test_goal_requires_contact_and_whole_ball_below_crossbar(self):
        m, robot, ball, feet = self.make_metrics(2)
        feet[0] = ball[0]
        # First step establishes a nearby foot; later rolling confirms displacement.
        self.step(m, robot, ball, feet)
        for x in np.arange(1.3, 7.31, .3):
            ball[:, 0] = x
            self.step(m, robot, ball, feet)
        report = m.report()
        self.assertEqual(report["validated_goal_count"], 1)
        self.assertEqual(report["contact_proxy_count"], 1)
        self.assertGreater(report["per_env"][0]["ball_displacement_after_contact_m"], 6.)
        self.assertEqual(report["per_env"][1]["ball_displacement_after_contact_m"], 0.)

    def test_scenario_covers_yaw_pi_and_relative_ball_range(self):
        poses = self.module.scenario_layout(64, "soccer", 123)
        self.assertTrue(np.any(np.isclose(np.abs(poses["yaw"]), np.pi)))
        distances = np.linalg.norm(poses["ball_xy"], axis=-1)
        self.assertTrue(np.all((distances >= .8 - 1e-8) & (distances <= 2. + 1e-8)))
        np.testing.assert_equal(poses["ball_xy"], self.module.scenario_layout(64, "soccer", 123)["ball_xy"])


if __name__ == "__main__":
    unittest.main()
