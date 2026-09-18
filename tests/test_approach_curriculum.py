import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_kick_amp as evaluation


class ApproachContracts(unittest.TestCase):
    def test_gate_rejects_falling_and_motion_without_alternating_support(self):
        good = dict(survived_first_episode=True, robot_displacement_m=.9,
                    minimum_robot_ball_distance_m=.4, alternating_support_switches=6, ball_discontinuity=False)
        report = {"cohort_size": 10, "first_episode_survival_fraction": 1., "per_env": [dict(good) for _ in range(10)]}
        self.assertTrue(evaluation.assess_approach(report)["passed"])
        for row in report["per_env"]: row["alternating_support_switches"] = 0
        self.assertFalse(evaluation.assess_approach(report)["passed"])
        for row in report["per_env"]:
            row.update(good); row["survived_first_episode"] = False
        self.assertEqual(evaluation.assess_approach(report)["success_count"], 0)

    def test_approach_layout_rotates_target_with_robot_heading(self):
        layout = evaluation.scenario_layout(64, "approach", 123)
        xy, yaw = layout["ball_xy"], layout["yaw"]
        forward = xy[:, 0]*np.cos(yaw) + xy[:, 1]*np.sin(yaw)
        self.assertTrue((forward > .7).all())
        self.assertEqual(len(np.unique(yaw)), 4)

    def test_support_switch_counter_requires_stability_and_stops_after_fall(self):
        robot=np.array([[0.,0.,.55]]); ball=np.array([[2.,0.,.11]])
        feet=np.array([[[0.,.1,.04],[0.,-.1,.04]]]); velocity=np.zeros((1,3))
        metrics=evaluation.CohortMetrics(robot,ball,feet,.02)
        def step(contacts, fallen=False):
            metrics.update(robot,ball,feet,velocity,[fallen],[False],{"base_contact":[fallen]},[contacts])
        for contacts in ([True,False],[False,True])*5: step(contacts)
        self.assertEqual(int(metrics.support_switches[0]),0)
        for contacts in ([True,False],[False,True],[True,False],[False,True],[True,False]):
            for _ in range(3): step(contacts)
        self.assertEqual(int(metrics.support_switches[0]),4)
        step([True,False],True)
        for _ in range(4): step([False,True])
        self.assertEqual(int(metrics.support_switches[0]),4)

    def test_phase_keeps_safety_terms_and_uses_only_walks(self):
        path = ROOT / "source/booster_train/booster_train/tasks/manager_based/kick_amp/curriculum.py"
        spec = importlib.util.spec_from_file_location("curriculum", path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            walk = Path(tmp)/"walk"; walk.mkdir(); (walk/"clip.npz").touch()
            names = ["kick_ball","side_kick_ball","face_ball_pitch","face_ball_yaw","survival","termination","pos_still","track_lin_vel_ball","collision","dof_pos_limits"]
            rewards = NS(**{n:NS(weight=-100.,params={}) for n in names})
            cfg = NS(commands=NS(soccer=NS(motion_dir=tmp)),rewards=rewards)
            agent = NS()
            module.configure_training_phase(cfg,agent,"approach")
            self.assertEqual(cfg.commands.soccer.motion_dir,str(walk))
            self.assertEqual(cfg.commands.soccer.reset_motion_fraction,0.)
            self.assertEqual(rewards.side_kick_ball.weight,0.)
            self.assertEqual(rewards.collision.weight,-100.)
            self.assertEqual(agent.advantage_coef,(0.,1.))


if __name__ == "__main__": unittest.main()
