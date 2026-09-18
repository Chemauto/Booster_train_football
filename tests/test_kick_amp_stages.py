"""Exercise orchestration with real CPU child processes instead of Isaac Sim."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_kick_amp_stages as stages


class StageExecution(unittest.TestCase):
    def run_case(self, root, trainer, phase="soccer", passing=False, extra=()):
        (root / "scripts/rsl_rl").mkdir(parents=True)
        (root / "scripts/rsl_rl/train_kick_amp.py").write_text(trainer)
        (root / "initial.pt").touch()
        evaluator = """import sys,json
from pathlib import Path
report={'cohort_size':1,'first_episode_survival_fraction':1.,'fall_count':0,'contact_proxy_count':0,'validated_goal_count':0,'per_env':[{'robot_displacement_m':.1,'robot_path_length_m':.2,'approach_progress_m':.05}]}
Path(sys.argv[sys.argv.index('--output')+1]).write_text(json.dumps(report))
"""
        if passing:
            evaluator = evaluator.replace(".1,'robot_path_length_m':.2", "1.,'robot_path_length_m':1.2")
            evaluator = evaluator.replace("'approach_progress_m':.05", "'approach_progress_m':1.,'minimum_robot_ball_distance_m':.4,'alternating_support_switches':6,'survived_first_episode':True")
        (root / "scripts/evaluate_kick_amp.py").write_text(evaluator)
        argv = ["stages", "--checkpoint", str(root / "initial.pt"), "--output", str(root / "reports"), "--stages", "2", "--updates", "4"]
        argv += ["--training_phase", phase, "--reset_optimization"]
        argv += list(extra)
        with patch.object(stages, "ROOT", root), patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            stages.main()
        return json.loads((root / "reports/status.json").read_text())

    def test_default_profile_preserves_training_randomization_and_perception(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'motions'; data.mkdir(); (data / 'clip.npz').write_bytes(b'fixture')
            self.run_case(root, """import sys
from pathlib import Path
for flag in ('--nominal_physics', '--perfect_perception', '--near_ball'):
    assert flag not in sys.argv, sys.argv
assert '--motion_dir' in sys.argv
p=Path('logs/run'); p.mkdir(parents=True,exist_ok=True)
(p/'model_104.pt').touch()
print('[train] log dir: logs/run')
""", extra=('--training_profile', 'default', '--motion_dir', str(data)))
            report = json.loads((root / 'reports/status.json').read_text())
            self.assertEqual(report['training_profile'], 'default')
            self.assertNotIn('--perfect_perception', report['command'])

    def test_uses_final_checkpoint_and_evaluates_each_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = self.run_case(Path(tmp), """from pathlib import Path
p=Path('logs/run'); p.mkdir(parents=True,exist_ok=True)
(p/'model_100.pt').touch(); (p/'model_104.pt').touch()
print('[train] log dir: logs/run')
""")
        self.assertEqual(state['status'], 'stages_finished_requires_review')
        self.assertEqual(len(state['stages']), 2)
        self.assertTrue(all(s['checkpoint'].endswith('model_104.pt') for s in state['stages']))
        self.assertEqual(state['stages'][0]['mean_robot_displacement_m'], .1)

    def test_approach_requires_confirmation_before_stopping_for_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            state=self.run_case(root, """from pathlib import Path
p=Path('logs/run'); p.mkdir(parents=True,exist_ok=True)
(p/'model_104.pt').touch()
print('[train] log dir: logs/run')
""", phase="approach", passing=True)
            self.assertTrue((root/'reports/stage_1_confirmation.json').exists())
        self.assertEqual(state['status'],'approach_gate_passed_requires_soccer_training')
        self.assertEqual(len(state['stages']),1)
        self.assertTrue(state['stages'][0]['confirmation_gate']['passed'])

    def test_failed_training_stops_without_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, 'exited 7'):
                self.run_case(root, 'raise SystemExit(7)')
            state = json.loads((root / 'reports/status.json').read_text())
            self.assertEqual(state['status'], 'failed')
            self.assertEqual(state['stages'], [])
            self.assertFalse((root / 'reports/stage_1_evaluation.log').exists())


if __name__ == '__main__':
    unittest.main()
