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


SUCCESSFUL_TRAINER = """import sys,json
from pathlib import Path
source=Path(sys.argv[sys.argv.index('--checkpoint')+1])
start=json.loads(source.read_text())['iter']
updates=int(sys.argv[sys.argv.index('--max_iterations')+1])
end=start+updates
p=Path('logs')/str(end); p.mkdir(parents=True,exist_ok=True)
checkpoint=p/f'model_{end}.pt'
checkpoint.write_text(json.dumps({'iter':end}))
(p/'model_9999.pt').write_text(json.dumps({'iter':0}))
print(f'[train] log dir: {p}')
if '--completion_file' in sys.argv:
    result={'status':'completed','start_iteration':start,'requested_updates':updates,
            'requested_end_iteration':end,'end_iteration':end,'completed_updates':updates,
            'checkpoint':str(checkpoint.resolve())}
    Path(sys.argv[sys.argv.index('--completion_file')+1]).write_text(json.dumps(result))
"""


class StageExecution(unittest.TestCase):
    def run_case(self, root, trainer, phase="soccer", passing=False, extra=(), reset=True):
        (root / "scripts/rsl_rl").mkdir(parents=True)
        (root / "scripts/rsl_rl/train_kick_amp.py").write_text(trainer)
        (root / "initial.pt").write_text(json.dumps({'iter':100}))
        evaluator = """import sys,json
from pathlib import Path
checkpoint=Path(sys.argv[sys.argv.index('--checkpoint')+1])

report={'cohort_size':1,'first_episode_survival_fraction':1.,'fall_count':0,'contact_proxy_count':0,'validated_goal_count':0,'per_env':[{'robot_displacement_m':.1,'robot_path_length_m':.2,'approach_progress_m':.05}]}
Path(sys.argv[sys.argv.index('--output')+1]).write_text(json.dumps(report))
"""
        if passing:
            evaluator = evaluator.replace(".1,'robot_path_length_m':.2", "1.,'robot_path_length_m':1.2")
            evaluator = evaluator.replace("'approach_progress_m':.05", "'approach_progress_m':1.,'minimum_robot_ball_distance_m':.4,'alternating_support_switches':6,'survived_first_episode':True")
        (root / "scripts/evaluate_kick_amp.py").write_text(evaluator)
        argv = ["stages", "--checkpoint", str(root / "initial.pt"), "--output", str(root / "reports"), "--stages", "2", "--updates", "4"]
        argv += ["--training_phase", phase]
        if reset:
            argv += ["--reset_optimization"]
        argv += list(extra)
        with patch.object(stages, "ROOT", root), patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            stages.main()
        return json.loads((root / "reports/status.json").read_text())

    def test_default_profile_preserves_training_randomization_and_perception(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'motions'; data.mkdir(); (data / 'clip.npz').write_bytes(b'fixture')
            self.run_case(root, SUCCESSFUL_TRAINER + """
for flag in ('--nominal_physics', '--perfect_perception', '--near_ball'):
    assert flag not in sys.argv, sys.argv
assert '--motion_dir' in sys.argv
""", extra=('--training_profile', 'default', '--motion_dir', str(data)))
            report = json.loads((root / 'reports/status.json').read_text())
            self.assertEqual(report['training_profile'], 'default')
            self.assertNotIn('--perfect_perception', report['command'])

    def test_task_weight_floor_is_forwarded_without_optimization_reset(self):
        trainer = SUCCESSFUL_TRAINER + """
assert float(sys.argv[sys.argv.index('--task_weight_floor')+1]) == .2
assert '--reset_optimization' not in sys.argv
"""
        with tempfile.TemporaryDirectory() as tmp:
            state = self.run_case(Path(tmp), trainer, extra=('--task_weight_floor', '.2'), reset=False)
            self.assertEqual(state['task_weight_floor'], .2)

    def test_invalid_task_weight_floor_is_rejected_before_launch(self):
        for value in ('-.1', '1.1', 'nan', 'inf'):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                    self.run_case(root, SUCCESSFUL_TRAINER, extra=('--task_weight_floor', value))
                self.assertEqual(exc.exception.code, 2)
                self.assertFalse((root / 'reports').exists())

    def test_uses_reported_checkpoint_and_evaluates_each_completed_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = self.run_case(Path(tmp), SUCCESSFUL_TRAINER)
        self.assertEqual(state['status'], 'stages_finished_requires_review')
        self.assertEqual(len(state['stages']), 2)
        self.assertTrue(state['stages'][0]['checkpoint'].endswith('model_104.pt'))
        self.assertTrue(state['stages'][1]['checkpoint'].endswith('model_108.pt'))
        self.assertEqual(state['stages'][0]['training']['completed_updates'], 4)
        self.assertEqual(state['stages'][1]['training']['start_iteration'], 104)
        self.assertEqual(state['stages'][0]['mean_robot_displacement_m'], .1)

    def test_approach_requires_confirmation_before_stopping_for_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            state=self.run_case(root, SUCCESSFUL_TRAINER, phase="approach", passing=True)
            self.assertTrue((root/'reports/stage_1_confirmation.json').exists())
        self.assertEqual(state['status'],'approach_gate_passed_requires_soccer_training')
        self.assertEqual(len(state['stages']),1)
        self.assertTrue(state['stages'][0]['confirmation_gate']['passed'])

    def assert_no_evaluation(self, root):
        state = json.loads((root / 'reports/status.json').read_text())
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(state['stages'], [])
        self.assertFalse((root / 'reports/stage_1_evaluation.log').exists())
        self.assertFalse((root / 'reports/stage_2_train.log').exists())

    def test_failed_training_stops_without_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, 'exited 7'):
                self.run_case(root, 'raise SystemExit(7)')
            self.assert_no_evaluation(root)

    def test_zero_exit_partial_checkpoint_without_completion_is_rejected(self):
        trainer = """from pathlib import Path
p=Path('logs/run'); p.mkdir(parents=True)
(p/'model_101.pt').write_text('{"iter":101}')
print('[train] log dir: logs/run')
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, 'completion'):
                self.run_case(root, trainer)
            self.assert_no_evaluation(root)

    def test_incomplete_counter_is_rejected_even_with_completed_label(self):
        trainer = SUCCESSFUL_TRAINER + """
if '--completion_file' in sys.argv:
    result.update(end_iteration=start+1,completed_updates=1)
    Path(sys.argv[sys.argv.index('--completion_file')+1]).write_text(json.dumps(result))
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                self.run_case(root, trainer)
            self.assert_no_evaluation(root)

    def test_child_graceful_stop_is_not_promoted_to_completed(self):
        trainer = SUCCESSFUL_TRAINER + """
if '--completion_file' in sys.argv:
    result.update(status='stopped')
    Path(sys.argv[sys.argv.index('--completion_file')+1]).write_text(json.dumps(result))
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                self.run_case(root, trainer)
            self.assert_no_evaluation(root)

    def test_controller_stop_preserves_graceful_stop_semantics(self):
        trainer = """import os,signal
signal.signal(signal.SIGINT, lambda *_: None)
os.kill(os.getppid(),signal.SIGTERM)
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self.run_case(root, trainer)
            self.assertEqual(state['status'], 'stopped')
            self.assertEqual(state['stages'], [])
            self.assertFalse((root / 'reports/stage_1_evaluation.log').exists())


if __name__ == '__main__':
    unittest.main()
