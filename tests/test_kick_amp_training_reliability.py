"""CPU checks for the exact training completion and history reuse code."""
import ast
import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / 'scripts/rsl_rl/train_kick_amp.py'
RUNNER = ROOT / 'source/booster_train/booster_train/rsl_rl/amp/runner.py'


class TrainingArguments(unittest.TestCase):
    def parse_arguments(self, value=None):
        tree = ast.parse(TRAIN.read_text())
        start = next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0]) == 'parser')
        end = next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0]) == 'app_launcher')
        scope = {'argparse':argparse, 'math':math, 'AppLauncher':NS(add_app_launcher_args=lambda _:None)}
        with patch.object(sys, 'argv', ['train'] + ([] if value is None else ['--task_weight_floor',value])):
            exec(compile(ast.Module(body=tree.body[start:end],type_ignores=[]),str(TRAIN),'exec'),scope)
        return scope['args_cli']

    def test_task_weight_floor_defaults_to_zero_and_accepts_unit_interval(self):
        self.assertEqual(self.parse_arguments().task_weight_floor, 0.)
        for value in ('0','.2','1'):
            self.assertEqual(self.parse_arguments(value).task_weight_floor, float(value))

    def test_task_weight_floor_rejects_nonfinite_and_out_of_range_before_simulator(self):
        for value in ('-.1','1.1','nan','inf'):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                self.parse_arguments(value)
            self.assertEqual(exc.exception.code, 2)

    def test_task_weight_floor_reaches_command_configuration(self):
        tree = ast.parse(TRAIN.read_text())
        nodes = [n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(
            ast.unparse(t) == 'env_cfg.commands.soccer.task_weight_floor' for t in n.targets)]
        self.assertEqual(len(nodes), 1)
        env_cfg = NS(commands=NS(soccer=NS(task_weight_floor=0.)))
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(TRAIN),'exec'),
             {'env_cfg':env_cfg,'args_cli':NS(task_weight_floor=.2)})
        self.assertEqual(env_cfg.commands.soccer.task_weight_floor,.2)


class TrainingCompletion(unittest.TestCase):
    def completion_function(self):
        tree = ast.parse(TRAIN.read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'learn_with_completion']
        self.assertEqual(len(nodes), 1, 'training must report the real runner counters after learn returns')
        scope = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TRAIN), 'exec'), scope)
        return scope['learn_with_completion']

    def run_training(self, root, updates, stopped=False, fail=False):
        marker = root / 'result.json'
        runner = NS(tot_iterations=1945, cfg=NS(max_iterations=500), log_dir=str(root), stop_requested=stopped)
        def learn():
            self.assertFalse(marker.exists(), 'completion must not be visible during training')
            runner.tot_iterations += updates
            if fail:
                raise RuntimeError('training failed')
            (root / f'model_{runner.tot_iterations}.pt').write_bytes(b'saved')
        runner.learn = learn
        self.completion_function()(runner, marker)
        return json.loads(marker.read_text())

    def test_full_run_records_actual_counters_and_saved_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.run_training(root, 500)
            self.assertEqual(result, {
                'status': 'completed', 'start_iteration': 1945, 'requested_updates': 500,
                'requested_end_iteration': 2445, 'end_iteration': 2445,
                'completed_updates': 500, 'checkpoint': str(root / 'model_2445.pt'),
            })

    def test_partial_graceful_run_keeps_actual_saved_iteration(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_training(Path(tmp), 55, stopped=True)
            self.assertEqual(result['status'], 'stopped')
            self.assertEqual(result['end_iteration'], 2000)
            self.assertEqual(result['requested_end_iteration'], 2445)
            self.assertEqual(result['completed_updates'], 55)

    def test_exception_never_writes_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, 'training failed'):
                self.run_training(root, 55, fail=True)
            self.assertFalse((root / 'result.json').exists())


class HistoryAllocation(unittest.TestCase):
    def test_mirroring_reuses_selected_batch_storage_and_preserves_values(self):
        # Execute the real selection and mirror statements, excluding Isaac Sim
        # initialization and unrelated PPO operations.
        tree = ast.parse(RUNNER.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AmpRunner')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_update')
        assignments = [n for n in ast.walk(method) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id in ('stacked_b', 'mirrored_stack_b') for t in n.targets)]
        self.assertEqual(len(assignments), 2)
        history = torch.arange(2*4*3*5).reshape(2,4,3,5).float()
        indices = torch.tensor([6,1,4])
        mirrored_inputs = []
        def mirror(batch, *_):
            mirrored_inputs.append(batch)
            return -batch
        scope = {'buf': {'stacked':history}, 'batch_idx':indices, 'mirror':mirror,
                 'self':NS(obs_dim=5,joint_names=[],device='cpu'), 'cfg':NS(num_stack=3)}
        exec(compile(ast.Module(body=assignments, type_ignores=[]), str(RUNNER), 'exec'), scope)
        self.assertEqual(mirrored_inputs[0].data_ptr(), scope['stacked_b'].data_ptr(),
                         'mirror must not allocate a second advanced-indexed history batch')
        torch.testing.assert_close(scope['mirrored_stack_b'], -history.flatten(0,1)[indices])
