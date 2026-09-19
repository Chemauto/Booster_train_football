"""The soccer recovery control changes only normalized task policy weighting."""
from types import SimpleNamespace as NS
import unittest
import torch
from test_kick_amp_contracts import functions, ROOT, MDP


class TaskWeight(unittest.TestCase):
    def policy_weight(self, step=3000*24, floor=0., phase='soccer'):
        scope = functions(MDP/'commands.py', ['task_curriculum', 'task_policy_weight'], TASK_RAMP_STEPS=16000*24)
        self.assertIn('task_policy_weight', scope, 'Missing independent task policy weighting control')
        env = NS(common_step_counter=step, cfg=NS(commands=NS(soccer=NS(training_phase=phase, task_weight_floor=floor))))
        return scope, env

    def test_default_preserves_schedule_and_full_weight_does_not_remove_gait(self):
        scope,env=self.policy_weight()
        self.assertEqual(scope['task_policy_weight'](env),.1875)
        env.cfg.commands.soccer.task_weight_floor=1.
        self.assertEqual(scope['task_policy_weight'](env),1.)
        self.assertEqual(scope['task_curriculum'](env),.1875)

    def test_floor_does_not_reduce_later_schedule_or_enable_approach_task(self):
        scope,env=self.policy_weight(step=12000*24,floor=.5)
        self.assertEqual(scope['task_policy_weight'](env),.75)
        env.cfg.commands.soccer.training_phase='approach'
        self.assertEqual(scope['task_policy_weight'](env),0.)

    def test_invalid_floor_rejected(self):
        for value in [-.1,1.1,float('nan'),float('inf')]:
            with self.subTest(value=value):
                scope,env=self.policy_weight(floor=value)
                with self.assertRaises(ValueError): scope['task_policy_weight'](env)

    def test_runner_applies_weight_after_normalization_without_changing_auxiliary(self):
        cls=functions(ROOT/'source/booster_train/booster_train/rsl_rl/amp/runner.py',['AmpRunner'])['AmpRunner']
        r=cls.__new__(cls); r.device='cpu';r.cfg=NS(advantage_coef=(2.,1.));r.env=NS(extras={'task_weight':.1875})
        adv=torch.tensor([[1.,2.],[-1.,3.]])
        before=r._policy_advantages(adv)
        r.env.extras['task_weight']=1.
        after=r._policy_advantages(adv)
        torch.testing.assert_close(before,torch.tensor([2.375,2.625]))
        torch.testing.assert_close(after,torch.tensor([4.,1.]))


if __name__=='__main__': unittest.main()
