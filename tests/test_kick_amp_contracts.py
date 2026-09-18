"""CPU regression checks on task functions, without starting Isaac Sim.

Load actual function ASTs to isolate pure tensor contracts from omni imports.
"""
import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
MDP = ROOT / 'source/booster_train/booster_train/tasks/manager_based/kick_amp/mdp'


def functions(path, names, **namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes, type_ignores=[])
    scope = {'torch': torch, **namespace}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), scope)
    return scope


class ActionBoundContracts(unittest.TestCase):
    def test_penalty_uses_radians_not_robot_specific_action_units(self):
        path = ROOT / 'source/booster_train/booster_train/rsl_rl/amp/runner.py'
        fn = functions(path, ['action_bound_loss'])['action_bound_loss']
        # Same physical target on T1 scale=1 and K1 ankle scale=.25.
        physical = torch.tensor([[.8, 1.2, -1.3]])
        expected = fn(physical, torch.ones(3), 1.)
        actual = fn(physical / .25, torch.full((3,), .25), 1.)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(fn(torch.tensor([[3.2]]), torch.tensor([.25]), 1.), torch.tensor(0.))
        self.assertGreater(float(expected), 0.)


class GeometryContracts(unittest.TestCase):
    def test_foot_yaw_is_heading_not_roll(self):
        yaw = torch.tensor([0., math.pi/2, math.pi, -math.pi/2])
        q = torch.stack((torch.cos(yaw/2), yaw*0, yaw*0, torch.sin(yaw/2)), -1)
        env = NS(num_envs=2, scene={'robot': NS(data=NS(body_quat_w=q.reshape(2,2,4)))})
        extra = {}
        geometry = MDP / 'geometry.py'
        if geometry.exists():
            extra.update(functions(geometry, ['yaw_from_quat']))
        f = functions(MDP/'rewards.py', ['_feet_yaw'], **extra)['_feet_yaw']
        actual = f(env, NS(feet_idx=[0,1])).flatten()
        torch.testing.assert_close(torch.cos(actual), torch.cos(yaw))
        torch.testing.assert_close(torch.sin(actual), torch.sin(yaw))

    def test_base_yaw_covers_full_circle(self):
        # Execute the exact state-to-heading assignment used by the command.
        tree = ast.parse((MDP/'commands.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='SoccerStateCommand')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name=='_update_command')
        assignment = next(n for n in ast.walk(method) if isinstance(n, ast.Assign) and any('self.base_yaw' in ast.unparse(t) for t in n.targets))
        yaw = torch.tensor([math.pi/2, math.pi, -math.pi/2, 2.5])
        root = torch.zeros(4,13); root[:,3]=torch.cos(yaw/2); root[:,6]=torch.sin(yaw/2)
        cmd=NS(base_yaw=torch.zeros(4))
        scope={'torch':torch,'self':cmd,'base_root':root}
        if (MDP/'geometry.py').exists(): scope.update(functions(MDP/'geometry.py',['yaw_from_quat']))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[assignment],type_ignores=[])),'command_yaw','exec'),scope)
        torch.testing.assert_close(torch.cos(cmd.base_yaw),torch.cos(yaw))
        torch.testing.assert_close(torch.sin(cmd.base_yaw),torch.sin(yaw))


class ContactContracts(unittest.TestCase):
    def test_sensor_order_independent_and_real_arm_collision_penalized(self):
        robot_names=['trunk','right_elbow_yaw_link','left_ankle_roll_link','right_ankle_roll_link']
        sensor_names=['left_ankle_roll_link','trunk','right_ankle_roll_link','right_elbow_yaw_link']
        forces=torch.zeros(2,4,3)
        forces[0,[0,2],2]=90  # standing on both feet, no penalty
        forces[1,3,2]=90      # actual elbow contact, must be penalized
        env=NS(scene={'robot':NS(body_names=robot_names),'contact_forces':NS(body_names=sensor_names,data=NS(net_forces_w=forces))})
        f=functions(MDP/'rewards.py',['collision'],_cmd=lambda _:NS(feet_idx=[2,3]),ARM_BODIES={'right_elbow_yaw_link'})['collision']
        torch.testing.assert_close(f(env),torch.tensor([0.,1.]))




class RunnerContracts(unittest.TestCase):
    def runner_class(self):
        import os, copy
        amp=ROOT/'source/booster_train/booster_train/rsl_rl/amp'
        mirrors=functions(amp/'mirror.py',['_mirror_partner','joint_mirror_matrix','obs_mirror_matrix','mirror','mirror_act'],NEGATIVE_TOKENS=('_roll_joint','_yaw_joint'),_OBS_MAT_CACHE={},_JOINT_MAT_CACHE={})
        return functions(amp/'runner.py',['AmpRunner'],os=os,copy=copy,mirror=mirrors['mirror'])['AmpRunner'], mirrors['mirror']

    def test_phase_reset_preserves_actor_and_discards_old_value_optimizer(self):
        import numpy as np
        amp=ROOT/'source/booster_train/booster_train/rsl_rl/amp'
        runner=functions(amp/'runner.py',['AmpRunner'],np=np)['AmpRunner']
        model_cls=functions(amp/'modules.py',['ActorCriticAMP'])['ActorCriticAMP']
        r=runner.__new__(runner);r.model=model_cls(2,2,1,2,1)
        r.optimizer=torch.optim.Adam(r.model.parameters(),lr=.001)
        actor={k:v.clone() for k,v in r.model.actor.state_dict().items()}
        critic={k:v.clone() for k,v in r.model.critics.state_dict().items()}
        previous_optimizer=r.optimizer
        r.reset_optimization_for_phase()
        for key,value in actor.items(): torch.testing.assert_close(value,r.model.actor.state_dict()[key])
        self.assertTrue(any(not torch.equal(v,r.model.critics.state_dict()[k]) for k,v in critic.items()))
        self.assertIsNot(r.optimizer,previous_optimizer)
        self.assertEqual(len(r.optimizer.state),0)
        torch.testing.assert_close(r.model._std(),torch.full((2,),.15))

    def normalizer(self, width):
        from rsl_rl.networks.normalization import EmpiricalNormalization
        return EmpiricalNormalization(width)

    def test_export_keeps_training_normalizer_updating(self):
        import tempfile
        cls,_=self.runner_class();r=cls.__new__(cls)
        class Model(torch.nn.Module):
            def forward(self, obs, history): return obs + history[:, -1]
        r.model=Model().train(); r.obs_norm=self.normalizer(2).train();r.obs_dim=2;r.device='cpu';r.cfg=NS(num_stack=2)
        with tempfile.TemporaryDirectory() as d: r.export(str(Path(d)/'policy.pt'))
        before=r.obs_norm.count.clone();r.obs_norm.update(torch.ones(3,2))
        self.assertTrue(r.model.training)
        self.assertEqual(int(r.obs_norm.count-before),3)

    def test_normalization_commutes_with_physical_mirror(self):
        cls,mirror=self.runner_class();r=cls.__new__(cls);r.device='cpu'
        r.joint_names=['left_hip_pitch_joint','right_hip_pitch_joint'];r.obs_dim=19;r.obs_norm=self.normalizer(19)
        # Deliberately asymmetric distribution (ball predominantly on one side).
        raw=torch.arange(19).float().repeat(8,1)+torch.randn(8,19)
        normalized=r._normalize_obs(raw, update=True)
        torch.testing.assert_close(mirror(normalized,r.joint_names,'cpu'),r.obs_norm(mirror(raw,r.joint_names,'cpu')))

    def test_decoder_target_is_current_state_before_step(self):
        # Check the executed collection block, not a duplicate formula: step
        # mutates critic_obs to a different state's privileged coordinates.
        tree=ast.parse((ROOT/'source/booster_train/booster_train/rsl_rl/amp/runner.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AmpRunner')
        learn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='learn')
        loop=next(n for n in ast.walk(learn) if isinstance(n,ast.For) and ast.unparse(n.target)=='n')
        assigns=[]
        for n in loop.body:
            if isinstance(n,ast.Assign):
                target=' '.join(ast.unparse(t) for t in n.targets)
                if 'buf[\'privileged\']' in target or 'buf["privileged"]' in target or target=='critic_obs': assigns.append(n)
        scope={'buf':{'privileged':torch.zeros(1,1,14)},'n':0,'critic_obs':torch.ones(1,93),'obs':{'critic_observations':torch.full((1,93),2.)},'extras':{'privileged_obs':torch.full((1,14),2.)},'self':NS(priv_dim=14)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=assigns,type_ignores=[])),'rollout','exec'),scope)
        torch.testing.assert_close(scope['buf']['privileged'][0],torch.ones(1,14))


class WalkingRewardContracts(unittest.TestCase):
    def test_intended_walking_speed_is_not_penalized_as_still(self):
        buf = torch.zeros(2, 50, 3)
        buf[1, :, 0] = torch.arange(50) * .02 * .4
        cmd = NS(base_pos_buffer=buf, buffer_idx=torch.tensor([49,49]),
                 buffer_count=torch.tensor([50,50]), relative_ball_pos=torch.tensor([[2.,0.],[2.,0.]]),
                 cfg=NS(training_phase="approach"))
        env = NS(num_envs=2, device="cpu")
        fn = functions(MDP/'rewards.py', ['pos_still'], _cmd=lambda _:cmd, task_curriculum=lambda _:1.)['pos_still']
        torch.testing.assert_close(fn(env), torch.tensor([1.,0.]))

    def test_approach_phase_never_fades_with_iteration_count(self):
        fn = functions(MDP/'commands.py', ['task_curriculum'], TASK_RAMP_STEPS=16000*24)['task_curriculum']
        env = NS(common_step_counter=1000000, cfg=NS(commands=NS(soccer=NS(training_phase="approach"))))
        self.assertEqual(fn(env), 0.)
        env.cfg.commands.soccer.training_phase = "soccer"
        self.assertEqual(fn(env), 1.)

    def test_clearance_does_not_reward_motionless_stance(self):
        # First env stands still, second swings a foot at desired height,
        # third swings equally fast too close to the floor.
        pos=torch.zeros(3,2,3);pos[:,:,2]=.04;pos[1,0,2]=.1
        vel=torch.zeros_like(pos);vel[1:,0,0]=.5
        force=torch.zeros_like(pos);force[:,1,2]=90;force[0,0,2]=90
        cmd=NS(feet_idx=[0,1],feet_sensor_idx=[0,1],relative_ball_pos=torch.ones(3,2))
        env=NS(scene={'robot':NS(data=NS(body_pos_w=pos,body_lin_vel_w=vel)),
                      'contact_forces':NS(data=NS(net_forces_w=force))})
        f=functions(MDP/'rewards.py',['feet_clearance','_far_from_ball'],_cmd=lambda _:cmd,task_curriculum=lambda _:0.)['feet_clearance']
        reward=f(env)
        self.assertEqual(float(reward[0]),0.)
        self.assertGreater(float(reward[1]),0.)
        self.assertGreater(float(reward[1]),float(reward[2]))

    def test_task_curriculum_gates_normalized_advantage(self):
        cls,_=RunnerContracts().runner_class();r=cls.__new__(cls)
        r.cfg=NS(advantage_coef=(2.,1.));r.device='cpu';r.env=NS(extras={'task_weight':0.})
        advantages=torch.tensor([[100.,2.],[-100.,3.]])
        torch.testing.assert_close(r._policy_advantages(advantages),torch.tensor([2.,3.]))
        r.env.extras['task_weight']=.5
        torch.testing.assert_close(r._policy_advantages(advantages),torch.tensor([102.,-97.]))


class ExplorationContracts(unittest.TestCase):
    def test_loaded_out_of_range_std_can_learn_again(self):
        module=ROOT/'source/booster_train/booster_train/rsl_rl/amp/modules.py'
        cls=functions(module,['ActorCriticAMP'])['ActorCriticAMP']
        model=cls(2,2,1,2,1)
        with torch.no_grad(): model.logstd.fill_(-3.)
        model.constrain_std()
        model._std().sum().backward()
        self.assertTrue(bool(torch.all(model.logstd.grad > 0)))
        torch.testing.assert_close(model.logstd,torch.full((2,),-2.5))

if __name__=='__main__': unittest.main()
