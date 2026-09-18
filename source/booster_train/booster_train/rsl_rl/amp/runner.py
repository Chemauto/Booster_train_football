"""AMP runner for the kick_amp task: PPO + WGAN-GP discriminator + encoder-decoder
+ mirror symmetry loss + two-critic advantages.

Port of /data/rl_robot/BoosterRobotics/code/utils/runner.py onto the Isaac Lab
manager-based env interface. The stacked-observation buffer lives here (not in
the env): (num_envs, num_stack, num_obs), zeroed on episode boundaries.

Env contract (all via env.step extras):
    obs["policy"]            (N, num_obs)          actor obs (POMDP)
    obs["critic_observations"] (N, num_critic_obs)  asymmetric critic obs
    extras["amp_obs"]        (N, AMP_OBS_DIM)      current-step AMP obs
    extras["privileged_obs"] (N, 14)               decoder reconstruction target
    extras["rew_groups"]     (N, 2)                goal-cluster reward, 0 placeholder
    extras["success"]        (N,)                  goal held >= 50 steps
    extras["time_outs"]      (N,)
and the env reward buffer fills reward group 1 (+ AMP style reward added here).
"""

from __future__ import annotations

import os
import time
import copy
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from isaaclab.utils import configclass

from .mirror import mirror, mirror_act
from .modules import ActorCriticAMP, Discriminator, Normalizer
from rsl_rl.networks.normalization import EmpiricalNormalization


class AmpReplayBuffer:
    """Stores (s_t, s_{t+1}) policy AMP pairs, without done-boundary pairs."""

    def __init__(self, obs_dim: int, buffer_size: int, device: str):
        self.device = device
        self.obs_dim = obs_dim
        self.buffer_size = buffer_size
        self.obs = torch.zeros(buffer_size, obs_dim, device=device)
        self.next_obs = torch.zeros(buffer_size, obs_dim, device=device)
        self.insert_ptr = 0
        self.filled = 0

    def insert(self, obs: torch.Tensor, next_obs: torch.Tensor, valid: torch.Tensor):
        idx = valid.nonzero(as_tuple=False).flatten()
        if len(idx) == 0:
            return
        n = len(idx)
        end = self.insert_ptr + n
        if end <= self.buffer_size:
            self.obs[self.insert_ptr:end] = obs[idx]
            self.next_obs[self.insert_ptr:end] = next_obs[idx]
        else:
            k = self.buffer_size - self.insert_ptr
            self.obs[self.insert_ptr:] = obs[idx[:k]]
            self.next_obs[self.insert_ptr:] = next_obs[idx[:k]]
            self.obs[:n - k] = obs[idx[k:]]
            self.next_obs[:n - k] = next_obs[idx[k:]]
        self.insert_ptr = end % self.buffer_size
        self.filled = min(self.filled + n, self.buffer_size)

    def feed_forward_generator(self, num_mini_batches: int, mini_batch_size: int):
        for _ in range(num_mini_batches):
            idx = torch.randint(0, self.filled, (mini_batch_size,), device=self.device)
            yield self.obs[idx], self.next_obs[idx]


def surrogate_loss(old_log_prob, log_prob, advantage, clip_param=0.2):
    ratio = torch.exp(log_prob - old_log_prob)
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
    return -torch.min(surr1, surr2).mean()


def action_bound_loss(action_mean, action_scale, bound_rad):
    """Penalize position-target deltas in radians, independent of action units.

    The reference T1 uses scale=1 and a one-radian bound. K1 has per-joint
    scales down to .268; penalizing raw actions at one silently shrinks the
    allowed ankle excursion almost fourfold.
    """
    delta = action_mean * action_scale
    return (delta - bound_rad).clamp(min=0.).square().mean() + (delta + bound_rad).clamp(max=0.).square().mean()


class AmpRunner:
    def __init__(self, env, cfg: "AmpRunnerCfg", train_mode=True):
        self.env = env
        self.cfg = cfg
        self.device = env.device
        if cfg.seed is not None:
            import random as _random
            _random.seed(cfg.seed)
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)
        self.num_envs = env.num_envs
        self.obs_dim = env.observation_space["policy"].shape[-1]
        self.critic_obs_dim = env.observation_space["critic_observations"].shape[-1]
        self.priv_dim = 14
        # single source of truth: a hardcoded copy here would silently mismatch
        # the env's amp_obs width and make the discriminator compare misaligned
        # columns instead of failing loudly
        from booster_train.tasks.manager_based.kick_amp.mdp.motion_lib import AMP_OBS_DIM
        self.amp_dim = AMP_OBS_DIM
        self.num_actions = env.action_manager.total_action_dim
        self.num_joints = self.num_actions
        action_term = env.action_manager.get_term("joint_pos")
        scale = torch.as_tensor(action_term._scale, device=self.device)
        self.action_scale = torch.broadcast_to(scale, (self.num_envs, self.num_actions))[0].clone()

        self.model = ActorCriticAMP(
            self.obs_dim, self.critic_obs_dim, self.priv_dim, self.num_actions,
            num_stack=cfg.num_stack,
        ).to(self.device)
        self.discriminator = Discriminator(self.amp_dim * 2).to(self.device)
        self.amp_normalizer = Normalizer(self.amp_dim, self.device)
        # running normalization of policy obs (meters-scale ball/goal terms);
        # critic obs stays raw, matching rsl_rl's empirical_normalization
        self.obs_norm = EmpiricalNormalization(self.obs_dim).to(self.device)

        self.optimizer = torch.optim.Adam(
            [{"name": "actor_critic", "params": self.model.parameters(), "lr": cfg.learning_rate}]
        )
        self.discriminator_optimizer = torch.optim.RMSprop(
            [
                {"name": "amp_trunk", "params": self.discriminator.trunk.parameters(),
                 "weight_decay": 1e-2, "lr": cfg.discriminator_lr},
                {"name": "amp_head", "params": self.discriminator.amp_linear.parameters(),
                 "weight_decay": 5e-1, "lr": cfg.discriminator_lr},
            ]
        )
        self.amp_storage = AmpReplayBuffer(self.amp_dim, cfg.replay_buffer_size, self.device)
        self.learning_rate = cfg.learning_rate

        # expert dataset comes from the env's soccer command (already loaded)
        self.motion_dataset = env.command_manager.get_term("soccer").motion_dataset

        # mirror matrices are built from the LIVE joint order (PhysX BFS order,
        # not URDF declaration order) -- never from a hardcoded list
        self.joint_names = list(env.scene["robot"].joint_names)

        # stacked obs buffer (zeroed on episode end)
        self.stacked_obs = torch.zeros(self.num_envs, cfg.num_stack, self.obs_dim, device=self.device)

        self.train_mode = train_mode
        log_root = os.path.join("logs", "rsl_rl")
        run_dir = os.path.join(log_root, cfg.experiment_name, time.strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(run_dir, exist_ok=True)
        self.log_dir = run_dir
        if train_mode:
            self.writer = SummaryWriter(log_dir=run_dir, max_queue=10, flush_secs=60)
            self._dump_provenance()
        self.tot_iterations = 0
        self.success_history = deque(maxlen=100)
        self.fall_history = deque(maxlen=100)
        self.stop_requested = False

    def _normalize_obs(self, obs: torch.Tensor, update: bool = False) -> torch.Tensor:
        if update:
            # Mirror symmetry is physical, not symmetry about arbitrary sample
            # means. Paired statistics make N(M(x)) == M(N(x)), including for
            # histories collected with earlier running statistics.
            reflected = mirror(obs, self.joint_names, self.device)
            self.obs_norm.update(torch.cat((obs, reflected), dim=0))
        return self.obs_norm(obs)

    def _policy_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        weights = torch.tensor(self.cfg.advantage_coef, device=self.device)
        weights[0] *= float(self.env.extras.get("task_weight", 1.0))
        return (advantages * weights).sum(dim=-1)

    def _dump_provenance(self):
        """Persist git state + env/agent configs for reproducibility."""
        import subprocess
        lines = []
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=os.path.dirname(__file__)).stdout.strip()
            lines.append(f"commit: {commit}")
            branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, cwd=os.path.dirname(__file__)).stdout.strip()
            lines.append(f"branch: {branch}")
            dirty = subprocess.run(["git", "status", "--short"], capture_output=True, text=True, cwd=os.path.dirname(__file__)).stdout.strip()
            lines.append(f"dirty files:\n{dirty or '(clean)'}")
            diff_stat = subprocess.run(["git", "diff", "--stat", "HEAD"], capture_output=True, text=True, cwd=os.path.dirname(__file__)).stdout.strip()
            lines.append(f"diff stat:\n{diff_stat or '(none)'}")
        except Exception as e:
            lines.append(f"git unavailable: {e}")
        provenance = "\n".join(lines)
        with open(os.path.join(self.log_dir, "git_info.txt"), "w") as f:
            f.write(provenance + "\n")
        # Dirty worktrees are common during RL experiments. Record content,
        # including untracked task files, instead of only a diff summary.
        from pathlib import Path
        import hashlib
        import tarfile
        project = Path(__file__).resolve().parents[5]
        sources = sorted((project / "source").rglob("*.py")) + sorted((project / "scripts").rglob("*.py"))
        with tarfile.open(os.path.join(self.log_dir, "source_snapshot.tgz"), "w:gz") as archive:
            for source in sources:
                archive.add(source, arcname=str(source.relative_to(project)))
        with open(os.path.join(self.log_dir, "source_sha256.txt"), "w") as f:
            for source in sources:
                f.write(f"{hashlib.sha256(source.read_bytes()).hexdigest()}  {source.relative_to(project)}\n")
        self.writer.add_text("provenance/git", f"```\n{provenance}\n```", 0)
        # config snapshot: agent cfg + env cfg (repr-based; full yaml via dataclasses.asdict where possible)
        from dataclasses import asdict
        try:
            agent_cfg = asdict(self.cfg)
        except Exception:
            agent_cfg = {k: getattr(self.cfg, k) for k in dir(self.cfg) if not k.startswith("_") and not callable(getattr(self.cfg, k))}
        with open(os.path.join(self.log_dir, "agent_cfg.yaml"), "w") as f:
            import yaml
            yaml.safe_dump(agent_cfg, f, default_flow_style=False)
        env_cfg_repr = repr(self.env.cfg)
        with open(os.path.join(self.log_dir, "env_cfg.txt"), "w") as f:
            f.write(env_cfg_repr + "\n")
        self.writer.add_text("provenance/agent_cfg", f"```\n{agent_cfg}\n```", 0)

    # ------------------------------------------------------------------ helpers

    def _push_obs(self, obs: torch.Tensor, dones: torch.Tensor):
        self.stacked_obs = torch.roll(self.stacked_obs, shifts=-1, dims=1)
        self.stacked_obs[:, -1, :] = obs
        self.stacked_obs[dones] = 0.0

    def export(self, path: str):
        """JIT-export the deployable graph: (raw_obs, stacked_of_normalized_obs) -> action.
        The normalizer is frozen into the graph; the deployment harness must maintain
        the stacked history on NORMALIZED obs (zero it on episode boundaries)."""
        class Deploy(torch.nn.Module):
            def __init__(self, model, obs_norm):
                super().__init__()
                self.model = model
                self.obs_norm = obs_norm

            def forward(self, obs, stacked_obs):
                return self.model(self.obs_norm(obs), stacked_obs)

        # Export must not put the live normalizer in eval mode: otherwise its
        # update() silently stops learning after the first checkpoint export.
        deploy = Deploy(copy.deepcopy(self.model), copy.deepcopy(self.obs_norm)).eval()
        example = (
            torch.zeros(1, self.obs_dim, device=self.device),
            torch.zeros(1, self.cfg.num_stack, self.obs_dim, device=self.device),
        )
        torch.jit.trace(deploy, example).save(path)

    def save(self, path: str):
        tmp_path = path + ".tmp"
        torch.save(
            {
                "model": self.model.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "amp_mean": self.amp_normalizer.mean,
                "amp_var": self.amp_normalizer.var,
                "amp_count": self.amp_normalizer.count,
                "iter": self.tot_iterations,
                "training_phase": self.env.cfg.commands.soccer.training_phase,
                "learning_rate": self.learning_rate,
                "obs_norm": self.obs_norm.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
            },
            tmp_path,
        )
        os.replace(tmp_path, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(data["model"])
        self.model.constrain_std()
        self.discriminator.load_state_dict(data["discriminator"])
        self.amp_normalizer.mean = data["amp_mean"].to(self.device)
        self.amp_normalizer.var = data["amp_var"].to(self.device)
        self.amp_normalizer.count = data["amp_count"]
        self.learning_rate = data.get("learning_rate", self.cfg.learning_rate)
        if "obs_norm" in data:  # v4 checkpoints; older ones start stats fresh
            self.obs_norm.load_state_dict(data["obs_norm"])
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate
        if "optimizer" in data:
            self.optimizer.load_state_dict(data["optimizer"])
        if "discriminator_optimizer" in data:
            self.discriminator_optimizer.load_state_dict(data["discriminator_optimizer"])
        self.tot_iterations = data.get("iter", 0)
        # curriculum gates read the env's step counter, which is process-local:
        # restore it or every resume silently replays the whole reset
        # curriculum (observed as rew_groups re-zeroed after resume).
        self.env.common_step_counter = self.tot_iterations * self.cfg.horizon_length

    def reset_optimization_for_phase(self):
        """Retain actor/history/normalization, discard values of the old objective."""
        for module in self.model.critics.modules():
            if isinstance(module, torch.nn.Linear):
                module.reset_parameters()
        with torch.no_grad():
            self.model.logstd.fill_(float(np.log(.15)))
        self.learning_rate = 1.e-4
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        print("[train] new phase: actor retained, critics/PPO optimizer reset, exploration sigma=0.15", flush=True)

    # -------------------------------------------------------------------- train

    def learn(self):
        cfg = self.cfg
        reset_out = self.env.reset()
        obs_dict = reset_out[0]
        obs = obs_dict["policy"]
        critic_obs = obs_dict["critic_observations"]
        self.model.train()
        self.obs_norm.train()
        prev_amp_obs = None  # set after the first env.step (reset extras have no amp_obs)

        T = cfg.horizon_length
        N = self.num_envs
        buf = {
            k: torch.zeros(T, N, *shape, device=self.device)
            for k, shape in {
                "obs": (self.obs_dim,), "critic_obs": (self.critic_obs_dim,),
                "stacked": (cfg.num_stack, self.obs_dim,), "mirrored": (self.obs_dim,),
                "actions": (self.num_actions,),
                "rewards": (2,), "privileged": (self.priv_dim,),
            }.items()
        }
        dones = torch.zeros(T, N, dtype=torch.bool, device=self.device)
        time_outs = torch.zeros(T, N, dtype=torch.bool, device=self.device)

        end_iteration = self.tot_iterations + cfg.max_iterations
        for it in range(cfg.max_iterations):
            iteration_start = time.perf_counter()
            ep_success = torch.zeros(N, dtype=torch.bool, device=self.device)
            ep_fall = torch.zeros(N, dtype=torch.bool, device=self.device)
            completed_lengths = []
            fall_count = 0
            d_stds = []  # v13 instrumentation: within-batch spread of D scores
            for n in range(T):
                obs_n = self._normalize_obs(obs, update=True)
                buf["obs"][n] = obs_n
                buf["critic_obs"][n] = critic_obs
                # The encoder reconstructs s_t from history at s_t. Using the
                # step extras here made it predict s_{t+1}, or unrelated reset
                # states on falls, instead of the current privileged state.
                buf["privileged"][n] = critic_obs[:, -self.priv_dim:]
                buf["stacked"][n] = self.stacked_obs
                buf["mirrored"][n] = mirror(obs_n, self.joint_names, self.device)

                with torch.no_grad():
                    dist, _ = self.model.act(obs_n, self.stacked_obs)
                    act = dist.sample()
                episode_lengths = self.env.episode_length_buf.clone() + 1
                obs, rew_env, terminated, time_outs_now, extras = self.env.step(act)
                critic_obs = obs["critic_observations"]
                obs = obs["policy"]

                done = terminated | time_outs_now
                buf["actions"][n] = act
                if done.any():
                    completed_lengths.append(episode_lengths[done])
                fall_count += int(terminated.sum())
                dones[n] = done
                time_outs[n] = time_outs_now
                ep_success |= extras["success"].bool()
                ep_fall |= terminated
                # keep the freshest per-episode env log (Episode Reward/* etc.)
                env_log = extras.get("log", None)

                # AMP pair (s_{t-1}, s_t) for both the replay buffer and the
                # style reward; skip the very first step and reset-straddling pairs
                amp_now = extras["amp_obs"]
                if prev_amp_obs is not None:
                    valid_pair = ~done
                    self.amp_storage.insert(prev_amp_obs, amp_now, valid_pair)
                    with torch.no_grad():
                        norm_prev = self.amp_normalizer.normalize(prev_amp_obs)
                        norm_now = self.amp_normalizer.normalize(amp_now)
                        d = self.discriminator(torch.cat([norm_prev, norm_now], dim=-1)).squeeze(-1)
                        amp_rew = (1 + torch.tanh(0.4 * d)) * valid_pair.float()
                        d_stds.append(d.std().item())
                else:
                    amp_rew = torch.zeros(N, device=self.device)
                prev_amp_obs = amp_now

                # rewards: group0 from the command extras, group1 = env reward + AMP style
                group1 = rew_env + cfg.amp_reward_coef * amp_rew
                buf["rewards"][n, :, 0] = extras["rew_groups"][:, 0]
                buf["rewards"][n, :, 1] = group1

                self._push_obs(self.obs_norm(obs), done)

            # episode stats (any env that ended during the horizon)
            self.success_history.append(ep_success.float().mean().item())
            self.fall_history.append(ep_fall.float().mean().item())

            self._update(buf, dones, time_outs, critic_obs)

            self.tot_iterations += 1
            self.writer.add_scalar("train/steps_per_second", T * N / (time.perf_counter() - iteration_start), self.tot_iterations)
            if completed_lengths:
                lengths = torch.cat(completed_lengths).float()
                self.writer.add_scalar("train/episode_length_s", float(lengths.mean()) * self.env.step_dt, self.tot_iterations)
                self.writer.add_scalar("train/completed_episodes", lengths.numel(), self.tot_iterations)
                self.writer.add_scalar("train/terminated_episode_fraction", fall_count / lengths.numel(), self.tot_iterations)
            # every iteration: forward the env's per-episode log (per-term rewards,
            # episode length, command metrics) into tensorboard
            if env_log:
                for key, value in env_log.items():
                    if isinstance(value, torch.Tensor):
                        value = value.float().mean().item()
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f"env/{key}", value, self.tot_iterations)
            if self.tot_iterations % 10 == 0:
                succ = np.mean(self.success_history) if self.success_history else 0.0
                fall = np.mean(self.fall_history) if self.fall_history else 0.0
                # success is printed in scientific notation: real early signal
                # lives at 1e-5..1e-3 and %.3f swallows it to a fake 0.000
                ep_seconds = f"{float(torch.cat(completed_lengths).float().mean()) * self.env.step_dt:.2f}" if completed_lengths else "n/a"
                print(f"iter {self.tot_iterations}/{end_iteration} success={succ:.1e} fall_window={fall:.3f} episode_s={ep_seconds} lr={self.learning_rate:.2e}", flush=True)
                self.writer.add_scalar("train/success_rate", succ, self.tot_iterations)
                self.writer.add_scalar("train/fall_rate", fall, self.tot_iterations)
                self.writer.add_scalar("train/rew_goal_group", float(buf["rewards"][:, :, 0].mean()) * T, self.tot_iterations)
                self.writer.add_scalar("train/rew_aux_group", float(buf["rewards"][:, :, 1].mean()) * T, self.tot_iterations)
                # v13: the style gradient only exists if D's score VARIES inside
                # the policy batch (constant reward = zero advantage). ~0 means
                # the teacher is inaudible regardless of amp_reward_coef.
                if d_stds:
                    self.writer.add_scalar("train/amp_d_std", float(np.mean(d_stds)), self.tot_iterations)
                self.writer.add_scalar("train/env_reward", float(self.env.reward_buf.mean()) * T, self.tot_iterations)
            if self.tot_iterations % cfg.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{self.tot_iterations}.pt"))
                # keep a deployable policy.pt current even if the run is killed
                self.export(os.path.join(self.log_dir, "policy.pt"))
            if self.stop_requested:
                print("[train] graceful stop requested; saving completed iteration", flush=True)
                break
        self.save(os.path.join(self.log_dir, f"model_{self.tot_iterations}.pt"))
        self.export(os.path.join(self.log_dir, "policy.pt"))
        self.writer.flush()
        self.writer.close()

    def _update(self, buf, dones, time_outs, last_critic_obs):
        cfg = self.cfg
        T, N = cfg.horizon_length, self.num_envs

        with torch.no_grad():
            old_dist, _ = self.model.act(buf["obs"].flatten(0, 1), buf["stacked"].flatten(0, 1))
            old_log_prob = old_dist.log_prob(buf["actions"].flatten(0, 1)).sum(dim=-1).reshape(T, N)

            values = self.model.est_values(buf["critic_obs"])          # (T, N, 2)
            last_values = self.model.est_values(last_critic_obs)       # (N, 2)
            rewards = buf["rewards"].clone()
            rewards[time_outs] += cfg.gamma * values[time_outs]        # V(s_t) bootstrap on time-outs (t1.py)
            dones_for_gae = (dones | time_outs).float()
            advantages = torch.zeros_like(rewards)
            last_adv = torch.zeros(N, 2, device=self.device)
            for t in reversed(range(T)):
                # transition t produced s_{t+1}; if t was terminal/time-out, the
                # post-reset value must not be bootstrapped (mask with dones[t])
                nt = (1.0 - dones_for_gae[t]).unsqueeze(-1)
                next_values = last_values if t == T - 1 else values[t + 1]
                delta = rewards[t] + cfg.gamma * nt * next_values - values[t]
                last_adv = delta + cfg.gamma * cfg.lam * nt * last_adv
                advantages[t] = last_adv
            returns = values + advantages
            advantages = (advantages - advantages.mean(dim=(0, 1))) / (advantages.std(dim=(0, 1)) + 1e-8)

        statistics = {}
        mini_batch_size = (N * T) // cfg.num_mini_batches
        indices = torch.randperm(mini_batch_size * cfg.num_mini_batches, device=self.device)
        amp_policy_gen = self.amp_storage.feed_forward_generator(
            cfg.num_learning_epochs * cfg.num_mini_batches, mini_batch_size
        )
        amp_expert_gen = self.motion_dataset.feed_forward_generator(
            cfg.num_learning_epochs * cfg.num_mini_batches, mini_batch_size
        )

        for _ in range(cfg.num_learning_epochs):
            for i in range(cfg.num_mini_batches):
                batch_idx = indices[i * mini_batch_size:(i + 1) * mini_batch_size]
                obs_b = buf["obs"].flatten(0, 1)[batch_idx]
                stacked_b = buf["stacked"].flatten(0, 1)[batch_idx]
                dist, priv_est = self.model.act(obs_b, stacked_b)
                values = self.model.est_values(buf["critic_obs"].flatten(0, 1)[batch_idx])
                value_loss = F.mse_loss(values, returns.flatten(0, 1)[batch_idx])

                log_prob = dist.log_prob(buf["actions"].flatten(0, 1)[batch_idx]).sum(dim=-1)
                adv = advantages.flatten(0, 1)[batch_idx]
                actor_loss = surrogate_loss(
                    old_log_prob.flatten(0, 1)[batch_idx],
                    log_prob,
                    self._policy_advantages(adv),
                )

                # AMP WGAN-GP
                policy_s, policy_ns = next(amp_policy_gen)
                expert_s, expert_ns = next(amp_expert_gen)
                # the normalizer must track the RAW distribution: feeding it its
                # own output locks the statistics at a distorted fixed point
                policy_s_raw, expert_s_raw = policy_s, expert_s
                with torch.no_grad():
                    policy_s = self.amp_normalizer.normalize(policy_s)
                    policy_ns = self.amp_normalizer.normalize(policy_ns)
                    expert_s = self.amp_normalizer.normalize(expert_s)
                    expert_ns = self.amp_normalizer.normalize(expert_ns)
                policy_d = self.discriminator(torch.cat([policy_s, policy_ns], dim=-1))
                expert_d = self.discriminator(torch.cat([expert_s, expert_ns], dim=-1))
                amp_loss = -torch.tanh(0.4 * expert_d).mean() + torch.tanh(0.4 * policy_d).mean()

                expert_pair = torch.cat([expert_s, expert_ns], dim=-1)
                policy_pair = torch.cat([policy_s, policy_ns], dim=-1)
                alpha = torch.rand(expert_pair.size(0), 1, device=self.device)
                interpolates = (alpha * expert_pair + (1 - alpha) * policy_pair).requires_grad_(True)
                interpolates_d = self.discriminator(interpolates)
                grad = torch.autograd.grad(
                    interpolates_d, interpolates,
                    grad_outputs=torch.ones_like(interpolates_d),
                    create_graph=True, retain_graph=True, only_inputs=True,
                )[0]
                grad_pen_loss = 10 * ((torch.sqrt(torch.sum(grad.view(grad.size(0), -1) ** 2, dim=1) + 1e-12) - 1) ** 2).mean()

                # action bound / entropy / reconstruction / symmetry
                bound_loss = action_bound_loss(dist.loc, self.action_scale, cfg.action_bound_rad)
                entropy = dist.entropy().sum(dim=-1).mean()
                reconstruction_loss = F.mse_loss(priv_est, buf["privileged"].flatten(0, 1)[batch_idx])
                mirrored_stack_b = mirror(
                    buf["stacked"].flatten(0, 1)[batch_idx].reshape(-1, self.obs_dim), self.joint_names, self.device
                ).reshape(-1, cfg.num_stack, self.obs_dim)
                mirrored_dist, _ = self.model.act(buf["mirrored"].flatten(0, 1)[batch_idx], mirrored_stack_b)
                symmetric_loss = F.mse_loss(dist.loc, mirror_act(mirrored_dist.loc, self.joint_names, self.device))

                loss = (
                    value_loss + actor_loss
                    + cfg.bound_coef * bound_loss
                    + cfg.entropy_coef * entropy
                    + cfg.reconstruction_coef * reconstruction_loss
                    + cfg.amp_coef * amp_loss
                    + cfg.grad_pen_coef * grad_pen_loss
                    + cfg.symmetric_coef * symmetric_loss
                )
                self.optimizer.zero_grad()
                self.discriminator_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.model.constrain_std()
                self.discriminator_optimizer.step()
                # (a sigma freeze to fill_(-2.5) lived here: it protected the
                # old stand-in-place curriculum from noise, but it also removed
                # the only exploration the policy had, and the 1/sigma^2 term
                # amplified every KL estimate ~148x. Both that curriculum and
                # the freeze are gone; the policy samples at its learned
                # logstd, clamped to sigma in [0.08, 0.30] in ActorCriticAMP.)

                self.amp_normalizer.update(policy_s_raw)
                self.amp_normalizer.update(expert_s_raw)

                with torch.no_grad():
                    old_scale = old_dist.scale[batch_idx]
                    old_loc = old_dist.loc[batch_idx]
                    kl = torch.sum(
                        torch.log(dist.scale / old_scale)
                        + 0.5 * (old_scale ** 2 + (dist.loc - old_loc) ** 2) / dist.scale ** 2 - 0.5,
                        axis=-1,
                    ).mean()
                # lr adjustment happens ONCE per iteration after the loops,
                # from the iteration-MEAN kl (the raise branch fires below
                # desired_kl, i.e. 2x more eager than T1's rule -- kept, the
                # dead rationale it replaced described the old buggy decision)

                for key in ("value_loss", "actor_loss", "amp_loss", "grad_pen_loss",
                            "symmetric_loss", "reconstruction_loss", "entropy"):
                    statistics[key] = statistics.get(key, 0.0) + float(locals()[key])
                statistics["expert_pred"] = statistics.get("expert_pred", 0.0) + float(expert_d.mean())
                statistics["policy_pred"] = statistics.get("policy_pred", 0.0) + float(policy_d.mean())
                statistics["kl"] = statistics.get("kl", 0.0) + float(kl)
                statistics["lr"] = statistics.get("lr", 0.0) + self.learning_rate

        # adjust from the iteration-MEAN kl (the same quantity logged to tb).
        # The previous last-minibatch decision ran ~2x the mean; with the old
        # desired_kl=0.01 override it decreased lr nearly every iteration and
        # pinned it at the 1e-4 floor -- frozen sigma amplifies KL ~148x
        # (1/sigma^2), so the decrease branch fired even with a healthy mean
        mean_kl = statistics["kl"] / (cfg.num_learning_epochs * cfg.num_mini_batches)
        if mean_kl > cfg.desired_kl * 2:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)  # floor 1e-5 = T1 runner
        elif mean_kl < cfg.desired_kl:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

        denom = cfg.num_learning_epochs * cfg.num_mini_batches
        for key, val in statistics.items():
            self.writer.add_scalar(f"loss/{key}", val / denom, self.tot_iterations)


@configclass
class AmpRunnerCfg:
    seed = 42
    num_envs = 4096
    experiment_name = "k1_kick_amp"
    max_iterations = 20000
    save_interval = 200
    replay_buffer_size = 1_000_000

    num_stack = 50
    horizon_length = 24
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1.0e-3
    discriminator_lr = 1.0e-4
    gamma = 0.995
    lam = 0.95
    action_bound_rad = 1.0
    bound_coef = 100.0
    reconstruction_coef = 1.0
    # paper Table 1 (via T1.yaml): -0.01 with loss += coef*entropy is the
    # standard entropy BONUS under gradient descent
    entropy_coef = -0.01
    symmetric_coef = 10.0
    amp_coef = 1.0
    amp_reward_coef = 0.3
    # effective 50 = internal 10* in the loss x this 5.0 (paper Table 1)
    grad_pen_coef = 5.0
    # paper Table 1: 0.01 (safe now: the adaptive-lr rule decides on the
    # iteration-mean kl, not the last minibatch)
    desired_kl = 0.01
    advantage_coef = (2.0, 1.0)
    # curriculum horizon, kept in sync with commands.CURRICULUM_FULL_STEPS by
    # the task cfg (an out-of-sync copy here was a silent-bug source)
    curriculum_full_steps = 1000 * 24
