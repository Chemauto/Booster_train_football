"""Actor-critic with encoder-decoder + discriminator for AMP training.

Direct port of /data/rl_robot/BoosterRobotics/code/utils/model.py, adapted to
K1 dimensions (num_obs=79, num_critic_obs=93, num_privileged=14, num_actions=22).
"""

from __future__ import annotations

import os

import torch


class ActorCriticAMP(torch.nn.Module):
    def __init__(self, num_obs: int, num_critic_obs: int, num_privileged_obs: int, num_actions: int,
                 num_stack: int, history_embedding_dim: int = 64,
                 actor_hidden=(256, 256, 128), critic_hidden=(256, 256, 128)):
        super().__init__()
        self.num_obs = num_obs
        self.num_actions = num_actions
        # two independent critics for the two reward groups (goal cluster / auxiliary)
        self.critics = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(num_critic_obs, critic_hidden[0]), torch.nn.ELU(),
                    torch.nn.Linear(critic_hidden[0], critic_hidden[1]), torch.nn.ELU(),
                    torch.nn.Linear(critic_hidden[1], critic_hidden[2]), torch.nn.ELU(),
                    torch.nn.Linear(critic_hidden[2], 1),
                )
                for _ in range(2)
            ]
        )
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(num_obs + history_embedding_dim, actor_hidden[0]), torch.nn.ELU(),
            torch.nn.Linear(actor_hidden[0], actor_hidden[1]), torch.nn.ELU(),
            torch.nn.Linear(actor_hidden[1], actor_hidden[2]), torch.nn.ELU(),
            torch.nn.Linear(actor_hidden[2], num_actions),
        )
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(num_obs * num_stack, 1024), torch.nn.ELU(),
            torch.nn.Linear(1024, 128), torch.nn.ELU(),
            torch.nn.Linear(128, history_embedding_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(history_embedding_dim, 128), torch.nn.ELU(),
            torch.nn.Linear(128, 128), torch.nn.ELU(),
            torch.nn.Linear(128, num_privileged_obs),
        )
        self.logstd = torch.nn.Parameter(torch.full((num_actions,), fill_value=-2.0))
        # guardrail on the learned exploration scale (sigma in [e^-2.5, e^-1.2]
        # ~= [0.08, 0.30]): keeps PPO from inflating action noise into
        # fall-inducing jitter when advantages are noisy
        self.logstd_min, self.logstd_max = -2.5, -1.2

    def _std(self) -> torch.Tensor:
        return torch.exp(self.logstd.clamp(self.logstd_min, self.logstd_max))

    @torch.no_grad()
    def constrain_std(self):
        """Project parameters, not just forward values, to avoid dead gradients."""
        self.logstd.clamp_(self.logstd_min, self.logstd_max)

    def act(self, obs: torch.Tensor, stacked_obs: torch.Tensor):
        """Returns (Normal dist, privileged-obs estimate)."""
        embedding = self.encoder(stacked_obs.flatten(start_dim=-2))
        action_mean = self.actor(torch.cat((obs, embedding), dim=-1))
        action_std = self._std().expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, action_std)
        privileged_obs_est = self.decoder(embedding)
        return dist, privileged_obs_est

    def est_values(self, critic_obs: torch.Tensor) -> torch.Tensor:
        """(N, 2) value estimates, one per reward-group critic."""
        return torch.stack([critic(critic_obs).squeeze(-1) for critic in self.critics], dim=-1)

    def forward(self, obs: torch.Tensor, stacked_obs: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(stacked_obs.flatten(start_dim=-2))
        return self.actor(torch.cat((obs, embedding), dim=-1))


class Discriminator(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512), torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(512, 512), torch.nn.LeakyReLU(0.2),
        )
        self.amp_linear = torch.nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.amp_linear(self.trunk(x))


class Normalizer:
    """Running mean/var normalizer for AMP observations (paper utils/model.py)."""

    def __init__(self, input_dim: int, device: str):
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.mean = torch.zeros(input_dim, dtype=torch.float, device=device)
        self.var = torch.ones(input_dim, dtype=torch.float, device=device)
        self.count = 0

    def update(self, batch: torch.Tensor):
        batch_mean = torch.mean(batch, dim=0)
        batch_var = torch.var(batch, dim=0, unbiased=False)
        batch_count = batch.shape[0] * self.world_size
        if self.world_size > 1:
            torch.distributed.all_reduce(batch_mean, op=torch.distributed.ReduceOp.AVG)
            torch.distributed.all_reduce(batch_var, op=torch.distributed.ReduceOp.AVG)
        self.var = (
            self.var * self.count
            + batch_var * batch_count
            + torch.square(batch_mean - self.mean) * self.count * batch_count / (self.count + batch_count)
        ) / (self.count + batch_count)
        self.mean += (batch_mean - self.mean) * batch_count / (self.count + batch_count)
        self.count += batch_count

    def normalize(self, batch: torch.Tensor) -> torch.Tensor:
        # divide by std, not var: var-dividing over-amplifies low-variance dims
        # (joint-space dims sit at var ~ 0.01-0.1) into the +-10 clamp; with
        # update() now tracking raw data the stats are true, so this is exact
        # unit-variance whitening
        return torch.clamp((batch - self.mean) / (torch.sqrt(self.var) + 1e-4), -10.0, 10.0)
