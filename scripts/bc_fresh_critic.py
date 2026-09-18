"""Rebuild the BC checkpoint with fresh critics + fresh optimizer state.

The seed-protection run (2026-09-17_12-53-03) still washed the gait out:
fall 0.39 -> 0.12 while planar speed 0.47 -> 0.094. Hypothesis (pre-registered
branch): the critics inherited from the standing policy know standing's value
(~3.6+) but undervalue walking states, so every advantage comparison favors
standing; the stale Adam moments carry the same bias. This script keeps the
BC actor + encoder + obs normalizer and re-initializes everything value-side.
"""

import copy

import torch

from booster_train.rsl_rl.amp.modules import ActorCriticAMP

SRC = "logs/bc/model_bc_init.pt"
DST = "logs/bc/model_bc_fresh.pt"

ck = torch.load(SRC, map_location="cpu", weights_only=True)
fresh = ActorCriticAMP(79, 93, 14, 22, num_stack=50)
fresh.load_state_dict(ck["model"])  # load everything, then overwrite critics
torch.nn.init.zeros_(fresh.critics[0][-1].weight); torch.nn.init.zeros_(fresh.critics[0][-1].bias)

# full re-init of both critics (default init), keeping actor/encoder from BC
ref = ActorCriticAMP(79, 93, 14, 22, num_stack=50)
sd = fresh.state_dict()
for name, p in ref.critics.named_parameters():
    sd[f"critics.{name}"].copy_(p)
for name, b in ref.critics.named_buffers():
    sd[f"critics.{name}"].copy_(b)
fresh.load_state_dict(sd)

out = dict(ck)
out["model"] = fresh.state_dict()
# fresh optimizer state (empty moments) matching the runner's param groups
out["optimizer"] = torch.optim.Adam(fresh.parameters(), lr=1e-4).state_dict()
out["note"] = "critics reinitialized + optimizer reset; actor/encoder = BC"
torch.save(out, DST)
print(f"saved {DST} (critics fresh, optimizer fresh, iter {ck['iter']})")
