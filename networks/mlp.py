# networks/mlp.py
"""
Shared MLP Backbone
===================

## Why a shared backbone?

In classical RL, the actor (policy) and critic (value function) were
completely separate concepts — Q(s,a) was one table, V(s) was another.

With neural networks you have a choice:

    Separate networks:
        obs → [MLP_actor]  → μ, σ        (actor)
        obs → [MLP_critic] → V(s)         (critic)

    Shared backbone:
        obs → [MLP_shared] → features → [head_actor] → μ, σ
                                       → [head_critic] → V(s)

The shared backbone is better here because:
1. The drone's observation (position, velocity, rays) needs to be
   understood the same way for both "what action to take" and
   "how good is this state". Learning it once and sharing is efficient.
2. Fewer parameters to train — faster convergence with limited data.
3. Gradients from both actor loss and critic loss flow through the
   shared layers, providing a richer training signal.

The trade-off: if actor and critic need very different representations
(e.g. image input where the actor needs edges and critic needs global
structure), separate networks can be better. For our low-dim state, shared is fine.

## Architecture

    obs (14,)
        ↓
    Linear(14, 64)  + Tanh
        ↓
    Linear(64, 64)  + Tanh
        ↓
    features (64,)
        ↓
    [ActorHead]          [CriticHead]
    Linear(64, 2*act)    Linear(64, 1)
    → μ (2,), σ (2,)     → V(s) scalar

Why Tanh activation?
    - Bounded output range (-1, 1) — keeps features well-scaled
    - Works well with Gaussian policy outputs
    - Standard in PPO implementations (ReLU also works but can cause
      dead neurons with negative inputs, which occur with normalised obs)

Why 64 hidden units?
    - Our state is 14-dim, action is 2-dim — not a complex problem
    - 64 is the standard PPO baseline for continuous control
    - Bigger nets don't help if the bottleneck is sample efficiency
"""

import torch
import torch.nn as nn
from torch import Tensor


def init_weights(module: nn.Module, gain: float = 1.0):
    """
    Orthogonal initialisation — standard for PPO.

    Random initialisation (default PyTorch) works but leads to
    slow early training because gradients are unbalanced across layers.

    Orthogonal init ensures the weight matrix preserves vector norms,
    which keeps gradients well-conditioned from the first update.

    gain controls the scale:
        gain = sqrt(2)  for Tanh layers  (compensates for the activation shrinkage)
        gain = 0.01     for the actor mean head  (start with near-zero actions)
        gain = 1.0      for the critic head
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)


class MLP(nn.Module):
    """
    Shared MLP trunk: obs → feature vector.

    Used by ActorCritic in networks/actor_critic.py.
    Not used standalone — always paired with actor/critic heads.
    """

    def __init__(
        self,
        input_dim:   int,
        hidden_dim:  int = 64,
        n_layers:    int = 2,
        activation = nn.Tanh,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for _ in range(n_layers):
            linear = nn.Linear(in_dim, hidden_dim)
            init_weights(linear, gain=torch.nn.init.calculate_gain('tanh'))
            layers += [linear, activation()]
            in_dim  = hidden_dim

        self.net        = nn.Sequential(*layers)
        self.output_dim = hidden_dim

    def forward(self, obs: Tensor) -> Tensor:
        """
        obs   : (batch, input_dim)  or  (input_dim,) for single obs
        return: (batch, hidden_dim) feature vector
        """
        return self.net(obs)