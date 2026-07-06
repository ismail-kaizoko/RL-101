# networks/actor_critic.py
"""
Actor-Critic Network — Gaussian Actor + Scalar Critic
======================================================

## The fundamental shift from classical RL

Classical RL:
    policy = table  π[s] = argmax_a Q[s,a]      deterministic, discrete

Now:
    policy = neural net  π_θ(a|s) = N(μ_θ(s), σ_θ(s))   stochastic, continuous

The actor doesn't output an action directly. It outputs the *parameters*
of a probability distribution over actions. Then we SAMPLE from that
distribution to get the actual action.

Why stochastic?
    1. Exploration is built-in — no need for ε-greedy
    2. The policy gradient theorem requires a differentiable distribution
       (you need ∇_θ log π_θ(a|s))
    3. In continuous spaces, a deterministic policy collapses to a single
       point — a distribution covers the space naturally

## The Gaussian policy for continuous actions

For action dimension i, the actor outputs:
    μ_i(s) = mean of the Gaussian       — the "intended" action
    σ_i(s) = std of the Gaussian        — the "confidence/exploration"

Action is sampled:
    a_i ~ N(μ_i(s), σ_i(s))

Log probability (needed for PPO):
    log π_θ(a|s) = Σ_i log N(a_i; μ_i, σ_i)
                 = Σ_i [ -0.5 log(2πσ_i²) - (a_i - μ_i)²/(2σ_i²) ]

For 2D actions [ax, ay], we use a diagonal Gaussian (independent dims):
    log π_θ(a|s) = log N(ax; μ_ax, σ_ax) + log N(ay; μ_ay, σ_ay)

## How σ is parameterised

Option A: σ = exp(log_std)  where log_std is a learned parameter vector
    — Not state-dependent: same exploration level everywhere
    — Simpler, more stable, standard PPO choice
    — log_std is initialised to 0 → σ=1, full exploration at start

Option B: σ = softplus(network_output)
    — State-dependent: can be more careful near obstacles
    — More complex, can collapse to zero (no exploration)

We use Option A — fixed log_std per action dim, not state-dependent.
This is what CleanRL, Stable-Baselines3, and most PPO papers use.

## Action squashing

Our drone env expects actions in [-1, 1] (normalised thrust).
The Gaussian has infinite support — it can sample any value.
Two common fixes:

    Option A: tanh squashing — action = tanh(sample)  → (-1, 1)
        Requires correcting log_prob: log_prob -= Σ log(1 - tanh²(x))
        Used by SAC (Phase 4).

    Option B: clamp — action = clamp(sample, -1, 1)
        Simple, slightly biased near the boundaries.
        Standard for PPO in bounded action spaces.

We use clamp for PPO (simpler), tanh for SAC later.

## The critic

The critic V_φ(s) is a scalar estimator of the expected return
from state s under the current policy:

    V_φ(s) ≈ E_π[G_t | s_t = s]

It's trained by minimising MSE against the GAE returns:
    L_critic = E[(V_φ(s_t) - G_t)²]

The critic is NOT used to select actions — only to compute advantages:
    A_t = G_t - V_φ(s_t)
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal
from typing import Tuple

from networks.mlp import MLP, init_weights


class ActorCritic(nn.Module):
    """
    Shared-backbone actor-critic for continuous action spaces.

    Architecture:
        obs → MLP (shared) → features
                               ├→ actor_mean  (Linear → μ)
                               └→ critic_head (Linear → V)
        log_std: learnable parameter vector, NOT state-dependent

    act(obs):   used during COLLECTION — returns action, log_prob, value
    evaluate(obs, action): used during UPDATE — recomputes log_prob, value,
                           entropy under the NEW policy θ
    """

    def __init__(
        self,
        obs_dim:    int,
        act_dim:    int,
        hidden_dim: int   = 64,
        n_layers:   int   = 2,
        log_std_init: float = 0.0,   # σ=1 at start → full exploration
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # ── Shared backbone ────────────────────────────────────────────────────
        self.backbone = MLP(obs_dim, hidden_dim, n_layers)
        feat_dim      = self.backbone.output_dim

        # ── Actor head — outputs μ(s) ──────────────────────────────────────────
        # Small gain (0.01) → near-zero initial actions
        # The agent starts by barely moving, then learns to act more decisively
        self.actor_mean = nn.Linear(feat_dim, act_dim)
        init_weights(self.actor_mean, gain=0.01)

        # log_std: one value per action dimension, not state-dependent
        # Stored as nn.Parameter so it's updated by the optimizer
        # exp(0.0) = 1.0 → unit std at initialisation
        self.log_std = nn.Parameter(
            torch.full((act_dim,), log_std_init)
        )

        # ── Critic head — outputs V(s) ─────────────────────────────────────────
        self.critic_head = nn.Linear(feat_dim, 1)
        init_weights(self.critic_head, gain=1.0)

    # ── Forward passes ────────────────────────────────────────────────────────

    def _features(self, obs: Tensor) -> Tensor:
        """Shared trunk: obs → feature vector."""
        return self.backbone(obs)

    def _distribution(self, features: Tensor) -> Normal:
        """
        Build the action distribution N(μ(s), σ) from features.

        σ = exp(log_std) — always positive, never zero
        μ = actor_mean(features) — state-dependent mean
        """
        mu  = self.actor_mean(features)
        std = self.log_std.exp().expand_as(mu)   # broadcast to batch size
        return Normal(mu, std)

    def _value(self, features: Tensor) -> Tensor:
        """Critic: features → scalar V(s). Shape: (batch,)"""
        return self.critic_head(features).squeeze(-1)

    # ── API used by rollout.py ─────────────────────────────────────────────────

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Sample an action from the current policy.
        Called at EVERY STEP during rollout collection.

        Returns numpy arrays (not tensors) since env.step() expects numpy.

        obs      → (obs_dim,)  numpy
        action   → (act_dim,)  numpy, clamped to [-1, 1]
        log_prob → float       log π_θ(action|obs)
        value    → float       V_φ(obs)
        """
        obs_t    = torch.FloatTensor(obs).unsqueeze(0)   # (1, obs_dim)
        features = self._features(obs_t)
        dist     = self._distribution(features)
        value    = self._value(features)

        action   = dist.sample()                         # (1, act_dim)

        # sum(-1): add log_probs across action dimensions
        # For independent Gaussian dims: log p(a) = Σ_i log p(a_i)
        log_prob = dist.log_prob(action).sum(-1)         # (1,)

        # Clamp to [-1, 1] for env compatibility
        action_np   = action.squeeze(0).numpy()
        action_np   = np.clip(action_np, -1.0, 1.0)
        log_prob_np = log_prob.item()
        value_np    = value.item()

        return action_np, log_prob_np, value_np

    # ── API used by ppo.py ─────────────────────────────────────────────────────

    def evaluate(
        self,
        obs:     Tensor,     # (batch, obs_dim)
        actions: Tensor,     # (batch, act_dim)
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Recompute log_prob, value, entropy for a batch of (obs, action) pairs.
        Called during the PPO UPDATE — under the NEW policy θ (not θ_old).

        This is the key operation that enables PPO:
            log_prob_new  (under θ)      → used to compute ratio r_t(θ)
            log_prob_old  (under θ_old)  → stored in buffer during collection

            r_t(θ) = exp(log_prob_new - log_prob_old)
                   = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)

        Returns:
            log_probs : (batch,)   log π_θ(a|s) under NEW policy
            values    : (batch,)   V_φ(s)
            entropy   : (batch,)   H[π_θ(·|s)] = -E[log π_θ(a|s)]
        """
        features = self._features(obs)
        dist     = self._distribution(features)
        values   = self._value(features)

        log_probs = dist.log_prob(actions).sum(-1)   # (batch,)

        # Entropy of a Gaussian: H = 0.5 * log(2πe σ²)
        # Used as a bonus in the PPO loss to prevent premature collapse
        # to a deterministic policy (encourages continued exploration)
        entropy = dist.entropy().sum(-1)             # (batch,)

        return log_probs, values, entropy

    def get_value(self, obs: np.ndarray) -> float:
        """
        Get critic value for a single observation.
        Used to bootstrap the last state value in GAE.
        """
        with torch.no_grad():
            obs_t    = torch.FloatTensor(obs).unsqueeze(0)
            features = self._features(obs_t)
            return self._value(features).item()