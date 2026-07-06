# training/rollout.py
"""
Rollout Buffer — Trajectory Collection + GAE Advantage Estimation
=================================================================

This is the data engine for all on-policy algorithms (REINFORCE, PPO, GRPO).

## What changed from classical RL

In Q-learning you updated after every single step:
    observe (s, a, r, s') → update Q[s,a] → next step

In policy gradient methods you collect a full batch of experience first,
then update, then throw the data away (on-policy: data is only valid
under the policy that generated it).

The training loop looks like:

    while not converged:
        rollout = collect_rollout(env, policy, T steps)   ← this file
        advantages = compute_gae(rollout)
        update_policy(rollout, advantages)                ← ppo.py
        discard rollout

## What this file produces

For every timestep t in the rollout:
    obs[t]      : observation s_t                         shape (obs_dim,)
    actions[t]  : action a_t sampled from π_θ(·|s_t)     shape (act_dim,)
    log_probs[t]: log π_θ(a_t|s_t)  — needed for PPO ratio
    values[t]   : V_φ(s_t)          — critic estimate
    rewards[t]  : r_t from environment
    dones[t]    : whether episode ended at t

After collection, we compute:
    returns[t]   : G_t = r_t + γ r_{t+1} + γ² r_{t+2} + ...
    advantages[t]: A_t^GAE (see below)

## GAE — Generalized Advantage Estimation (Schulman et al. 2015)

The advantage A_t measures: how much better was action a_t
than the average action the policy would take in state s_t?

    A_t = Q(s_t, a_t) - V(s_t)

We don't have Q directly. We estimate it using TD errors:

    δ_t = r_t + γ V(s_{t+1}) - V(s_t)     ← one-step TD error

δ_t is a biased but low-variance estimate of A_t.
The full return G_t - V(s_t) is unbiased but high-variance.

GAE interpolates with λ ∈ [0,1]:

    A_t^GAE = Σ_{k=0}^{∞} (γλ)^k δ_{t+k}
            = δ_t + γλ δ_{t+1} + (γλ)² δ_{t+2} + ...

Computed efficiently backwards:
    A_T     = δ_T
    A_{t-1} = δ_{t-1} + γλ · A_t

λ=0 → pure TD (low var, high bias)
λ=1 → pure MC (high var, low bias)
λ=0.95 → standard PPO setting

## Why log_prob matters

In Q-learning you never needed to know "how likely was this action?"
Now you do, for two reasons:

1. The policy gradient theorem requires ∇_θ log π_θ(a|s)
2. PPO needs the ratio:
       r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
              = exp(log π_θ(a_t|s_t) - log π_θ_old(a_t|s_t))

log_prob is stored at collection time (under old policy),
then recomputed at update time (under new policy) to form the ratio.
"""

import numpy as np
import torch
from torch import Tensor
from typing import Optional


class RolloutBuffer:
    """
    Stores one batch of on-policy experience and computes
    GAE advantages + discounted returns.

    Usage:
        buffer = RolloutBuffer(obs_dim=14, act_dim=2, T=2048, gamma=0.99, gae_lambda=0.95)
        buffer.collect(env, actor_critic)
        for batch in buffer.get_batches(batch_size=64):
            ppo_update(batch)
        buffer.reset()
    """

    def __init__(
        self,
        obs_dim:    int,
        act_dim:    int,
        T:          int   = 2048,    # steps per rollout — how many steps before one PPO update
        gamma:      float = 0.99,    # discount factor
        gae_lambda: float = 0.95,    # GAE λ — 0=TD, 1=MC
        device:     str   = "cpu",
    ):
        self.obs_dim    = obs_dim
        self.act_dim    = act_dim
        self.T          = T
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.device     = device

        # Pre-allocate storage — faster than appending
        self.obs       = np.zeros((T, obs_dim), dtype=np.float32)
        self.actions   = np.zeros((T, act_dim), dtype=np.float32)
        self.log_probs = np.zeros((T,),         dtype=np.float32)
        self.values    = np.zeros((T,),         dtype=np.float32)
        self.rewards   = np.zeros((T,),         dtype=np.float32)
        self.dones     = np.zeros((T,),         dtype=np.float32)

        # Filled after collection by compute_gae()
        self.advantages = np.zeros((T,), dtype=np.float32)
        self.returns    = np.zeros((T,), dtype=np.float32)

        self.ptr = 0         # current write position
        self.full = False

    def reset(self):
        """Clear buffer for next rollout. Call after each PPO update."""
        self.ptr  = 0
        self.full = False

    def store(
        self,
        obs:      np.ndarray,
        action:   np.ndarray,
        log_prob: float,
        value:    float,
        reward:   float,
        done:     bool,
    ):
        """
        Store one timestep of experience.
        Called inside the collection loop at every env.step().
        """
        assert self.ptr < self.T, "Buffer full — call reset() before storing more"

        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = action
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr]    = value
        self.rewards[self.ptr]   = reward
        self.dones[self.ptr]     = float(done)

        self.ptr += 1
        if self.ptr == self.T:
            self.full = True

    def compute_gae(self, last_value: float = 0.0):
        """
        Compute GAE advantages and discounted returns.

        Must be called AFTER collect() finishes (all T steps stored).

        last_value: V(s_T) — the critic's estimate of the value of the
                    state AFTER the last stored step. This is needed to
                    bootstrap the return if the episode didn't terminate.
                    Pass 0.0 if the last episode ended (done=True).

        The backward pass:
            δ_t   = r_t + γ(1-d_t) V(s_{t+1}) - V(s_t)
            A_t   = δ_t + γλ(1-d_t) A_{t+1}

        (1 - d_t) masks the bootstrap when the episode ends:
            if done: next state value = 0 (no future)
            if not done: next state value = V(s_{t+1})
        """
        T = self.ptr   # may be less than self.T at end of training

        gae = 0.0
        for t in reversed(range(T)):
            # Bootstrap: use last_value after the final step,
            # use stored value for all other steps
            if t == T - 1:
                next_value = last_value
                next_done  = 0.0       # we handle termination via last_value=0
            else:
                next_value = self.values[t + 1]
                next_done  = self.dones[t + 1]

            # TD error δ_t = r_t + γ(1-d_{t+1}) V(s_{t+1}) - V(s_t)
            delta = (
                self.rewards[t]
                + self.gamma * next_value * (1.0 - next_done)
                - self.values[t]
            )

            # GAE: A_t = δ_t + γλ(1-d_{t+1}) A_{t+1}
            gae = delta + self.gamma * self.gae_lambda * (1.0 - next_done) * gae

            self.advantages[t] = gae

        # Returns = advantages + values
        # This is equivalent to the discounted return G_t,
        # used as the critic's regression target:
        #   V_φ(s_t) → G_t    (minimise MSE)
        self.returns[:T] = self.advantages[:T] + self.values[:T]

        # Normalise advantages over the batch — critical for training stability
        # Zero-mean, unit-variance advantages prevent exploding gradients
        # and make the clipping in PPO meaningful
        adv = self.advantages[:T]
        self.advantages[:T] = (adv - adv.mean()) / (adv.std() + 1e-8)

    def collect(self, env, actor_critic, render: bool = False):
        """
        Run the environment for T steps using actor_critic, fill the buffer.

        actor_critic: networks.actor_critic.ActorCritic
            Must implement:
                act(obs) → (action, log_prob, value)
                    obs: np.ndarray (obs_dim,)
                    action: np.ndarray (act_dim,) — sampled from π_θ(·|obs)
                    log_prob: float — log π_θ(action|obs)
                    value: float — V_φ(obs)

        The collection loop respects episode boundaries (done=True resets
        the env) but does NOT reset the buffer — we collect T steps
        across as many episodes as needed to fill the buffer.
        """
        self.reset()

        obs, _   = env.reset()
        episode_rewards = []
        ep_reward       = 0.0

        while not self.full:
            # Get action, log_prob, value from the current policy
            # All of these are computed under θ_old (the policy before this update)
            with torch.no_grad():
                action, log_prob, value = actor_critic.act(obs)

            # Step the environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            self.store(obs, action, log_prob, value, reward, done)

            ep_reward += reward
            obs        = next_obs

            if render:
                env.render()

            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                obs, _    = env.reset()

        # Bootstrap last value for GAE
        # If the last step was terminal, last_value = 0
        # If truncated (time limit), bootstrap with V(s_T)
        if done and terminated:
            last_value = 0.0
        else:
            with torch.no_grad():
                _, _, last_value = actor_critic.act(obs)

        self.compute_gae(last_value=last_value)

        return {
            "n_episodes":    len(episode_rewards),
            "mean_reward":   np.mean(episode_rewards) if episode_rewards else 0.0,
            "min_reward":    np.min(episode_rewards)  if episode_rewards else 0.0,
            "max_reward":    np.max(episode_rewards)  if episode_rewards else 0.0,
        }

    def get_batches(self, batch_size: int):
        """
        Yield randomised mini-batches for the PPO update epochs.

        Why shuffle?
        PPO does K epochs over the same rollout data. If you feed data
        in the same order every epoch, the network sees correlated batches
        and overfits to the first few steps. Shuffling breaks the correlation.

        Why mini-batches?
        Full-batch gradient descent on T=2048 steps is memory-intensive
        and gives noisier gradients than mini-batches. Mini-batches
        also allow more gradient steps per rollout.

        Yields dicts of torch.Tensors, ready for the PPO loss computation.
        """
        T       = self.ptr
        indices = np.random.permutation(T)

        for start in range(0, T, batch_size):
            idx = indices[start : start + batch_size]
            yield {
                "obs":        torch.FloatTensor(self.obs[idx]).to(self.device),
                "actions":    torch.FloatTensor(self.actions[idx]).to(self.device),
                "log_probs":  torch.FloatTensor(self.log_probs[idx]).to(self.device),
                "values":     torch.FloatTensor(self.values[idx]).to(self.device),
                "advantages": torch.FloatTensor(self.advantages[idx]).to(self.device),
                "returns":    torch.FloatTensor(self.returns[idx]).to(self.device),
            }