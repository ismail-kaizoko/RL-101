# algos/classical/dynamic_programming.py
"""
Dynamic Programming — Policy Iteration & Value Iteration
=========================================================
Requires a full model: P(s'|s,a) and R(s,a).
We approximate it by exhaustively sampling the discrete env.

Bellman Optimality:
    V*(s) = max_a [ R(s,a) + γ Σ_s' P(s,a,s') V*(s') ]
    π*(s) = argmax_a [ R(s,a) + γ Σ_s' P(s,a,s') V*(s') ]

See algos/classical/THEORY.md §3 for full derivation.
"""

import numpy as np
from typing import Tuple
from tqdm import tqdm


# ── Model builder ──────────────────────────────────────────────────────────────

def build_model(env, n_samples: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate P(s'|s,a) and R(s,a) by sampling the env.

    P : (n_states, n_actions, n_states)  transition probabilities
    R : (n_states, n_actions)            expected immediate reward
    """
    n_s, n_a = env.n_states, env.n_actions
    P = np.zeros((n_s, n_a, n_s), dtype=np.float32)
    R = np.zeros((n_s, n_a),      dtype=np.float32)

    print("Building environment model...")
    for s in tqdm(range(n_s), desc="States"):
        for a in range(n_a):
            for _ in range(n_samples):
                env.env.reset()
                env.env.unwrapped.pos = env.state_to_pos(s).copy()
                env.env.unwrapped.vel = np.zeros(2)
                s2, r, term, trunc, _ = env.step(a)
                s2 = s if (term or trunc) else s2
                P[s, a, s2] += 1.0 / n_samples
                R[s, a]     += r   / n_samples
    return P, R


# ── Policy evaluation ──────────────────────────────────────────────────────────

def policy_evaluation(
    policy: np.ndarray,   # (n_states, n_actions)  probability distribution
    P:      np.ndarray,
    R:      np.ndarray,
    gamma:  float = 0.99,
    theta:  float = 1e-4,
) -> np.ndarray:
    """
    Iteratively compute V^π:
        V(s) ← Σ_a π(a|s) [ R(s,a) + γ Σ_s' P(s,a,s') V(s') ]
    """
    V = np.zeros(P.shape[0])
    for i in range(10_000):
        delta = 0.0
        for s in range(P.shape[0]):
            v     = V[s]
            V[s]  = np.sum(policy[s] * (R[s] + gamma * (P[s] @ V)))
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            print(f"  Policy evaluation converged in {i+1} sweeps")
            break
    return V


# ── Policy improvement ─────────────────────────────────────────────────────────

def policy_improvement(
    V: np.ndarray, P: np.ndarray, R: np.ndarray, gamma: float = 0.99
) -> Tuple[np.ndarray, bool]:
    """
    π'(s) = argmax_a [ R(s,a) + γ Σ_s' P(s,a,s') V(s') ]
    Returns (new_policy, policy_stable).
    """
    n_s, n_a  = P.shape[0], P.shape[1]
    old_best  = np.argmax(np.ones((n_s, n_a)), axis=1)   # placeholder
    Q         = R + gamma * (P @ V)                       # (n_s, n_a)
    new_best  = np.argmax(Q, axis=1)
    stable    = np.all(old_best == new_best)
    policy    = np.zeros((n_s, n_a))
    policy[np.arange(n_s), new_best] = 1.0
    return policy, stable


# ── Policy iteration ───────────────────────────────────────────────────────────

def policy_iteration(
    P: np.ndarray, R: np.ndarray, gamma: float = 0.99, theta: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Alternate evaluation → improvement until policy stops changing.
    Guaranteed to converge to π* for finite MDPs.

    Returns (policy, V)
    """
    n_s, n_a = P.shape[0], P.shape[1]
    policy   = np.ones((n_s, n_a)) / n_a   # start: uniform random

    print("\n=== Policy Iteration ===")
    for i in range(200):
        print(f"  Iteration {i+1}")
        V              = policy_evaluation(policy, P, R, gamma, theta)
        policy, stable = policy_improvement(V, P, R, gamma)
        if stable:
            print(f"  Converged after {i+1} iterations.")
            break
    return policy, V


# ── Value iteration ────────────────────────────────────────────────────────────

def value_iteration(
    P: np.ndarray, R: np.ndarray, gamma: float = 0.99,
    theta: float = 1e-4, max_iter: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bellman optimality update, repeated until convergence:
        V(s) ← max_a [ R(s,a) + γ Σ_s' P(s,a,s') V(s') ]

    Faster than policy iteration — no inner evaluation loop.
    Returns (policy, V)
    """
    n_s, n_a = P.shape[0], P.shape[1]
    V        = np.zeros(n_s)

    print("\n=== Value Iteration ===")
    for i in range(max_iter):
        Q     = R + gamma * (P @ V)     # (n_s, n_a) — vectorised Bellman
        V_new = np.max(Q, axis=1)
        delta = np.max(np.abs(V_new - V))
        V     = V_new
        if i % 50 == 0:
            print(f"  Sweep {i:4d} | Δ = {delta:.6f}")
        if delta < theta:
            print(f"  Converged in {i+1} sweeps.")
            break

    policy = np.zeros((n_s, n_a))
    policy[np.arange(n_s), np.argmax(R + gamma * (P @ V), axis=1)] = 1.0
    return policy, V