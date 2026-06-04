# algos/classical/dynamic_programming.py
"""
Dynamic Programming — Policy Iteration & Value Iteration
=========================================================
Requires a full model of the environment: P(s'|s,a) and r(s,a,s').
We build an approximate model by exhaustively running the discrete env.

Key formulas (see THEORY.md §3):

  Policy Evaluation:
    V(s) ← Σ_a π(a|s) Σ_s' P(s'|s,a) [r + γ V(s')]

  Policy Improvement:
    π'(s) = argmax_a Σ_s' P(s'|s,a) [r + γ V(s')]

  Value Iteration (combines both):
    V(s) ← max_a Σ_s' P(s'|s,a) [r + γ V(s')]
"""

import numpy as np
from typing import Dict, Tuple
from tqdm import tqdm


# ── Model Builder ──────────────────────────────────────────────────────────────

def build_model(
    env,
    n_samples_per_sa: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate the transition model P(s'|s,a) and R(s,a) by sampling.

    Since our env is stochastic only in reset() (random obstacle placement),
    we treat each (s, a) pair as near-deterministic and average a few samples.

    Returns
    -------
    P : np.ndarray, shape (n_states, n_actions, n_states)
        P[s, a, s'] = probability of transitioning to s' from (s,a)
    R : np.ndarray, shape (n_states, n_actions)
        R[s, a] = expected immediate reward for (s, a)
    """
    n_s = env.n_states
    n_a = env.n_actions
    P   = np.zeros((n_s, n_a, n_s), dtype=np.float32)
    R   = np.zeros((n_s, n_a),      dtype=np.float32)

    print("Building environment model (this takes a moment)...")

    for s in tqdm(range(n_s), desc="Sweeping states"):
        for a in range(n_a):
            next_states = []
            rewards     = []

            for _ in range(n_samples_per_sa):
                # Force the env into state s by setting drone position
                env.env.reset()
                cell_pos = env.state_to_pos(s)
                env.env.unwrapped.pos = cell_pos.copy()
                env.env.unwrapped.vel = np.zeros(2)

                s_next, r, terminated, truncated, _ = env.step(a)

                if terminated or truncated:
                    # Terminal state maps to itself with zero future value
                    next_states.append(s)
                    rewards.append(r)
                else:
                    next_states.append(s_next)
                    rewards.append(r)

            # Build transition distribution from samples
            for s_next, r in zip(next_states, rewards):
                P[s, a, s_next] += 1.0 / n_samples_per_sa
                R[s, a]         += r   / n_samples_per_sa

    return P, R


# ── Policy Evaluation ──────────────────────────────────────────────────────────

def policy_evaluation(
    policy: np.ndarray,    # shape (n_states, n_actions) — probability distribution
    P:      np.ndarray,    # transition model
    R:      np.ndarray,    # reward model
    gamma:  float = 0.99,
    theta:  float = 1e-4,  # convergence threshold
    max_iter: int = 1000,
) -> np.ndarray:
    """
    Iteratively compute V^π for a given policy using the Bellman expectation.

    V(s) ← Σ_a π(a|s) [ R(s,a) + γ Σ_s' P(s'|s,a) V(s') ]

    We iterate this update for all states until the max change is < theta.
    This is called a 'sweep' — one full pass over all states.
    """
    n_states = P.shape[0]
    V        = np.zeros(n_states)

    for iteration in range(max_iter):
        delta = 0.0   # track max change this sweep

        for s in range(n_states):
            v_old = V[s]

            # Bellman expectation update
            # V(s) = Σ_a π(a|s) * [ R(s,a) + γ * Σ_s' P(s,a,s') * V(s') ]
            v_new = 0.0
            for a in range(policy.shape[1]):
                future = np.dot(P[s, a], V)          # Σ_s' P(s,a,s') V(s')
                v_new += policy[s, a] * (R[s, a] + gamma * future)

            V[s]  = v_new
            delta = max(delta, abs(v_old - v_new))

        if delta < theta:
            print(f"  Policy evaluation converged in {iteration + 1} sweeps (Δ={delta:.2e})")
            break

    return V


# ── Policy Improvement ─────────────────────────────────────────────────────────

def policy_improvement(
    V:     np.ndarray,
    P:     np.ndarray,
    R:     np.ndarray,
    gamma: float = 0.99,
) -> Tuple[np.ndarray, bool]:
    """
    Given V^π, compute a greedy improved policy π'.

    π'(s) = argmax_a [ R(s,a) + γ Σ_s' P(s,a,s') V(s') ]

    Returns the new deterministic policy (one-hot) and whether it changed.
    """
    n_states, n_actions = P.shape[0], P.shape[1]
    policy_stable = True
    new_policy    = np.zeros((n_states, n_actions))

    for s in range(n_states):
        # Compute Q(s, a) for all actions
        q_values = R[s] + gamma * P[s] @ V    # shape (n_actions,)

        best_a   = np.argmax(q_values)
        old_best = np.argmax(new_policy[s]) if new_policy[s].sum() > 0 else -1

        new_policy[s]          = 0.0
        new_policy[s, best_a]  = 1.0   # deterministic greedy policy

        if old_best != best_a:
            policy_stable = False

    return new_policy, policy_stable


# ── Policy Iteration ───────────────────────────────────────────────────────────

def policy_iteration(
    env,
    P:     np.ndarray,
    R:     np.ndarray,
    gamma: float = 0.99,
    theta: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Policy Iteration: alternate evaluation and improvement until convergence.

    1. Start with random policy
    2. Evaluate it fully  → V^π
    3. Improve greedily   → π'
    4. If π' == π, stop. Else go to 2.

    Guaranteed to converge to π* in finite steps for finite MDPs.

    Returns (optimal_policy, optimal_V)
    """
    n_states  = P.shape[0]
    n_actions = P.shape[1]

    # Start with uniform random policy
    policy = np.ones((n_states, n_actions)) / n_actions

    print("\n=== Policy Iteration ===")
    for i in range(100):   # max outer iterations
        print(f"\nIteration {i + 1}")
# algos/classical/dynamic_programming.py
"""
Dynamic Programming — Policy Iteration & Value Iteration
=========================================================
Requires a full model of the environment: P(s'|s,a) and r(s,a,s').
We build an approximate model by exhaustively running the discrete env.

Key formulas (see THEORY.md §3):

  Policy Evaluation:
    V(s) ← Σ_a π(a|s) Σ_s' P(s'|s,a) [r + γ V(s')]

  Policy Improvement:
    π'(s) = argmax_a Σ_s' P(s'|s,a) [r + γ V(s')]

  Value Iteration (combines both):
    V(s) ← max_a Σ_s' P(s'|s,a) [r + γ V(s')]
"""

import numpy as np
from typing import Dict, Tuple
from tqdm import tqdm


# ── Model Builder ──────────────────────────────────────────────────────────────

def build_model(
    env,
    n_samples_per_sa: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate the transition model P(s'|s,a) and R(s,a) by sampling.

    Since our env is stochastic only in reset() (random obstacle placement),
    we treat each (s, a) pair as near-deterministic and average a few samples.

    Returns
    -------
    P : np.ndarray, shape (n_states, n_actions, n_states)
        P[s, a, s'] = probability of transitioning to s' from (s,a)
    R : np.ndarray, shape (n_states, n_actions)
        R[s, a] = expected immediate reward for (s, a)
    """
    n_s = env.n_states
    n_a = env.n_actions
    P   = np.zeros((n_s, n_a, n_s), dtype=np.float32)
    R   = np.zeros((n_s, n_a),      dtype=np.float32)

    print("Building environment model (this takes a moment)...")

    for s in tqdm(range(n_s), desc="Sweeping states"):
        for a in range(n_a):
            next_states = []
            rewards     = []

            for _ in range(n_samples_per_sa):
                # Force the env into state s by setting drone position
                env.env.reset()
                cell_pos = env.state_to_pos(s)
                env.env.unwrapped.pos = cell_pos.copy()
                env.env.unwrapped.vel = np.zeros(2)

                s_next, r, terminated, truncated, _ = env.step(a)

                if terminated or truncated:
                    # Terminal state maps to itself with zero future value
                    next_states.append(s)
                    rewards.append(r)
                else:
                    next_states.append(s_next)
                    rewards.append(r)

            # Build transition distribution from samples
            for s_next, r in zip(next_states, rewards):
                P[s, a, s_next] += 1.0 / n_samples_per_sa
                R[s, a]         += r   / n_samples_per_sa

    return P, R


# ── Policy Evaluation ──────────────────────────────────────────────────────────

def policy_evaluation(
    policy: np.ndarray,    # shape (n_states, n_actions) — probability distribution
    P:      np.ndarray,    # transition model
    R:      np.ndarray,    # reward model
    gamma:  float = 0.99,
    theta:  float = 1e-4,  # convergence threshold
    max_iter: int = 1000,
) -> np.ndarray:
    """
    Iteratively compute V^π for a given policy using the Bellman expectation.

    V(s) ← Σ_a π(a|s) [ R(s,a) + γ Σ_s' P(s'|s,a) V(s') ]

    We iterate this update for all states until the max change is < theta.
    This is called a 'sweep' — one full pass over all states.
    """
    n_states = P.shape[0]
    V        = np.zeros(n_states)

    for iteration in range(max_iter):
        delta = 0.0   # track max change this sweep

        for s in range(n_states):
            v_old = V[s]

            # Bellman expectation update
            # V(s) = Σ_a π(a|s) * [ R(s,a) + γ * Σ_s' P(s,a,s') * V(s') ]
            v_new = 0.0
            for a in range(policy.shape[1]):
                future = np.dot(P[s, a], V)          # Σ_s' P(s,a,s') V(s')
                v_new += policy[s, a] * (R[s, a] + gamma * future)

            V[s]  = v_new
            delta = max(delta, abs(v_old - v_new))

        if delta < theta:
            print(f"  Policy evaluation converged in {iteration + 1} sweeps (Δ={delta:.2e})")
            break

    return V


# ── Policy Improvement ─────────────────────────────────────────────────────────

def policy_improvement(
    V:     np.ndarray,
    P:     np.ndarray,
    R:     np.ndarray,
    gamma: float = 0.99,
) -> Tuple[np.ndarray, bool]:
    """
    Given V^π, compute a greedy improved policy π'.

    π'(s) = argmax_a [ R(s,a) + γ Σ_s' P(s,a,s') V(s') ]

    Returns the new deterministic policy (one-hot) and whether it changed.
    """
    n_states, n_actions = P.shape[0], P.shape[1]
    policy_stable = True
    new_policy    = np.zeros((n_states, n_actions))

    for s in range(n_states):
        # Compute Q(s, a) for all actions
        q_values = R[s] + gamma * P[s] @ V    # shape (n_actions,)

        best_a   = np.argmax(q_values)
        old_best = np.argmax(new_policy[s]) if new_policy[s].sum() > 0 else -1

        new_policy[s]          = 0.0
        new_policy[s, best_a]  = 1.0   # deterministic greedy policy

        if old_best != best_a:
            policy_stable = False

    return new_policy, policy_stable


# ── Policy Iteration ───────────────────────────────────────────────────────────

def policy_iteration(
    env,
    P:     np.ndarray,
    R:     np.ndarray,
    gamma: float = 0.99,
    theta: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Policy Iteration: alternate evaluation and improvement until convergence.

    1. Start with random policy
    2. Evaluate it fully  → V^π
    3. Improve greedily   → π'
    4. If π' == π, stop. Else go to 2.

    Guaranteed to converge to π* in finite steps for finite MDPs.

    Returns (optimal_policy, optimal_V)
    """
    n_states  = P.shape[0]
    n_actions = P.shape[1]

    # Start with uniform random policy
    policy = np.ones((n_states, n_actions)) / n_actions

    print("\n=== Policy Iteration ===")
    for i in range(100):   # max outer iterations
        print(f"\nIteration {i + 1}")

        # Step 1: Evaluate current policy
        V = policy_evaluation(policy, P, R, gamma, theta)

        # Step 2: Improve policy greedily
        policy, stable = policy_improvement(V, P, R, gamma)

        if stable:
            print(f"  Policy converged after {i + 1} iterations.")
            break

    return policy, V


# ── Value Iteration ────────────────────────────────────────────────────────────

def value_iteration(
    env,
    P:       np.ndarray,
    R:       np.ndarray,
    gamma:   float = 0.99,
    theta:   float = 1e-4,
    max_iter: int  = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Value Iteration: directly compute V* without waiting for policy convergence.

    V(s) ← max_a [ R(s,a) + γ Σ_s' P(s,a,s') V(s') ]

    One equation, repeated until convergence. Faster than policy iteration
    because we don't do full policy evaluation each step.

    Returns (optimal_policy, optimal_V)
    """
    n_states  = P.shape[0]
    n_actions = P.shape[1]
    V         = np.zeros(n_states)

    print("\n=== Value Iteration ===")
    for iteration in range(max_iter):
        delta = 0.0

        for s in range(n_states):
            v_old = V[s]

            # Q(s, a) for all actions at once — fast vectorised form
            q_values = R[s] + gamma * (P[s] @ V)   # shape (n_actions,)
            V[s]     = np.max(q_values)             # Bellman optimality update

            delta = max(delta, abs(v_old - V[s]))

        if iteration % 50 == 0:
            print(f"  Sweep {iteration:4d} | Δ = {delta:.6f}")

        if delta < theta:
            print(f"  Value Iteration converged in {iteration + 1} sweeps.")
            break

    # Extract greedy policy from V*
    policy = np.zeros((n_states, n_actions))
    for s in range(n_states):
        q_values          = R[s] + gamma * (P[s] @ V)
        policy[s, np.argmax(q_values)] = 1.0

    return policy, V
        # Step 1: Evaluate current policy
        V = policy_evaluation(policy, P, R, gamma, theta)

        # Step 2: Improve policy greedily
        policy, stable = policy_improvement(V, P, R, gamma)

        if stable:
            print(f"  Policy converged after {i + 1} iterations.")
            break

    return policy, V


# ── Value Iteration ────────────────────────────────────────────────────────────

def value_iteration(
    env,
    P:       np.ndarray,
    R:       np.ndarray,
    gamma:   float = 0.99,
    theta:   float = 1e-4,
    max_iter: int  = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Value Iteration: directly compute V* without waiting for policy convergence.

    V(s) ← max_a [ R(s,a) + γ Σ_s' P(s,a,s') V(s') ]

    One equation, repeated until convergence. Faster than policy iteration
    because we don't do full policy evaluation each step.

    Returns (optimal_policy, optimal_V)
    """
    n_states  = P.shape[0]
    n_actions = P.shape[1]
    V         = np.zeros(n_states)

    print("\n=== Value Iteration ===")
    for iteration in range(max_iter):
        delta = 0.0

        for s in range(n_states):
            v_old = V[s]

            # Q(s, a) for all actions at once — fast vectorised form
            q_values = R[s] + gamma * (P[s] @ V)   # shape (n_actions,)
            V[s]     = np.max(q_values)             # Bellman optimality update

            delta = max(delta, abs(v_old - V[s]))

        if iteration % 50 == 0:
            print(f"  Sweep {iteration:4d} | Δ = {delta:.6f}")

        if delta < theta:
            print(f"  Value Iteration converged in {iteration + 1} sweeps.")
            break

    # Extract greedy policy from V*
    policy = np.zeros((n_states, n_actions))
    for s in range(n_states):
        q_values          = R[s] + gamma * (P[s] @ V)
        policy[s, np.argmax(q_values)] = 1.0

    return policy, V