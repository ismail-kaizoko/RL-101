# Classical Reinforcement Learning — Theory

## The Core Problem

An agent interacts with an environment over discrete timesteps.
At each step it observes a **state** $s$, takes an **action** $a$,
receives a **reward** $r$, and transitions to a new state $s'$.

The goal: find a **policy** $\pi(a|s)$ that maximises the expected
**discounted cumulative reward** (the *return*):

$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

where $\gamma \in [0, 1)$ is the **discount factor**.
- $\gamma \approx 0$ → agent is myopic, only cares about immediate reward
- $\gamma \approx 1$ → agent plans far into the future

---

## 1. Value Functions

### State-Value Function $V^\pi(s)$

Expected return when starting in state $s$ and following policy $\pi$:

$$V^\pi(s) = \mathbb{E}_\pi\left[ G_t \mid s_t = s \right]$$

### Action-Value Function $Q^\pi(s, a)$

Expected return when starting in state $s$, taking action $a$,
then following policy $\pi$:

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[ G_t \mid s_t = s,\ a_t = a \right]$$

### Relationship between V and Q

$$V^\pi(s) = \sum_a \pi(a|s)\, Q^\pi(s, a)$$

$$Q^\pi(s, a) = \mathbb{E}\left[ r + \gamma V^\pi(s') \right]$$

---

## 2. The Bellman Equations

The Bellman equations are the **recursive definition** of value functions.
They are the backbone of every algorithm in this phase.

### Bellman Expectation (for a given policy $\pi$)

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)\left[ r(s,a,s') + \gamma V^\pi(s') \right]$$

$$Q^\pi(s,a) = \sum_{s'} P(s'|s,a)\left[ r(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a') \right]$$

### Bellman Optimality (for the optimal policy $\pi^*$)

$$V^*(s) = \max_a \sum_{s'} P(s'|s,a)\left[ r(s,a,s') + \gamma V^*(s') \right]$$

$$Q^*(s,a) = \sum_{s'} P(s'|s,a)\left[ r(s,a,s') + \gamma \max_{a'} Q^*(s',a') \right]$$

The **optimal policy** is then simply:

$$\pi^*(s) = \arg\max_a Q^*(s, a)$$

---

## 3. Dynamic Programming

Requires a **known model** of the environment: $P(s'|s,a)$ and $r(s,a,s')$.
We iterate the Bellman equations directly until convergence.

### Policy Evaluation (compute $V^\pi$ for a fixed $\pi$)

Repeat until $\max_s |V_{new}(s) - V_{old}(s)| < \theta$:

$$V(s) \leftarrow \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)\left[ r + \gamma V(s') \right]$$

### Policy Improvement

$$\pi'(s) = \arg\max_a \sum_{s'} P(s'|s,a)\left[ r + \gamma V(s') \right]$$

### Policy Iteration

Alternate Policy Evaluation → Policy Improvement until the policy stops changing.
**Guaranteed to converge** to $\pi^*$ in finite steps for finite MDPs.

### Value Iteration

Collapse evaluation + improvement into one sweep.
Repeat until convergence:

$$V(s) \leftarrow \max_a \sum_{s'} P(s'|s,a)\left[ r + \gamma V(s') \right]$$

Faster in practice because it doesn't wait for full policy evaluation.

---

## 4. Monte Carlo Methods

**No model needed.** Learn from complete episodes of experience.

### MC Prediction (estimating $V^\pi$)

After each episode, for every visited state $s$:

$$V(s) \leftarrow V(s) + \alpha \left[ G_t - V(s) \right]$$

where $G_t$ is the **actual return** observed from that state.

Two variants:
- **First-visit MC** — only update on the first visit to $s$ per episode
- **Every-visit MC** — update on every visit

### MC Control (finding $\pi^*$) — $\varepsilon$-greedy

Learn $Q(s,a)$ from episodes, then improve greedily.
Must use **exploring starts** or $\varepsilon$-greedy to ensure all
$(s,a)$ pairs are visited.

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ G_t - Q(s,a) \right]$$

Policy: with prob $\varepsilon$ take random action, else $\arg\max_a Q(s,a)$.

### Key MC insight
MC uses the **full return** $G_t$ — no bootstrapping.
This means:
- (+) Unbiased estimates of $V$ and $Q$
- (-) High variance (one unlucky episode throws off the estimate)
- (-) Must wait until episode ends before updating

---

## 5. Temporal Difference Learning — TD(0)

Combines the model-free advantage of MC with the online learning
advantage of DP. **The most important idea in classical RL.**

Update after **every step**, without waiting for the episode to end:

$$V(s) \leftarrow V(s) + \alpha \underbrace{\left[ r + \gamma V(s') - V(s) \right]}_{\text{TD error } \delta}$$

The quantity $\delta = r + \gamma V(s') - V(s)$ is the **TD error**:
how wrong was our current estimate compared to what we just observed?

- $r + \gamma V(s')$ → **TD target** (our new estimate of what $V(s)$ should be)
- $V(s)$ → **old estimate** (what we thought before)

### Bias-Variance tradeoff

| Method | Bias | Variance | Updates |
|--------|------|----------|---------|
| MC     | None | High     | End of episode |
| TD(0)  | Yes (bootstraps) | Low | Every step |
| TD(n)  | Intermediate | Intermediate | Every n steps |

---

## 6. SARSA — On-Policy TD Control

Learns $Q(s,a)$ while **following the same policy** it's improving.
Name comes from the quintuple: $(S, A, R, S', A')$.

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma Q(s', a') - Q(s,a) \right]$$

where $a'$ is the action **actually taken** in $s'$ (from the current $\varepsilon$-greedy policy).

- **On-policy**: the policy being evaluated = the policy generating data
- Converges to the **optimal $\varepsilon$-soft policy**
- Safer in environments with dangerous states (it accounts for its own exploration)

---

## 7. Q-Learning — Off-Policy TD Control

The classic algorithm. Learns $Q^*$ regardless of which policy generates the data.

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s,a) \right]$$

The key difference from SARSA: we use $\max_{a'} Q(s', a')$ — the **best possible** 
action in $s'$ — regardless of what the agent actually does next.

- **Off-policy**: the policy being evaluated ≠ the policy generating data
- **Behaviour policy**: $\varepsilon$-greedy (explores)
- **Target policy**: greedy (always takes best action)
- Converges directly to $Q^*$ (with decaying $\alpha$ and $\varepsilon$)

### SARSA vs Q-Learning

```
State: drone near a wall

SARSA:   knows it might explore randomly → penalises states near danger
Q-Learning: always imagines taking the best action → more aggressive, 
            may fly closer to walls
```

---

## 8. Discretisation for the Drone

The drone lives in a **continuous** state space $(x, y, v_x, v_y, \ldots)$.
Classical tabular methods need a **finite** state space.

### Approach: Uniform Grid

Divide each dimension into $n$ bins:

$$s_{discrete} = \left\lfloor \frac{s_{continuous} - s_{min}}{s_{max} - s_{min}} \cdot n \right\rfloor$$

For our drone we discretise only position $(x, y)$ for simplicity:
- Grid: $20 \times 20 = 400$ states
- Actions: 8 directions + hover = 9 discrete actions

### Tile Coding (better for Q-learning)

Instead of one grid, use multiple overlapping grids (tiles) offset from each other.
Each tile votes independently. This gives:
- Smoother value function approximation
- Generalisation across similar states
- Still linear — fast to compute

$$Q(s, a) = \sum_{i=1}^{n_{tiles}} w_i \cdot \mathbf{1}[s \in \text{tile}_i]$$

---

## 9. Convergence Conditions

All TD methods converge under the **Robbins-Monro conditions**:

$$\sum_{t=0}^{\infty} \alpha_t = \infty \qquad \text{and} \qquad \sum_{t=0}^{\infty} \alpha_t^2 < \infty$$

A common schedule that satisfies this:

$$\alpha_t = \frac{\alpha_0}{1 + \text{decay} \cdot t}$$

For $\varepsilon$-greedy exploration, we also need $\varepsilon \to 0$ over time
(but slowly enough that all state-action pairs are visited infinitely often).

---

## Summary — When to Use What

| Algorithm | Model needed? | Updates | Converges to |
|-----------|--------------|---------|--------------|
| Policy Iteration | Yes | Per sweep | $\pi^*$ exactly |
| Value Iteration | Yes | Per sweep | $V^*$ exactly |
| Monte Carlo | No | End of episode | $Q^*$ (unbiased) |
| TD(0) / SARSA | No | Every step | $Q^{\pi}$ (on-policy) |
| Q-Learning | No | Every step | $Q^*$ (off-policy) |