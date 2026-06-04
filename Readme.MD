<!-- Experimenting repository to implement almost all RL-Algorithms from calssic one (Q-Learning/SARSA) to the latest one (PPO / Actor-Critic ).
Algorithms are used to guide a drone that moves in 2D space, from point S to destination S, while trying to avoid obstacles, by giving velocity commands : Vx and Vy.

<img src="docs/imgs/example.png"/>

# Environement & Agent:

The RL frameork is used in two main reasons : when we don't have serialized dataset, or when we don't have a required loss-function to modelize such desire (like the case for human preference in desired LLM responses, hence the use of RLHF techniques to align with the model ).
in this framwork, data is required through some interaction between our system that we want to train, named the **Agent**, that performs some actions in a **Environement**, and based on the actions the env either rewards him positively or negatively.

For our case, we want to train a drone policy to navigate in an environemnt without crashing, and to attend a desired destination in the fastest way possible.

<img src="docs/imgs/env.png"/>

- The drone is commanded by Velocity vector : (Vx,Vy). the drone scenario tells that there is a gravity field in y-axis, meaning if no command $V_y=0$ the drone will just fall and crash.

# RL-Vocabulary (under this example) :

- policy : the policy Function is the function that gives you wich action to take given a stat $\pi(a|s)$. the aim of RL algorithms is to find the best policy $\pi^*(a|s)$, that will give us the shortest and safest path in our example here.

- Reward : The reawrd is the key wining for RL, can be seen as the loss-function, and must be wisely chosen deoending of the scenario, for our drone example :
  - _progress_ : to encourage moving closer to Destination. if deosn't exist, the algo will find hard timeto know where to start.
    computed as (dist_prev - dist_now) \* scale. the scale controlls how effective this reward is : if null, the drone will wander not knowing from wich direction start moving, if too high, the drone won't consider to maybe go back to find maybe a shorter path. set to +1
  - _time penalty_ : this is a negative reward, pushing the algo to arrive faster so he donsn't loose much from this penalty. set to -1
  - _collision penalty_ : another negative reawrd, but high in magnitude, since we definetly never want to do a collision. set to -10
  - _goal_bonus_ : this gives a huge bonus when destination is reached. focusing that the the big objective is to arrive to D. set to +100
  - final reward function : r = progress - time_penalty - collision_penality + goal_bonus.
    to tune the values on your preference, visit : "env/reward.py"

- Value function

- Horizon -->

# RL101

**RL101** is an experimentation repository for implementing and understanding Reinforcement Learning algorithms — from classical methods (**Q-Learning**, **SARSA**) to modern approaches (**Actor-Critic**, **PPO**).

The goal is to train a **2D drone navigation policy** that moves from a start position **S** to a destination **D** while avoiding obstacles.

The drone is controlled through Acceleration commands $ (a_x, a_y) $

<img src="docs/imgs/example.png"/>

---

# Environment & Agent

Reinforcement Learning (RL) is useful when:

- No labeled/serialized dataset is available
- The desired behavior cannot be expressed through a standard loss function

In RL, an **Agent** interacts with an **Environment**:

- The **Agent** performs actions
- The **Environment** returns rewards or penalties
- The Agent learns a policy that maximizes long-term reward

In this project, the objective is to train a drone that:

- Reaches the destination safely
- Avoids obstacles
- Minimizes travel time

<img src="docs/imgs/env.png"/>

### Drone Dynamics

The drone receives velocity commands: $(\mathcal{a}_x, \mathcal{a}_y)$

The environment simulates **gravity along the y-axis**.  
If: $\mathcal{a}_y = 0$. the drone naturally falls and eventually crashes.


### Covered Topics : 
- **Classic RL** : dynamic_programming / Q-learning / TD-learning / Monte-Carlo.
  - these algorithms run on a finite set of state and actions, a discretiation of the drone environement is required and done with the script : `env/discrete_wrapper.py`
- **Deep RL** (coming)
- **Policy Gradient** (coming)

---

# RL Vocabulary (Using this Drone)

### I. Policy

A **policy** determines which action to take given a state $\pi(a|s)$
RL aims to learn the **optimal policy**: $\pi^*(a|s)$

that produces the safest and shortest trajectory.

---

### II. Reward Function

The reward function defines the desired behavior.

For this drone task:

| Reward Component      | Purpose                              |  Value |
| --------------------- | ------------------------------------ | -----: |
| **Progress Reward**   | Encourages moving toward destination |   `+1` |
| **Time Penalty**      | Encourages faster arrival            |   `-1` |
| **Collision Penalty** | Strongly discourages crashes         |  `-10` |
| **Goal Bonus**        | Reward for reaching destination      | `+100` |

Final reward:

\[
r = progress - timePenalty - collisionPenalty + goalBonus
\]

*Reward tuning can be modified in*: `env/reward.py`

---

### III. Value Function

The **value function** estimates the expected future reward from a state: $V(s)$

It helps the agent evaluate whether a state is promising or dangerous.

---

### IV. Horizon

The **horizon** defines how far into the future the agent considers rewards.

- **Short horizon** → greedy behavior
- **Long horizon** → better long-term planning
