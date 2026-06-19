# env/discrete_wrapper.py
"""
Wraps DroneEnv2D into a fully discrete (s, a) space for tabular classical RL.

State  : (x, y) position binned onto a grid_size × grid_size grid
Action : 8 compass directions + hover  (9 discrete actions)

Why only (x, y)?
  A full discretisation of all 14 state dims at even 10 bins/dim gives 10^14
  states — impossible to tabulate. This deliberate simplification is the
  teaching moment that motivates deep RL in Phase 2.

Action map:
    7  0  1
    6  8  2     (8 = hover: zero thrust)
    5  4  3
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DiscreteWrapper(gym.Wrapper):

    G = 9.8

    AX_VALUES = np.array(
        [-2*G, -G, 0.0, G, 2*G],
        dtype=np.float32
    )

    AY_VALUES = np.array(
        [-2*G, -G, 0.0, G, 2*G],
        dtype=np.float32
    )


    def __init__(self, env: gym.Env, grid_size: int = 20):
        super().__init__(env)
        self.grid_size  = grid_size
        self.world_size = env.unwrapped.world_size
        # 25 discrete velocity actions
        self.action_table = np.array(
            [
                [ax, ay]
                for ay in self.AY_VALUES
                for ax in self.AX_VALUES
            ],
            dtype=np.float32
        )

        self.n_actions  = len(self.action_table)
        self.n_states   = grid_size * grid_size


        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space      = spaces.Discrete(self.n_actions)

    # ── Discretisation ─────────────────────────────────────────────────────────

    def pos_to_state(self, pos: np.ndarray) -> int:
        bx = int(np.clip(pos[0] / self.world_size * self.grid_size, 0, self.grid_size - 1))
        by = int(np.clip(pos[1] / self.world_size * self.grid_size, 0, self.grid_size - 1))
        return by * self.grid_size + bx

    def state_to_pos(self, state: int) -> np.ndarray:
        """Centre of the grid cell in world coordinates."""
        by, bx  = divmod(state, self.grid_size)
        cell    = self.world_size / self.grid_size
        return np.array([(bx + 0.5) * cell, (by + 0.5) * cell])

    def _action_to_thrust(self, action: int) -> np.ndarray:
        return self.action_table[action]

    def _discrete_state(self) -> int:
        return self.pos_to_state(self.env.unwrapped.pos)

    def goal_state(self) -> int:
        return self.pos_to_state(self.env.unwrapped.goal)

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        return self._discrete_state(), info

    def step(self, action: int):
        thrust   = self._action_to_thrust(action)
        norm_act = thrust / self.env.unwrapped.physics.max_thrust
        _, reward, terminated, truncated, info = self.env.step(norm_act)
        return self._discrete_state(), reward, terminated, truncated, info
    
    def step(self, action: int):
        thrust   = self._action_to_thrust(action)     # [ax, ay] in m/s²
        norm_act = thrust / self.env.unwrapped.physics.max_thrust   # normalize to [-1,1] for DroneEnv2D.step()
        _, reward, terminated, truncated, info = self.env.step(norm_act)
        return self._discrete_state(), reward, terminated, truncated, info