# env/reward.py
import numpy as np
from typing import Tuple

class RewardFunction:
    """
    Reward = (progress toward goal)
             - (time penalty)
             - (collision penalty)
             + (goal bonus)

    The key design insight: reward the *change* in distance,
    not the raw distance. This gives a dense reward signal
    every timestep instead of only at the goal.
    """

    def __init__(
        self,
        goal_radius:      float = 1.0,    # how close = "reached"
        collision_penalty: float = -100.0,  # large negative for hitting walls
        time_penalty:      float = -0.1,  # small nudge to be fast
        goal_bonus:        float = 100.0,  # big reward for arriving
        progress_scale:    float = 1.0,    # weight for distance progress
    ):
        self.goal_radius       = goal_radius
        self.collision_penalty = collision_penalty
        self.time_penalty      = time_penalty
        self.goal_bonus        = goal_bonus
        self.progress_scale    = progress_scale

    def compute(
        self,
        pos:           np.ndarray,   # current position after step
        prev_pos:      np.ndarray,   # position before step
        goal:          np.ndarray,   # goal position
        collided:      bool,
    ) -> Tuple[float, bool]:
        """
        Returns (reward, done)
        """
        if collided:
            return self.collision_penalty, True   # episode ends on collision

        dist_now  = np.linalg.norm(pos - goal)
        dist_prev = np.linalg.norm(prev_pos - goal)

        # Progress reward: positive when moving closer, negative when drifting away
        progress = (dist_prev - dist_now) * self.progress_scale

        reward = progress + self.time_penalty

        # Check goal reached
        if dist_now < self.goal_radius:
            reward += self.goal_bonus
            return reward, True   # episode ends on success

        return reward, False