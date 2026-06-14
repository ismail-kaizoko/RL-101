# env/drone_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .physics   import DronePhysics
from .obstacles import generate_obstacles, Rect
from .reward    import RewardFunction
from typing import Optional


class DroneEnv2D(gym.Env):
    """
    2D drone navigation environment.

    Observation (state):
        [x, y, vx, vy, goal_dx, goal_dy, *obstacle_distances]
        - (x, y)           : drone position, normalised to [0, 1]
        - (vx, vy)         : current velocity
        - (goal_dx, goal_dy): vector from drone to goal (normalised)
        - obstacle_distances: distances to N nearest obstacles (or world edge)

    Action:
        [dax, day] — acceleration command, clipped to [-1, 1]

    This follows the gymnasium API so every RL library works out of the box.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        world_size:    float = 20.0,   # world is a square [0, world_size]^2
        max_steps:     int   = 500,    # episode truncation
        level: int = 0,     # 0 = empty, 3 = dense
        n_obstacle_rays: int = 8,      # how many obstacle distance sensors
        render_mode:   Optional[str] = None,
        fps:           int   = 30,     # rendering frame rate (lower = slow motion)
        seed:          Optional[int] = None,
    ):
        super().__init__()

        self.world_size       = world_size
        self.max_steps        = max_steps
        self.level = level
        self.n_obstacle_rays  = n_obstacle_rays
        self.render_mode      = render_mode
        self.fps              = fps

        # Sub-modules
        self.physics  = DronePhysics(max_speed=5.0, drag=0.1, dt=0.1)
        self.reward_fn = RewardFunction()

        # RNG — seeded for reproducibility
        self.rng = np.random.default_rng(seed)

        # ── Spaces ──────────────────────────────────────────────
        # Observation: [x, y, vx, vy, dx_goal, dy_goal, d_obs_0 ... d_obs_n]
        obs_dim = 6 + n_obstacle_rays
        self.observation_space = spaces.Box(
            low  = -np.ones(obs_dim, dtype=np.float32),
            high =  np.ones(obs_dim, dtype=np.float32),
        )

        # Actions are normalised to [-1, 1] and scaled inside physics.step()
        # So action [0, -1] = full upward thrust, [0, 1] = full downward thrust
        self.action_space = spaces.Box(
            low  = np.array([-1.0, -1.0], dtype=np.float32),
            high = np.array([ 1.0,  1.0], dtype=np.float32),
        )

        # State (reset initialises these)
        self.pos:       np.ndarray = np.zeros(2)
        self.vel:       np.ndarray = np.zeros(2)
        self.goal:      np.ndarray = np.zeros(2)
        self.obstacles: list       = []
        self.step_count: int       = 0

        # Renderer state (cached each step for visual feedback)
        self._last_thrust:        np.ndarray = np.zeros(2)
        self._last_ray_distances: np.ndarray = np.ones(n_obstacle_rays, dtype=np.float32)
        self._last_collided:      bool       = False
        self._episode_reward:     float      = 0.0

        self.renderer = None   # lazy-init when render() is called

    # ── Gymnasium API ─────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count       = 0
        self._episode_reward  = 0.0
        self._last_collided   = False
        self._last_thrust     = np.zeros(2)
        self.obstacles = generate_obstacles(self.level, self.world_size, self.rng)

        # Place drone and goal away from obstacles and each other
        self.pos  = self._random_free_position()
        self.goal = self._random_free_position(min_dist_from=self.pos, min_dist=5.0)
        self.vel  = np.zeros(2)

        obs  = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0)
        thrust = action * self.physics.max_thrust
        self._last_thrust = thrust.copy()

        prev_pos = self.pos.copy()
        self.pos, self.vel = self.physics.step(self.pos, self.vel, thrust)

        # Collision check on raw physics position — walls are real obstacles
        collided = self._check_collision()

        # Clip so the drone stays in the renderable area even after a crash
        self.pos[0] = float(np.clip(self.pos[0], 0.0, self.world_size))
        self.pos[1] = min(float(self.pos[1]), self.world_size)
        self._last_collided = collided

        # Reward
        reward, terminated = self.reward_fn.compute(
            self.pos, prev_pos, self.goal, collided
        )
        self._episode_reward += reward

        # Truncation (ran out of steps — not a failure, just time limit)
        truncated = (self.step_count >= self.max_steps)

        obs  = self._get_obs()
        info = {
            "dist_to_goal": float(np.linalg.norm(self.pos - self.goal)),
            "collided":     collided,
            "step":         self.step_count,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return
        if self.renderer is None:
            from .renderer import DroneRenderer
            self.renderer = DroneRenderer(self.world_size, fps=self.fps)
        return self.renderer.render(
            self.pos, self.vel, self.goal, self.obstacles, self.render_mode,
            thrust=self._last_thrust,
            ray_distances=self._last_ray_distances,
            n_rays=self.n_obstacle_rays,
            dist_to_goal=float(np.linalg.norm(self.pos - self.goal)),
            step_count=self.step_count,
            episode_reward=self._episode_reward,
            collided=self._last_collided,
        )

    def close(self):
        if self.renderer:
            self.renderer.close()

    # ── Internal helpers ──────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Build the observation vector, all values normalised to ~[-1, 1]."""
        s = self.world_size

        # Normalised position
        pos_norm = (self.pos / s) * 2 - 1   # maps [0, world] → [-1, 1]

        # Normalised velocity
        vel_norm = self.vel / self.physics.max_speed

        # Direction to goal (normalised by world size)
        goal_delta = (self.goal - self.pos) / s

        # Obstacle distances via ray casting
        obs_distances = self._ray_cast_obstacles()
        self._last_ray_distances = obs_distances   # cache for renderer

        return np.concatenate([
            pos_norm, vel_norm, goal_delta, obs_distances
        ]).astype(np.float32)

    def _ray_cast_obstacles(self) -> np.ndarray:
        """
        Cast N rays evenly around the drone.
        Each ray returns a distance in [0, 1] (normalised by world_size).
        If no obstacle is hit, returns 1.0 (max range).

        This gives the agent local spatial awareness without
        needing to know the full obstacle layout.
        """
        distances = []
        angles = np.linspace(0, 2 * np.pi, self.n_obstacle_rays, endpoint=False)
        max_range = self.world_size

        for angle in angles:
            direction = np.array([np.cos(angle), np.sin(angle)])
            dist = self._cast_ray(self.pos, direction, max_range)
            distances.append(dist / max_range)   # normalise

        return np.array(distances, dtype=np.float32)

    def _cast_ray(self, origin, direction, max_range, steps=50) -> float:
        """Simple ray march: step along direction until hitting an obstacle."""
        for i in range(steps):
            t = (i + 1) * max_range / steps
            point = origin + direction * t
            # Wall hit: left, right, bottom — top is open sky
            if point[0] < 0 or point[0] > self.world_size or point[1] > self.world_size:
                return t
            # Obstacle hit
            for obs in self.obstacles:
                if obs.contains(point):
                    return t
        return max_range

    def _check_collision(self) -> bool:
        """Returns True if drone is inside any obstacle or hits a wall.
        Top boundary (y < 0) is open sky — the drone can fly above it freely.
        """
        x, y = self.pos
        if x < 0 or x > self.world_size or y > self.world_size:
            return True
        for obs in self.obstacles:
            if obs.contains(self.pos):
                return True
        return False

    def _random_free_position(
        self,
        min_dist_from: Optional[np.ndarray] = None,
        min_dist: float = 0.0,
        max_tries: int = 100
    ) -> np.ndarray:
        """Sample a random position not inside any obstacle."""
        margin = 1.0
        for _ in range(max_tries):
            pos = self.rng.uniform(margin, self.world_size - margin, size=2)
            if self._check_collision_at(pos):
                continue
            if min_dist_from is not None:
                if np.linalg.norm(pos - min_dist_from) < min_dist:
                    continue
            return pos
        # Fallback: corners are always free in level 0
        return np.array([1.0, 1.0]) if min_dist_from is None else np.array([self.world_size - 1, self.world_size - 1])

    def _check_collision_at(self, pos: np.ndarray) -> bool:
        for obs in self.obstacles:
            if obs.contains(pos):
                return True
        return False