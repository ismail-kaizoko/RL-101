# env/obstacles.py
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class Rect:
    x: float      # top-left corner
    y: float
    w: float      # width
    h: float      # height

    def contains(self, point: np.ndarray) -> bool:
        """Returns True if point [px, py] is inside this rectangle."""
        px, py = point
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def distance_to(self, point: np.ndarray) -> float:
        """Shortest distance from point to the rectangle boundary."""
        px, py = point
        # clamp point to rect, then measure distance
        cx = np.clip(px, self.x, self.x + self.w)
        cy = np.clip(py, self.y, self.y + self.h)
        return np.sqrt((px - cx)**2 + (py - cy)**2)


def generate_obstacles(level: int, world_size: float, rng: np.random.Generator) -> List[Rect]:
    """
    Curriculum generator.
    level 0 → empty world (great for classical RL / sanity checks)
    level 1 → 3 random obstacles
    level 2 → 6 obstacles, some near the path
    level 3 → dense field, 12 obstacles
    """
    counts = [0, 3, 6, 12]
    n = counts[min(level, len(counts) - 1)]

    obstacles = []
    for _ in range(n):
        # random position, keep away from borders
        margin = world_size * 0.1
        x = rng.uniform(margin, world_size - margin - 4)
        y = rng.uniform(margin, world_size - margin - 4)
        w = rng.uniform(2, 5)
        h = rng.uniform(2, 5)
        obstacles.append(Rect(x, y, w, h))

    return obstacles