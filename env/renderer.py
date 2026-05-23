# env/renderer.py
import numpy as np

class DroneRenderer:
    """Pygame renderer. Lazy-imported so the env works headless (for training)."""

    def __init__(self, world_size: float, window_px: int = 600):
        import pygame
        self.pygame     = pygame
        self.world_size = world_size
        self.window_px  = window_px
        self.scale      = window_px / world_size

        pygame.init()
        self.screen = pygame.display.set_mode((window_px, window_px))
        pygame.display.set_caption("DroneEnv2D")
        self.clock  = pygame.time.Clock()

    def _w(self, pos):
        """World coords → screen pixels."""
        return (int(pos[0] * self.scale), int(pos[1] * self.scale))

    def render(self, pos, vel, goal, obstacles, mode="human"):
        import pygame
        self.screen.fill((15, 15, 20))   # dark background

        # Draw obstacles (grey rectangles)
        for obs in obstacles:
            rect = pygame.Rect(
                int(obs.x * self.scale),
                int(obs.y * self.scale),
                int(obs.w * self.scale),
                int(obs.h * self.scale),
            )
            pygame.draw.rect(self.screen, (80, 80, 100), rect)
            pygame.draw.rect(self.screen, (120, 120, 140), rect, 1)

        # Draw goal (green circle)
        pygame.draw.circle(self.screen, (50, 200, 100), self._w(goal), 8)
        pygame.draw.circle(self.screen, (100, 255, 150), self._w(goal), 3)

        # Draw drone (white circle + velocity arrow)
        drone_px = self._w(pos)
        pygame.draw.circle(self.screen, (220, 220, 255), drone_px, 7)

        # Velocity arrow
        if np.linalg.norm(vel) > 0.1:
            vel_end = pos + vel * 1.5
            pygame.draw.line(
                self.screen, (150, 200, 255),
                drone_px, self._w(vel_end), 2
            )

        if mode == "human":
            pygame.display.flip()
            self.clock.tick(30)
            return None
        else:
            # rgb_array mode for video recording
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        self.pygame.quit()