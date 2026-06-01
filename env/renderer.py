# env/renderer.py
import math
import numpy as np


class DroneRenderer:
    """Arcade-style pygame renderer — neon sci-fi theme."""

    MAX_TRAIL = 30

    def __init__(self, world_size: float, window_px: int = 700):
        import pygame
        self.pygame = pygame
        self.world_size = world_size
        self.window_px = window_px
        self.scale = window_px / world_size

        pygame.init()
        self.screen = pygame.display.set_mode((window_px, window_px))
        pygame.display.set_caption("DroneNav — Arcade")
        self.clock = pygame.time.Clock()

        self.tick = 0
        self.rotor_angle = 0.0
        self.flash_frames = 0
        self.trail: list = []

        # Palette
        self.C_BG       = (6,   8,  18)
        self.C_GRID     = (18, 22, 42)
        self.C_DRONE    = (210, 225, 255)
        self.C_ARM      = (130, 155, 215)
        self.C_ROTOR    = (90,  155, 255)
        self.C_VEL      = (70,  130, 255)
        self.C_TRAIL    = (55,  85, 175)
        self.C_GOAL     = (40,  230, 110)
        self.C_OBS_FILL = (42,  46,  62)
        self.C_OBS_LIT  = (82,  88, 118)
        self.C_OBS_SHD  = (22,  26,  38)
        self.C_RAY      = (28,  65,  48)
        self.C_RAY_HIT  = (55, 115,  75)
        self.C_THRUST   = (255, 140,  40)
        self.C_HUD      = (155, 185, 255)
        self.C_FLASH    = (255,  40,  40)

        self.font    = pygame.font.SysFont("Courier New", 16, bold=True)
        self.font_sm = pygame.font.SysFont("Courier New", 12)

    # ── Coordinate helpers ────────────────────────────────────────

    def _w(self, pos):
        """World → screen pixels."""
        return (int(pos[0] * self.scale), int(pos[1] * self.scale))

    def _circle_alpha(self, color_rgba, center_px, radius: int):
        """Draw a circle with per-pixel alpha onto the main screen."""
        if radius <= 0:
            return
        surf = self.pygame.Surface((radius * 2, radius * 2), self.pygame.SRCALPHA)
        self.pygame.draw.circle(surf, color_rgba, (radius, radius), radius)
        self.screen.blit(surf, (center_px[0] - radius, center_px[1] - radius))

    # ── Draw layers ───────────────────────────────────────────────

    def _draw_grid(self):
        step = max(1, int(self.scale * 2))
        for x in range(0, self.window_px + 1, step):
            self.pygame.draw.line(self.screen, self.C_GRID, (x, 0), (x, self.window_px))
        for y in range(0, self.window_px + 1, step):
            self.pygame.draw.line(self.screen, self.C_GRID, (0, y), (self.window_px, y))

    def _draw_obstacles(self, obstacles):
        for obs in obstacles:
            rx = int(obs.x * self.scale)
            ry = int(obs.y * self.scale)
            rw = max(2, int(obs.w * self.scale))
            rh = max(2, int(obs.h * self.scale))
            self.pygame.draw.rect(self.screen, self.C_OBS_FILL, (rx, ry, rw, rh))
            # lit top-left edges
            self.pygame.draw.line(self.screen, self.C_OBS_LIT, (rx, ry), (rx + rw, ry), 2)
            self.pygame.draw.line(self.screen, self.C_OBS_LIT, (rx, ry), (rx, ry + rh), 2)
            # shadow bottom-right edges
            self.pygame.draw.line(self.screen, self.C_OBS_SHD, (rx, ry + rh), (rx + rw, ry + rh), 1)
            self.pygame.draw.line(self.screen, self.C_OBS_SHD, (rx + rw, ry), (rx + rw, ry + rh), 1)

    def _draw_rays(self, pos, ray_distances, n_rays: int):
        """Sensor ray cast visualization."""
        angles = np.linspace(0, 2 * math.pi, n_rays, endpoint=False)
        for angle, dist_norm in zip(angles, ray_distances):
            dist = float(dist_norm) * self.world_size
            end = (pos[0] + math.cos(angle) * dist,
                   pos[1] + math.sin(angle) * dist)
            self.pygame.draw.line(self.screen, self.C_RAY, self._w(pos), self._w(end), 1)
            self.pygame.draw.circle(self.screen, self.C_RAY_HIT, self._w(end), 2)

    def _draw_trail(self):
        n = len(self.trail)
        for i, pt in enumerate(self.trail):
            frac = i / max(n - 1, 1)
            alpha = int(110 * frac)
            r = max(1, int(4 * frac))
            color = (int(self.C_TRAIL[0] * frac),
                     int(self.C_TRAIL[1] * frac),
                     int(self.C_TRAIL[2] * frac),
                     alpha)
            self._circle_alpha(color, self._w(pt), r)

    def _draw_goal(self, goal):
        t = self.tick * 0.06
        gx, gy = self._w(goal)
        # pulsing concentric rings
        for ring in range(4):
            pulse = math.sin(t - ring * 0.7) * 0.5 + 0.5
            r = int(10 + ring * 9 + pulse * 5)
            alpha = max(0, int(55 - ring * 12))
            self._circle_alpha((*self.C_GOAL, alpha), (gx, gy), r)
        # core dot
        self.pygame.draw.circle(self.screen, self.C_GOAL, (gx, gy), 6)
        self.pygame.draw.circle(self.screen, (200, 255, 215), (gx, gy), 3)
        # crosshair
        cs = 13
        self.pygame.draw.line(self.screen, self.C_GOAL, (gx - cs, gy), (gx + cs, gy), 1)
        self.pygame.draw.line(self.screen, self.C_GOAL, (gx, gy - cs), (gx, gy + cs), 1)

    def _draw_drone(self, pos, vel, thrust):
        px, py = self._w(pos)

        thrust_mag = float(np.linalg.norm(thrust)) if thrust is not None else 0.0
        thrust_frac = min(1.0, thrust_mag / 15.0)

        # rotor spin speed proportional to thrust
        self.rotor_angle += 0.08 + 0.18 * thrust_frac

        # thrust glow under the drone
        if thrust_frac > 0.05:
            glow_r = int(18 + thrust_frac * 22)
            alpha  = int(85 * thrust_frac)
            self._circle_alpha((*self.C_THRUST, alpha), (px, py), glow_r)

        # 4 arms at 45°/135°/225°/315°
        arm_len = 13
        rotor_r = 5
        for deg in (45, 135, 225, 315):
            rad = math.radians(deg)
            ax = int(px + math.cos(rad) * arm_len)
            ay = int(py + math.sin(rad) * arm_len)
            self.pygame.draw.line(self.screen, self.C_ARM, (px, py), (ax, ay), 2)
            self.pygame.draw.circle(self.screen, self.C_ROTOR, (ax, ay), rotor_r, 1)
            # spinning blade tick
            bx = int(ax + math.cos(self.rotor_angle + math.radians(deg)) * rotor_r * 0.75)
            by = int(ay + math.sin(self.rotor_angle + math.radians(deg)) * rotor_r * 0.75)
            self.pygame.draw.line(self.screen, self.C_ROTOR, (ax, ay), (bx, by), 1)

        # body hub
        self.pygame.draw.circle(self.screen, self.C_DRONE, (px, py), 5)
        self.pygame.draw.circle(self.screen, (240, 245, 255), (px, py), 2)

        # velocity arrow with arrowhead
        speed = float(np.linalg.norm(vel))
        if speed > 0.4:
            arrow_len = min(2.5, speed * 0.25)   # world units, capped
            dir_v  = vel / speed
            perp_v = np.array([-dir_v[1], dir_v[0]])
            tip  = pos + dir_v * arrow_len
            base1 = tip - dir_v * 0.3 + perp_v * 0.15
            base2 = tip - dir_v * 0.3 - perp_v * 0.15
            self.pygame.draw.line(self.screen, self.C_VEL, (px, py), self._w(tip), 2)
            self.pygame.draw.polygon(self.screen, self.C_VEL,
                                     [self._w(tip), self._w(base1), self._w(base2)])

    def _draw_collision_flash(self):
        if self.flash_frames > 0:
            alpha = int(150 * self.flash_frames / 8)
            overlay = self.pygame.Surface((self.window_px, self.window_px), self.pygame.SRCALPHA)
            overlay.fill((*self.C_FLASH, alpha))
            self.screen.blit(overlay, (0, 0))
            self.flash_frames -= 1

    def _draw_hud(self, dist_to_goal: float, speed: float,
                  step_count: int, episode_reward: float):
        pad, line_h = 10, 19

        def row(i, label, value, color=None):
            surf = self.font.render(f"{label:<7}{value}", True, color or self.C_HUD)
            self.screen.blit(surf, (pad, pad + i * line_h))

        row(0, "STEP",  f"{step_count:>4d}")
        row(1, "DIST",  f"{dist_to_goal:>5.1f}")
        row(2, "SPEED", f"{speed:>4.1f}")
        reward_color = (75, 215, 95) if episode_reward >= 0 else (215, 75, 75)
        row(3, "SCORE", f"{episode_reward:>+7.1f}", reward_color)

    # ── Public API ────────────────────────────────────────────────

    def render(self, pos, vel, goal, obstacles, mode="human",
               thrust=None, ray_distances=None, n_rays: int = 8,
               dist_to_goal: float = None, step_count: int = 0,
               episode_reward: float = 0.0, collided: bool = False):
        import pygame

        self.tick += 1
        if collided:
            self.flash_frames = 8

        self.trail.append(pos.copy())
        if len(self.trail) > self.MAX_TRAIL:
            self.trail.pop(0)

        self.screen.fill(self.C_BG)
        self._draw_grid()

        if ray_distances is not None:
            self._draw_rays(pos, ray_distances, n_rays)

        self._draw_obstacles(obstacles)
        self._draw_trail()
        self._draw_goal(goal)
        self._draw_drone(pos, vel, thrust)
        self._draw_collision_flash()

        speed = float(np.linalg.norm(vel))
        if dist_to_goal is None:
            dist_to_goal = float(np.linalg.norm(pos - goal))
        self._draw_hud(dist_to_goal, speed, step_count, episode_reward)

        if mode == "human":
            pygame.event.pump()   # keep OS from marking window as frozen
            pygame.display.flip()
            self.clock.tick(30)
            return None
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        self.pygame.quit()
