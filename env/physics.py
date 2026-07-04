# env/physics.py
import numpy as np

class DronePhysics:
    """
    2D drone dynamics with gravity and thrust.

    The agent commands acceleration [ax, ay] (thrust).
    Gravity pulls the drone down every timestep.
    Velocity and position integrate from acceleration.

    This means:
      - The drone has inertia — actions have delayed effects
      - The agent must learn to fight gravity to hover or climb
      - Trajectories are smooth curves, not instant direction changes
    """

    def __init__(
        self,
        max_thrust: float = 20.0,  # max acceleration the drone can apply (m/s²)
        gravity:    float = 9.8,   # downward acceleration (m/s²)
        drag:       float = 0.05,  # air resistance coefficient (0 = vacuum)
        max_speed:  float = 30.0,  # velocity clamp (m/s)
        dt:         float = 0.05,  # timestep (seconds) — smaller = more accurate
    ):
        self.max_thrust = max_thrust
        self.gravity    = gravity
        self.drag       = drag
        self.max_speed  = max_speed
        self.dt         = dt

        # Gravity as a constant acceleration vector (always downward = +y in screen coords)
        # We use +y = down (screen convention). Flip sign if you use +y = up (math convention).
        self.gravity_vec = np.array([0.0, self.gravity])  # +y is down in screen space

    def step(self, pos: np.ndarray, vel: np.ndarray, action: np.ndarray) -> tuple:
        """
        pos    : [x, y]     current position (m)
        vel    : [vx, vy]   current velocity (m/s)
        action : [ax, ay]   thrust command, clipped to [-max_thrust, max_thrust]

        Returns (new_pos, new_vel)

        Physics pipeline each timestep:
          1. Clip thrust to drone's physical limits
          2. Net acceleration = thrust + gravity + drag
          3. Integrate velocity:  v += a * dt
          4. Clamp speed
          5. Integrate position:  p += v * dt
        """

        # 1. Clip thrust — drone can't produce infinite force
        thrust = np.clip(action, -self.max_thrust, self.max_thrust)

        # 2. Net acceleration
        #    drag opposes velocity (like air resistance: F_drag = -c * v)
        drag_acc = -self.drag * vel
        net_acc  = thrust + self.gravity_vec + drag_acc

        # 3. Integrate velocity  (Euler integration)
        vel = vel + net_acc * self.dt

        # 4. Clamp speed — prevents runaway velocity in long falls
        speed = np.linalg.norm(vel)
        if speed > self.max_speed:
            vel = vel / speed * self.max_speed

        # 5. Integrate position
        pos = pos + vel * self.dt

        return pos, vel

    @property
    def hover_thrust(self) -> float:
        """
        The upward thrust needed to exactly cancel gravity.
        Useful for reward shaping and as a sanity check.
        In screen coords where +y is down, hover means ay = -gravity.
        """
        return self.gravity  # agent needs ay = -9.8 to hover in place