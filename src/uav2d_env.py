import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math


class UAV2DEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode=None,
        world_size=10.0,
        dt=0.1,
        max_steps=500,
        uav_radius=0.2,
        goal_radius=0.4,
        max_speed=1.5,

        # LiDAR
        lidar_num_rays=36,
        lidar_max_range=6.0,
        lidar_fov_deg=180,
        lidar_noise_std=0.03,
        lidar_dropout_prob=0.02,

        # Obstacles
        num_obstacles_circle_static=3,
        num_obstacles_circle_dynamic=2,
        num_obstacles_rect_static=2,
        num_obstacles_rect_dynamic=2,

        # Wind
        wind_enabled=True,
        wind_max_force=0.4,

        seed=None
    ):
        super().__init__()

        self.render_mode = render_mode
        self.world_size = float(world_size)
        self.dt = float(dt)
        self.max_steps = int(max_steps)

        self.uav_radius = float(uav_radius)
        self.goal_radius = float(goal_radius)
        self.max_speed = float(max_speed)

        # LiDAR settings
        self.lidar_num_rays = int(lidar_num_rays)
        self.lidar_max_range = float(lidar_max_range)
        self.lidar_fov_deg = float(lidar_fov_deg)
        self.lidar_noise_std = float(lidar_noise_std)
        self.lidar_dropout_prob = float(lidar_dropout_prob)

        # Obstacles
        self.num_obstacles_circle_static = num_obstacles_circle_static
        self.num_obstacles_circle_dynamic = num_obstacles_circle_dynamic
        self.num_obstacles_rect_static = num_obstacles_rect_static
        self.num_obstacles_rect_dynamic = num_obstacles_rect_dynamic

        # Wind
        self.wind_enabled = wind_enabled
        self.wind_max_force = float(wind_max_force)

        self._rng = np.random.default_rng(seed)

        self.uav_pos = np.zeros(2, dtype=np.float32)
        self.uav_vel = np.zeros(2, dtype=np.float32)
        self.goal_pos = np.zeros(2, dtype=np.float32)

        self.obstacles = []
        self.step_count = 0

        self.wind = np.zeros(2, dtype=np.float32)

        # Heading (direction of UAV)
        self.heading = 0.0

        # Progress tracking
        self.prev_dist_to_goal = 0.0

        # Action space: acceleration ax, ay
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Observation:
        # [uav_x, uav_y, uav_vx, uav_vy, goal_dx, goal_dy] + lidar
        obs_dim = 6 + self.lidar_num_rays
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Rendering
        self.screen = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.step_count = 0

        # Spawn UAV and Goal
        self.uav_pos = self._random_position()
        self.uav_vel = np.zeros(2, dtype=np.float32)

        self.goal_pos = self._random_position(min_dist_from=self.uav_pos, min_dist=4.0)

        # Init heading
        self.heading = 0.0

        # Wind
        if self.wind_enabled:
            self.wind = self._rng.uniform(
                low=-self.wind_max_force,
                high=self.wind_max_force,
                size=(2,)
            ).astype(np.float32)
        else:
            self.wind = np.zeros(2, dtype=np.float32)

        # Spawn obstacles
        self.obstacles = []
        self._spawn_obstacles()

        # Init distance memory
        self.prev_dist_to_goal = float(np.linalg.norm(self.uav_pos - self.goal_pos))

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.step_count += 1

        action = np.clip(action, -1.0, 1.0)

        # Scale acceleration
        accel = action.astype(np.float32) * 1.2

        # Add wind
        if self.wind_enabled:
            accel += self.wind

        # Update velocity
        self.uav_vel += accel * self.dt

        # Clip speed
        speed = np.linalg.norm(self.uav_vel)
        if speed > self.max_speed:
            self.uav_vel = (self.uav_vel / (speed + 1e-8)) * self.max_speed

        # Update heading based on velocity
        if np.linalg.norm(self.uav_vel) > 1e-4:
            self.heading = math.atan2(self.uav_vel[1], self.uav_vel[0])

        # Update position
        self.uav_pos += self.uav_vel * self.dt
        self.uav_pos = np.clip(self.uav_pos, 0.0, self.world_size)

        # Update obstacles
        self._update_obstacles()

        # Reward
        reward = self._compute_reward()

        terminated = False
        truncated = False

        # Goal reached
        if np.linalg.norm(self.uav_pos - self.goal_pos) <= self.goal_radius:
            reward += 250.0
            terminated = True

        # Collision
        if self._check_collision(self.uav_pos):
            reward -= 300.0
            terminated = True

        # Timeout
        if self.step_count >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # -------------------------
    # Observation
    # -------------------------
    def _get_obs(self):
        uav_x = self.uav_pos[0] / self.world_size
        uav_y = self.uav_pos[1] / self.world_size

        uav_vx = self.uav_vel[0] / self.max_speed
        uav_vy = self.uav_vel[1] / self.max_speed

        goal_dx = (self.goal_pos[0] - self.uav_pos[0]) / self.world_size
        goal_dy = (self.goal_pos[1] - self.uav_pos[1]) / self.world_size

        lidar = self._lidar_scan()

        obs = np.concatenate([
            np.array([uav_x, uav_y, uav_vx, uav_vy, goal_dx, goal_dy], dtype=np.float32),
            lidar
        ]).astype(np.float32)

        return np.clip(obs, -1.0, 1.0)

    # -------------------------
    # Reward
    # -------------------------
    def _compute_reward(self):
        dist_to_goal = float(np.linalg.norm(self.uav_pos - self.goal_pos))

        # Progress reward
        progress = self.prev_dist_to_goal - dist_to_goal
        self.prev_dist_to_goal = dist_to_goal

        reward = 15.0 * progress

        # Step penalty
        reward -= 0.02

        # Speed penalty
        reward -= 0.01 * float(np.linalg.norm(self.uav_vel))

        # LiDAR safety penalty
        lidar = self._lidar_scan()
        min_lidar = float(np.min(lidar))

        if min_lidar < 0.25:
            reward -= (0.25 - min_lidar) * 50.0

        return reward

    # -------------------------
    # Obstacles
    # -------------------------
    def _spawn_obstacles(self):
        for _ in range(self.num_obstacles_circle_static):
            self.obstacles.append(self._create_circle(dynamic=False))

        for _ in range(self.num_obstacles_circle_dynamic):
            self.obstacles.append(self._create_circle(dynamic=True))

        for _ in range(self.num_obstacles_rect_static):
            self.obstacles.append(self._create_rect(dynamic=False))

        for _ in range(self.num_obstacles_rect_dynamic):
            self.obstacles.append(self._create_rect(dynamic=True))

    def _create_circle(self, dynamic=False):
        r = float(self._rng.uniform(0.35, 0.75))
        pos = self._random_position(min_dist_from=self.uav_pos, min_dist=1.5)

        vx, vy = 0.0, 0.0
        if dynamic:
            vx = float(self._rng.uniform(-0.8, 0.8))
            vy = float(self._rng.uniform(-0.8, 0.8))

        return {"type": "circle", "x": float(pos[0]), "y": float(pos[1]), "r": r, "vx": vx, "vy": vy}

    def _create_rect(self, dynamic=False):
        w = float(self._rng.uniform(0.7, 1.4))
        h = float(self._rng.uniform(0.7, 1.4))
        pos = self._random_position(min_dist_from=self.uav_pos, min_dist=1.5)

        vx, vy = 0.0, 0.0
        if dynamic:
            vx = float(self._rng.uniform(-0.6, 0.6))
            vy = float(self._rng.uniform(-0.6, 0.6))

        return {"type": "rect", "x": float(pos[0]), "y": float(pos[1]), "w": w, "h": h, "vx": vx, "vy": vy}

    def _update_obstacles(self):
        for obs in self.obstacles:
            if abs(obs["vx"]) < 1e-6 and abs(obs["vy"]) < 1e-6:
                continue

            obs["x"] += obs["vx"] * self.dt
            obs["y"] += obs["vy"] * self.dt

            # Bounce
            if obs["x"] < 0.0 or obs["x"] > self.world_size:
                obs["vx"] *= -1.0
                obs["x"] = float(np.clip(obs["x"], 0.0, self.world_size))

            if obs["y"] < 0.0 or obs["y"] > self.world_size:
                obs["vy"] *= -1.0
                obs["y"] = float(np.clip(obs["y"], 0.0, self.world_size))

    # -------------------------
    # Collision
    # -------------------------
    def _check_collision(self, pos):
        px, py = float(pos[0]), float(pos[1])

        for obs in self.obstacles:
            if obs["type"] == "circle":
                dx = px - obs["x"]
                dy = py - obs["y"]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist <= (obs["r"] + self.uav_radius):
                    return True

            elif obs["type"] == "rect":
                half_w = obs["w"] / 2.0
                half_h = obs["h"] / 2.0

                closest_x = min(max(px, obs["x"] - half_w), obs["x"] + half_w)
                closest_y = min(max(py, obs["y"] - half_h), obs["y"] + half_h)

                dx = px - closest_x
                dy = py - closest_y
                dist = math.sqrt(dx * dx + dy * dy)

                if dist <= self.uav_radius:
                    return True

        return False

    # -------------------------
    # LiDAR (FOV + Noise + Dropout)
    # -------------------------
    def _lidar_scan(self):
        fov = math.radians(self.lidar_fov_deg)

        start_angle = self.heading - fov / 2.0
        end_angle = self.heading + fov / 2.0

        angles = np.linspace(start_angle, end_angle, self.lidar_num_rays, endpoint=True)

        lidar = np.ones(self.lidar_num_rays, dtype=np.float32)

        ox, oy = float(self.uav_pos[0]), float(self.uav_pos[1])

        for i, ang in enumerate(angles):
            # Dropout probability
            if self._rng.random() < self.lidar_dropout_prob:
                lidar[i] = 1.0
                continue

            dx = math.cos(ang)
            dy = math.sin(ang)

            best_dist = self.lidar_max_range

            for obs in self.obstacles:
                if obs["type"] == "circle":
                    d = self._ray_circle(ox, oy, dx, dy, obs["x"], obs["y"], obs["r"])
                    if d is not None and d < best_dist:
                        best_dist = d

                elif obs["type"] == "rect":
                    d = self._ray_rect(ox, oy, dx, dy, obs["x"], obs["y"], obs["w"], obs["h"])
                    if d is not None and d < best_dist:
                        best_dist = d

            # Add noise if hit something
            if best_dist < self.lidar_max_range:
                best_dist += float(self._rng.normal(0.0, self.lidar_noise_std))

            best_dist = float(np.clip(best_dist, 0.0, self.lidar_max_range))

            lidar[i] = best_dist / self.lidar_max_range

        return lidar

    def _ray_circle(self, ox, oy, dx, dy, cx, cy, r):
        fx = ox - cx
        fy = oy - cy

        a = dx * dx + dy * dy
        b = 2.0 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - r * r

        disc = b * b - 4 * a * c
        if disc < 0:
            return None

        disc_sqrt = math.sqrt(disc)
        t1 = (-b - disc_sqrt) / (2 * a)
        t2 = (-b + disc_sqrt) / (2 * a)

        if t1 >= 0:
            return t1
        if t2 >= 0:
            return t2
        return None

    def _ray_rect(self, ox, oy, dx, dy, rx, ry, w, h):
        half_w = w / 2.0
        half_h = h / 2.0

        min_x = rx - half_w
        max_x = rx + half_w
        min_y = ry - half_h
        max_y = ry + half_h

        tmin = -1e9
        tmax = 1e9

        # X slab
        if abs(dx) < 1e-8:
            if ox < min_x or ox > max_x:
                return None
        else:
            tx1 = (min_x - ox) / dx
            tx2 = (max_x - ox) / dx
            tmin = max(tmin, min(tx1, tx2))
            tmax = min(tmax, max(tx1, tx2))

        # Y slab
        if abs(dy) < 1e-8:
            if oy < min_y or oy > max_y:
                return None
        else:
            ty1 = (min_y - oy) / dy
            ty2 = (max_y - oy) / dy
            tmin = max(tmin, min(ty1, ty2))
            tmax = min(tmax, max(ty1, ty2))

        if tmax < tmin:
            return None
        if tmax < 0:
            return None

        if tmin >= 0:
            return tmin
        return tmax

    # -------------------------
    # Random Position
    # -------------------------
    def _random_position(self, min_dist_from=None, min_dist=0.0):
        for _ in range(500):
            pos = self._rng.uniform(0.7, self.world_size - 0.7, size=(2,))
            pos = pos.astype(np.float32)

            if min_dist_from is not None:
                if np.linalg.norm(pos - min_dist_from) < min_dist:
                    continue

            return pos

        return np.array([1.0, 1.0], dtype=np.float32)

    # -------------------------
    # Render
    # -------------------------
    def render(self):
        if self.render_mode is None:
            return

        import pygame

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((700, 700))
            pygame.display.set_caption("UAV2D LiDAR Navigation (v3.1)")
            self.clock = pygame.time.Clock()

        self.clock.tick(self.metadata["render_fps"])
        self.screen.fill((20, 20, 25))

        scale = 700 / self.world_size

        def world_to_screen(p):
            return int(p[0] * scale), int(700 - p[1] * scale)

        # Goal
        gx, gy = world_to_screen(self.goal_pos)
        pygame.draw.circle(self.screen, (50, 220, 50), (gx, gy), int(self.goal_radius * scale), 2)

        # Obstacles
        for obs in self.obstacles:
            if obs["type"] == "circle":
                x, y = world_to_screen((obs["x"], obs["y"]))
                pygame.draw.circle(self.screen, (220, 70, 70), (x, y), int(obs["r"] * scale))

            elif obs["type"] == "rect":
                half_w = obs["w"] / 2.0
                half_h = obs["h"] / 2.0

                top_left = (obs["x"] - half_w, obs["y"] + half_h)
                sx, sy = world_to_screen(top_left)

                rect_w = int(obs["w"] * scale)
                rect_h = int(obs["h"] * scale)

                pygame.draw.rect(self.screen, (200, 120, 40), pygame.Rect(sx, sy, rect_w, rect_h))

        # LiDAR rays (FOV)
        lidar = self._lidar_scan()
        fov = math.radians(self.lidar_fov_deg)

        start_angle = self.heading - fov / 2.0
        end_angle = self.heading + fov / 2.0

        angles = np.linspace(start_angle, end_angle, self.lidar_num_rays, endpoint=True)

        ux, uy = world_to_screen(self.uav_pos)

        for i, ang in enumerate(angles):
            dist = lidar[i] * self.lidar_max_range
            end_x = self.uav_pos[0] + math.cos(ang) * dist
            end_y = self.uav_pos[1] + math.sin(ang) * dist
            ex, ey = world_to_screen((end_x, end_y))
            pygame.draw.line(self.screen, (80, 80, 120), (ux, uy), (ex, ey), 1)

        # UAV
        pygame.draw.circle(self.screen, (70, 150, 255), (ux, uy), int(self.uav_radius * scale))

        # Heading line
        hx = self.uav_pos[0] + math.cos(self.heading) * 0.8
        hy = self.uav_pos[1] + math.sin(self.heading) * 0.8
        hsx, hsy = world_to_screen((hx, hy))
        pygame.draw.line(self.screen, (255, 255, 255), (ux, uy), (hsx, hsy), 2)

        # Wind vector
        if self.wind_enabled:
            wx = int(ux + self.wind[0] * 120)
            wy = int(uy - self.wind[1] * 120)
            pygame.draw.line(self.screen, (150, 150, 255), (ux, uy), (wx, wy), 3)

        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.quit()
            self.screen = None
            self.clock = None
