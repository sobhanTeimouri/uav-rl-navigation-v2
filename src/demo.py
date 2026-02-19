import time
from stable_baselines3 import PPO

from uav2d_env import UAV2DEnv


if __name__ == "__main__":
    env = UAV2DEnv(
        render_mode="human",
        world_size=10.0,
        dt=0.1,
        max_steps=500,

        # lidar
        lidar_num_rays=36,
        lidar_max_range=6.0,

        # obstacles
        num_obstacles_circle_static=3,
        num_obstacles_circle_dynamic=2,
        num_obstacles_rect_static=2,
        num_obstacles_rect_dynamic=2,

        # wind
        wind_enabled=True,
        wind_max_force=0.4
    )

    model = PPO.load("models/ppo_uav2d_lidar_v3")

    obs, info = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            time.sleep(0.8)
            obs, info = env.reset()
