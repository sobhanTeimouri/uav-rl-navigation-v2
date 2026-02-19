import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from uav2d_env import UAV2DEnv


def make_env():
    env = UAV2DEnv(
        render_mode=None,
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

    env = Monitor(env)
    return env


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="logs",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    total_timesteps = 1_000_000

    model.learn(total_timesteps=total_timesteps)

    model.save("models/ppo_uav2d_lidar_v3")

    print("\n✅ Training Finished!")
    print("Model saved to: models/ppo_uav2d_lidar_v3.zip")
