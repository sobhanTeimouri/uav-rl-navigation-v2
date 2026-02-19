from uav2d_env import UAV2DEnv
import numpy as np
import time

env = UAV2DEnv(render_mode="human")

obs, info = env.reset()

for _ in range(300):
    action = env.action_space.sample()  # random action
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

    time.sleep(0.02)

env.close()
