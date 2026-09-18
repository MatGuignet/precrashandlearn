import sys
import warnings
import numpy as np

import gymnasium
sys.modules.setdefault("gym", gymnasium)

from stable_baselines3 import PPO
from env import RacingEnv

warnings.filterwarnings("error", category=RuntimeWarning)

model = PPO.load(sys.argv[1] if len(sys.argv) > 1 else "/app/logs_bc/bc_pretrained_lowstd.zip", device="cpu")

env = RacingEnv()
obs, info = env.reset()
print("map:", info["map"])

for i in range(30000):
    action, _ = model.predict(obs, deterministic=False)
    try:
        obs, reward, terminated, truncated, info = env.step(action)
    except RuntimeWarning as e:
        print(f"CRASH at step {i}")
        print("action:", action)
        print("prev obs velocity/steering/progress:", obs[100:103])
        print("prev obs lidar min/max:", obs[:100].min(), obs[:100].max())
        print("error:", e)
        raise
    if i % 500 == 0:
        print(i, "velocity=", obs[100], "progress=", obs[102])
    if terminated or truncated:
        print(f"episode ended cleanly at step {i}, status={info['status']}")
        obs, info = env.reset()
        print("new map:", info["map"])

env.close()
print("completed 6000 steps with no instability")
