import sys
import numpy as np

import gymnasium  # noqa: F401
sys.modules.setdefault("gym", gymnasium)

from env_champion import ChampionRacingEnv

env = ChampionRacingEnv(opponent_model_path="/app/model.onnx")
print("observation_space:", env.observation_space.shape)
print("action_space:", env.action_space.shape)

obs, info = env.reset(options={"map_name": "Monza"})
print("reset OK, obs shape:", obs.shape, "num_cars:", info["num_cars"])

for i in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if i % 10 == 0:
        print(f"step {i}: reward={reward:.3f} terminated={terminated} truncated={truncated}")
    if terminated or truncated:
        print("episode ended early at step", i)
        break

env.close()
print("smoke test passed")
