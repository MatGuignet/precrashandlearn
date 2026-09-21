"""features.py — Pure-numpy preprocessing shared by env.py (training) and agent.py (inference).

This module has no dependency outside numpy, so it can be imported unchanged
by both worlds without pulling gymnasium or torch into the tournament process
(see agent.py / env.py docstrings for why that boundary matters).

Anything that must stay identical between training and inference — LiDAR
normalization, velocity/steering scaling, action decoding — lives here once,
instead of being duplicated and risking silent drift.
"""

import numpy as np

# ==========================================================================
# Physical constants (must match env_simulation's contract)
# ==========================================================================

N_LIDAR = 100

LIDAR_MAX = 15.0

MIN_SPEED = -2.0
MAX_SPEED = 10.0

MAX_STEERING = 0.4189

# ==========================================================================
# Observation space bounds
# ==========================================================================
#
# 100 LiDAR values : [0, 1]
# velocity         : [-0.2, 1]
# steering         : [-1, 1]
# progress         : [0, 1]

OBS_LOW = np.concatenate(
    [
        np.zeros(N_LIDAR, dtype=np.float32),
        np.array([-0.2, -1.0, 0.0], dtype=np.float32),
    ]
)

OBS_HIGH = np.concatenate(
    [
        np.ones(N_LIDAR, dtype=np.float32),
        np.array([1.0, 1.0, 1.0], dtype=np.float32),
    ]
)

OBS_SIZE = N_LIDAR + 3


def build_observation(raw_obs: dict) -> np.ndarray:
    """Convert a raw get_obs()-style dict into the normalized 103-dim vector
    consumed by the policy. Solo-car only (Part 1): opponents are ignored.
    """

    lidar = np.asarray(raw_obs["lidar"], dtype=np.float32)
    lidar = np.clip(lidar / LIDAR_MAX, 0.0, 1.0)

    velocity = np.float32(
        np.clip(raw_obs["velocity"] / MAX_SPEED, -0.2, 1.0)
    )

    steering = np.float32(
        np.clip(raw_obs["steering"] / MAX_STEERING, -1.0, 1.0)
    )

    progress = np.float32(
        np.clip(raw_obs["progress"], 0.0, 1.0)
    )

    observation = np.concatenate(
        [lidar, np.array([velocity, steering, progress], dtype=np.float32)]
    )

    return observation.astype(np.float32)


def action_to_physical(action) -> tuple[float, float]:
    """Convert a normalized RL action [-1, 1]^2 into the physical simulation
    action (target_speed in m/s, steering in rad).
    """

    action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

    normalized_speed = float(action[0])
    normalized_steering = float(action[1])

    # Map [-1, 1] -> [MIN_SPEED, MAX_SPEED]
    target_speed = MIN_SPEED + (normalized_speed + 1.0) * (MAX_SPEED - MIN_SPEED) / 2.0

    # Map [-1, 1] -> [-MAX_STEERING, MAX_STEERING]
    steering = normalized_steering * MAX_STEERING

    return float(target_speed), float(steering)
