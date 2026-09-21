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


# ==========================================================================
# Opponent encoding (Part 3 champion only — Part 1 solo agents never call this)
# ==========================================================================
#
# Fixed 4 slots regardless of how many opponents are actually active, so the
# observation shape never changes: each slot is
# [x_rel, y_rel, yaw_rel, speed, active], normalized.
#
# Field names are read defensively under BOTH names seen in this codebase:
# env_simulation.get_obs() itself uses x_rel/y_rel/yaw_rel/speed/status, but
# agent_loader.py's dry-run dummy obs (used to validate agent.py at load time,
# in every team's process, before a single real observation ever arrives)
# uses rel_x/rel_y/rel_yaw/velocity/active instead. An agent.py that only
# understood one spelling would raise KeyError on whichever dummy/real dict
# it wasn't written for - silently failing tournament load in one case.

N_OPPONENT_SLOTS = 4
OPPONENT_FEATURES_PER_SLOT = 5
OBS_SIZE_MULTI = OBS_SIZE + N_OPPONENT_SLOTS * OPPONENT_FEATURES_PER_SLOT

OPPONENT_RANGE = LIDAR_MAX  # meters, same normalization scale as LiDAR


def _encode_opponent(opp: dict) -> np.ndarray:
    x_rel = float(opp.get("x_rel", opp.get("rel_x", 0.0)))
    y_rel = float(opp.get("y_rel", opp.get("rel_y", 0.0)))
    yaw_rel = float(opp.get("yaw_rel", opp.get("rel_yaw", 0.0)))
    speed = float(opp.get("speed", opp.get("velocity", 0.0)))

    status = opp.get("status")
    if status is not None:
        active = 1.0 if int(status) == 1 else 0.0
    else:
        active = float(opp.get("active", 0.0))

    x_norm = np.clip(x_rel / OPPONENT_RANGE, -1.0, 1.0)
    y_norm = np.clip(y_rel / OPPONENT_RANGE, -1.0, 1.0)
    yaw_norm = np.clip(yaw_rel / np.pi, -1.0, 1.0)
    speed_norm = np.clip(speed / MAX_SPEED, -0.2, 1.0)

    return np.array([x_norm, y_norm, yaw_norm, speed_norm, active], dtype=np.float32)


def build_observation_multi(raw_obs: dict) -> np.ndarray:
    """Like build_observation(), plus 4 fixed opponent slots (Part 3 champion)."""

    base = build_observation(raw_obs)
    opponents = raw_obs.get("opponents", {})
    slots = [_encode_opponent(opponents.get(i, {})) for i in range(N_OPPONENT_SLOTS)]
    return np.concatenate([base] + slots).astype(np.float32)


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
