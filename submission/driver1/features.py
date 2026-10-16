import numpy as np

# constantes physiques, doivent coller à env_simulation
N_LIDAR = 100

LIDAR_MAX = 15.0

MIN_SPEED = -2.0
MAX_SPEED = 10.0

MAX_STEERING = 0.4189

# bornes de l'obs :
# lidar x100 -> [0, 1], vitesse -> [-0.2, 1], steering -> [-1, 1], progress -> [0, 1]

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
    """dict get_obs() -> vecteur normalisé de 103 valeurs (solo, partie 1)."""

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

    # np.clip laisse passer les NaN. La physique f110 en sort parfois un, et ça
    # m'a fait crasher un run de 15M steps vers 600k (NaN jusque dans le Normal
    # de torch). Du coup on nettoie ici.
    return np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)


# Adversaires (champion partie 3 uniquement)
#
# Toujours 4 slots, même s'il y a moins de voitures, pour garder une taille
# d'obs fixe. Un slot = [x_rel, y_rel, yaw_rel, speed, active].
#
# Attention aux noms de clés : get_obs() renvoie x_rel/y_rel/yaw_rel/speed/status
# mais l'obs bidon de agent_loader.py (le dry-run au chargement) utilise
# rel_x/rel_y/rel_yaw/velocity/active. On lit les deux sinon KeyError au
# chargement du tournoi.

N_OPPONENT_SLOTS = 4
OPPONENT_FEATURES_PER_SLOT = 5
OBS_SIZE_MULTI = OBS_SIZE + N_OPPONENT_SLOTS * OPPONENT_FEATURES_PER_SLOT

OPPONENT_RANGE = LIDAR_MAX  # même échelle que le lidar


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
    """Pareil que build_observation() + les 4 slots adversaires (123 valeurs)."""

    base = build_observation(raw_obs)
    opponents = raw_obs.get("opponents", {})
    slots = [_encode_opponent(opponents.get(i, {})) for i in range(N_OPPONENT_SLOTS)]
    observation = np.concatenate([base] + slots)
    return np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)


def action_to_physical(action) -> tuple[float, float]:
    """Action du réseau dans [-1, 1]^2 -> (vitesse cible en m/s, steering en rad)."""

    # même souci de NaN que plus haut, on nettoie avant le clip
    action = np.nan_to_num(np.asarray(action, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
    action = np.clip(action, -1.0, 1.0)

    normalized_speed = float(action[0])
    normalized_steering = float(action[1])

    # [-1, 1] -> [MIN_SPEED, MAX_SPEED]
    target_speed = MIN_SPEED + (normalized_speed + 1.0) * (MAX_SPEED - MIN_SPEED) / 2.0

    # [-1, 1] -> [-MAX_STEERING, MAX_STEERING]
    steering = normalized_steering * MAX_STEERING

    return float(target_speed), float(steering)
