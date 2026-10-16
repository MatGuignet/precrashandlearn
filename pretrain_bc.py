import argparse
import sys

import gymnasium
sys.modules.setdefault("gym", gymnasium)

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import env_simulation
from env import RacingEnv


class _DemoEnv(RacingEnv):
    # juste pour récupérer l'obs brute (lidar en mètres) via env_method()

    def _raw_obs_for_heuristic(self):
        return env_simulation.get_obs(0)

# PPO from scratch n'arrivait jamais à apprendre à passer un virage (voiture
# figée, qui recule, volant bloqué...). Donc on pré-entraîne le réseau en
# imitant une heuristique lidar simple, puis on fine-tune avec train.py --resume.
#
#   python pretrain_bc.py --steps 300000 --out logs_bc/bc_pretrained
#   python train.py --resume logs_bc/bc_pretrained.zip ...

# Heuristique follow-the-gap
# 100 rayons sur 360°, l'indice ~49-50 regarde droit devant

_N_LIDAR = 100
_FOV = 2.0 * np.pi
_ANGLES = -_FOV / 2.0 + np.arange(_N_LIDAR) * (_FOV / (_N_LIDAR - 1))

_FORWARD_HALF_ANGLE = np.deg2rad(90)   # on cherche le trou seulement devant
_STEER_HALF_ANGLE = np.deg2rad(70)     # 70° = braquage max
_STRAIGHT_HALF_ANGLE = np.deg2rad(15)  # cône pour mesurer la place devant
_FORWARD_MASK = np.abs(_ANGLES) <= _FORWARD_HALF_ANGLE
_STRAIGHT_MASK = np.abs(_ANGLES) <= _STRAIGHT_HALF_ANGLE

_BUBBLE_RAYS = 6        # rayons mis à 0 autour de l'obstacle le plus proche
_GAP_THRESHOLD = 2.5    # m, au-dessus le rayon est "libre"
_CLEAR_DISTANCE = 8.0   # m devant = voie libre
_MIN_SPEED_NORM = -0.4  # ~1.5 m/s, ne recule jamais
_MAX_SPEED_NORM = 0.8   # ~8.8 m/s


def _widest_gap_center(lidar_1d: np.ndarray) -> int:
    """Indice du milieu du plus grand trou (suite de rayons > _GAP_THRESHOLD).
    Moins sensible au bruit que viser juste le rayon le plus long.
    """
    safe = lidar_1d.copy()
    closest_idx = int(np.argmin(safe))
    lo = max(0, closest_idx - _BUBBLE_RAYS)
    hi = min(len(safe), closest_idx + _BUBBLE_RAYS + 1)
    safe[lo:hi] = 0.0

    free = safe > _GAP_THRESHOLD
    best_start, best_len = 0, 0
    cur_start, cur_len = None, 0
    for i, is_free in enumerate(free):
        if is_free:
            if cur_start is None:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_start, cur_len = None, 0

    if best_len == 0:
        return int(np.argmax(lidar_1d))
    return best_start + best_len // 2


def heuristic_action(raw_obs: dict) -> np.ndarray:
    """Vise le milieu du trou, accélère selon la place devant et ralentit
    quand on braque. Avance toujours.
    """
    lidar = np.asarray(raw_obs["lidar"], dtype=np.float32)

    forward_lidar = lidar[_FORWARD_MASK]
    forward_angles = _ANGLES[_FORWARD_MASK]
    target_idx = _widest_gap_center(forward_lidar)
    target_angle = forward_angles[target_idx]

    steer_norm = float(np.clip(target_angle / _STEER_HALF_ANGLE, -1.0, 1.0))

    straight_clear = float(np.min(lidar[_STRAIGHT_MASK]))
    clear_ratio = np.clip(straight_clear / _CLEAR_DISTANCE, 0.0, 1.0)
    speed_norm = _MIN_SPEED_NORM + clear_ratio * (_MAX_SPEED_NORM - _MIN_SPEED_NORM)

    # sans ça elle arrive trop vite dans les virages et tape le mur
    turn_severity = abs(steer_norm)
    speed_norm *= (1.0 - 0.6 * turn_severity)

    return np.array([speed_norm, steer_norm], dtype=np.float32)


def collect_demonstrations(n_steps: int, n_envs: int, seed: int):
    """Fait rouler l'heuristique dans le vrai env et enregistre (obs normalisée, action)."""

    def make_env():
        return _DemoEnv()

    vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    env = make_vec_env(make_env, n_envs=n_envs, seed=seed, vec_env_cls=vec_env_cls)

    observations = []
    actions = []

    obs = env.reset()
    collected = 0
    while collected < n_steps:
        # l'heuristique a besoin du lidar en mètres, pas de l'obs normalisée
        raw_obs_batch = env.env_method("_raw_obs_for_heuristic")
        action_batch = np.stack([heuristic_action(o) for o in raw_obs_batch])

        observations.append(obs.copy())
        actions.append(action_batch)
        collected += n_envs

        obs, rewards, dones, infos = env.step(action_batch)

        if collected % (n_envs * 2000) == 0:
            print(f"  collected {collected}/{n_steps} transitions...")

    env.close()

    observations = np.concatenate(observations, axis=0).astype(np.float32)
    actions = np.concatenate(actions, axis=0).astype(np.float32)
    return observations[:n_steps], actions[:n_steps]


def behavior_clone(model: PPO, observations: np.ndarray, actions: np.ndarray, epochs: int, batch_size: int):
    device = model.device
    obs_t = th.as_tensor(observations, device=device)
    act_t = th.as_tensor(actions, device=device)

    n = obs_t.shape[0]
    optimizer = model.policy.optimizer

    for epoch in range(epochs):
        perm = th.randperm(n, device=device)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            obs_batch = obs_t[idx]
            act_batch = act_t[idx]

            # deterministic=True -> moyenne de la gaussienne, c'est ce qu'on exporte en onnx
            pred_actions, _values, _log_prob = model.policy(obs_batch, deterministic=True)

            loss = th.nn.functional.mse_loss(pred_actions, act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        print(f"  BC epoch {epoch + 1}/{epochs}: mse={total_loss / n_batches:.5f}")

    # le BC ne touche pas log_std qui reste à std=1 : au début du fine-tune les
    # actions étaient quasi aléatoires et faisaient planter la physique (overflow).
    # On le baisse à 0.15.
    with th.no_grad():
        model.policy.log_std.fill_(np.log(0.15))


def parse_args():
    parser = argparse.ArgumentParser(description="Behavior-cloning warm start for the racing policy")
    parser.add_argument("--steps", type=int, default=300_000, help="Nb de transitions à collecter")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20, help="Nb d'epochs de BC")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="logs_bc/bc_pretrained", help="Chemin de sortie (sans .zip)")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Collecting {args.steps} demonstration transitions from the LiDAR heuristic...")
    observations, actions = collect_demonstrations(args.steps, args.n_envs, args.seed)
    print(f"Collected {observations.shape[0]} transitions. "
          f"action mean={actions.mean(axis=0)}, action std={actions.std(axis=0)}")

    # env juste pour créer le modèle PPO avec les bons espaces, jamais steppé
    template_env = make_vec_env(lambda: RacingEnv(), n_envs=1, seed=args.seed)

    model = PPO(
        "MlpPolicy",
        template_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=args.seed,
        device="cpu",
        verbose=1,
    )

    print("Behavior-cloning the policy network on the demonstrations...")
    behavior_clone(model, observations, actions, args.epochs, args.batch_size)

    template_env.close()

    model.save(args.out)
    print(f"Saved BC-pretrained checkpoint to {args.out}.zip")
    print("Resume RL fine-tuning with: python train.py --resume "
          f"{args.out}.zip --timesteps <N> ...")


if __name__ == "__main__":
    main()
