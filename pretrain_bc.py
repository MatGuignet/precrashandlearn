"""pretrain_bc.py — Behavior-cloning warm start for the Part 1 solo agent.

Every failure mode hit while debugging train.py this session (frozen car,
reverse-lock, steering saturation, entropy collapse) was really the same
root problem wearing different clothes: PPO exploring from a random policy
essentially never stumbles onto "steer through this corner successfully"
often enough for gradient descent to reinforce it, on a 103-dim continuous
control task across 5 different tracks.

This script sidesteps that cold-start exploration problem instead of
tuning around it further: a simple reactive LiDAR heuristic (no learning,
pure numpy) that reliably drives forward without crashing is used to
generate demonstrations, and the PPO policy network is pretrained on those
demonstrations via plain supervised regression (behavior cloning) before
any RL update ever runs. RL fine-tuning (train.py --resume) then starts
from "already knows how to drive forward reasonably," not from scratch.

Usage:
    docker compose run --rm sim python pretrain_bc.py --steps 300000 --out logs_bc/bc_pretrained
    docker compose run --rm sim python train.py --resume logs_bc/bc_pretrained.zip --timesteps 3000000 ...
"""

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
    """RacingEnv + one extra accessor so env_method() can fetch the raw,
    un-normalized obs dict the heuristic needs. Kept local to this
    data-collection script rather than added to env.py, which stays exactly
    the training/inference contract with nothing collection-specific in it.
    """

    def _raw_obs_for_heuristic(self):
        return env_simulation.get_obs(0)

# ==========================================================================
# Reactive LiDAR heuristic — no learning, just geometry
# ==========================================================================
#
# 100 rays over a full 360 deg field of view (see env_simulation._LIDAR_FOV),
# angle[i] = -pi + i * (2*pi / 99), so index ~49-50 point straight ahead.

_N_LIDAR = 100
_FOV = 2.0 * np.pi
_ANGLES = -_FOV / 2.0 + np.arange(_N_LIDAR) * (_FOV / (_N_LIDAR - 1))

_FORWARD_HALF_ANGLE = np.deg2rad(90)   # hemisphere considered for gap-finding
_STEER_HALF_ANGLE = np.deg2rad(70)     # angle range mapped to full steering lock
_STRAIGHT_HALF_ANGLE = np.deg2rad(15)  # cone used to gauge forward clearance
_FORWARD_MASK = np.abs(_ANGLES) <= _FORWARD_HALF_ANGLE
_STRAIGHT_MASK = np.abs(_ANGLES) <= _STRAIGHT_HALF_ANGLE

_BUBBLE_RAYS = 6        # rays zeroed out around the single closest obstacle
_GAP_THRESHOLD = 2.5    # meters - a ray counts as "open" above this distance
_CLEAR_DISTANCE = 8.0   # meters of straight-ahead clearance counted as "fully clear"
_MIN_SPEED_NORM = -0.4  # normalized speed floor (~1.5 m/s) — never reverses
_MAX_SPEED_NORM = 0.8   # normalized speed ceiling (~8.8 m/s) when fully clear


def _widest_gap_center(lidar_1d: np.ndarray) -> int:
    """Follow-The-Gap: index at the center of the widest contiguous run of
    rays above _GAP_THRESHOLD, after bubbling out the closest obstacle.
    Standard F1TENTH reactive baseline — more robust to single noisy rays
    than steering at the single farthest ray.
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
    """Reactive Follow-The-Gap controller: steer toward the center of the
    widest open corridor within a forward hemisphere, throttle proportional
    to clearance straight ahead and inversely to how hard we're steering.
    Always commands forward motion — this is precisely the behavior PPO
    kept failing to discover on its own.
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

    # Slow down further the sharper the turn ahead — without this, the
    # heuristic barrels toward corners at "clear straight ahead" speed right
    # up until the moment it also needs to steer hard, clipping the apex.
    turn_severity = abs(steer_norm)
    speed_norm *= (1.0 - 0.6 * turn_severity)

    return np.array([speed_norm, steer_norm], dtype=np.float32)


# ==========================================================================
# Demonstration collection
# ==========================================================================

def collect_demonstrations(n_steps: int, n_envs: int, seed: int):
    """Roll out the heuristic in the real training env (domain randomization
    included) and record (normalized_observation, heuristic_action) pairs.
    """

    def make_env():
        return _DemoEnv()

    vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    env = make_vec_env(make_env, n_envs=n_envs, seed=seed, vec_env_cls=vec_env_cls)

    observations = []
    actions = []

    obs = env.reset()
    collected = 0
    while collected < n_steps:
        # Each subprocess env wraps its own env_simulation singleton; the
        # heuristic needs the RAW obs (physical lidar in meters), not the
        # normalized 103-dim vector env.step() returns. env_method() calls
        # env_simulation.get_obs(0) inside each subprocess directly.
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


# ==========================================================================
# Behavior cloning
# ==========================================================================

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

            # ActorCriticPolicy.forward(deterministic=True) returns the raw
            # (unclipped, differentiable) Gaussian mean action - exactly
            # what agent.py/features.action_to_physical will clip and use
            # at inference, so this is the right target to fit.
            pred_actions, _values, _log_prob = model.policy(obs_batch, deterministic=True)

            loss = th.nn.functional.mse_loss(pred_actions, act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        print(f"  BC epoch {epoch + 1}/{epochs}: mse={total_loss / n_batches:.5f}")

    # BC only fits the mean-action pathway above; log_std (the Gaussian
    # action distribution's exploration width) is untouched and stays at
    # its PPO default (std=1.0 - roughly the whole [-1,1] action range).
    # Fine-tuning's very first rollout SAMPLES from this distribution
    # (deterministic=False), so a well-trained mean gets swamped by std=1.0
    # noise, clipped to erratic extreme actions, which was enough to send
    # env_simulation's RK4 integrator into repeated numerical overflow on
    # the first attempt. Shrink it to a modest exploration level so
    # fine-tuning starts from "confidently drives, a bit of noise to
    # refine with" instead of "coin-flip actions around a good mean".
    with th.no_grad():
        model.policy.log_std.fill_(np.log(0.15))


def parse_args():
    parser = argparse.ArgumentParser(description="Behavior-cloning warm start for the racing policy")
    parser.add_argument("--steps", type=int, default=300_000, help="Demonstration transitions to collect")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20, help="Supervised passes over the demonstration set")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="logs_bc/bc_pretrained", help="Output path (no .zip suffix)")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Collecting {args.steps} demonstration transitions from the LiDAR heuristic...")
    observations, actions = collect_demonstrations(args.steps, args.n_envs, args.seed)
    print(f"Collected {observations.shape[0]} transitions. "
          f"action mean={actions.mean(axis=0)}, action std={actions.std(axis=0)}")

    # Single-env instance just to construct a PPO model with the right
    # observation/action spaces - never stepped, BC only touches the
    # network via direct tensors.
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
        verbose=0,
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
