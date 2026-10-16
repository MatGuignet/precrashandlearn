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
from env_champion import ChampionRacingEnv
from pretrain_bc import heuristic_action, behavior_clone


# Même idée que pretrain_bc.py mais pour le champion : même heuristique
# (elle voit déjà les autres voitures comme des obstacles dans le lidar),
# env avec adversaires et obs de 123.
#
#   python pretrain_bc_champion.py --opponent-model model.onnx --steps 300000

def parse_args():
    parser = argparse.ArgumentParser(description="Behavior-cloning warm start for the Part 3 champion")
    parser.add_argument("--opponent-model", type=str, required=True,
                        help="Modèle ONNX des adversaires")
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="logs_bc_champion/bc_pretrained")
    return parser.parse_args()


def collect_demonstrations(opponent_model: str, n_steps: int, n_envs: int, seed: int):

    class _DemoEnv(ChampionRacingEnv):
        def _raw_obs_for_heuristic(self):
            return env_simulation.get_obs(0)

    def make_env():
        return _DemoEnv(opponent_model_path=opponent_model)

    vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    env = make_vec_env(make_env, n_envs=n_envs, seed=seed, vec_env_cls=vec_env_cls)

    observations = []
    actions = []

    obs = env.reset()
    collected = 0
    while collected < n_steps:
        raw_obs_batch = env.env_method("_raw_obs_for_heuristic")
        action_batch = np.stack([heuristic_action(o) for o in raw_obs_batch])

        observations.append(obs.copy())
        actions.append(action_batch)
        collected += n_envs

        obs, rewards, dones, infos = env.step(action_batch)

        if collected % (n_envs * 2000) == 0:
            print(f"  collected {collected}/{n_steps} transitions...", flush=True)

    env.close()

    observations = np.concatenate(observations, axis=0).astype(np.float32)
    actions = np.concatenate(actions, axis=0).astype(np.float32)
    return observations[:n_steps], actions[:n_steps]


def main():
    args = parse_args()

    print(f"Collecting {args.steps} demonstration transitions "
          f"(opponents driven by {args.opponent_model})...", flush=True)
    observations, actions = collect_demonstrations(args.opponent_model, args.steps, args.n_envs, args.seed)
    print(f"Collected {observations.shape[0]} transitions, obs dim={observations.shape[1]}. "
          f"action mean={actions.mean(axis=0)}, action std={actions.std(axis=0)}", flush=True)

    template_env = make_vec_env(
        lambda: ChampionRacingEnv(opponent_model_path=args.opponent_model), n_envs=1, seed=args.seed
    )

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

    print("Behavior-cloning the policy network on the demonstrations...", flush=True)
    behavior_clone(model, observations, actions, args.epochs, args.batch_size)

    template_env.close()

    model.save(args.out)
    print(f"Saved BC-pretrained checkpoint to {args.out}.zip", flush=True)


if __name__ == "__main__":
    main()
