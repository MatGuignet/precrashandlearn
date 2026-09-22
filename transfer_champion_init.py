"""transfer_champion_init.py — Warm-start the Part 3 champion from the Part 1
solo model's weights, instead of behavior-cloning a fresh network from a
generic heuristic.

Why: pretrain_bc_champion.py's BC-from-heuristic approach discards all the
driving competence the solo model spent ~9M steps acquiring, and rebuilds it
from a generic Follow-The-Gap imitation on a brand-new network. Diagnostics
after 1M + 2.1M further RL steps still showed progress capped at 8-16% on
every map, at a consistent per-map stopping point even with zero collisions
- consistent with a still-undertrained network giving up at the first
genuinely hard corner on each track, not an opponent-avoidance failure per
se (Part 2/3's actual target skill).

The champion's observation is the solo observation (103 dims) plus 20 new
opponent-slot dims appended at the end (features.build_observation_multi).
That means the first Linear layer's weight matrix is the ONLY tensor whose
shape actually differs between the two networks (input width 103 -> 123);
every other layer matches exactly. Copying the solo weights directly and
zero-initializing only the 20 new input columns makes the champion, at
initialization, mathematically IDENTICAL to the solo model (the opponent
features are multiplied by zero and contribute nothing) - i.e. it starts
already able to finish/nearly-finish every training map solo, and PPO
fine-tuning only has to learn to *additionally* react to opponents on top
of that, rather than re-deriving solo driving competence from scratch.

Usage:
    docker compose run --rm sim python transfer_champion_init.py \\
        --solo-model logs_weather/best_model.zip --out logs_champion3/transferred_init
    docker compose run --rm sim python train_champion.py --resume logs_champion3/transferred_init.zip \\
        --opponent-model model.onnx --timesteps 2000000 ...
"""

import argparse
import sys

import gymnasium
sys.modules.setdefault("gym", gymnasium)

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from env_champion import ChampionRacingEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Transfer solo model weights into a champion-shaped network")
    parser.add_argument("--solo-model", type=str, required=True, help="Path to the Part 1 solo model .zip")
    parser.add_argument("--opponent-model", type=str, required=True,
                        help="ONNX model for opponents (only used to construct a valid ChampionRacingEnv)")
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="logs_champion3/transferred_init")
    return parser.parse_args()


def main():
    args = parse_args()

    solo_model = PPO.load(args.solo_model, device="cpu")
    solo_obs_dim = solo_model.observation_space.shape[0]

    template_env = make_vec_env(
        lambda: ChampionRacingEnv(opponent_model_path=args.opponent_model), n_envs=1, seed=args.seed
    )

    champion_model = PPO(
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

    solo_sd = solo_model.policy.state_dict()
    champ_sd = champion_model.policy.state_dict()

    copied, expanded, skipped = [], [], []

    for key, champ_tensor in champ_sd.items():
        if key not in solo_sd:
            skipped.append((key, "not in solo model"))
            continue
        solo_tensor = solo_sd[key]

        if solo_tensor.shape == champ_tensor.shape:
            champ_sd[key] = solo_tensor.clone()
            copied.append(key)
        elif (
            solo_tensor.dim() == 2 and champ_tensor.dim() == 2
            and solo_tensor.shape[0] == champ_tensor.shape[0]
            and solo_tensor.shape[1] == solo_obs_dim
        ):
            # The one expected mismatch: this layer's input width grew from
            # 103 to 123. Copy the original 103 columns, zero the 20 new
            # (opponent-feature) ones so they start fully inert.
            new_tensor = champ_tensor.clone()
            with th.no_grad():
                new_tensor[:, :solo_obs_dim] = solo_tensor
                new_tensor[:, solo_obs_dim:] = 0.0
            champ_sd[key] = new_tensor
            expanded.append(key)
        else:
            skipped.append((key, f"shape {tuple(solo_tensor.shape)} vs {tuple(champ_tensor.shape)}"))

    champion_model.policy.load_state_dict(champ_sd)

    print(f"Copied unchanged: {len(copied)} tensors")
    print(f"Expanded (input layer, zero-padded for opponents): {expanded}")
    if skipped:
        print(f"Skipped (kept randomly initialized): {skipped}")
    else:
        print("Nothing skipped - full transfer.")

    template_env.close()
    champion_model.save(args.out)
    print(f"Saved transferred champion init to {args.out}.zip")


if __name__ == "__main__":
    main()
