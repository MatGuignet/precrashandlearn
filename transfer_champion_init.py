import argparse
import sys

import gymnasium
sys.modules.setdefault("gym", gymnasium)

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from env_champion import ChampionRacingEnv


# Init du champion à partir des poids du modèle solo au lieu du BC.
# Avec le BC le champion plafonnait à 8-16% de progress même après 3M steps,
# on jetait tout ce que le solo avait appris.
#
# Seule la 1re couche change de taille (103 -> 123 entrées). On copie les poids
# du solo et on met à 0 les 20 nouvelles colonnes : au départ le champion conduit
# exactement comme le solo, et le fine-tune n'a plus qu'à apprendre les adversaires.
#
#   python transfer_champion_init.py --solo-model logs_weather/best_model.zip \
#       --opponent-model model.onnx --out logs_champion3/transferred_init
#   python train_champion.py --resume logs_champion3/transferred_init.zip ...

def parse_args():
    parser = argparse.ArgumentParser(description="Transfer solo model weights into a champion-shaped network")
    parser.add_argument("--solo-model", type=str, required=True, help="Modèle solo (.zip)")
    parser.add_argument("--opponent-model", type=str, required=True,
                        help="Modèle ONNX adversaire (juste pour construire l'env)")
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
            # la couche d'entrée : on garde les 103 colonnes, les 20 nouvelles à 0
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
