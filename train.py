import argparse
import os
import sys

# Le dossier gym/ à la racine (moteur f110) fait que `import gym` tombe dessus
# au lieu de lever une ImportError, et ça casse model.save() de sb3.
# On force "gym" -> gymnasium, AVANT d'importer stable_baselines3.
import gymnasium

sys.modules.setdefault("gym", gymnasium)

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env import RacingEnv


class BestModelCallback(BaseCallback):
    """Garde le meilleur modèle vu pendant l'entraînement (best_model.zip).

    PPO ne s'améliore pas en continu : sur deux longs runs le modèle était bon
    puis s'est dégradé (collisions qui reviennent), et j'ai dû retrouver le bon
    checkpoint à la main. Ici on évalue tous les eval_freq steps sur les 5 maps
    et on garde le meilleur score.
    """

    def __init__(self, eval_freq: int, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.best_score = -float("inf")
        self._eval_env = None
        self._last_eval_at = 0

    def _on_training_start(self) -> None:
        self._eval_env = RacingEnv()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_at >= self.eval_freq:
            self._last_eval_at = self.num_timesteps
            self._evaluate()
        return True

    def _evaluate(self) -> None:
        finished = 0
        total_progress = 0.0
        total_wall_hits = 0

        for map_name in self._eval_env.MAPS:
            obs, info = self._eval_env.reset(options={"map_name": map_name})
            wall_hits = 0
            max_progress = 0.0
            for _ in range(6000):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, info = self._eval_env.step(action)
                sim = info["simulation"]
                wall_hits += int(sim["collisions"]["wall"][0])
                max_progress = max(max_progress, sim["max_progress"][0])
                if terminated or truncated:
                    break
            if info["status"] == 2:
                finished += 1
            total_progress += max_progress
            total_wall_hits += wall_hits

        avg_progress = total_progress / len(self._eval_env.MAPS)
        # finir une map compte le plus, les collisions départagent.
        # Poids au pif, ça sert juste à comparer les checkpoints entre eux.
        score = finished * 100 + avg_progress * 10 - total_wall_hits * 0.1

        print(
            f"[BestModelCallback] step={self.num_timesteps} finished={finished}/5 "
            f"avg_progress={avg_progress:.3f} wall_hits={total_wall_hits} "
            f"score={score:.2f} (best={self.best_score:.2f})",
            flush=True,
        )

        if score > self.best_score:
            self.best_score = score
            path = os.path.join(self.save_path, "best_model")
            self.model.save(path)
            print(f"[BestModelCallback] new best - saved to {path}.zip", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Part-1 solo racing PPO agent")
    parser.add_argument("--timesteps", type=int, default=3_000_000, help="Nombre total de timesteps")
    parser.add_argument("--n-envs", type=int, default=4, help="Nb d'envs en parallèle (4 max sinon OOM sous Docker)")
    parser.add_argument("--logdir", type=str, default="logs", help="Dossier TensorBoard + checkpoints")
    parser.add_argument("--out", type=str, default="model.onnx", help="Chemin du .onnx exporté")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint .zip à reprendre")
    parser.add_argument(
        "--eval-freq", type=int, default=100_000,
        help="Timesteps entre deux évals (5 maps). Sauvegarde logdir/best_model.zip "
             "quand le score s'améliore.",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.999,
        help="Discount. 0.99 = horizon ~100 steps (5s à 20Hz), trop court pour un "
             "tour de 1000+ steps. 0.999 = ~1000 steps.",
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.005,
        help="Coef d'entropie. 0.005 marche bien (le bug de la voiture qui recule "
             "venait de REWARD_DNF, pas de ça).",
    )
    parser.add_argument(
        "--reward-progress-scale", type=float, default=None,
        help="Remplace REWARD_PROGRESS_SCALE (défaut 100). Plus haut = plus agressif. "
             "Sert à donner un style différent aux 4 drivers.",
    )
    parser.add_argument(
        "--reward-wall-proximity-scale", type=float, default=None,
        help="Remplace REWARD_WALL_PROXIMITY_SCALE (défaut 0.5). Plus haut = plus prudent près des murs.",
    )
    parser.add_argument(
        "--wall-proximity-margin", type=float, default=None,
        help="Remplace WALL_PROXIMITY_MARGIN en mètres (défaut 0.2).",
    )
    parser.add_argument(
        "--reward-steering-jerk-scale", type=float, default=None,
        help="Remplace REWARD_STEERING_JERK_SCALE (défaut 0.02). Plus haut = conduite plus douce.",
    )
    return parser.parse_args()


def make_env(reward_overrides=None):
    # env_simulation est un singleton au niveau du module, donc 2 envs dans
    # le même process se marchent dessus -> SubprocVecEnv obligatoire si n_envs > 1
    env = RacingEnv()
    for key, value in (reward_overrides or {}).items():
        setattr(env, key, value)  # écrase la valeur de la classe pour cette instance
    return Monitor(env)


def export_onnx(model: PPO, path: str):
    """Exporte l'action déterministe de la policy en ONNX.

    Le clip dans [-1, 1] n'est pas dans le graphe, c'est agent.py qui le fait
    (features.action_to_physical), comme predict() de sb3.
    """

    class OnnxablePolicy(th.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, observation: th.Tensor) -> th.Tensor:
            actions, _values, _log_prob = self.policy(observation, deterministic=True)
            return actions

    onnx_policy = OnnxablePolicy(model.policy).to("cpu")
    onnx_policy.eval()

    dummy_input = th.zeros(1, *model.observation_space.shape)
    th.onnx.export(
        onnx_policy,
        dummy_input,
        path,
        input_names=["observation"],
        output_names=["action"],
        opset_version=17,
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        # l'exporteur dynamo (défaut depuis torch 2.6) plante sur la policy sb3
        # à cause du torch.distributions.Normal, donc on reste sur l'ancien
        dynamo=False,
    )
    print(f"Exported ONNX policy to {path}")


# usage : docker compose run --rm sim python train.py --timesteps 3000000 --n-envs 4
def main():
    args = parse_args()

    reward_overrides = {}
    if args.reward_progress_scale is not None:
        reward_overrides["REWARD_PROGRESS_SCALE"] = args.reward_progress_scale
    if args.reward_wall_proximity_scale is not None:
        reward_overrides["REWARD_WALL_PROXIMITY_SCALE"] = args.reward_wall_proximity_scale
    if args.wall_proximity_margin is not None:
        reward_overrides["WALL_PROXIMITY_MARGIN"] = args.wall_proximity_margin
    if args.reward_steering_jerk_scale is not None:
        reward_overrides["REWARD_STEERING_JERK_SCALE"] = args.reward_steering_jerk_scale
    if reward_overrides:
        print(f"Reward overrides for this run: {reward_overrides}", flush=True)

    # pas de VecNormalize : l'obs est déjà normalisée dans features.py et
    # agent.py n'aurait pas les stats mean/std au tournoi
    vec_env_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    env = make_vec_env(
        lambda: make_env(reward_overrides), n_envs=args.n_envs, seed=args.seed, vec_env_cls=vec_env_cls
    )

    if args.resume is not None:
        model = PPO.load(args.resume, env=env, device="cpu")
        # PPO.load() reprend gamma/ent_coef du checkpoint et ignore les args.
        # gamma est aussi copié dans le rollout buffer, faut changer les deux.
        model.gamma = args.gamma
        model.rollout_buffer.gamma = args.gamma
        model.ent_coef = args.ent_coef
        # sinon verbose=0 repris du checkpoint et on ne voit rien s'afficher
        model.verbose = 1
        print(f"Resumed with overridden gamma={args.gamma}, ent_coef={args.ent_coef}", flush=True)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=args.gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=args.logdir,
            seed=args.seed,
            device="cpu",
            verbose=1,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 1),
        save_path=os.path.join(args.logdir, "checkpoints"),
        name_prefix="ppo_racing",
    )
    best_model_callback = BestModelCallback(
        eval_freq=args.eval_freq,
        save_path=args.logdir,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=CallbackList([checkpoint_callback, best_model_callback]),
        reset_num_timesteps=args.resume is None,
    )

    model.save(os.path.join(args.logdir, "ppo_racing_final"))

    # on exporte le best_model plutôt que le dernier (cf. BestModelCallback),
    # sauf s'il n'existe pas (run trop court)
    best_model_path = os.path.join(args.logdir, "best_model.zip")
    if os.path.exists(best_model_path):
        print(f"Exporting best_model.zip (score={best_model_callback.best_score:.2f}), "
              f"not the final-step model, as {args.out}", flush=True)
        try:
            export_onnx(PPO.load(best_model_path, device="cpu"), args.out)
        except Exception as e:
            # eu une fois une erreur d'import torch bizarre en rechargeant ici.
            # Pour pas perdre tout le run, on exporte le modèle final et
            # best_model.zip reste dispo pour l'exporter à la main.
            print(f"WARNING: could not load/export best_model.zip ({e}). "
                  f"Exporting the final-step model instead; best_model.zip "
                  f"is still on disk at {best_model_path} for manual export.", flush=True)
            export_onnx(model, args.out)
    else:
        export_onnx(model, args.out)

    env.close()


if __name__ == "__main__":
    main()
