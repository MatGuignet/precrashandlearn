"""train_champion.py — PPO training loop for the Part 3 champion (multi-agent).

Same PPO setup as train.py (Part 1), pointed at ChampionRacingEnv instead of
RacingEnv: the ego car shares the circuit with 1-3 opponents driven by a
frozen reference policy (--opponent-model, typically the Part 1 champion),
and the observation/reward now include real opponent awareness (see
env_champion.py and Part 2's diagnosis for why this is a structurally
different problem from Part 1, not a continuation of it).

Usage:
    docker compose run --rm sim python train_champion.py \\
        --resume logs_bc_champion/bc_pretrained.zip \\
        --opponent-model model.onnx --timesteps 3000000 --n-envs 4
"""

import argparse
import os
import sys

import gymnasium
sys.modules.setdefault("gym", gymnasium)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env_champion import ChampionRacingEnv
from train import export_onnx


class ChampionBestModelCallback(BaseCallback):
    """Same safety net as train.py's BestModelCallback (see its docstring),
    adapted for the multi-car eval: score now also penalizes
    vehicle-vehicle collisions, since an ego that "finishes" by plowing
    through opponents is not what Part 3 is asking for.
    """

    def __init__(self, opponent_model_path: str, eval_freq: int, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.opponent_model_path = opponent_model_path
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.best_score = -float("inf")
        self._eval_env = None
        self._last_eval_at = 0

    def _on_training_start(self) -> None:
        self._eval_env = ChampionRacingEnv(opponent_model_path=self.opponent_model_path)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_at >= self.eval_freq:
            self._last_eval_at = self.num_timesteps
            self._evaluate()
        return True

    def _evaluate(self) -> None:
        finished = 0
        total_progress = 0.0
        total_wall_hits = 0
        total_vehicle_hits = 0

        for map_name in self._eval_env.MAPS:
            obs, info = self._eval_env.reset(options={"map_name": map_name})
            wall_hits = 0
            vehicle_hits = 0
            max_progress = 0.0
            for _ in range(6000):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, info = self._eval_env.step(action)
                sim = info["simulation"]
                wall_hits += int(sim["collisions"]["wall"][0])
                vehicle_hits += int(sim["collisions"]["vehicle"][0])
                max_progress = max(max_progress, sim["max_progress"][0])
                if terminated or truncated:
                    break
            if info["status"] == 2:
                finished += 1
            total_progress += max_progress
            total_wall_hits += wall_hits
            total_vehicle_hits += vehicle_hits

        avg_progress = total_progress / len(self._eval_env.MAPS)
        score = (
            finished * 100
            + avg_progress * 10
            - total_wall_hits * 0.1
            - total_vehicle_hits * 0.2
        )

        print(
            f"[ChampionBestModelCallback] step={self.num_timesteps} finished={finished}/5 "
            f"avg_progress={avg_progress:.3f} wall_hits={total_wall_hits} "
            f"vehicle_hits={total_vehicle_hits} score={score:.2f} (best={self.best_score:.2f})",
            flush=True,
        )

        if score > self.best_score:
            self.best_score = score
            path = os.path.join(self.save_path, "best_model")
            self.model.save(path)
            print(f"[ChampionBestModelCallback] new best - saved to {path}.zip", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Part-3 multi-agent champion")
    parser.add_argument("--opponent-model", type=str, required=True,
                        help="ONNX model driving the 1-3 opponent cars each episode")
    parser.add_argument("--timesteps", type=int, default=3_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--logdir", type=str, default="logs_champion")
    parser.add_argument("--out", type=str, default="champion_model.onnx")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    return parser.parse_args()


def main():
    args = parse_args()

    def make_env():
        return Monitor(ChampionRacingEnv(opponent_model_path=args.opponent_model))

    vec_env_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    env = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed, vec_env_cls=vec_env_cls)

    if args.resume is not None:
        model = PPO.load(args.resume, env=env, device="cpu")
        model.gamma = args.gamma
        model.rollout_buffer.gamma = args.gamma
        model.ent_coef = args.ent_coef
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
        name_prefix="ppo_champion",
    )
    best_model_callback = ChampionBestModelCallback(
        opponent_model_path=args.opponent_model,
        eval_freq=args.eval_freq,
        save_path=args.logdir,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=CallbackList([checkpoint_callback, best_model_callback]),
        reset_num_timesteps=args.resume is None,
    )

    model.save(os.path.join(args.logdir, "ppo_champion_final"))

    best_model_path = os.path.join(args.logdir, "best_model.zip")
    if os.path.exists(best_model_path):
        print(f"Exporting best_model.zip (score={best_model_callback.best_score:.2f}), "
              f"not the final-step model, as {args.out}", flush=True)
        try:
            export_onnx(PPO.load(best_model_path, device="cpu"), args.out)
        except Exception as e:
            print(f"WARNING: could not load/export best_model.zip ({e}). "
                  f"Exporting the final-step model instead; best_model.zip "
                  f"is still on disk at {best_model_path} for manual export.", flush=True)
            export_onnx(model, args.out)
    else:
        export_onnx(model, args.out)

    env.close()


if __name__ == "__main__":
    main()
