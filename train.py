"""train.py — PPO training loop for the Crash & Learn solo driving licence (Part 1).

Trains one policy able to loop any of the 5 training circuits (randomly
reselected every episode by RacingEnv.reset(), under env_simulation's own
weather/friction/LiDAR-noise randomization) and exports it to model.onnx.

Usage (inside the provided Docker image, which has torch/sb3/onnx installed):

    docker compose run --rm sim python train.py --timesteps 3000000 --n-envs 4

n-envs=4 is a practical ceiling, not a tuning choice: env_simulation's parallel
envs each run in their own subprocess (see make_env() below), and on a
Docker Desktop VM capped around 4GB RAM, 8 parallel envs reliably OOM-kill a
subprocess (confirmed: EOFError from SubprocVecEnv.reset() on 8 envs, clean
run on 4). Raise it only if the container has more memory available.

Why PPO:
    - On-policy: each batch of rollouts is collected under whatever
      friction/map/noise regime env_simulation happens to be sampling at that
      moment, so there is no stale off-policy replay buffer mixing
      incompatible dynamics regimes together (unlike SAC/TD3's replay buffer).
    - Robust default hyperparameters and no target networks / no additional
      moving parts to destabilize training given a single, deterministic
      run per team.
    - Clipped surrogate objective keeps updates conservative, which matters
      here because the reward already mixes several shaped terms (progress,
      proximity, jerk, collisions) — large policy updates on a multi-term
      reward are a common source of degenerate "loophole" policies.

IMPORTANT — no VecNormalize:
    env.py already hand-normalizes every observation component into a small,
    fixed range (see features.py). Wrapping the env in SB3's VecNormalize
    would additionally track running mean/std statistics that are needed at
    inference time too — but agent.py has no access to that running state
    (it only loads model.onnx). Using VecNormalize here would train a policy
    that looks fine in training curves and then silently degrades at
    tournament time, exactly the drift failure mode the subject warns about.
"""

import argparse
import os
import sys

# The repo ships a folder literally named "gym/" at its root (the f110_gym
# engine, added to PYTHONPATH by the Dockerfile). Since the working
# directory is on sys.path, a bare `import gym` resolves to that folder as
# an empty namespace package instead of raising ImportError — which crashes
# stable_baselines3's model.save() (it does `import gym as openai_gym` for
# version reporting and doesn't guard against a "gym" with no __version__).
# Pre-seeding sys.modules with gymnasium under the "gym" name avoids the
# collision; it must happen before stable_baselines3 is imported.
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
    """Track and save the best-performing checkpoint across training.

    PPO fine-tuning over many millions of steps does NOT improve
    monotonically: two long unsupervised runs this project drifted from a
    genuinely good multi-map policy into a much worse one (collisions and
    stalls reappearing on some maps) well before total_timesteps was
    reached, and only manual post-hoc bisection through CheckpointCallback's
    snapshots recovered the good state. This automates that safety net: every
    `eval_freq` steps, run one deterministic rollout per training map (the
    same check used throughout this project's debugging) and keep the
    highest-scoring snapshot as `best_model.zip`, independent of whatever
    state training happens to be in when it eventually stops.
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
        # Finishing a map dominates the score; collisions are a tie-breaking
        # penalty. Weights are deliberately coarse - this only needs to rank
        # snapshots relative to each other, not produce a calibrated metric.
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
    parser.add_argument("--timesteps", type=int, default=3_000_000, help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4, help="Parallel environments (see module docstring: 4 is the observed RAM ceiling)")
    parser.add_argument("--logdir", type=str, default="logs", help="TensorBoard + checkpoint directory")
    parser.add_argument("--out", type=str, default="model.onnx", help="Output ONNX path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None, help="Path to a .zip checkpoint to resume from")
    parser.add_argument(
        "--eval-freq", type=int, default=100_000,
        help="Timesteps between BestModelCallback evaluations (5-map deterministic "
             "rollout). Saves logdir/best_model.zip whenever the score improves, so "
             "a long run's eventual drift/regression never costs the best state "
             "actually reached.",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.999,
        help="Discount factor. At 20Hz decision rate, gamma=0.99 gives an effective "
             "horizon of only 1/(1-gamma)=100 steps (5s) - too short to credit a "
             "~1000+ step lap. 0.999 gives ~1000 steps (50s).",
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.005,
        help="Entropy coefficient. Investigated as a suspect when training "
             "collapsed to a 'reverse away from every corner' local optimum, "
             "but the actual cause was REWARD_DNF being large enough to make "
             "a failed forward attempt cost more than never attempting one "
             "(see env.py's REWARD_DNF comment) - 0.005 trains fine once "
             "that's fixed. Lower this only if std runs away again.",
    )
    return parser.parse_args()


def make_env():
    # NOTE: env_simulation keeps its simulator behind a single MODULE-LEVEL
    # singleton (see env_simulation._sim_instance). Two RacingEnv() instances
    # sharing the same Python process would silently stomp on each other's
    # state. Parallelism therefore MUST go through SubprocVecEnv (one
    # process per env, hence one singleton per env) rather than
    # DummyVecEnv/threads whenever n_envs > 1.
    return Monitor(RacingEnv())


def export_onnx(model: PPO, path: str):
    """Export the trained policy's deterministic action head to ONNX.

    agent.py runs this exact graph through onnxruntime and clips its output
    to [-1, 1] itself (features.action_to_physical) — matching the clipping
    stable_baselines3.BaseAlgorithm.predict() applies outside policy.forward(),
    which is intentionally NOT baked into this graph.
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
        # Force the legacy TorchScript-tracing exporter: the newer
        # torch.export/dynamo-based exporter (default since torch 2.6+)
        # cannot symbolically trace SB3's ActorCriticPolicy.forward(), which
        # constructs a torch.distributions.Normal object internally.
        dynamo=False,
    )
    print(f"Exported ONNX policy to {path}")


def main():
    args = parse_args()

    vec_env_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    env = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed, vec_env_cls=vec_env_cls)

    if args.resume is not None:
        model = PPO.load(args.resume, env=env, device="cpu")
        # PPO.load() restores gamma/ent_coef/gae_lambda from the checkpoint
        # itself, not from this run's CLI args. gamma is baked into the
        # rollout buffer at _setup_model() time rather than re-read from
        # self.gamma each rollout, so both must be patched explicitly for a
        # hyperparameter change to actually take effect on resume.
        model.gamma = args.gamma
        model.rollout_buffer.gamma = args.gamma
        model.ent_coef = args.ent_coef
        # verbose is also restored from the checkpoint (e.g. pretrain_bc.py
        # used to save with verbose=0) - force it back on so training
        # progress is actually visible instead of silently running for
        # however long total_timesteps needs with zero printed output.
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

    # The final-step model is NOT necessarily the best one reached (PPO does
    # not improve monotonically over long runs - see BestModelCallback). Export
    # whichever is best according to the 5-map eval, falling back to the final
    # model only if BestModelCallback never ran (e.g. a very short --timesteps).
    best_model_path = os.path.join(args.logdir, "best_model.zip")
    if os.path.exists(best_model_path):
        print(f"Exporting best_model.zip (score={best_model_callback.best_score:.2f}), "
              f"not the final-step model, as {args.out}", flush=True)
        try:
            export_onnx(PPO.load(best_model_path, device="cpu"), args.out)
        except Exception as e:
            # Loading a second checkpoint in this same process has, once,
            # hit a flaky torch lazy-import error unrelated to this script's
            # own logic (reproduced independent of it, see project notes).
            # Don't let that crash the whole training run at the very last
            # step - fall back to exporting the in-memory final model, and
            # leave best_model.zip on disk to export by hand if it matters:
            #   python -c "from stable_baselines3 import PPO; ..." (no gym
            #   namespace workaround in that one-off process).
            print(f"WARNING: could not load/export best_model.zip ({e}). "
                  f"Exporting the final-step model instead; best_model.zip "
                  f"is still on disk at {best_model_path} for manual export.", flush=True)
            export_onnx(model, args.out)
    else:
        export_onnx(model, args.out)

    env.close()


if __name__ == "__main__":
    main()
