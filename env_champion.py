"""env_champion.py — Part 3 multi-agent training environment.

Part 2's diagnosis established why a Part 1 solo policy can't hold a shared
circuit: it never observed another car as anything but a wall, and
REWARD_VEHICLE_COLLISION was structurally inert under num_cars=1. This
environment fixes both: the ego car (slot 0) is driven by the policy being
trained, slots 1..(num_cars-1) are driven internally by a frozen reference
ONNX policy (the Part 1 champion), and the observation now includes the
opponents field via features.build_observation_multi.

Design choice — frozen opponents, not simultaneous self-play: fully
simultaneous multi-agent PPO (every car's policy updated together every
step) is a substantially harder, more unstable learning problem (the
environment's dynamics keep shifting under every agent as every other agent
also changes). Treating opponents as environment dynamics driven by a fixed,
already-competent policy keeps this a standard single-agent RL problem from
the ego's point of view, which is what the rest of this project's PPO/BC
infrastructure is built for. The frozen policy is a reasonable stand-in for
"some other skilled solo driver" - not identical to what other teams will
field at tournament time, but not required to be: the ego only needs to
learn "detect nearby cars, avoid/overtake them", which generalizes across
whatever the opponent actually is.

num_cars is randomized per episode (2..4) - domain randomization over
*opponent count*, the same principle as Part 1's map/weather randomization,
so the ego doesn't overfit to always racing against exactly 3 others.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import onnxruntime

import env_simulation
import features
from env import RacingEnv


class ChampionRacingEnv(RacingEnv):

    MIN_CARS = 2
    MAX_CARS = 4

    def __init__(self, opponent_model_path, max_episode_steps=6000):
        self._opponent_session = onnxruntime.InferenceSession(opponent_model_path)
        self._opponent_input_name = self._opponent_session.get_inputs()[0].name
        self.num_cars = self.MAX_CARS

        super().__init__(max_episode_steps=max_episode_steps)

        # Overwrite the Part 1 (103-dim) observation space with the
        # opponent-augmented one - everything else about RacingEnv.__init__
        # (action space, reward constants) is inherited unchanged.
        self.observation_space = spaces.Box(
            low=np.concatenate([features.OBS_LOW, np.full(20, -1.0, dtype=np.float32)]),
            high=np.concatenate([features.OBS_HIGH, np.full(20, 1.0, dtype=np.float32)]),
            dtype=np.float32,
        )

    def _opponent_action(self, car_id: int) -> tuple[float, float]:
        raw_obs = env_simulation.get_obs(car_id)
        # Opponents see the plain solo observation (they ARE solo policies) -
        # only the ego needs the opponent-augmented one.
        obs = features.build_observation(raw_obs)
        raw_action = self._opponent_session.run(
            None, {self._opponent_input_name: obs[None, :]}
        )[0][0]
        return features.action_to_physical(raw_action)

    def reset(self, *, seed=None, options=None):
        # RacingEnv.reset() calls env_simulation.set_map(..., num_cars=1) and
        # env_simulation.reset(num_cars=1) - both hardcode 1. Re-run the
        # equivalent logic here with self.num_cars instead of duplicating
        # RacingEnv.reset()'s full body.
        gym.Env.reset(self, seed=seed)

        self.num_cars = int(self.np_random.integers(self.MIN_CARS, self.MAX_CARS + 1))

        self.step_count = 0
        self.previous_steering = 0.0
        self.wall_contact_streak = 0

        if options is not None and "map_name" in options:
            selected_map = options["map_name"]
            if selected_map not in self.MAPS:
                raise ValueError(f"Unknown map '{selected_map}'. Available maps: {self.MAPS}")
        else:
            selected_map = self._select_map()
        self.current_map = selected_map

        env_simulation.set_map(selected_map, num_cars=self.num_cars)
        env_simulation.reset(num_cars=self.num_cars)

        raw_obs = env_simulation.get_obs(0)
        step_info = env_simulation.get_step_info()
        observation = features.build_observation_multi(raw_obs)
        info = {
            "map": self.current_map,
            "step_count": self.step_count,
            "num_cars": self.num_cars,
            "simulation": step_info,
        }
        return observation, info

    def step(self, action):
        normalized_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        target_speed, steering = features.action_to_physical(normalized_action)
        env_simulation.apply_action(0, target_speed, steering)

        for car_id in range(1, self.num_cars):
            opp_speed, opp_steering = self._opponent_action(car_id)
            env_simulation.apply_action(car_id, opp_speed, opp_steering)

        env_simulation.simulation_step()
        self.step_count += 1

        step_info = env_simulation.get_step_info()
        raw_obs = env_simulation.get_obs(0)

        wall_collision_now = bool(step_info["collisions"]["wall"][0])
        self.wall_contact_streak = self.wall_contact_streak + 1 if wall_collision_now else 0
        forced_dnf = self.wall_contact_streak >= self.WALL_CONTACT_DNF_STEPS

        # Reused verbatim from RacingEnv (Part 1): progress/time/proximity/
        # collision/jerk/lap/finish/DNF terms. REWARD_VEHICLE_COLLISION is
        # now actually reachable, since real opponents exist.
        reward = self._compute_reward(
            step_info, raw_obs["lidar"], float(normalized_action[1]), forced_dnf,
        )

        observation = features.build_observation_multi(raw_obs)

        status = int(step_info["agent_status"][0])
        terminated = status in (0, 2) or forced_dnf
        truncated = self.step_count >= self.max_episode_steps and not terminated

        info = {
            "map": self.current_map,
            "step_count": self.step_count,
            "num_cars": self.num_cars,
            "status": status,
            "simulation": step_info,
        }
        return observation, reward, terminated, truncated, info
