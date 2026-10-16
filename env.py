import numpy as np
import gymnasium as gym
from gymnasium import spaces

import env_simulation
import features


class RacingEnv(gym.Env):
    """Env gymnasium pour la partie 1 : une seule voiture, map tirée au hasard
    parmi les 5 à chaque reset. Obs = 100 lidar + vitesse + steering + progress.
    """

    metadata = {"render_modes": []}

    MAPS = [
        "Monza",
        "Zandvoort",
        "YasMarina",
        "Spielberg",
        "Shanghai",
    ]

    # définies dans features.py, gardées ici pour le code qui fait env.LIDAR_MAX
    LIDAR_MAX = features.LIDAR_MAX
    MIN_SPEED = features.MIN_SPEED
    MAX_SPEED = features.MAX_SPEED
    MAX_STEERING = features.MAX_STEERING

    # Reward. En attributs de classe pour pouvoir les changer depuis
    # train.py (--reward-*) sans toucher au code.
    REWARD_PROGRESS_SCALE = 100.0
    REWARD_TIME_PENALTY = 0.05          # par step
    REWARD_WALL_COLLISION = 0.5         # par step contre un mur
    REWARD_VEHICLE_COLLISION = 0.5
    # était à 0.1 : la voiture apprenait à bloquer le volant (pénalité nulle
    # si on ne tourne plus). 0.02 suffit pour éviter le zigzag.
    REWARD_STEERING_JERK_SCALE = 0.02
    REWARD_LAP_COMPLETE = 20.0
    REWARD_FINISH = 50.0
    # était à 50 : essayer d'avancer et se planter coûtait plus cher que rester
    # sur place, du coup la voiture reculait / ne bougeait plus.
    REWARD_DNF = 12.0

    # pénalité qui monte quand on passe sous WALL_PROXIMITY_MARGIN d'un mur,
    # pour apprendre à les éviter avant de les toucher
    WALL_PROXIMITY_MARGIN = 0.2         # en m
    REWARD_WALL_PROXIMITY_SCALE = 0.5

    # toucher un mur ne termine pas l'épisode dans env_simulation, la voiture
    # peut rester collée jusqu'à 4s et prendre la pénalité à chaque step.
    # On coupe nous-mêmes après 1s de contact.
    WALL_CONTACT_DNF_STEPS = 20         # 1s à 20Hz

    def __init__(self, max_episode_steps=6000):
        super().__init__()

        self.max_episode_steps = max_episode_steps

        self.step_count = 0
        self.previous_steering = 0.0
        self.current_map = None
        self.wall_contact_streak = 0

        # action = [vitesse, steering] dans [-1, 1]
        # vitesse : -1 -> -2 m/s, 0 -> 4 m/s, 1 -> 10 m/s
        # steering : -1 -> -0.4189 rad, 1 -> 0.4189 rad
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        # bornes dans features.py
        self.observation_space = spaces.Box(
            low=features.OBS_LOW,
            high=features.OBS_HIGH,
            dtype=np.float32,
        )

    def _select_map(self):
        return self.np_random.choice(self.MAPS)

    def _compute_reward(self, step_info, raw_lidar, normalized_steering, forced_dnf=False):
        """Reward d'un step. + progress, tour, arrivée / - temps, murs,
        collisions, à-coups de volant, DNF (forced_dnf = coupé après contact mur).
        """

        agent_id = 0

        progress_delta = float(
            step_info["progress_delta"][agent_id]
        )

        wall_collision = bool(
            step_info["collisions"]["wall"][agent_id]
        )

        vehicle_collision = bool(
            step_info["collisions"]["vehicle"][agent_id]
        )

        lap_complete = bool(
            step_info["lap_complete"][agent_id]
        )

        status = int(
            step_info["agent_status"][agent_id]
        )

        finished = status == 2
        dnf = status == 0 or forced_dnf

        # en unités normalisées
        steering_jerk = abs(
            normalized_steering
            - self.previous_steering
        )

        # 0 au-delà de la marge, puis linéaire jusqu'à REWARD_WALL_PROXIMITY_SCALE au contact
        min_wall_dist = float(np.min(raw_lidar))

        proximity_penalty = 0.0
        if min_wall_dist < self.WALL_PROXIMITY_MARGIN:
            proximity_penalty = self.REWARD_WALL_PROXIMITY_SCALE * (
                (self.WALL_PROXIMITY_MARGIN - min_wall_dist)
                / self.WALL_PROXIMITY_MARGIN
            )

        reward = 0.0

        reward += self.REWARD_PROGRESS_SCALE * progress_delta

        reward -= self.REWARD_TIME_PENALTY

        reward -= proximity_penalty

        if wall_collision:
            reward -= self.REWARD_WALL_COLLISION

        if vehicle_collision:
            reward -= self.REWARD_VEHICLE_COLLISION

        reward -= self.REWARD_STEERING_JERK_SCALE * steering_jerk

        if lap_complete:
            reward += self.REWARD_LAP_COMPLETE

        if finished:
            reward += self.REWARD_FINISH

        if dnf:
            reward -= self.REWARD_DNF

        self.previous_steering = normalized_steering

        return float(reward)

    def reset(self, *, seed=None, options=None):
        """Map au hasard, ou forcée avec options={"map_name": "Monza"}
        (pratique pour tester map par map).
        """

        super().reset(seed=seed)

        self.step_count = 0
        self.previous_steering = 0.0
        self.wall_contact_streak = 0

        if options is not None and "map_name" in options:
            selected_map = options["map_name"]

            if selected_map not in self.MAPS:
                raise ValueError(
                    f"Unknown map '{selected_map}'. "
                    f"Available maps: {self.MAPS}"
                )
        else:
            selected_map = self._select_map()

        self.current_map = selected_map

        env_simulation.set_map(
            selected_map,
            num_cars=1,
        )

        env_simulation.reset(
            num_cars=1,
        )

        raw_obs = env_simulation.get_obs(0)

        step_info = env_simulation.get_step_info()

        observation = features.build_observation(
            raw_obs
        )

        info = {
            "map": self.current_map,
            "step_count": self.step_count,
            "simulation": step_info,
        }

        return observation, info

    def step(self, action):
        # 1 step = 1 simulation_step() = 5 steps physiques = 50 ms (20 Hz)

        normalized_action = np.clip(
            np.asarray(action, dtype=np.float32),
            -1.0,
            1.0,
        )

        target_speed, steering = features.action_to_physical(
            normalized_action
        )

        env_simulation.apply_action(
            0,
            target_speed,
            steering,
        )

        env_simulation.simulation_step()

        self.step_count += 1

        step_info = env_simulation.get_step_info()

        raw_obs = env_simulation.get_obs(0)

        # contact mur prolongé -> DNF (cf. WALL_CONTACT_DNF_STEPS)
        wall_collision_now = bool(
            step_info["collisions"]["wall"][0]
        )

        if wall_collision_now:
            self.wall_contact_streak += 1
        else:
            self.wall_contact_streak = 0

        forced_dnf = self.wall_contact_streak >= self.WALL_CONTACT_DNF_STEPS

        reward = self._compute_reward(
            step_info,
            raw_obs["lidar"],
            float(normalized_action[1]),
            forced_dnf,
        )

        observation = features.build_observation(
            raw_obs
        )

        status = int(
            step_info["agent_status"][0]
        )

        # status : 0 = DNF, 1 = en course, 2 = fini
        terminated = status in (0, 2) or forced_dnf

        # timeout à nous -> truncated
        truncated = (
            self.step_count >= self.max_episode_steps
            and not terminated
        )

        info = {
            "map": self.current_map,
            "step_count": self.step_count,
            "status": status,
            "simulation": step_info,
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def close(self):
        env_simulation.close()


# test rapide : python env.py

if __name__ == "__main__":

    env = RacingEnv(
        max_episode_steps=6000
    )

    print("Testing all training maps...\n")

    for map_name in env.MAPS:

        print("=" * 60)
        print(f"MAP: {map_name}")
        print("=" * 60)

        observation, info = env.reset(
            options={
                "map_name": map_name
            }
        )

        print(
            "Selected map:",
            info["map"]
        )

        print(
            "Observation shape:",
            observation.shape
        )

        print(
            "Observation dtype:",
            observation.dtype
        )

        print(
            "Observation bounds:",
            observation.min(),
            observation.max()
        )

        # action nulle
        zero_action = np.array(
            [0.0, 0.0],
            dtype=np.float32,
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(zero_action)

        print(
            "Zero action:",
            f"reward={reward:.4f}",
            f"status={info['status']}"
        )

        # plein gaz tout droit, la vitesse doit monter
        forward_action = np.array(
            [1.0, 0.0],
            dtype=np.float32,
        )

        velocities = []

        for _ in range(20):

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(forward_action)

            simulation_obs = env_simulation.get_obs(0)

            velocities.append(
                float(
                    simulation_obs["velocity"]
                )
            )

            if terminated or truncated:
                break

        print(
            "Forward test:",
            f"velocity_start={velocities[0]:.3f}",
            f"velocity_end={velocities[-1]:.3f}"
        )

        assert observation.shape == (103,)

        assert observation.dtype == np.float32

        assert np.all(
            observation >= env.observation_space.low
        )

        assert np.all(
            observation <= env.observation_space.high
        )

        print(
            "Observation bounds: OK"
        )

    env.close()

    print(
        "\nAll map tests completed successfully."
    )