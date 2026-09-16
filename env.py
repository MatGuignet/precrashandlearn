import numpy as np
import gymnasium as gym
from gymnasium import spaces

import env_simulation


class RacingEnv(gym.Env):
    """
    Gymnasium environment for the Crash & Learn racing simulation.

    Part 1:
    - One car only
    - Randomly selects one of the five training maps at reset
    - Observation: 103 values
        * 100 LiDAR values
        * velocity
        * steering
        * progress
    - Normalized observation and action spaces
    """

    metadata = {"render_modes": []}

    MAPS = [
        "Monza",
        "Zandvoort",
        "YasMarina",
        "Spielberg",
        "Shanghai",
    ]

    LIDAR_MAX = 15.0

    MIN_SPEED = -2.0
    MAX_SPEED = 10.0

    MAX_STEERING = 0.4189

    def __init__(self, max_episode_steps=4000):
        super().__init__()

        self.max_episode_steps = max_episode_steps

        self.step_count = 0
        self.previous_steering = 0.0
        self.current_map = None

        # ============================================================
        # ACTION SPACE
        # ============================================================
        #
        # action[0] = normalized target speed
        # action[1] = normalized steering
        #
        # Both values are in [-1, 1].
        #
        # Speed:
        #   -1 -> -2 m/s
        #    0 ->  4 m/s
        #   +1 -> 10 m/s
        #
        # Steering:
        #   -1 -> -0.4189 rad
        #    0 ->  0 rad
        #   +1 -> +0.4189 rad
        #
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        # ============================================================
        # OBSERVATION SPACE
        # ============================================================
        #
        # 100 LiDAR values:
        #   [0, 1]
        #
        # velocity:
        #   [-0.2, 1]
        #
        # steering:
        #   [-1, 1]
        #
        # progress:
        #   [0, 1]
        #
        low = np.concatenate(
            [
                np.zeros(100, dtype=np.float32),
                np.array(
                    [-0.2, -1.0, 0.0],
                    dtype=np.float32,
                ),
            ]
        )

        high = np.concatenate(
            [
                np.ones(100, dtype=np.float32),
                np.array(
                    [1.0, 1.0, 1.0],
                    dtype=np.float32,
                ),
            ]
        )

        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

    # ==================================================================
    # MAP MANAGEMENT
    # ==================================================================

    def _select_map(self):
        """
        Randomly select a training map.
        """
        return self.np_random.choice(self.MAPS)

    # ==================================================================
    # OBSERVATION PREPROCESSING
    # ==================================================================

    def _get_observation(self, obs):
        """
        Convert the raw simulation observation into the 103-dimensional
        normalized observation used by the RL agent.
        """

        # --------------------------------------------------------------
        # LiDAR
        # --------------------------------------------------------------
        #
        # Raw range: [0.1, 15]
        # Normalized range: [0, 1]
        #
        lidar = np.asarray(
            obs["lidar"],
            dtype=np.float32,
        )

        lidar = np.clip(
            lidar / self.LIDAR_MAX,
            0.0,
            1.0,
        )

        # --------------------------------------------------------------
        # Velocity
        # --------------------------------------------------------------
        #
        # Physical range:
        #   [-2, 10] m/s
        #
        # We normalize by MAX_SPEED = 10.
        #
        velocity = np.float32(
            np.clip(
                obs["velocity"] / self.MAX_SPEED,
                -0.2,
                1.0,
            )
        )

        # --------------------------------------------------------------
        # Steering
        # --------------------------------------------------------------

        steering = np.float32(
            np.clip(
                obs["steering"] / self.MAX_STEERING,
                -1.0,
                1.0,
            )
        )

        # --------------------------------------------------------------
        # Progress
        # --------------------------------------------------------------

        progress = np.float32(
            np.clip(
                obs["progress"],
                0.0,
                1.0,
            )
        )

        # --------------------------------------------------------------
        # Final observation
        # --------------------------------------------------------------

        observation = np.concatenate(
            [
                lidar,
                np.array(
                    [
                        velocity,
                        steering,
                        progress,
                    ],
                    dtype=np.float32,
                ),
            ]
        )

        return observation.astype(np.float32)

    # ==================================================================
    # ACTION PREPROCESSING
    # ==================================================================

    def _action_to_physical(self, action):
        """
        Convert the normalized RL action [-1, 1] into the physical
        simulation action.

        Input:
            action[0] = normalized target speed
            action[1] = normalized steering

        Output:
            target_speed in [-2, 10] m/s
            steering in [-0.4189, 0.4189] rad
        """

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        normalized_speed = action[0]
        normalized_steering = action[1]

        # Map [-1, 1] -> [-2, 10]
        target_speed = (
            self.MIN_SPEED
            + (
                (normalized_speed + 1.0)
                * (self.MAX_SPEED - self.MIN_SPEED)
                / 2.0
            )
        )

        # Map [-1, 1] -> [-0.4189, 0.4189]
        steering = (
            normalized_steering
            * self.MAX_STEERING
        )

        return (
            float(target_speed),
            float(steering),
        )

    # ==================================================================
    # REWARD
    # ==================================================================

    def _compute_reward(self, step_info, current_steering):
        """
        Compute the reward for one simulation decision step.

        Current reward:

            +100 * progress_delta
            -0.05 time penalty
            -2 wall collision
            -2 vehicle collision
            -0.05 steering variation
            +20 completed lap
            +50 finish
            -50 DNF
        """

        agent_id = 0

        # --------------------------------------------------------------
        # Backend information
        # --------------------------------------------------------------

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
        dnf = status == 0

        # --------------------------------------------------------------
        # Steering smoothness
        # --------------------------------------------------------------

        steering_change = abs(
            current_steering
            - self.previous_steering
        )

        # --------------------------------------------------------------
        # Reward
        # --------------------------------------------------------------

        reward = 0.0

        # Main objective: make progress
        reward += 100.0 * progress_delta

        # Small time penalty
        reward -= 0.05

        # Avoid walls
        if wall_collision:
            reward -= 2.0

        # Avoid other vehicles
        if vehicle_collision:
            reward -= 2.0

        # Encourage smooth steering
        reward -= 0.05 * steering_change

        # Lap completed
        if lap_complete:
            reward += 20.0

        # Race finished
        if finished:
            reward += 50.0

        # DNF / stagnation
        if dnf:
            reward -= 50.0

        # Store steering for the next step
        self.previous_steering = current_steering

        return float(reward)

    # ==================================================================
    # RESET
    # ==================================================================

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment.

        Normal behaviour:
            env.reset()
            -> randomly selects one of the five maps.

        Forced map:
            env.reset(options={"map_name": "Monza"})
            -> forces Monza.

        This is useful for deterministic map-by-map validation while
        keeping random map selection during training.
        """

        super().reset(seed=seed)

        self.step_count = 0
        self.previous_steering = 0.0

        # --------------------------------------------------------------
        # Select map
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Reset backend simulation
        # --------------------------------------------------------------

        env_simulation.set_map(
            selected_map,
            num_cars=1,
        )

        env_simulation.reset(
            num_cars=1,
        )

        # --------------------------------------------------------------
        # Initial observation
        # --------------------------------------------------------------

        raw_obs = env_simulation.get_obs(0)

        step_info = env_simulation.get_step_info()

        observation = self._get_observation(
            raw_obs
        )

        info = {
            "map": self.current_map,
            "step_count": self.step_count,
            "simulation": step_info,
        }

        return observation, info

    # ==================================================================
    # STEP
    # ==================================================================

    def step(self, action):
        """
        Execute one RL decision step.

        One call to simulation_step() corresponds to:
            5 physics steps
            50 ms
            20 Hz decision frequency
        """

        # --------------------------------------------------------------
        # Convert normalized action to physical action
        # --------------------------------------------------------------

        target_speed, steering = self._action_to_physical(
            action
        )

        # --------------------------------------------------------------
        # Send action to backend
        # --------------------------------------------------------------

        env_simulation.apply_action(
            0,
            target_speed,
            steering,
        )

        # --------------------------------------------------------------
        # Advance simulation
        # --------------------------------------------------------------

        env_simulation.simulation_step()

        self.step_count += 1

        # --------------------------------------------------------------
        # Retrieve backend information
        # --------------------------------------------------------------

        step_info = env_simulation.get_step_info()

        # --------------------------------------------------------------
        # Reward
        # --------------------------------------------------------------

        reward = self._compute_reward(
            step_info,
            steering,
        )

        # --------------------------------------------------------------
        # Observation
        # --------------------------------------------------------------

        raw_obs = env_simulation.get_obs(0)

        observation = self._get_observation(
            raw_obs
        )

        # --------------------------------------------------------------
        # Termination
        # --------------------------------------------------------------

        status = int(
            step_info["agent_status"][0]
        )

        # Backend:
        #   0 = DNF
        #   1 = active
        #   2 = finished
        terminated = status in (0, 2)

        # Gymnasium truncation is used for our own timeout.
        truncated = (
            self.step_count >= self.max_episode_steps
            and not terminated
        )

        # --------------------------------------------------------------
        # Info
        # --------------------------------------------------------------

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

    # ==================================================================
    # CLOSE
    # ==================================================================

    def close(self):
        """
        Close the backend simulation.
        """
        env_simulation.close()


# ======================================================================
# MANUAL VALIDATION
# ======================================================================

if __name__ == "__main__":

    env = RacingEnv(
        max_episode_steps=4000
    )

    print("Testing all training maps...\n")

    for map_name in env.MAPS:

        print("=" * 60)
        print(f"MAP: {map_name}")
        print("=" * 60)

        # --------------------------------------------------------------
        # Force the map through Gymnasium reset options
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Zero action
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Forward action
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

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