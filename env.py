import numpy as np
import gymnasium as gym
from gymnasium import spaces

import env_simulation
import features


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

    # Physical constants, single source of truth in features.py (shared with
    # agent.py at inference time). Re-exposed here as class attributes so
    # existing code/tests referencing env.LIDAR_MAX etc. keep working.
    LIDAR_MAX = features.LIDAR_MAX
    MIN_SPEED = features.MIN_SPEED
    MAX_SPEED = features.MAX_SPEED
    MAX_STEERING = features.MAX_STEERING

    # ============================================================
    # REWARD SHAPING CONSTANTS
    # ============================================================
    #
    # Kept as named class attributes (rather than inline magic numbers)
    # so they're easy to retune independently without touching the reward
    # logic itself.
    #
    REWARD_PROGRESS_SCALE = 100.0       # per unit of normalized lap progress
    REWARD_TIME_PENALTY = 0.05          # flat cost per decision step
    REWARD_WALL_COLLISION = 0.5         # per step in contact with a wall
    REWARD_VEHICLE_COLLISION = 0.5      # per step in contact with another car
    # Kept small deliberately: this penalty only fires on CHANGE in steering,
    # so once steering settles on any constant value (including a hard
    # lock), the penalty drops to exactly 0 forever after — rewarding "never
    # turn the wheel" regardless of which value it got stuck on. A previous
    # value of 0.1 was likely strong enough (relative to a still-weak, noisy
    # progress signal early in training) to make that a genuine local
    # optimum. Kept nonzero only to still discourage genuine violent zigzag.
    REWARD_STEERING_JERK_SCALE = 0.02   # per unit of |normalized steering delta|
    REWARD_LAP_COMPLETE = 20.0          # bonus on completing a lap
    REWARD_FINISH = 50.0                # bonus on finishing the race
    # Kept well below REWARD_LAP_COMPLETE/REWARD_FINISH so finishing still
    # dominates, but deliberately NOT so large that it swamps the modest
    # progress reward a still-learning policy can bank before an early
    # failure. An earlier value of 50 here made "attempt and partially
    # progress, then fail" cost more in expectation than "never attempt to
    # move" — a strong incentive to freeze/reverse rather than explore.
    REWARD_DNF = 12.0                   # penalty on stagnation-based DNF / early wall-contact bail-out

    # Dense wall-avoidance shaping: below this distance (meters) from the
    # nearest LiDAR hit, incur a penalty growing linearly to
    # REWARD_WALL_PROXIMITY_SCALE as the car gets closer to the wall. This
    # gives a gradient *before* an actual collision happens, instead of
    # relying only on the sparse REWARD_WALL_COLLISION signal.
    WALL_PROXIMITY_MARGIN = 0.2         # meters
    REWARD_WALL_PROXIMITY_SCALE = 0.5

    # A wall collision does NOT by itself end the episode in env_simulation:
    # a car can stay wedged against a wall, accruing REWARD_WALL_COLLISION
    # every step, until the stagnation detector eventually fires (up to 80
    # steps / 4s later) and also applies REWARD_DNF on top. That combination
    # can make a *failed* attempt to drive cost several times more than
    # never attempting to move in the first place (bail out fast, pay
    # REWARD_DNF once) — a strong incentive to freeze rather than explore.
    # Bailing out ourselves after a short streak of continuous wall contact
    # bounds that downside to roughly the same order as standing still.
    WALL_CONTACT_DNF_STEPS = 20         # 1s of continuous wall contact @ 20Hz

    def __init__(self, max_episode_steps=6000):
        super().__init__()

        self.max_episode_steps = max_episode_steps

        self.step_count = 0
        self.previous_steering = 0.0
        self.current_map = None
        self.wall_contact_streak = 0

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
        # Bounds live in features.py (single source of truth, shared with
        # agent.py at inference time).
        #
        self.observation_space = spaces.Box(
            low=features.OBS_LOW,
            high=features.OBS_HIGH,
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
    # OBSERVATION / ACTION PREPROCESSING
    # ==================================================================
    #
    # Delegated to features.py, the pure-numpy module shared with agent.py.
    # See features.build_observation() and features.action_to_physical().

    # ==================================================================
    # REWARD
    # ==================================================================

    def _compute_reward(self, step_info, raw_lidar, normalized_steering, forced_dnf=False):
        """
        Compute the reward for one simulation decision step.

        Components (see REWARD_* class constants for the exact weights):

            + progress reward               (main objective)
            - flat time penalty             (discourage crawling)
            - dense wall-proximity penalty  (anticipate collisions)
            - wall collision penalty
            - vehicle collision penalty
            - steering-jerk penalty         (smooth driving)
            + lap-complete bonus
            + finish bonus
            - DNF penalty (stagnation, or forced_dnf: a wall-contact bail-out)
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
        dnf = status == 0 or forced_dnf

        # --------------------------------------------------------------
        # Steering smoothness (normalized action units, scale-independent
        # of MAX_STEERING)
        # --------------------------------------------------------------

        steering_jerk = abs(
            normalized_steering
            - self.previous_steering
        )

        # --------------------------------------------------------------
        # Dense wall-proximity shaping
        # --------------------------------------------------------------
        #
        # A sparse collision penalty only fires once contact already
        # happened. Adding a continuous penalty as the car gets close to
        # a wall gives the policy an earlier gradient to steer away,
        # instead of learning to avoid walls purely from crash events.
        #
        min_wall_dist = float(np.min(raw_lidar))

        proximity_penalty = 0.0
        if min_wall_dist < self.WALL_PROXIMITY_MARGIN:
            proximity_penalty = self.REWARD_WALL_PROXIMITY_SCALE * (
                (self.WALL_PROXIMITY_MARGIN - min_wall_dist)
                / self.WALL_PROXIMITY_MARGIN
            )

        # --------------------------------------------------------------
        # Reward
        # --------------------------------------------------------------

        reward = 0.0

        # Main objective: make progress
        reward += self.REWARD_PROGRESS_SCALE * progress_delta

        # Small time penalty
        reward -= self.REWARD_TIME_PENALTY

        # Anticipate walls before actually hitting them
        reward -= proximity_penalty

        # Avoid walls
        if wall_collision:
            reward -= self.REWARD_WALL_COLLISION

        # Avoid other vehicles
        if vehicle_collision:
            reward -= self.REWARD_VEHICLE_COLLISION

        # Encourage smooth steering
        reward -= self.REWARD_STEERING_JERK_SCALE * steering_jerk

        # Lap completed
        if lap_complete:
            reward += self.REWARD_LAP_COMPLETE

        # Race finished
        if finished:
            reward += self.REWARD_FINISH

        # DNF / stagnation
        if dnf:
            reward -= self.REWARD_DNF

        # Store steering for the next step
        self.previous_steering = normalized_steering

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
        self.wall_contact_streak = 0

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

        observation = features.build_observation(
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

        normalized_action = np.clip(
            np.asarray(action, dtype=np.float32),
            -1.0,
            1.0,
        )

        target_speed, steering = features.action_to_physical(
            normalized_action
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

        raw_obs = env_simulation.get_obs(0)

        # --------------------------------------------------------------
        # Wall-contact streak (see WALL_CONTACT_DNF_STEPS docstring)
        # --------------------------------------------------------------

        wall_collision_now = bool(
            step_info["collisions"]["wall"][0]
        )

        if wall_collision_now:
            self.wall_contact_streak += 1
        else:
            self.wall_contact_streak = 0

        forced_dnf = self.wall_contact_streak >= self.WALL_CONTACT_DNF_STEPS

        # --------------------------------------------------------------
        # Reward
        # --------------------------------------------------------------

        reward = self._compute_reward(
            step_info,
            raw_obs["lidar"],
            float(normalized_action[1]),
            forced_dnf,
        )

        # --------------------------------------------------------------
        # Observation
        # --------------------------------------------------------------

        observation = features.build_observation(
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
        terminated = status in (0, 2) or forced_dnf

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
        max_episode_steps=6000
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