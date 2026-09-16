"""demo.py — Simulation runner and smoke test.

Executes simulation steps using submission/agent.py (if present) or a fallback demo policy.
Loops until lap completion, stagnation / DNF, or maximum steps limit.
Validates the env_simulation API contract.

Usage:
    python demo.py [--steps 1000] [--num-cars 1] [--all-laps]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from env_simulation import (
    apply_action,
    close,
    get_obs,
    get_space_info,
    get_step_info,
    reset,
    simulation_step,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Crash&Learn Grand Prix — Simulation Demo")
    parser.add_argument("--steps", type=int, default=1000, help="Maximum decision steps (default: 1000)")
    parser.add_argument("--num-cars", type=int, default=1, help="Number of cars in simulation (1..4, default: 1)")
    parser.add_argument("--all-laps", action="store_true", help="Continue running across all laps instead of stopping on first lap complete")
    parser.add_argument("--log-freq", type=int, default=40, help="Logging frequency in decision steps (default: 40)")
    return parser.parse_args()


def load_submission_agent():
    """Attempt to dynamically load submission.agent.Agent."""
    try:
        current_dir = Path(__file__).resolve().parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        
        from submission.agent import Agent
        agent = Agent()
        print("  [Controller] Successfully loaded Agent from submission/agent.py")
        return agent
    except Exception as e:
        print(f"  [Controller] No submission/agent.py active ({e}) — using open-loop sinusoidal controller")
        return None


def main():
    args = parse_args()
    print("=== env_simulation demo ===\n")
    print("Space info:")
    space = get_space_info()
    freq = space["decision_freq_hz"]
    print(f"  decision_freq_hz : {freq}")
    print(f"  lidar shape      : {space['observations']['lidar']['shape']}")
    print(f"  action bounds    : speed {space['actions']['target_speed']['bounds']}, "
          f"steer {space['actions']['steering']['bounds']}")
    print()

    agent = load_submission_agent()

    reset(num_cars=args.num_cars)
    print(f"Reset with {args.num_cars} car(s). Running up to {args.steps} steps...\n")

    obs = get_obs(0)
    info = get_step_info()

    for step in range(args.steps):
        if agent is not None:
            try:
                speed, steering = agent.predict(obs, info)
            except Exception as e:
                print(f"[ERROR] agent.predict failed at step {step}: {e}")
                break
        else:
            speed = 3.0
            steering = 0.05 * np.sin(step * 0.1)

        apply_action(0, float(speed), float(steering))
        simulation_step()
        info = get_step_info()
        obs = get_obs(0)

        # Periodic telemetry log
        if step % args.log_freq == 0:
            print(
                f"  step={info['step_count']:4d} | "
                f"vel={obs['velocity']:+.2f} m/s | "
                f"steer={obs['steering']:+.3f} rad | "
                f"progress={obs['progress']:.4f} | "
                f"lap={obs['lap_count']} | "
                f"wall={info['collisions']['wall'][0]} | "
                f"veh={info['collisions']['vehicle'][0]} | "
                f"friction={info['friction_current']:.3f}"
            )

        # Lap completion detection
        if info["lap_complete"][0]:
            lap_times = info.get("lap_times", {}).get(0, [])
            lap_t = lap_times[-1] if len(lap_times) > 0 else (step + 1) / freq
            print(
                f"\n🏁 [LAP COMPLETE] Lap {obs['lap_count']} finished in {lap_t:.2f}s "
                f"at step {info['step_count']} (avg speed: {obs['velocity']:.2f} m/s)!"
            )
            if not args.all_laps:
                break

        # Stagnation / crash detection
        if info["stagnation"][0]:
            print(
                f"\n💥 [DNF / STAGNATION] Vehicle stopped/stuck at step {info['step_count']} "
                f"(progress reached: {obs['progress']:.4f})"
            )
            break

    print("\nFinal obs keys  :", sorted(obs.keys()))
    print("lidar shape     :", obs["lidar"].shape)
    print("lidar dtype     :", obs["lidar"].dtype)
    print("velocity (scalar):", type(obs["velocity"]).__name__)
    print("steering (scalar):", type(obs["steering"]).__name__)
    print("progress (scalar):", type(obs["progress"]).__name__)
    print("lap_count (int)  :", type(obs["lap_count"]).__name__)
    print("rank (int)       :", type(obs["rank"]).__name__)
    print("opponents keys  :", sorted(obs["opponents"].keys()))

    final_info = get_step_info()
    print("\nFinal info keys :", sorted(final_info.keys()))
    print("opponents_mask  :", final_info["opponents_mask"])
    print("agent_status    :", final_info["agent_status"])
    print("step_count      :", final_info["step_count"])

    assert obs["lidar"].shape == (100,), "LiDAR shape contract violated"
    assert isinstance(obs["velocity"], float), "velocity must be float scalar"
    assert isinstance(obs["steering"], float), "steering must be float scalar"
    assert isinstance(obs["progress"], float), "progress must be float scalar"
    assert isinstance(obs["lap_count"], int), "lap_count must be int scalar"
    assert set(obs["opponents"].keys()) == {0, 1, 2, 3}, "opponents must have keys 0..3"
    assert "done" not in final_info, "info must NOT contain 'done'"
    assert "wall" in final_info["collisions"] and "vehicle" in final_info["collisions"]

    close()
    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
