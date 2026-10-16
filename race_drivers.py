"""race_drivers.py — Race any combination of submission/ folders together on the
5 training circuits (real multi-car races, not a single dummy predict() call).

By default races the 4 solo drivers (Part 2's "Grand Crash Auto" scenario: every
team member's solo model, simultaneously, no champion). Pass --drivers to race
any other combination instead, e.g. the champion against some/all drivers:
    python race_drivers.py --drivers champion,driver1,driver2,driver3
(the engine caps at 4 cars total - see env_champion.py's MAX_CARS - so at most
4 names are used regardless of how many are passed).

Runs back-to-back episodes on each of the 5 maps, cycling until a global
wall-clock time budget is spent, and reports per-driver / per-map finish
rate, DNF rate, average progress and collision counts.
"""

import argparse
import sys
import time

import numpy as np

from agent_loader import validate_and_load_agent
from env_simulation import (
    set_map, reset, get_obs, apply_action, simulation_step, get_step_info, close,
)

MAPS = ["Monza", "Zandvoort", "YasMarina", "Spielberg", "Shanghai"]
DEFAULT_DRIVERS = ["driver1", "driver2", "driver3", "driver4"]
MAX_CARS = 4  # engine limit, matches env_champion.py's MAX_CARS
MAX_STEPS_PER_EPISODE = 6000  # matches env.py's max_episode_steps


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=30.0)
    parser.add_argument(
        "--drivers", type=str, default=",".join(DEFAULT_DRIVERS),
        help="Comma-separated submission/ subfolder names to race together, e.g. "
             "'champion,driver1,driver2,driver3'. Capped at 4 (engine limit).",
    )
    args = parser.parse_args()

    names_requested = [d.strip() for d in args.drivers.split(",") if d.strip()]
    if len(names_requested) > MAX_CARS:
        print(f"[WARNING] {len(names_requested)} names given, engine caps at {MAX_CARS} - "
              f"using the first {MAX_CARS}: {names_requested[:MAX_CARS]}", flush=True)
        names_requested = names_requested[:MAX_CARS]

    budget_s = args.minutes * 60.0
    per_map_budget = budget_s / len(MAPS)

    agents = []
    for i, dname in enumerate(names_requested):
        d = f"submission/{dname}"
        name, agent = validate_and_load_agent(d, team_name=dname, unique_id=i)
        agents.append((name, agent))
        print(f"Loaded {name} from {d}", flush=True)

    num_cars = len(agents)
    stats = {name: {m: {"episodes": 0, "finished": 0, "dnf": 0, "progress_sum": 0.0,
                         "wall_hits": 0, "vehicle_hits": 0}
                     for m in MAPS} for name, _ in agents}

    t_global_start = time.perf_counter()

    for map_name in MAPS:
        print(f"\n=== MAP: {map_name} ===", flush=True)
        set_map(map_name, num_cars=num_cars)
        map_deadline = time.perf_counter() + per_map_budget

        while time.perf_counter() < map_deadline:
            reset(num_cars=num_cars)
            finished_cars = set()
            dnf_cars = set()
            best_progress = [0.0] * num_cars
            wall_hits = [0] * num_cars
            vehicle_hits = [0] * num_cars

            prev_obs = None
            prev_info = None

            for step_i in range(MAX_STEPS_PER_EPISODE):
                if time.perf_counter() >= map_deadline:
                    break

                if prev_obs is not None:
                    for cid in range(num_cars):
                        if cid in finished_cars or cid in dnf_cars:
                            apply_action(cid, 0.0, 0.0)
                            continue
                        name, agent = agents[cid]
                        try:
                            spd, steer = agent.predict(prev_obs[cid], prev_info)
                        except Exception as e:
                            print(f"[ERROR] {name} predict failed on {map_name} step {step_i}: {e}", flush=True)
                            spd, steer = 0.0, 0.0
                        apply_action(cid, spd, steer)

                simulation_step()
                info = get_step_info()
                obs_list = [get_obs(i) for i in range(num_cars)]

                for cid in range(num_cars):
                    status = int(info["agent_status"][cid])
                    best_progress[cid] = max(best_progress[cid], float(obs_list[cid]["progress"]))
                    if info["collisions"]["wall"].get(cid):
                        wall_hits[cid] += 1
                    if info["collisions"]["vehicle"].get(cid):
                        vehicle_hits[cid] += 1
                    if status == 2:
                        finished_cars.add(cid)
                    elif status == 0:
                        dnf_cars.add(cid)

                prev_obs = obs_list
                prev_info = info

                if len(finished_cars) + len(dnf_cars) == num_cars:
                    break

            for cid, (name, _) in enumerate(agents):
                s = stats[name][map_name]
                s["episodes"] += 1
                s["progress_sum"] += best_progress[cid]
                s["wall_hits"] += wall_hits[cid]
                s["vehicle_hits"] += vehicle_hits[cid]
                if cid in finished_cars:
                    s["finished"] += 1
                elif cid in dnf_cars:
                    s["dnf"] += 1

        print(f"  ({map_name} done, {time.perf_counter() - (map_deadline - per_map_budget):.0f}s elapsed)", flush=True)

    close()

    total_elapsed = time.perf_counter() - t_global_start
    print(f"\n=== SUMMARY (total {total_elapsed/60:.1f} min) ===")
    header = f"{'driver':<10} {'map':<10} {'eps':>4} {'finish':>7} {'dnf':>4} {'avg_prog':>9} {'wall':>5} {'veh':>4}"
    print(header)
    print("-" * len(header))
    for name, _ in agents:
        for m in MAPS:
            s = stats[name][m]
            eps = s["episodes"] or 1
            print(f"{name:<10} {m:<10} {s['episodes']:>4} {s['finished']:>7} {s['dnf']:>4} "
                  f"{s['progress_sum']/eps:>9.3f} {s['wall_hits']:>5} {s['vehicle_hits']:>4}")

    print("\n=== PER-DRIVER TOTALS ===")
    for name, _ in agents:
        tot_eps = sum(stats[name][m]["episodes"] for m in MAPS)
        tot_fin = sum(stats[name][m]["finished"] for m in MAPS)
        tot_dnf = sum(stats[name][m]["dnf"] for m in MAPS)
        avg_prog = sum(stats[name][m]["progress_sum"] for m in MAPS) / max(tot_eps, 1)
        print(f"{name}: episodes={tot_eps} finished={tot_fin} dnf={tot_dnf} avg_progress={avg_prog:.3f}")


if __name__ == "__main__":
    run()
