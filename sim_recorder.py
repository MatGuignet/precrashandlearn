"""
sim_recorder.py — Run a full episode at maximum speed, dump all obs/state to .npz.

Usage:
    python sim_recorder.py [--steps 500] [--num-cars 1] [--model agent.onnx] [--out episode.npz]
"""
import argparse
import glob
import math
import os
import sys
import time

import numpy as np

_SIM_DT = 0.05  # 1 / _DECISION_FREQ_HZ (20 Hz) — must match env_simulation

OUTPUT_DIR = "/app/recordings"

parser = argparse.ArgumentParser(description="Record a sim episode to .npz")
parser.add_argument("--steps",          type=int, default=2000)
parser.add_argument("--laps",           type=int, default=2, help="Target number of laps to complete per car (default: 1)")
parser.add_argument("--num-cars",       type=int, default=1)
parser.add_argument("--model",          type=str, default="", help="ONNX model for car 0")
parser.add_argument("--submissions",    type=str, default="", help="Path to a submission directory or parent directory with team subfolders")
parser.add_argument("--controller",     type=str, default="auto", choices=["auto", "submission", "random", "onnx", "pure_pursuit"],
                    help="Controller policy for car 0 (default: auto, uses submission if available)")
parser.add_argument("--recordings-dir", type=str, default="", help="Directory for saved episodes (default: /app/recordings or recordings/)")
parser.add_argument("--out",            type=str, default=None, help="Output path (default: <recordings-dir>/episode_XXX.npz)")
args = parser.parse_args()

# Determine recordings directory and ensure parent directories exist
rec_dir = args.recordings_dir.strip()
if not rec_dir:
    rec_dir = "/app/recordings" if os.path.exists("/app") else "recordings"

if args.out is None:
    os.makedirs(rec_dir, exist_ok=True)
    existing = glob.glob(os.path.join(rec_dir, "episode_*.npz"))
    episode_id = len(existing) + 1
    args.out = os.path.join(rec_dir, f"episode_{episode_id:04d}.npz")
else:
    # If args.out is a directory, place auto-numbered episode inside it
    if os.path.isdir(args.out) or args.out.endswith(os.sep) or args.out.endswith("/"):
        os.makedirs(args.out, exist_ok=True)
        existing = glob.glob(os.path.join(args.out, "episode_*.npz"))
        episode_id = len(existing) + 1
        args.out = os.path.join(args.out, f"episode_{episode_id:04d}.npz")
    else:
        out_parent = os.path.dirname(args.out)
        if out_parent:
            os.makedirs(out_parent, exist_ok=True)

num_cars = max(1, min(args.num_cars, 4))
T = args.steps

# ---------------------------------------------------------------------------
# Policy / controller selection & Multi-agent loader
# ---------------------------------------------------------------------------
_SPEED   = 2.0
_LOOKAHEAD = 1.0

from env_simulation import _PARAMS as _EP
_WHEELBASE = _EP["lf"] + _EP["lr"]
_SMIN, _SMAX = _EP["s_min"], _EP["s_max"]


def _random_policy(step_i):
    return _SPEED, 0.08 * math.sin(step_i * 0.12)


def _pure_pursuit_policy(step_i, obs, waypoints, arc, total_arc, state):
    """Pure pursuit: find lookahead waypoint by walking from closest point, compute steering as angle to carrot."""
    x, y, yaw = float(state[0]), float(state[1]), float(state[4])

    # Find closest waypoint on centerline
    dists = (waypoints[:, 0] - x) ** 2 + (waypoints[:, 1] - y) ** 2
    idx = int(np.argmin(dists))

    # Walk from closest point, accumulating distance until >= lookahead
    N = len(waypoints)
    cum = 0.0
    target_idx = idx  # fallback: closest point itself
    for i in range(1, N):
        wi = (idx + i) % N
        j = (idx + i - 1) % N
        dx_w = waypoints[wi, 0] - waypoints[j, 0]
        dy_w = waypoints[wi, 1] - waypoints[j, 1]
        cum += math.sqrt(dx_w * dx_w + dy_w * dy_w)
        if cum >= _LOOKAHEAD:
            target_idx = wi
            break

    tx, ty = waypoints[target_idx, 0], waypoints[target_idx, 1]

    # Angle from car's orientation to the carrot
    dx, dy = tx - x, ty - y
    angle_to_target = math.atan2(dy, dx)
    angle_diff = angle_to_target - yaw

    # Normalize to [-pi, pi]
    while angle_diff > math.pi:
        angle_diff -= 2 * math.pi
    while angle_diff < -math.pi:
        angle_diff += 2 * math.pi

    # Use the angle directly as steering (clipped to bounds)
    steer = max(_SMIN, min(_SMAX, angle_diff))
    return _SPEED, steer


def _onnx_policy(session, obs):
    feat = np.concatenate([
        obs["lidar"].astype(np.float32),
        np.array([obs["velocity"], obs["steering"], obs["progress"]], dtype=np.float32),
    ]).reshape(1, -1)
    inp = session.get_inputs()[0].name
    out = session.run(None, {inp: feat})[0][0]
    return float(out[0]), float(out[1])


# Team agents list: [(team_name, agent_instance or None), ...]
team_agents = []
team_names = [f"Car_{i}" for i in range(num_cars)]

from agent_loader import discover_and_load_submissions

sub_path = args.submissions.strip()
if not sub_path and args.controller in ("auto", "submission"):
    if os.path.exists("submission"):
        sub_path = "submission"

if sub_path:
    try:
        loaded = discover_and_load_submissions(sub_path, max_teams=num_cars)
        if len(loaded) == 1 and num_cars > 1:
            # Self-play mode: clone single submission across all cars
            single_name, single_agent = loaded[0]
            print(f"Loaded submission '{single_name}' — replicated across {num_cars} cars (self-play)")
            for i in range(num_cars):
                team_names[i] = f"{single_name}_{i}"[:10]
                team_agents.append((team_names[i], single_agent))
        else:
            for i in range(num_cars):
                if i < len(loaded):
                    tname, tagent = loaded[i]
                    team_names[i] = tname[:10]
                    team_agents.append((team_names[i], tagent))
                    print(f"  [Car {i}] Assigned to Team '{team_names[i]}'")
                else:
                    team_agents.append((f"Car_{i}", None))
    except Exception as e:
        print(f"[ERROR] Failed loading submissions from '{sub_path}':\n{e}")
        if args.submissions:
            sys.exit(1)

session = None
if args.model:
    import onnxruntime
    session = onnxruntime.InferenceSession(args.model)
    print(f"Loaded ONNX model for car 0: {args.model}")

# ---------------------------------------------------------------------------
# Sim import
# ---------------------------------------------------------------------------
from env_simulation import (
    reset, get_obs, apply_action, simulation_step, get_step_info, close, _get_sim,
    get_current_map,
)

# ---------------------------------------------------------------------------
# Pre-allocate
# ---------------------------------------------------------------------------
lidar     = np.zeros((T, num_cars, 100), dtype=np.float32)
velocity  = np.zeros((T, num_cars),      dtype=np.float32)
steering  = np.zeros((T, num_cars),      dtype=np.float32)
progress  = np.zeros((T, num_cars),      dtype=np.float32)
lap_count = np.zeros((T, num_cars),      dtype=np.int32)
poses     = np.zeros((T, num_cars, 3),   dtype=np.float32)  # x, y, theta
actions   = np.zeros((T, num_cars, 2),   dtype=np.float32)  # speed, steer
status    = np.zeros((T, num_cars),      dtype=np.int32)
ranks     = np.zeros((T, num_cars),      dtype=np.int32)
friction  = np.zeros((T,),               dtype=np.float32)
max_prog  = np.zeros((T, num_cars),      dtype=np.float32)
compute_time = np.zeros((T, num_cars),   dtype=np.float32)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
reset(num_cars=num_cars)
sim_obj = _get_sim()

t_start = time.perf_counter()
actual_steps = 0
prev_obs = None  # observations from previous step (used for action inputs)
prev_info = None

finished_cars = {}  # {cid: finish_step}
dnf_cars = set()

for step_i in range(T):
    # --- controller helpers (pure_pursuit needs waypoints/state from sim_obj) ---
    use_pp = args.controller == "pure_pursuit"
    wpts = sim_obj._waypoints if use_pp else None
    arc  = sim_obj._arc      if use_pp else None
    tac  = sim_obj._total_arc if use_pp else None

    # --- controller: pick speed/steer from prev_obs (or skip on step 0) ---
    decisions = {}  # {cid: (spd, steer)}
    if prev_obs is not None:
        for cid, obs in enumerate(prev_obs):
            if cid in finished_cars:
                decisions[cid] = (0.0, 0.0)  # car already completed target laps
                continue

            t_calc_0 = time.perf_counter()
            spd, steer = _random_policy(step_i)
            if cid < len(team_agents) and team_agents[cid][1] is not None:
                try:
                    spd, steer = team_agents[cid][1].predict(obs, prev_info if prev_info is not None else {})
                except Exception as e:
                    print(f"[ERROR] Agent '{team_names[cid]}' (Car {cid}) predict failed at step {step_i}: {e}")
            elif args.controller == "onnx" and cid == 0 and session is not None:
                spd, steer = _onnx_policy(session, obs)
            elif args.controller == "pure_pursuit" and use_pp:
                state = sim_obj._sim.agents[cid].state
                spd, steer = _pure_pursuit_policy(step_i, obs, wpts, arc, tac, state)
            compute_time[step_i, cid] = time.perf_counter() - t_calc_0
            decisions[cid] = (spd, steer)

    # --- actions (apply controller decisions; skip on step 0 — car starts idle) ---
    for cid, (spd, steer) in decisions.items():
        apply_action(cid, spd, steer)
        actions[step_i, cid, 0] = spd
        actions[step_i, cid, 1] = steer

    # --- step (THIS is where lap_count increments and DNF can fire) ---
    simulation_step()
    info = get_step_info()

    # --- obs — capture AFTER step so lap_count reflects any completed laps ---
    obs_list = [get_obs(i) for i in range(num_cars)]

    # --- store obs (all from post-step observations, consistent source) ---
    for cid, obs in enumerate(obs_list):
        lidar[step_i, cid]     = obs["lidar"]
        velocity[step_i, cid]  = obs["velocity"]
        steering[step_i, cid]  = obs["steering"]
        progress[step_i, cid]  = obs["progress"]
        lap_count[step_i, cid] = obs["lap_count"]
        status[step_i, cid]    = info["agent_status"][cid]
        ranks[step_i, cid]     = int(obs["rank"])

    # --- poses (from sim state after step) ---
    for cid in range(num_cars):
        agent = sim_obj._sim.agents[cid]
        poses[step_i, cid, 0] = float(agent.state[0])
        poses[step_i, cid, 1] = float(agent.state[1])
        poses[step_i, cid, 2] = float(agent.state[4])

    friction[step_i] = info["friction_current"]
    for cid in range(num_cars):
        max_prog[step_i, cid] = float(info["max_progress"][cid])

    actual_steps = step_i + 1

    # --- check lap completions and DNFs per car ---
    for cid, obs in enumerate(obs_list):
        if cid not in finished_cars and obs["lap_count"] >= args.laps:
            finished_cars[cid] = actual_steps
            sim_obj._finish(cid)
            status[step_i, cid] = 2
            poses[step_i, cid, 0] = float(sim_obj._sim.agents[cid].state[0])
            poses[step_i, cid, 1] = float(sim_obj._sim.agents[cid].state[1])
            poses[step_i, cid, 2] = float(sim_obj._sim.agents[cid].state[4])
            t_fin = actual_steps * _SIM_DT
            #print(f"🏁 [FINISH] Team '{team_names[cid]}' (Car {cid}) completed {args.laps} lap(s) at step {actual_steps} ({t_fin:.2f}s) — teleported to paddock")

        if info["agent_status"][cid] == 0 and cid not in finished_cars and cid not in dnf_cars:
            dnf_cars.add(cid)
            #print(f"💥 [DNF] Team '{team_names[cid]}' (Car {cid}) DNF at step {actual_steps} (progress: {obs['progress']:.3f}, laps: {obs['lap_count']})")

    # --- save obs/info for next iteration's controller inputs ---
    prev_obs = obs_list
    prev_info = info

    # --- progress print ---
    if actual_steps % 100 == 0:
        elapsed = time.perf_counter() - t_start
        rt = (actual_steps * _SIM_DT) / elapsed
        fin_count = len(finished_cars)
        dnf_count = len(dnf_cars)
        active_count = num_cars - fin_count - dnf_count
        print(f"step={actual_steps:4d}  sim_t={actual_steps*_SIM_DT:.1f}s  "
              f"active={active_count} fin={fin_count} dnf={dnf_count}  RT×{rt:.1f}")

    # --- exit when all cars have reached target laps or suffered DNF ---
    if all(cid in finished_cars or info["agent_status"][cid] != 1 for cid in range(num_cars)):
        print(f"\n🏁 Race complete: all cars reached target laps ({args.laps}) or DNF at step {actual_steps}")
        break

# Print final classification table
if False:
    print("\n" + "=" * 62)
    print(f"🏁 FINAL RACE CLASSIFICATION ({args.laps} lap{'s' if args.laps > 1 else ''})")
    print("=" * 62)
    results = []
    for cid in range(num_cars):
        tname = team_names[cid]
        if cid in finished_cars:
            f_step = finished_cars[cid]
            results.append((0, f_step, cid, tname, f"FINISHED ({f_step * _SIM_DT:.2f}s)", obs_list[cid]["lap_count"]))
        else:
            results.append((1, -float(max_prog[actual_steps - 1, cid]), cid, tname, "DNF", obs_list[cid]["lap_count"]))

    results.sort(key=lambda x: (x[0], x[1]))
    for rank, (_, _, cid, tname, st_label, lcount) in enumerate(results, start=1):
        print(f"  P{rank} | Car {cid} ({tname:10s}) : {st_label:<20s} | {lcount} lap(s)")
    print("=" * 62 + "\n")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
wall_total = time.perf_counter() - t_start
rt_total = (actual_steps * _SIM_DT) / wall_total

np.savez_compressed(
    args.out,
    lidar     = lidar[:actual_steps],
    velocity  = velocity[:actual_steps],
    steering  = steering[:actual_steps],
    progress  = progress[:actual_steps],
    lap_count = lap_count[:actual_steps],
    poses     = poses[:actual_steps],
    actions   = actions[:actual_steps],
    status    = status[:actual_steps],
    ranks     = ranks[:actual_steps],
    friction  = friction[:actual_steps],
    max_prog  = max_prog[:actual_steps],
    compute_time = compute_time[:actual_steps],
    sim_dt    = np.float32(_SIM_DT),
    num_cars  = np.int32(num_cars),
    map_name  = np.str_(get_current_map()),
    team_names = np.array(team_names, dtype="U10"),
)

close()
print(f"\nDone. {actual_steps} steps in {wall_total:.2f}s  RT×{rt_total:.1f}")
print(f"Saved → {args.out}")

