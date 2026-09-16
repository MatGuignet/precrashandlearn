"""
test_integration.py — Slow, physics-driven, map-dependent tests.
Run with: python test_integration.py
"""
import math
import numpy as np

from _harness import run_all, run_test, ok, fail

# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------
from env_simulation import (
    get_space_info, reset, get_obs, apply_action, simulation_step, get_step_info, close,
    _LIDAR_RAYS, _MAX_SLOTS, _get_sim, _DNF_WINDOW_STEPS, _FRICTION_HI, _PARAMS,
)
from f110_gym.envs.base_classes import RaceCar

# ---------------------------------------------------------------------------
# T05 — vehicle-vehicle collision: no termination, knockback visible
# ---------------------------------------------------------------------------
def t05_vehicle_collision_no_terminate_knockback():
    """
    Place 2 cars on a direct collision course in open space.
    Verify:
    - collision flag fires
    - episode does NOT terminate (no done, no status change)
    - after contact, distance between cars increases (knockback)
    """
    reset(2)
    sim_obj = _get_sim()
    assert sim_obj._sim is not None
    engine = sim_obj._sim

    # Override start positions: face each other ~1.0 m apart
    engine.agents[0].state[:] = 0.0
    engine.agents[0].state[0] = 8.0
    engine.agents[0].state[1] = 0.0
    engine.agents[0].state[4] = 0.0   # facing +x

    engine.agents[1].state[:] = 0.0
    engine.agents[1].state[0] = 9.2
    engine.agents[1].state[1] = 0.0
    engine.agents[1].state[4] = math.pi  # facing -x

    collision_detected = False
    dist_prev = None
    dist      = None

    for i in range(60):
        apply_action(0, _PARAMS["v_max"], 0.0)
        apply_action(1, _PARAMS["v_max"], 0.0)
        simulation_step()
        info = get_step_info()

        x0 = float(engine.agents[0].state[0])
        y0 = float(engine.agents[0].state[1])
        x1 = float(engine.agents[1].state[0])
        y1 = float(engine.agents[1].state[1])
        dist = math.hypot(x1 - x0, y1 - y0)

        if info["collisions"]["vehicle"][0] and not collision_detected:
            collision_detected = True
            # No termination: status must still be ACTIVE
            assert info["agent_status"][0] == 1, "car 0 must be ACTIVE after collision"
            assert info["agent_status"][1] == 1, "car 1 must be ACTIVE after collision"
            assert dist >= dist_prev, (
                f"cars interpenetrating after knockback: at_hit={dist_prev:.4f}, after={dist:.4f}"
            )

        if collision_detected and i > 30:
            x0 = float(engine.agents[0].state[0])
            y0 = float(engine.agents[0].state[1])
            x1 = float(engine.agents[1].state[0])
            y1 = float(engine.agents[1].state[1])
            break

        dist_prev = dist

    assert collision_detected, "vehicle-vehicle collision not detected"
    assert dist is not None
    

# ---------------------------------------------------------------------------
# T10 — DNF'd agent doesn't corrupt active car's collisions or scan
# ---------------------------------------------------------------------------
def t10_dnf_no_corruption():
    """
    Force car 1 into DNF via net-displacement window, then verify car 0's
    collision_idx stays clear and its scan is unaffected (DNF car moved off-map).
    """
    reset(2)
    sim_obj = _get_sim()
    engine  = sim_obj._sim

    # Force car 1 to be frozen at its current position (simulate stagnation)
    # by injecting position directly and marking it DNF
    x1 = float(engine.agents[1].state[0])
    y1 = float(engine.agents[1].state[1])

    for _ in range(_DNF_WINDOW_STEPS + 5):
        # Car 1 commanded at 0 — stays still → net disp ≈ 0 → triggers DNF
        apply_action(0, 2.0, 0.0)
        apply_action(1, 0.0, 0.0)
        simulation_step()

    info = get_step_info()
    # Car 1 should be DNF
    assert info["agent_status"][1] == 0, f"car 1 should be DNF, got status={info['agent_status'][1]}"
    assert info["opponents_mask"][1] == False

    # Car 0 must NOT show a vehicle collision with the DNF'd car
    for _ in range(5):
        apply_action(0, 2.0, 0.0)
        simulation_step()
        info2 = get_step_info()
        assert not info2["collisions"]["vehicle"][0], \
            "DNF'd car should not trigger vehicle collision for active car"

    # Car 0's scan should be valid (not corrupted by off-map DNF position)
    obs0 = get_obs(0)
    assert obs0["lidar"].shape == (_LIDAR_RAYS,)
    assert all(obs0["lidar"] >= 0.1)
    assert all(obs0["lidar"] <= 15.0)

# ---------------------------------------------------------------------------
# T-wall — drive into nearest wall: engine iTTC fires, episode does NOT terminate
# ---------------------------------------------------------------------------
def t_wall_hit_physics():
    """Geometry-independent wall hit: orient the car at its min-range LiDAR beam and
    accelerate. collisions.wall[0] must fire and status must stay ACTIVE (no done-on-collision)."""
    reset(1)
    scan = get_obs(0)["lidar"]
    sim_obj = _get_sim()
    st = sim_obj._sim.agents[0].state
    k = int(np.argmin(scan))
    st[4] = float(st[4]) + float(RaceCar.scan_angles[k])   # yaw = state[4]; face nearest wall
    st[3] = 0.0                                            # start from rest; thrust accelerates
    fired = False
    for _ in range(80):
        apply_action(0, _PARAMS["v_max"], 0.0)
        simulation_step()
        info = get_step_info()
        if bool(info["collisions"]["wall"][0]):
            fired = True
            assert info["agent_status"][0] == 1, "wall hit must NOT terminate the episode"
            break
    assert fired, "wall collision never fired while driving into the nearest wall"

# ---------------------------------------------------------------------------
# T15 — progress in [0,1], progress_delta small and plausible
# ---------------------------------------------------------------------------
def t15_progress_bounds():
    reset(1)
    for _ in range(50):
        apply_action(0, 3.0, 0.0)
        simulation_step()
    obs = get_obs(0)
    assert 0.0 <= obs["progress"] < 1.0, f"progress out of [0,1): {obs['progress']}"
    info = get_step_info()
    assert abs(info["progress_delta"][0]) < 0.1, \
        f"progress_delta implausibly large: {info['progress_delta'][0]}"

# ---------------------------------------------------------------------------
# T23 — friction starts at 1.0 immediately
# ---------------------------------------------------------------------------
def t23_friction_initial_value():
    """Friction must be 1.0 (not the engine default 1.0489) right after reset."""
    reset(1)
    info = get_step_info()
    assert info["friction_current"] == _FRICTION_HI, \
        f"Initial friction should be {_FRICTION_HI}, got {info['friction_current']}"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_all([
        ("T05 vehicle collision: no terminate, knockback", t05_vehicle_collision_no_terminate_knockback),
        ("T10 DNF no corruption of active cars",          t10_dnf_no_corruption),
        ("T-wall drive into nearest wall (iTTC)",         t_wall_hit_physics),
        ("T15 progress in [0,1], delta plausible",        t15_progress_bounds),
        ("T23 friction starts at 1.0 immediately",       t23_friction_initial_value),
    ])
