"""
viz_replay.py — Offline top-down replay from a .npz episode file.

No sim dependency. Requires X11/display.

Usage:
    python viz_replay.py [--npz episode.npz] [--map examples/example_map.yaml] [--follow]

Controls:
    Play/Pause button — real-time playback (uses sim_dt)
    Left/Right arrow  — step by 1
    Shift+Left/Right  — step by 10
    Slider            — seek to any step
    Mouse wheel       — zoom in / zoom out centered on cursor
    Click + Drag      — pan / move map view
    Double-click / R  — reset zoom & pan to full track
"""
import argparse
import glob
import math
import os
import sys
import time

import numpy as np
from PIL import Image
import yaml

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
RECORD_DIR = "/app/recordings"
_maps_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps", "examples")
_default_map = os.path.join(_maps_dir, "example_map.yaml") if os.path.exists(_maps_dir) else None

parser = argparse.ArgumentParser(description="Replay a recorded episode")
parser.add_argument("--npz", default=None, help="Episode NPZ (default: latest in /app/recordings/)")
parser.add_argument("--map", default=_default_map, help="Map YAML (auto-detected from episode if available)")
parser.add_argument("--follow", action="store_true", help="Re-center view on car 0 at each step")
parser.add_argument("--no-display", action="store_true",
                    help="Headless: save a PNG snapshot instead of opening a window")
args = parser.parse_args()

# Backend selection AFTER argparse (FIX C11): Agg when headless, TkAgg otherwise.
import matplotlib
_HEADLESS = args.no_display or not os.environ.get("DISPLAY")
matplotlib.use("Agg" if _HEADLESS else "TkAgg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from matplotlib.widgets import Slider, Button

if args.npz is None:
    existing = sorted(glob.glob(os.path.join(RECORD_DIR, "episode_*.npz")))
    if not existing:
        sys.exit("No episode files in /app/recordings/. Use --npz <file>.")
    args.npz = existing[-1]
    print(f"Loading latest episode: {args.npz}")

# ---------------------------------------------------------------------------
# Constants & Pre-computed Geometry
# ---------------------------------------------------------------------------
_LIDAR_FOV   = 2.0 * math.pi
_CAR_HALF_L  = 0.29
_CAR_HALF_W  = 0.155
_CAR_CORNERS = np.array([
    [ _CAR_HALF_L,  _CAR_HALF_W],
    [ _CAR_HALF_L, -_CAR_HALF_W],
    [-_CAR_HALF_L, -_CAR_HALF_W],
    [-_CAR_HALF_L,  _CAR_HALF_W],
], dtype=np.float32)
_COLORS      = ["#FF6B35", "#4CC9F0", "#7BF178", "#F72585"]
_BG          = "#0D0D0D"

# ---------------------------------------------------------------------------
# Load episode
# ---------------------------------------------------------------------------
print(f"Loading {args.npz} ...")
ep = np.load(args.npz)
lidar     = ep["lidar"]       # (T, C, 100)
velocity  = ep["velocity"]    # (T, C)
steering  = ep["steering"]    # (T, C)
progress  = ep["progress"]    # (T, C)
lap_count = ep["lap_count"]   # (T, C)
poses     = ep["poses"]       # (T, C, 3)  x y theta
actions   = ep["actions"]     # (T, C, 2)  spd steer
status    = ep["status"]      # (T, C)
ranks     = ep["ranks"] if "ranks" in ep else None
friction  = ep["friction"]    # (T,)
max_prog  = ep["max_prog"]    # (T, C)
compute_time = ep["compute_time"] if "compute_time" in ep else (ep["compute_times"] if "compute_times" in ep else None)
sim_dt    = float(ep["sim_dt"])
num_cars  = int(ep["num_cars"])
T         = lidar.shape[0]

calc_stats = {}
if compute_time is not None:
    for cid in range(num_cars):
        active_steps = np.where((status[:, cid] == 1) & (np.arange(T) > 0))[0]
        if len(active_steps) > 0:
            times_ms = compute_time[active_steps, cid] * 1000.0
            calc_stats[cid] = (float(np.mean(times_ms)), float(np.std(times_ms)))
        else:
            calc_stats[cid] = (0.0, 0.0)

if "team_names" in ep:
    team_names = [str(name).strip() for name in ep["team_names"]]
else:
    team_names = [f"Car_{i}" for i in range(num_cars)]

print(f"  {T} steps, {num_cars} car(s) ({', '.join(team_names)}), sim_dt={sim_dt}s")

# ---------------------------------------------------------------------------
# Resolve map (NPZ map_name overrides --map if file exists)
# ---------------------------------------------------------------------------
map_yaml = args.map
if "map_name" in ep:
    auto = os.path.join(_maps_dir, f"{str(ep['map_name']).strip()}.yaml")
    if os.path.exists(auto):
        map_yaml = auto
        print(f"Auto-detected map: {auto}")

if map_yaml is None:
    sys.exit("No map found. Use --map <path>.")

with open(map_yaml) as f:
    meta = yaml.safe_load(f)
res = float(meta["resolution"])
ox, oy = float(meta["origin"][0]), float(meta["origin"][1])

img = np.array(Image.open(map_yaml.replace(".yaml", ".png")))
if img.ndim == 3:
    img = img[..., 0]
h, w = img.shape[:2]
extent = [ox, ox + w * res, oy, oy + h * res]

# Auto-compute tight bounding box of track content (crop out empty background padding)
vals, counts = np.unique(img, return_counts=True)
bg_val = vals[np.argmax(counts)]
track_y, track_x = np.where(img != bg_val)
if len(track_x) == 0:
    track_y, track_x = np.where(img > 0)

if len(track_x) > 0:
    pad_m = 3.0  # 3 meters comfort margin around circuit
    track_xmin = max(extent[0], ox + track_x.min() * res - pad_m)
    track_xmax = min(extent[1], ox + track_x.max() * res + pad_m)
    track_ymin = max(extent[2], oy + (h - 1 - track_y.max()) * res - pad_m)
    track_ymax = min(extent[3], oy + (h - 1 - track_y.min()) * res + pad_m)
    track_extent = [track_xmin, track_xmax, track_ymin, track_ymax]
else:
    track_extent = extent[:]

half_win = max(track_extent[1] - track_extent[0], track_extent[3] - track_extent[2]) / 4.0  # follow half-window

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
_n_lidar_beams = lidar.shape[2]
_LIDAR_REL_ANGLES = -_LIDAR_FOV / 2.0 + np.arange(_n_lidar_beams) * (_LIDAR_FOV / max(_n_lidar_beams - 1, 1))


def _car_vertices(x, y, c, s):
    return np.column_stack([
        _CAR_CORNERS[:, 0] * c - _CAR_CORNERS[:, 1] * s + x,
        _CAR_CORNERS[:, 0] * s + _CAR_CORNERS[:, 1] * c + y,
    ])


def _lidar_segments(x, y, theta, scan):
    """(N, 2, 2) for LineCollection."""
    angles = theta + _LIDAR_REL_ANGLES
    seg = np.empty((len(scan), 2, 2))
    seg[:, 0, 0] = x
    seg[:, 0, 1] = y
    seg[:, 1, 0] = x + scan * np.cos(angles)
    seg[:, 1, 1] = y + scan * np.sin(angles)
    return seg

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor(_BG)

ax_map = fig.add_axes([0.01, 0.08, 0.98, 0.90])   # map: nearly full window
ax_tel = fig.add_axes([0.01, 0.08, 0.24, 0.90])   # telemetry & HUD overlay (top-left)
ax_sld = fig.add_axes([0.01, 0.03, 0.79, 0.035])  # slider
ax_btn = fig.add_axes([0.83, 0.025, 0.14, 0.045]) # play/pause button

for ax in (ax_map, ax_tel):
    ax.set_facecolor(_BG)
    ax.axis("off")

ax_map.set_xlim(track_extent[0], track_extent[1])
ax_map.set_ylim(track_extent[2], track_extent[3])
ax_map.set_aspect("equal")
ax_map.imshow(np.flipud(img), extent=extent, origin="lower",
              cmap="gray_r", alpha=0.85, zorder=1, interpolation="nearest")

if num_cars == 1:
    for cid in range(num_cars):
        color = _COLORS[cid % len(_COLORS)]
        # Only plot trajectory points while the car is active on track (exclude off-map teleport poses)
        active_idx = np.where((status[:, cid] == 1) & (poses[:, cid, 0] < 5000.0))[0]
        if len(active_idx) > 1:
            ax_map.plot(poses[active_idx, cid, 0], poses[active_idx, cid, 1],
                        color=color, lw=2.0, alpha=0.5, zorder=2)

# dynamic artists
car_polys, car_arrows, traj_markers = [], [], []
for cid in range(num_cars):
    color = _COLORS[cid % len(_COLORS)]
    poly = Polygon(np.zeros((4, 2)), closed=True, fc=color, ec="white",
                   lw=0.8, alpha=0.9, zorder=4)
    ax_map.add_patch(poly)
    arrow, = ax_map.plot([], [], color="white", lw=1.2, zorder=5, solid_capstyle="butt")
    mk, = ax_map.plot([], [], "o", color=color, ms=3, zorder=6,
                      markeredgecolor="white", markeredgewidth=0.8)
    car_polys.append(poly)
    car_arrows.append(arrow)
    traj_markers.append(mk)

lidar_lc = LineCollection([], color="#00BB55", lw=1.5, alpha=0.7, zorder=3)
ax_map.add_collection(lidar_lc)

tel_fontsize = 12 if num_cars == 1 else (10 if num_cars == 2 else (8.5 if num_cars == 3 else 7.5))
tel_text = ax_tel.text(0.02, 0.99, "", transform=ax_tel.transAxes, va="top", ha="left",
                       fontsize=tel_fontsize, color="white", family="monospace",
                       bbox=dict(boxstyle="round,pad=0.35", fc="#0D0D0D", ec="#333333", alpha=0.80))

slider = Slider(ax_sld, "Step", 0, T - 1, valinit=0, valstep=1,
                color="#1A6FD4", track_color="#CCCCCC")
ax_sld.xaxis.label.set_color("white")
ax_sld.tick_params(colors="white")

button = Button(ax_btn, "\u25b6  Play", color="#444444", hovercolor="#555555")
button.label.set_color("white")
button.label.set_fontweight("bold")
_playing = False

# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------
def _update(step_i):
    step_i = int(np.clip(step_i, 0, T - 1))

    for cid in range(num_cars):
        x, y, theta = poses[step_i, cid]
        if status[step_i, cid] == 1 and x < 5000.0:
            c, s = math.cos(theta), math.sin(theta)
            car_polys[cid].set_xy(_car_vertices(x, y, c, s))
            car_polys[cid].set_visible(True)
            car_arrows[cid].set_data([x, x + 0.45 * c], [y, y + 0.45 * s])
            car_arrows[cid].set_visible(True)
            traj_markers[cid].set_data([x], [y])
        else:
            car_polys[cid].set_visible(False)
            car_arrows[cid].set_visible(False)
            traj_markers[cid].set_data([], [])

    if num_cars == 1 and status[step_i, 0] == 1 and poses[step_i, 0, 0] < 5000.0:
        x0, y0, th0 = poses[step_i, 0]
        lidar_lc.set_segments(_lidar_segments(x0, y0, th0, lidar[step_i, 0]))
    else:
        lidar_lc.set_segments([])

    if args.follow:
        ax_map.set_xlim(x0 - half_win, x0 + half_win)
        ax_map.set_ylim(y0 - half_win, y0 + half_win)

    lines = [f"Step {step_i:<4d}  |  t = {step_i * sim_dt:.2f}s", ""]

    # Live classification leaderboard
    if num_cars > 1:
        lines.append("── CLASSEMENT ──")
        if ranks is not None:
            order = sorted(range(num_cars), key=lambda c: (ranks[step_i, c], -lap_count[step_i, c], -progress[step_i, c]))
        else:
            order = sorted(
                range(num_cars),
                key=lambda c: (
                    0 if status[step_i, c] == 2 else (1 if status[step_i, c] == 1 else 2),
                    -lap_count[step_i, c],
                    -progress[step_i, c]
                )
            )

        for p_idx, c in enumerate(order, start=1):
            st_code = status[step_i, c]
            st = "active" if st_code == 1 else ("FINISHED" if st_code == 2 else "DNF")
            tname = team_names[c]
            lines.append(f" {p_idx} - {tname:<10s} L{lap_count[step_i, c]} ({st})")
        lines.append("")
    for cid in range(num_cars):
        st_code = status[step_i, cid]
        if st_code == 1:
            st = "active"
        elif st_code == 2:
            st = "FINISHED"
        else:
            st = "DNF"
        tname = team_names[cid]
        lines += [
            f"─── {tname} ({st}) ───", "",
            f"velocity : {velocity[step_i, cid]:+.2f} m/s",
            f"steering : {steering[step_i, cid]:+.4f} rad",
            f"cmd speed: {actions[step_i, cid, 0]:+.2f}",
            f"cmd steer: {actions[step_i, cid, 1]:+.4f}", "",
            f"progress : {progress[step_i, cid]:.3f}",
            f"max prog : {max_prog[step_i, cid]:.3f}",
            f"lap      : {lap_count[step_i, cid]}", "",
        ]
        if compute_time is not None:
            cur_ms = compute_time[step_i, cid] * 1000.0
            mean_ms, std_ms = calc_stats.get(cid, (0.0, 0.0))
            lines += [
                f"calc time: {cur_ms:4.2f} ms ({mean_ms:.2f}±{std_ms:.2f}ms)", "",
            ]
    lines.append(f"friction : {friction[step_i]:.3f}")
    tel_text.set_text("\n".join(lines))

    fig.canvas.draw_idle()


slider.on_changed(_update)

# ---------------------------------------------------------------------------
# Play / pause toggle
# ---------------------------------------------------------------------------
def _toggle(event=None):
    global _playing
    _playing = not _playing
    if _playing:
        if int(slider.val) >= T - 1:
            slider.set_val(0)
        button.label.set_text("\u23f8  Pause")
        button.ax.set_facecolor("#1A6FD4")
    else:
        button.label.set_text("\u25b6  Play")
        button.ax.set_facecolor("#444444")
    fig.canvas.draw_idle()

button.on_clicked(_toggle)

# ---------------------------------------------------------------------------
# Mouse interaction (Zoom & Pan in normal mode)
# ---------------------------------------------------------------------------
_pan_start = None


def _reset_view():
    ax_map.set_xlim(track_extent[0], track_extent[1])
    ax_map.set_ylim(track_extent[2], track_extent[3])
    fig.canvas.draw_idle()


def _on_scroll(event):
    if event.inaxes != ax_map or args.follow:
        return
    # Mouse wheel zoom centered on cursor
    scale = 0.82 if event.button == "up" else 1.22
    cur_xlim = ax_map.get_xlim()
    cur_ylim = ax_map.get_ylim()
    xdata = event.xdata if event.xdata is not None else (cur_xlim[0] + cur_xlim[1]) / 2.0
    ydata = event.ydata if event.ydata is not None else (cur_ylim[0] + cur_ylim[1]) / 2.0

    new_w = (cur_xlim[1] - cur_xlim[0]) * scale
    new_h = (cur_ylim[1] - cur_ylim[0]) * scale

    relx = (cur_xlim[1] - xdata) / max(cur_xlim[1] - cur_xlim[0], 1e-6)
    rely = (cur_ylim[1] - ydata) / max(cur_ylim[1] - cur_ylim[0], 1e-6)

    ax_map.set_xlim([xdata - new_w * (1 - relx), xdata + new_w * relx])
    ax_map.set_ylim([ydata - new_h * (1 - rely), ydata + new_h * rely])
    fig.canvas.draw_idle()


def _on_press(event):
    global _pan_start
    if event.inaxes == ax_map and not args.follow:
        if event.dblclick:
            _reset_view()
            return
        if event.button in (1, 2, 3):
            _pan_start = (event.x, event.y, ax_map.get_xlim(), ax_map.get_ylim())


def _on_release(event):
    global _pan_start
    _pan_start = None


def _on_motion(event):
    global _pan_start
    if _pan_start is None or event.inaxes != ax_map or args.follow:
        return
    x0, y0, (xlim0, xlim1), (ylim0, ylim1) = _pan_start
    dx = event.x - x0
    dy = event.y - y0
    bbox = ax_map.get_window_extent()
    if bbox.width <= 0 or bbox.height <= 0:
        return
    scale_x = (xlim1 - xlim0) / bbox.width
    scale_y = (ylim1 - ylim0) / bbox.height
    ax_map.set_xlim([xlim0 - dx * scale_x, xlim1 - dx * scale_x])
    ax_map.set_ylim([ylim0 - dy * scale_y, ylim1 - dy * scale_y])
    fig.canvas.draw_idle()


fig.canvas.mpl_connect("scroll_event", _on_scroll)
fig.canvas.mpl_connect("button_press_event", _on_press)
fig.canvas.mpl_connect("button_release_event", _on_release)
fig.canvas.mpl_connect("motion_notify_event", _on_motion)

# ---------------------------------------------------------------------------
# Keyboard — pauses on manual step
# ---------------------------------------------------------------------------
def _on_key(event):
    global _playing
    if event.key in ("r", "R") and not args.follow:
        _reset_view()
        return

    step = {"left": -1, "right": 1, "shift+left": -10, "shift+right": 10}.get(event.key)
    if step is None:
        return
    if _playing:
        _toggle()
    slider.set_val(int(np.clip(slider.val + step, 0, T - 1)))

fig.canvas.mpl_connect("key_press_event", _on_key)

# ---------------------------------------------------------------------------
# Main loop — flush_events is X11-friendly, no timer/thread
# ---------------------------------------------------------------------------
_update(0)

if _HEADLESS:
    snap_path = args.npz.replace(".npz", "_snapshot.png")
    fig.savefig(snap_path, dpi=150)
    print(f"Headless mode: snapshot saved to {snap_path}")
    sys.exit(0)

plt.show(block=False)

_last_step_time = time.perf_counter()
while plt.fignum_exists(fig.number):
    if _playing:
        now = time.perf_counter()
        if now - _last_step_time >= sim_dt:
            _last_step_time = now
            v = int(slider.val)
            if v >= T - 1:
                _toggle()   # auto-stop at end
            else:
                slider.set_val(v + 1)   # triggers _update via on_changed
    else:
        _last_step_time = time.perf_counter()
    fig.canvas.flush_events()
    time.sleep(0.005)