"""
Fire-Rescue Robot -- DYNAMIC Pathfinding Simulation
====================================================
Fork of fire_rescue_pathfinding.py with three key dynamic features:

1. DYNAMIC FIRE   -- Fire spreads realistically from ignition points each
                     time-step using a radial cellular-automata model.
2. DYNAMIC SOURCE -- The robot moves one cell per time-step along its
                     planned path, observing the changing environment.
3. DYNAMIC ALGO   -- "Dynamic Risk-Aware A*" replans every step using
                     the latest temperature map.  Compared against the
                     static planners that commit to their initial path.

Compares four strategies:
  - Dijkstra (static)     -- plans once, shortest path, ignores heat
  - A* (static)           -- plans once, shortest path, fewer nodes explored
  - Risk-Aware A* (static)-- plans once with heat penalty, follows blindly
  - Dynamic Risk-Aware A* -- replans every step on the live temperature map
"""

import heapq, itertools, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib import animation

# =============================================
# 1. ENVIRONMENT SETUP
# =============================================
GRID_SIZE = 20
START = (0, 0)
GOAL  = (19, 19)
ALPHA = 1.2          # heat-cost weight for risk-aware variants

# --- Initial temperature: cool everywhere, hot only near ignition ---
temperature_init = np.ones((GRID_SIZE, GRID_SIZE), dtype=float)

# Realistic fire ignitions: near the primary path gaps to cut them off
IGNITION_POINTS = [(19, 19), (8, 7), (14, 13)]

for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        # Distance-based heat from nearest ignition point
        min_dist = min(np.sqrt((r - ir)**2 + (c - ic)**2)
                       for ir, ic in IGNITION_POINTS)
        if min_dist <= 1.5:
            temperature_init[r, c] = 450 + np.random.randint(-30, 30)
        elif min_dist <= 3.0:
            temperature_init[r, c] = 300 + np.random.randint(-40, 40)
        elif min_dist <= 5.0:
            temperature_init[r, c] = 120 + np.random.randint(-30, 30)
        elif min_dist <= 7.0:
            temperature_init[r, c] = 40 + np.random.randint(-15, 15)
        else:
            temperature_init[r, c] = 1   # ambient / cool

# Small secondary heat source near the middle-right (makes static planners
# walk into trouble as fire spreads towards it)
for r in range(9, 12):
    for c in range(15, 18):
        temperature_init[r, c] = max(temperature_init[r, c],
                                      80 + np.random.randint(0, 40))

temperature_init[START] = 1
temperature_init = np.clip(temperature_init, 1, 500)

# --- Walls (same layout as original) ---
walls = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

# Barrier 1: row 7 full width. Primary gap at 7, alternate gap at 18
for c in range(GRID_SIZE):
    walls[7, c] = True
walls[7, 7] = False
walls[7, 18] = False

# Barrier 2: row 13 full width. Primary gap at 13, alternate gap at 2
for c in range(GRID_SIZE):
    walls[13, c] = True
walls[13, 13] = False
walls[13, 2] = False

# Top-edge partial wall
for c in range(5, 16):
    walls[0, c] = True

# Right-edge partial wall (upper)
for r in range(0, 5):
    walls[r, 19] = True

# Interior accent walls
walls[3, 13] = True; walls[3, 14] = True
walls[16, 4] = True; walls[16, 5] = True
walls[START] = False; walls[GOAL] = False


# =============================================
# 2. REALISTIC FIRE SPREAD MODEL
# =============================================
SPREAD_PROB  = 0.18     # base chance per hot neighbour per tick
HEAT_GAIN    = 30       # heat added per spread event
COOL_RATE    = 0.995    # very mild ambient cooling
FIRE_THRESH  = 120      # cells above this can spread
MAX_TEMP     = 600

fire_rng = np.random.RandomState(42)

def spread_fire(temp, step):
    """
    Advance fire one time-step.  Fire spreads radially from hot cells
    with stochastic irregularity for realistic flame shapes.
    """
    new_temp = temp.copy()

    # 1. Hot cells attempt to ignite 4-connected neighbours
    hot_r, hot_c = np.where((temp > FIRE_THRESH) & (~walls))
    for r, c in zip(hot_r, hot_c):
        intensity = temp[r, c] / MAX_TEMP          # hotter cells spread more
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and not walls[nr, nc]:
                prob = SPREAD_PROB * (0.5 + intensity)
                if fire_rng.random() < prob:
                    new_temp[nr, nc] = min(new_temp[nr, nc] + HEAT_GAIN, MAX_TEMP)

    # 2. Diagonal spread (lower probability -- gives irregular shape)
    for r, c in zip(hot_r, hot_c):
        for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and not walls[nr, nc]:
                if fire_rng.random() < SPREAD_PROB * 0.3:
                    new_temp[nr, nc] = min(new_temp[nr, nc] + HEAT_GAIN * 0.5,
                                           MAX_TEMP)

    # 3. Ignition points stay hot (fuel source)
    for ir, ic in IGNITION_POINTS:
        new_temp[ir, ic] = min(new_temp[ir, ic] + 5, MAX_TEMP)

    # 4. Very mild ambient cooling for distant cells
    cool_mask = (new_temp < FIRE_THRESH * 0.5) & (~walls)
    new_temp[cool_mask] *= COOL_RATE
    new_temp[cool_mask] = np.maximum(new_temp[cool_mask], 1)

    return new_temp


# =============================================
# 3. PATH-FINDING ALGORITHMS
# =============================================
def neighbors(node):
    r, c = node
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and not walls[nr, nc]:
            yield (nr, nc)

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def reconstruct(came_from, node):
    path = [node]
    while node in came_from:
        node = came_from[node]; path.append(node)
    path.reverse()
    return path


def dijkstra(start, goal, temp_grid):
    """Standard Dijkstra -- uniform cost, ignores temperature."""
    cnt  = itertools.count()
    heap = [(0, next(cnt), start)]
    g    = {start: 0}
    par  = {}
    vis  = set()
    while heap:
        cost, _, u = heapq.heappop(heap)
        if u in vis:
            continue
        vis.add(u)
        if u == goal:
            return reconstruct(par, u), vis
        for v in neighbors(u):
            nc = g[u] + 1
            if v not in g or nc < g[v]:
                g[v] = nc; par[v] = u
                heapq.heappush(heap, (nc, next(cnt), v))
    return None, vis


def a_star(start, goal, temp_grid, alpha=0.0):
    """
    A* with optional heat penalty.
      alpha = 0   --> standard A*
      alpha > 0   --> risk-aware A*
    """
    cnt  = itertools.count()
    heap = [(heuristic(start, goal), 0, next(cnt), start)]
    g    = {start: 0}
    par  = {}
    vis  = set()

    while heap:
        _, neg_g, _, u = heapq.heappop(heap)
        if u in vis:
            continue
        vis.add(u)
        if u == goal:
            return reconstruct(par, u), vis

        for v in neighbors(u):
            step_cost = 1 + alpha * temp_grid[v[0], v[1]]
            nc = g[u] + step_cost
            if v not in g or nc < g[v]:
                g[v] = nc
                par[v] = u
                f = nc + heuristic(v, goal)
                heapq.heappush(heap, (f, -nc, next(cnt), v))

    return None, vis


# =============================================
# 4. SIMULATION RUNNER
# =============================================
def simulate(planner_name, algo_fn, replan, alpha, max_steps=200):
    """
    Simulate the robot moving through a dynamically-changing fire grid.

    Parameters
    ----------
    planner_name : str
    algo_fn      : callable -- dijkstra or a_star
    replan       : bool     -- if True, recompute path every step
    alpha        : float    -- heat weight (only used by a_star)
    max_steps    : int      -- safety limit

    Returns
    -------
    dict with trajectory, heat_history, temp_snapshots, etc.
    """
    temp = temperature_init.copy()
    pos  = START
    trajectory   = [pos]
    heat_accum   = 0.0
    temp_snaps   = [temp.copy()]
    path_snaps   = []
    replans      = 0

    # initial plan
    if algo_fn == dijkstra:
        path, explored = dijkstra(pos, GOAL, temp)
    else:
        path, explored = a_star(pos, GOAL, temp, alpha)

    if path is None:
        return dict(trajectory=trajectory, heat=float("inf"),
                    temp_snaps=temp_snaps, path_snaps=[], replans=0,
                    steps=0, success=False, explored_total=len(explored))

    path_snaps.append(list(path))
    step_idx = 1
    explored_total = len(explored)

    for t in range(1, max_steps + 1):
        # --- advance fire ---
        temp = spread_fire(temp, t)
        temp_snaps.append(temp.copy())

        # --- replan if requested ---
        if replan:
            if algo_fn == dijkstra:
                new_path, exp = dijkstra(pos, GOAL, temp)
            else:
                new_path, exp = a_star(pos, GOAL, temp, alpha)
            explored_total += len(exp)
            replans += 1
            if new_path is not None:
                path = new_path
                step_idx = 1

        path_snaps.append(list(path))

        # --- move one step ---
        if step_idx < len(path):
            pos = path[step_idx]
            step_idx += 1

        heat_accum += temp[pos[0], pos[1]]
        trajectory.append(pos)

        if pos == GOAL:
            return dict(trajectory=trajectory, heat=heat_accum,
                        temp_snaps=temp_snaps, path_snaps=path_snaps,
                        replans=replans, steps=t, success=True,
                        explored_total=explored_total)

    return dict(trajectory=trajectory, heat=heat_accum,
                temp_snaps=temp_snaps, path_snaps=path_snaps,
                replans=replans, steps=max_steps, success=False,
                explored_total=explored_total)


# =============================================
# 5. RUN ALL FOUR STRATEGIES
# =============================================
strategies = {
    "Dijkstra (static)":    dict(algo_fn=dijkstra, replan=False, alpha=0.0),
    "A* (static)":          dict(algo_fn=a_star,    replan=False, alpha=0.0),
    "Risk-Aware A* (static)": dict(algo_fn=a_star,  replan=False, alpha=ALPHA),
    "Dynamic Risk-Aware":   dict(algo_fn=a_star,    replan=True,  alpha=ALPHA),
}

print("\n" + "=" * 70)
print("  DYNAMIC FIRE-RESCUE SIMULATION")
print("=" * 70)

results = {}
for name, kw in strategies.items():
    t0 = time.perf_counter()
    res = simulate(name, **kw)
    res["time_ms"] = (time.perf_counter() - t0) * 1000
    results[name] = res

# =============================================
# 6. CONSOLE REPORT
# =============================================
hdr = (f"{'Strategy':<28} {'OK':>4} {'Steps':>6} {'Replans':>8} "
       f"{'Heat':>10} {'Explored':>10} {'Time(ms)':>10}")
sep = "-" * len(hdr)
print(sep)
print(hdr)
print(sep)
for name, d in results.items():
    ok = "Y" if d["success"] else "N"
    print(f"{name:<28} {ok:>4} {d['steps']:>6} {d['replans']:>8} "
          f"{d['heat']:>10.0f} {d['explored_total']:>10} {d['time_ms']:>10.2f}")
print(sep + "\n")


# =============================================
# 7. VISUALIZATION -- 4-panel comparison
# =============================================
heat_cmap = mcolors.LinearSegmentedColormap.from_list(
    "fire", ["#1a1a2e", "#e94560", "#f5a623", "#f7e733"])

fig, axes = plt.subplots(2, 2, figsize=(15, 14), facecolor="#0f0f1a")
fig.suptitle("Dynamic Fire-Rescue  --  Trajectories & Final Fire State",
             fontsize=16, fontweight="bold", color="white", y=0.97)

pcol = {
    "Dijkstra (static)":      "#00e5ff",
    "A* (static)":            "#76ff03",
    "Risk-Aware A* (static)": "#ff9100",
    "Dynamic Risk-Aware":     "#e040fb",
}

for ax, (name, d) in zip(axes.flat, results.items()):
    ax.set_facecolor("#0f0f1a")

    # show FINAL temperature snapshot
    final_temp = d["temp_snaps"][-1].astype(float).copy()
    final_temp[walls] = np.nan
    ax.imshow(final_temp, cmap=heat_cmap, origin="upper",
              vmin=0, vmax=MAX_TEMP, interpolation="bilinear", alpha=0.85)

    # walls
    wr, wc = np.where(walls)
    ax.scatter(wc, wr, marker="s", s=120, color="#2d2d44",
               edgecolors="#6c6c8a", linewidths=0.6, zorder=3)

    # trajectory
    traj = np.array(d["trajectory"])
    ax.plot(traj[:,1], traj[:,0], color=pcol[name], linewidth=2.4,
            marker="o", markersize=3, markerfacecolor=pcol[name],
            markeredgecolor="white", markeredgewidth=0.3, zorder=5,
            alpha=0.9)

    # start & goal
    ax.plot(*START[::-1], marker="^", markersize=12, color="#00e676",
            markeredgecolor="white", markeredgewidth=1, zorder=6)
    ax.plot(*GOAL[::-1], marker="*", markersize=14, color="#ff1744",
            markeredgecolor="white", markeredgewidth=1, zorder=6)

    ok = "Y" if d["success"] else "N"
    ax.set_title(f"{name}  [{ok}]", fontsize=12, fontweight="bold",
                 color="white", pad=8)
    ax.text(0.5, -0.07,
            f"Steps: {d['steps']}  |  Heat: {d['heat']:.0f}  |  "
            f"Explored: {d['explored_total']}  |  Replans: {d['replans']}",
            transform=ax.transAxes, ha="center", fontsize=8,
            color="#b0bec5", fontstyle="italic")

    ax.set_xlim(-0.5, GRID_SIZE-0.5)
    ax.set_ylim(GRID_SIZE-0.5, -0.5)
    ax.set_xticks(range(0, GRID_SIZE, 5))
    ax.set_yticks(range(0, GRID_SIZE, 5))
    ax.tick_params(colors="#555555", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#333355")

# colorbar
cax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
dummy = axes[0,0].imshow(np.zeros((1,1)), cmap=heat_cmap, vmin=0, vmax=MAX_TEMP)
cb = fig.colorbar(dummy, cax=cax)
cb.set_label("Temperature", color="white", fontsize=10)
cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=8)

fig.legend(handles=[
    Patch(facecolor="#2d2d44", edgecolor="#6c6c8a", label="Wall"),
    Patch(facecolor="#00e676", label="Start (0,0)"),
    Patch(facecolor="#ff1744", label="Goal (19,19) -- Fire"),
    Patch(facecolor="#00e5ff", label="Dijkstra"),
    Patch(facecolor="#76ff03", label="A*"),
    Patch(facecolor="#ff9100", label="Risk-Aware A*"),
    Patch(facecolor="#e040fb", label="Dynamic RA*"),
], loc="lower center", ncol=7, fontsize=8,
    frameon=False, labelcolor="white", handlelength=1.5)

plt.subplots_adjust(left=0.04, right=0.91, top=0.91, bottom=0.07,
                    wspace=0.12, hspace=0.22)
save_path = (r"c:\Users\Arav Kilak\OneDrive\Documents\2nd Semester\MDP"
             r"\fire_rescue_dynamic_comparison.png")
plt.savefig(save_path, dpi=180, facecolor=fig.get_facecolor(),
            bbox_inches="tight")
plt.show()
print(f"Static comparison saved: {save_path}")


# =============================================
# 8. ANIMATED VISUALIZATION (Dynamic Risk-Aware)
# =============================================
print("\nGenerating animation for Dynamic Risk-Aware strategy...")

dyn = results["Dynamic Risk-Aware"]
n_frames = min(dyn["steps"] + 1, len(dyn["temp_snaps"]))

fig2, ax2 = plt.subplots(figsize=(8, 8), facecolor="#0f0f1a")
ax2.set_facecolor("#0f0f1a")
ax2.set_xlim(-0.5, GRID_SIZE-0.5)
ax2.set_ylim(GRID_SIZE-0.5, -0.5)
ax2.set_xticks(range(0, GRID_SIZE, 5))
ax2.set_yticks(range(0, GRID_SIZE, 5))
ax2.tick_params(colors="#555555", labelsize=7)
for sp in ax2.spines.values():
    sp.set_color("#333355")

title_txt = ax2.set_title("", fontsize=14, fontweight="bold",
                           color="white", pad=12)
info_txt  = ax2.text(0.5, -0.04, "", transform=ax2.transAxes,
                      ha="center", fontsize=9, color="#b0bec5",
                      fontstyle="italic")

# persistent elements
wr, wc = np.where(walls)
ax2.scatter(wc, wr, marker="s", s=180, color="#2d2d44",
            edgecolors="#6c6c8a", linewidths=0.8, zorder=3)
ax2.plot(*START[::-1], marker="^", markersize=14, color="#00e676",
         markeredgecolor="white", markeredgewidth=1.2, zorder=8)
ax2.plot(*GOAL[::-1], marker="*", markersize=16, color="#ff1744",
         markeredgecolor="white", markeredgewidth=1.2, zorder=8)

# elements that update each frame
im = ax2.imshow(np.zeros((GRID_SIZE, GRID_SIZE)), cmap=heat_cmap,
                origin="upper", vmin=0, vmax=MAX_TEMP,
                interpolation="bilinear", alpha=0.85, zorder=1)
trail_line, = ax2.plot([], [], color="#e040fb", linewidth=2.2,
                        alpha=0.7, zorder=5)
plan_line,  = ax2.plot([], [], color="#ffffff", linewidth=1.2,
                        linestyle="--", alpha=0.4, zorder=4)
robot_dot,  = ax2.plot([], [], marker="D", markersize=10,
                        color="#00e5ff", markeredgecolor="white",
                        markeredgewidth=1.5, zorder=9)

cax2 = fig2.add_axes([0.92, 0.15, 0.015, 0.65])
cb2  = fig2.colorbar(im, cax=cax2)
cb2.set_label("Temperature", color="white", fontsize=10)
cb2.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=8)


def animate(frame):
    # temperature
    tmp = dyn["temp_snaps"][frame].astype(float).copy()
    tmp[walls] = np.nan
    im.set_data(tmp)

    # robot position
    pos = dyn["trajectory"][frame]
    robot_dot.set_data([pos[1]], [pos[0]])

    # trail so far
    trail = np.array(dyn["trajectory"][:frame+1])
    trail_line.set_data(trail[:,1], trail[:,0])

    # planned path at this step
    if frame < len(dyn["path_snaps"]):
        plan = np.array(dyn["path_snaps"][frame])
        plan_line.set_data(plan[:,1], plan[:,0])

    # accum heat
    heat_so_far = sum(dyn["temp_snaps"][t][dyn["trajectory"][t][0],
                       dyn["trajectory"][t][1]]
                      for t in range(1, frame+1)) if frame > 0 else 0

    title_txt.set_text(f"Dynamic Risk-Aware A*  --  Step {frame}/{dyn['steps']}")
    info_txt.set_text(f"Position: {pos}  |  Accumulated Heat: {heat_so_far:.0f}")

    return im, robot_dot, trail_line, plan_line, title_txt, info_txt


anim = animation.FuncAnimation(fig2, animate, frames=n_frames,
                                interval=120, blit=False, repeat=False)
anim_path = (r"c:\Users\Arav Kilak\OneDrive\Documents\2nd Semester\MDP"
             r"\fire_rescue_dynamic_animation.gif")
anim.save(anim_path, writer="pillow", fps=8, dpi=120)
plt.show()
print(f"Animation saved: {anim_path}")
print("\nDone! All outputs generated.")
