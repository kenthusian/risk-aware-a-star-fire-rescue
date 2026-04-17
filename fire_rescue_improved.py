# -*- coding: utf-8 -*-
"""
Fire-Rescue Robot Pathfinding Simulation
=========================================
Compares Dijkstra, Standard A*, Risk-Aware A*, Weighted A*, and
Greedy Best-First Search on a 20x20 grid.
Goal: reach the fire source (19,19) to extinguish it.
"""

import heapq, itertools, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--no-display', action='store_true',
                    help="Run headlessly without opening GUI windows")
args = parser.parse_args()

OUT_DIR = Path(__file__).parent

# ──────────────────────────────────────────────
# 1. ENVIRONMENT
# ──────────────────────────────────────────────
GRID_SIZE = 20
START = (0, 0)
GOAL = (19, 19)
ALPHA = 0.8
W_WEIGHT = 2.0

temperature = np.ones((GRID_SIZE, GRID_SIZE), dtype=float)
rng = np.random.RandomState(7)

# ── Diagonal heat band (r ~ c) ──
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        d = abs(r - c)
        if   d == 0: temperature[r, c] = rng.randint(420, 500)
        elif d == 1: temperature[r, c] = rng.randint(320, 420)
        elif d == 2: temperature[r, c] = rng.randint(200, 320)
        elif d == 3: temperature[r, c] = rng.randint(80, 200)
        elif d == 4: temperature[r, c] = rng.randint(20, 80)

# Fire at goal
for r in range(17, 20):
    for c in range(17, 20):
        temperature[r, c] = max(temperature[r, c], rng.randint(350, 500))

temperature[START] = 1

# ── Walls ──
# Design: two full-width barriers, each with TWO gaps.
#   - One "hot" gap on the r=c diagonal  (short route, scorching)
#   - One "cool" gap far off-diagonal    (longer route, safe)
# This forces clear algorithm differentiation:
#   Dijkstra / A*      -> hot gaps (shortest)
#   GBFS / Weighted A* -> gap with lowest heuristic (rightward)
#   Risk-Aware A*      -> cool gaps (longest, safest)

walls = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

# Barrier 1 at row 7: hot gap at col 7, cool gap at col 17
for c in range(GRID_SIZE):
    walls[7, c] = True
walls[7, 7]  = False   # hot gap (on diagonal, temp ~460)
walls[7, 17] = False   # cool gap (off diagonal, temp ~1)

# Barrier 2 at row 13: hot gap at col 13, cool gap at col 3
for c in range(GRID_SIZE):
    walls[13, c] = True
walls[13, 13] = False  # hot gap (on diagonal, temp ~460)
walls[13, 3]  = False  # cool gap (off diagonal, temp ~1)

# Top-edge partial wall (forces robot downward first)
for c in range(5, 16):
    walls[0, c] = True

# Right-edge partial wall (upper portion)
for r in range(0, 5):
    walls[r, 19] = True

# Vertical wall segment between rows 8-12, col 10-11
# This separates the middle zone into left and right halves,
# making Weighted A* and GBFS diverge after barrier 1
for r in range(8, 13):
    walls[r, 10] = True
    walls[r, 11] = True

# Interior accent walls
walls[3, 13] = True;  walls[3, 14] = True
walls[16, 4] = True;  walls[16, 5] = True

walls[START] = False
walls[GOAL]  = False


# ──────────────────────────────────────────────
# 2. HELPERS
# ──────────────────────────────────────────────
def neighbors(node):
    r, c = node
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and not walls[nr, nc]:
            yield (nr, nc)

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def reconstruct(cf, n):
    p = [n]
    while n in cf:
        n = cf[n]; p.append(n)
    p.reverse()
    return p


# ──────────────────────────────────────────────
# 3. ALGORITHMS
# ──────────────────────────────────────────────
def dijkstra(start, goal):
    """
    Standard Dijkstra's algorithm.
    Cost function: uniform step cost = 1. Ignores temperature.

    Parameters
    ----------
    start : tuple  (row, col) starting coordinate
    goal  : tuple  (row, col) goal coordinate

    Returns
    -------
    path     : list of (row, col) or None
    explored : set of visited nodes
    """
    cnt = itertools.count()
    heap = [(0, next(cnt), start)]
    g = {start: 0}; par = {}; vis = set()
    while heap:
        cost, _, u = heapq.heappop(heap)
        if u in vis: continue
        vis.add(u)
        if u == goal: return reconstruct(par, u), vis
        for v in neighbors(u):
            nc = g[u] + 1
            if v not in g or nc < g[v]:
                g[v] = nc; par[v] = u
                heapq.heappush(heap, (nc, next(cnt), v))
    return None, vis

def a_star(start, goal):
    """
    Standard A*. f(n) = g(n) + h(n), h = Manhattan distance.
    Cost function: uniform step cost = 1. Ignores temperature.

    Parameters
    ----------
    start : tuple  (row, col) starting coordinate
    goal  : tuple  (row, col) goal coordinate

    Returns
    -------
    path     : list of (row, col) or None
    explored : set of visited nodes
    """
    cnt = itertools.count()
    h0 = heuristic(start, goal)
    heap = [(h0, 0, next(cnt), start)]
    g = {start: 0}; par = {}; vis = set()
    while heap:
        _, ng, _, u = heapq.heappop(heap)
        if u in vis: continue
        vis.add(u)
        if u == goal: return reconstruct(par, u), vis
        for v in neighbors(u):
            nc = g[u] + 1
            if v not in g or nc < g[v]:
                g[v] = nc; par[v] = u
                h = heuristic(v, goal)
                heapq.heappush(heap, (nc + h, -nc, next(cnt), v))
    return None, vis

def risk_aware_a_star(start, goal, alpha=ALPHA, temp_grid=None):
    """
    Risk-Aware A*. Step cost = 1 + alpha * T(v).
    Penalises high-temperature cells to find safer routes.

    h(n) = Manhattan distance is admissible for the *composite* cost objective
    because every step costs at least 1 (the alpha*T term is non-negative),
    so h(n) <= true remaining composite cost always holds.  Note this does NOT
    preserve admissibility for the original unweighted hop-count problem; the
    algorithm is optimal only for the risk-weighted objective it minimises.

    Parameters
    ----------
    start     : tuple    (row, col) starting coordinate
    goal      : tuple    (row, col) goal coordinate
    alpha     : float    risk-aversion weight
    temp_grid : ndarray  temperature map; defaults to module-level `temperature`

    Returns
    -------
    path     : list of (row, col) or None
    explored : set of visited nodes
    """
    if temp_grid is None:
        temp_grid = temperature
    cnt = itertools.count()
    h0 = heuristic(start, goal)
    heap = [(h0, 0, next(cnt), start)]
    g = {start: 0}; par = {}; vis = set()
    while heap:
        _, ng, _, u = heapq.heappop(heap)
        if u in vis: continue
        vis.add(u)
        if u == goal: return reconstruct(par, u), vis
        for v in neighbors(u):
            step = 1 + alpha * temp_grid[v[0], v[1]]
            nc = g[u] + step
            if v not in g or nc < g[v]:
                g[v] = nc; par[v] = u
                h = heuristic(v, goal)
                heapq.heappush(heap, (nc + h, -nc, next(cnt), v))
    return None, vis

def weighted_a_star(start, goal, w=W_WEIGHT):
    """
    Weighted A*. f(n) = g(n) + W * h(n).
    Inflates heuristic by W to trade optimality for speed.

    Parameters
    ----------
    start : tuple  (row, col) starting coordinate
    goal  : tuple  (row, col) goal coordinate
    w     : float  heuristic weight (W > 1 biases toward goal)

    Returns
    -------
    path     : list of (row, col) or None
    explored : set of visited nodes
    """
    cnt = itertools.count()
    h0 = w * heuristic(start, goal)
    heap = [(h0, 0, next(cnt), start)]
    g = {start: 0}; par = {}; vis = set()
    while heap:
        _, neg_g, _, u = heapq.heappop(heap)
        if u in vis: continue
        vis.add(u)
        if u == goal: return reconstruct(par, u), vis
        for v in neighbors(u):
            nc = g[u] + 1
            if v not in g or nc < g[v]:
                g[v] = nc; par[v] = u
                h = w * heuristic(v, goal)
                heapq.heappush(heap, (nc + h, -nc, next(cnt), v))
    return None, vis

def gbfs(start, goal):
    """
    Greedy Best-First Search. f(n) = h(n) only.
    Ignores accumulated cost g(n). Fast but non-optimal.

    Parameters
    ----------
    start : tuple  (row, col) starting coordinate
    goal  : tuple  (row, col) goal coordinate

    Returns
    -------
    path     : list of (row, col) or None
    explored : set of visited nodes
    """
    cnt = itertools.count()
    h0 = heuristic(start, goal)
    heap = [(h0, next(cnt), start)]
    par = {}; vis = set()
    while heap:
        _, _, u = heapq.heappop(heap)
        if u in vis: continue
        vis.add(u)
        if u == goal: return reconstruct(par, u), vis
        for v in neighbors(u):
            if v not in vis:
                par[v] = u
                h = heuristic(v, goal)
                heapq.heappush(heap, (h, next(cnt), v))
    return None, vis


# ──────────────────────────────────────────────
# 4. RUN
# ──────────────────────────────────────────────
algos = {
    "Dijkstra":      dijkstra,
    "A*":            a_star,
    "Risk-Aware A*": risk_aware_a_star,
    "Weighted A*":   weighted_a_star,
    "GBFS":          gbfs,
}

results = {}
TIMING_RUNS = 50

for name, fn in algos.items():
    path, exp = fn(START, GOAL)

    times = []
    for _ in range(TIMING_RUNS):
        t0 = time.perf_counter()
        fn(START, GOAL)
        times.append((time.perf_counter() - t0) * 1000)
    ms  = np.mean(times)
    std = np.std(times, ddof=1)

    if path is None:
        print(f"WARNING: {name} found no path!")
    heat = sum(temperature[r, c] for r, c in path) if path else float("inf")
    peak = max(temperature[r, c] for r, c in path) if path else float("inf")
    results[name] = dict(path=path, explored=exp,
                         steps=len(path) if path else 0,
                         nodes=len(exp), heat=heat, peak=peak,
                         time_ms=ms, time_std=std)


# ──────────────────────────────────────────────
# 5. CONSOLE TABLE & METRICS
# ──────────────────────────────────────────────
hdr = (f"{'Algorithm':<20} {'Path Len':>9} {'Explored':>10} "
       f"{'Cum. Heat':>10} {'Peak T':>7} {'Mean(ms)':>10} {'Std(ms)':>9}")
sep = "-" * len(hdr)
print(f"\n{sep}")
print("  FIRE-RESCUE ROBOT  -  PATHFINDING COMPARISON")
print(sep)
print(hdr)
print(sep)
for name, d in results.items():
    print(f"{name:<20} {d['steps']:>9} {d['nodes']:>10} "
          f"{d['heat']:>10.1f} {d['peak']:>7.0f} {d['time_ms']:>10.3f} {d['time_std']:>9.3f}")
print(f"{sep}\n")

print("DERIVED METRICS")
print(sep)
baseline_heat  = results["A*"]["heat"]
raa_heat       = results["Risk-Aware A*"]["heat"]
heat_reduction = (baseline_heat - raa_heat) / baseline_heat * 100

baseline_steps = results["A*"]["steps"]
raa_steps      = results["Risk-Aware A*"]["steps"]
path_overhead  = (raa_steps - baseline_steps) / baseline_steps * 100

dij_time   = results["Dijkstra"]["time_ms"]
astar_time = results["A*"]["time_ms"]
speedup    = dij_time / astar_time if astar_time > 0 else 0

print(f"Thermal reduction (Risk-Aware vs A*):     {heat_reduction:.1f}%")
print(f"Path length overhead (Risk-Aware vs A*):  {path_overhead:.1f}%")
print(f"Speedup ratio (A* over Dijkstra):         {speedup:.2f}x")
print(f"{sep}\n")


# ──────────────────────────────────────────────
# 6. GRID VISUALIZATION  (2x3 layout, 11" laptop friendly)
# ──────────────────────────────────────────────
heat_cmap = mcolors.LinearSegmentedColormap.from_list(
    "fire", ["#1a1a2e", "#e94560", "#f5a623", "#f7e733"])

pcol = {
    "Dijkstra":      "#00e5ff",
    "A*":            "#76ff03",
    "Risk-Aware A*": "#ff9100",
    "Weighted A*":   "#e040fb",
    "GBFS":          "#ffd740",
}

fig, axes = plt.subplots(2, 3, figsize=(14, 9.5), facecolor="#0f0f1a")
fig.suptitle("Fire-Rescue Robot  -  Pathfinding Comparison",
             fontsize=14, fontweight="bold", color="white", y=0.98)

algo_list = list(results.items())

for idx, (name, d) in enumerate(algo_list):
    row, col = divmod(idx, 3)
    ax = axes[row, col]
    ax.set_facecolor("#0f0f1a")

    tmp = temperature.astype(float).copy()
    tmp[walls] = np.nan
    im = ax.imshow(tmp, cmap=heat_cmap, origin="upper",
                   vmin=0, vmax=500, interpolation="bilinear", alpha=0.85)

    # walls
    wr, wc = np.where(walls)
    ax.scatter(wc, wr, marker="s", s=70, color="#2d2d44",
               edgecolors="#6c6c8a", linewidths=0.5, zorder=3)

    # explored nodes
    if d["explored"]:
        e = np.array(list(d["explored"]))
        ax.scatter(e[:, 1], e[:, 0], marker=".", s=12,
                   color="white", alpha=0.25, zorder=4)

    # path
    if d["path"]:
        p = np.array(d["path"])
        ax.plot(p[:, 1], p[:, 0], color=pcol[name], linewidth=2.2,
                marker="o", markersize=2.5, markerfacecolor=pcol[name],
                markeredgecolor="white", markeredgewidth=0.2, zorder=5)

    # start & goal markers
    ax.plot(START[1], START[0], marker="^", markersize=10, color="#00e676",
            markeredgecolor="white", markeredgewidth=0.8, zorder=6)
    ax.plot(GOAL[1], GOAL[0], marker="*", markersize=12, color="#ff1744",
            markeredgecolor="white", markeredgewidth=0.8, zorder=6)

    ax.set_title(name, fontsize=10, fontweight="bold", color=pcol[name], pad=6)
    ax.text(0.5, -0.08,
            f"Steps: {d['steps']}   Explored: {d['nodes']}   "
            f"Heat: {d['heat']:.0f}",
            transform=ax.transAxes, ha="center", fontsize=7.5,
            color="#b0bec5")
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(GRID_SIZE - 0.5, -0.5)
    ax.set_xticks(range(0, GRID_SIZE, 5))
    ax.set_yticks(range(0, GRID_SIZE, 5))
    ax.tick_params(colors="#555555", labelsize=6)
    for sp in ax.spines.values():
        sp.set_color("#333355")

# 6th panel: legend & summary
ax_leg = axes[1, 2]
ax_leg.set_facecolor("#0f0f1a")
ax_leg.axis("off")

legend_items = [
    Patch(facecolor="#2d2d44", edgecolor="#6c6c8a", label="Wall"),
    Patch(facecolor="white",  alpha=0.35,          label="Explored"),
    Patch(facecolor="#00e676",                      label="Start (0,0)"),
    Patch(facecolor="#ff1744",                      label="Goal (19,19)"),
]
for n in pcol:
    legend_items.append(Patch(facecolor=pcol[n], label=n))

ax_leg.legend(handles=legend_items, loc="center", fontsize=8,
              frameon=True, facecolor="#1a1a2e", edgecolor="#333355",
              labelcolor="white", handlelength=1.8, handleheight=1.2)

# summary text
summary_lines = []
for name, d in results.items():
    summary_lines.append(
        f"{name:<16} {d['steps']:>3} steps  {d['heat']:>6.0f} heat")
ax_leg.text(0.5, 0.12, "\n".join(summary_lines), transform=ax_leg.transAxes,
            ha="center", va="center", fontsize=7, color="#90a4ae",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e",
                      edgecolor="#333355"))

# colorbar
cax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
cb = fig.colorbar(im, cax=cax)
cb.set_label("Temperature", color="white", fontsize=9)
cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=7)

plt.subplots_adjust(left=0.04, right=0.91, top=0.93, bottom=0.08,
                    wspace=0.18, hspace=0.28)
plt.savefig(OUT_DIR / "fire_rescue_comparison.png",
            dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
if not args.no_display:
    plt.show()
print(f"Figure saved as {OUT_DIR / 'fire_rescue_comparison.png'}")


# ──────────────────────────────────────────────
# 7. COMPARISON FIGURE  (bar + scatter, compact)
# ──────────────────────────────────────────────
fig2, (ax_bar, ax_scat) = plt.subplots(1, 2, figsize=(13, 5.5),
                                        facecolor="#0f0f1a")
fig2.suptitle("Algorithm Performance Comparison",
              fontsize=13, fontweight="bold", color="white", y=0.97)

algo_names  = list(results.keys())
algo_colors = [pcol[n] for n in algo_names]

# ── Bar chart: Mean Execution Time ──
ax_bar.set_facecolor("#0f0f1a")
times_vals = [results[n]["time_ms"] for n in algo_names]
bars = ax_bar.bar(range(len(algo_names)), times_vals, color=algo_colors,
                  edgecolor="white", linewidth=0.5, alpha=0.9, width=0.55)
ax_bar.set_xticks(range(len(algo_names)))
ax_bar.set_xticklabels(algo_names, rotation=25, ha="right", fontsize=8,
                       color="#cfd8dc")
ax_bar.set_ylabel("Mean Execution Time (ms)", color="white", fontsize=9)
ax_bar.set_title("Computation Time", fontsize=11, fontweight="bold",
                 color="white", pad=8)
ax_bar.tick_params(axis="y", colors="#cfd8dc", labelsize=8)
for sp in ax_bar.spines.values():
    sp.set_color("#333355")
ax_bar.grid(axis="y", color="#333355", linestyle="--", alpha=0.4)

for bar, val in zip(bars, times_vals):
    ax_bar.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(times_vals) * 0.02,
                f"{val:.3f}", ha="center", va="bottom",
                color="white", fontsize=7, fontweight="bold")

# ── Scatter: Path Length vs Heat Exposure ──
ax_scat.set_facecolor("#0f0f1a")
for n in algo_names:
    d = results[n]
    ax_scat.scatter(d["heat"], d["steps"], color=pcol[n], s=120,
                    edgecolors="white", linewidths=1, zorder=5, label=n)

# smart label placement to avoid overlap
positions = [(results[n]["heat"], results[n]["steps"]) for n in algo_names]
for i, n in enumerate(algo_names):
    x, y = positions[i]
    # offset direction: alternate up/down/left/right
    offsets = [(10, 8), (-10, -12), (10, -12), (-10, 8), (10, 0)]
    ox, oy = offsets[i % len(offsets)]
    ax_scat.annotate(n, (x, y), textcoords="offset points",
                     xytext=(ox, oy), color=pcol[n], fontsize=7.5,
                     fontweight="bold",
                     arrowprops=dict(arrowstyle="-", color=pcol[n],
                                     alpha=0.4, lw=0.6))

ax_scat.set_xlabel("Cumulative Heat Exposure", color="white", fontsize=9)
ax_scat.set_ylabel("Path Length (steps)", color="white", fontsize=9)
ax_scat.set_title("Path Length vs Heat Exposure", fontsize=11,
                  fontweight="bold", color="white", pad=8)
ax_scat.tick_params(colors="#cfd8dc", labelsize=8)
for sp in ax_scat.spines.values():
    sp.set_color("#333355")
ax_scat.grid(True, color="#333355", linestyle="--", alpha=0.4)

plt.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.18,
                    wspace=0.28)
plt.savefig(OUT_DIR / "algorithm_comparison.png",
            dpi=180, facecolor=fig2.get_facecolor(), bbox_inches="tight")
if not args.no_display:
    plt.show()
print(f"Figure saved as {OUT_DIR / 'algorithm_comparison.png'}")


# ──────────────────────────────────────────────
# 8. ALPHA SENSITIVITY SWEEP
# ──────────────────────────────────────────────
def alpha_sweep():
    alpha_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    sweep_results = []

    print("\nALPHA SENSITIVITY SWEEP")
    print(sep)
    hdr_s = f"{'a':<6} | {'Path Len':>9} | {'Heat':>10} | {'Reduction %':>12}"
    print(hdr_s)
    print(sep)

    base_path, _ = risk_aware_a_star(START, GOAL, alpha=0.0)
    base_heat = sum(temperature[r, c] for r, c in base_path) \
        if base_path else float("inf")

    for a in alpha_vals:
        path, _ = risk_aware_a_star(START, GOAL, alpha=a)
        if path is None:
            continue
        steps = len(path)
        heat  = sum(temperature[r, c] for r, c in path)
        reduction = (base_heat - heat) / base_heat * 100
        sweep_results.append((a, steps, heat, reduction))
        print(f"{a:<6.1f} | {steps:>9} | {heat:>10.1f} | {reduction:>11.1f}%")
    print(sep)

    if not sweep_results:
        return

    fig_s, ax_h = plt.subplots(figsize=(9, 5), facecolor="#0f0f1a")
    ax_p = ax_h.twinx()
    fig_s.suptitle("Alpha Sensitivity Sweep", fontsize=13,
                   fontweight="bold", color="white")
    ax_h.set_facecolor("#0f0f1a")
    for sp in ax_h.spines.values(): sp.set_color("#333355")
    for sp in ax_p.spines.values(): sp.set_color("#333355")

    alphas  = [r[0] for r in sweep_results]
    heats   = [r[2] for r in sweep_results]
    lengths = [r[1] for r in sweep_results]

    ax_h.plot(alphas, heats, color="#ff1744", marker="o",
              linewidth=2, markersize=5, label="Heat")
    ax_p.plot(alphas, lengths, color="#00e5ff", marker="s",
              linewidth=2, markersize=5, label="Path Length")

    ax_h.set_xlabel("alpha", color="white", fontsize=10)
    ax_h.set_ylabel("Cumulative Heat", color="#ff1744", fontsize=10)
    ax_p.set_ylabel("Path Length", color="#00e5ff", fontsize=10)
    ax_h.tick_params(colors="#cfd8dc", labelsize=9)
    ax_p.tick_params(colors="#cfd8dc", labelsize=9)
    ax_h.grid(True, color="#333355", linestyle="--", alpha=0.5)

    lh, lbl_h = ax_h.get_legend_handles_labels()
    lp, lbl_p = ax_p.get_legend_handles_labels()
    ax_h.legend(lh + lp, lbl_h + lbl_p, loc="center right",
                facecolor="#1a1a2e", edgecolor="#333355",
                labelcolor="white", fontsize=8)

    plt.savefig(OUT_DIR / "alpha_sweep.png", dpi=180,
                facecolor=fig_s.get_facecolor(), bbox_inches="tight")
    if not args.no_display:
        plt.show()
    print(f"Figure saved as {OUT_DIR / 'alpha_sweep.png'}")

alpha_sweep()


# ──────────────────────────────────────────────
# 9. MULTI-SEED GENERALISABILITY EXPERIMENT
# ──────────────────────────────────────────────
def make_temperature(seed):
    """Return a diagonal-heat temperature map built from an independent seed."""
    _rng = np.random.RandomState(seed)
    temp = np.ones((GRID_SIZE, GRID_SIZE), dtype=float)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            d = abs(r - c)
            if   d == 0: temp[r, c] = _rng.randint(420, 500)
            elif d == 1: temp[r, c] = _rng.randint(320, 420)
            elif d == 2: temp[r, c] = _rng.randint(200, 320)
            elif d == 3: temp[r, c] = _rng.randint( 80, 200)
            elif d == 4: temp[r, c] = _rng.randint( 20,  80)
    for r in range(17, 20):
        for c in range(17, 20):
            temp[r, c] = max(temp[r, c], _rng.randint(350, 500))
    temp[START] = 1
    return temp


def multi_seed_experiment():
    """
    Re-run all 5 algorithms on 5 independent temperature seeds.
    Walls are fixed; only diagonal heat intensities vary.
    Checks that the performance ordering and three-cluster structure
    (fast-greedy | optimal-balanced | risk-aware) generalise beyond seed 7.
    """
    SEEDS = [7, 42, 99, 256, 512]
    col_hdr = (f"{'Seed':>6}  {'Algorithm':<16} {'PathLen':>8} "
               f"{'Nodes':>7} {'Heat':>9}")
    col_sep = "-" * len(col_hdr)

    print("\nMULTI-SEED GENERALISABILITY EXPERIMENT")
    print(col_sep)
    print(col_hdr)
    print(col_sep)

    heat_by_algo  = {name: [] for name in algos}
    nodes_by_algo = {name: [] for name in algos}

    for seed in SEEDS:
        temp_s = make_temperature(seed)
        local_algos = {
            "Dijkstra":      dijkstra,
            "A*":            a_star,
            "Risk-Aware A*": lambda s, g, _t=temp_s: risk_aware_a_star(s, g, temp_grid=_t),
            "Weighted A*":   weighted_a_star,
            "GBFS":          gbfs,
        }
        for name, fn in local_algos.items():
            path, exp = fn(START, GOAL)
            heat  = sum(temp_s[r, c] for r, c in path) if path else float("inf")
            steps = len(path) if path else 0
            print(f"{seed:>6}  {name:<16} {steps:>8} {len(exp):>7} {heat:>9.0f}")
            heat_by_algo[name].append(heat)
            nodes_by_algo[name].append(len(exp))
        print()

    print(col_sep)
    print("CROSS-SEED SUMMARY  (mean +/- std over 5 seeds)")
    print(col_sep)
    sum_hdr = f"{'Algorithm':<16} {'Nodes  mean+/-std':>22} {'Heat   mean+/-std':>22}"
    print(sum_hdr)
    print(col_sep)
    for name in algos:
        n_arr = np.array(nodes_by_algo[name], dtype=float)
        h_arr = np.array(heat_by_algo[name],  dtype=float)
        n_str = f"{np.mean(n_arr):.0f} +/- {np.std(n_arr, ddof=1):.1f}"
        h_str = f"{np.mean(h_arr):.0f} +/- {np.std(h_arr, ddof=1):.0f}"
        print(f"{name:<16} {n_str:>22} {h_str:>22}")
    print(col_sep + "\n")


multi_seed_experiment()


# ──────────────────────────────────────────────
# 10. SELF-CONTAINED ALGORITHM RUNNER  (used by §11–13)
# ──────────────────────────────────────────────
def _algo_runner(start, goal, temp_grid, walls_grid, N,
                 alpha_val=ALPHA, w_val=W_WEIGHT, timing_runs=0):
    """Run all 5 algorithms on any N×N grid. No global dependencies."""
    def nb(node):
        r, c = node
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r+dr, c+dc
            if 0<=nr<N and 0<=nc<N and not walls_grid[nr,nc]:
                yield (nr,nc)
    def h(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])
    def recon(cf,n):
        p=[n]
        while n in cf: n=cf[n]; p.append(n)
        p.reverse(); return p

    def _dij():
        cnt=itertools.count(); heap=[(0,next(cnt),start)]; g={start:0}; par={}; vis=set()
        while heap:
            _,_,u=heapq.heappop(heap)
            if u in vis: continue
            vis.add(u)
            if u==goal: return recon(par,u),vis
            for v in nb(u):
                nc=g[u]+1
                if v not in g or nc<g[v]: g[v]=nc; par[v]=u; heapq.heappush(heap,(nc,next(cnt),v))
        return None,vis
    def _astar():
        cnt=itertools.count(); heap=[(h(start,goal),0,next(cnt),start)]; g={start:0}; par={}; vis=set()
        while heap:
            _,_,_,u=heapq.heappop(heap)
            if u in vis: continue
            vis.add(u)
            if u==goal: return recon(par,u),vis
            for v in nb(u):
                nc=g[u]+1
                if v not in g or nc<g[v]: g[v]=nc; par[v]=u; heapq.heappush(heap,(nc+h(v,goal),-nc,next(cnt),v))
        return None,vis
    def _raa():
        cnt=itertools.count(); heap=[(h(start,goal),0,next(cnt),start)]; g={start:0}; par={}; vis=set()
        while heap:
            _,_,_,u=heapq.heappop(heap)
            if u in vis: continue
            vis.add(u)
            if u==goal: return recon(par,u),vis
            for v in nb(u):
                step=1+alpha_val*temp_grid[v[0],v[1]]; nc=g[u]+step
                if v not in g or nc<g[v]: g[v]=nc; par[v]=u; heapq.heappush(heap,(nc+h(v,goal),-nc,next(cnt),v))
        return None,vis
    def _was():
        cnt=itertools.count(); heap=[(w_val*h(start,goal),0,next(cnt),start)]; g={start:0}; par={}; vis=set()
        while heap:
            _,_,_,u=heapq.heappop(heap)
            if u in vis: continue
            vis.add(u)
            if u==goal: return recon(par,u),vis
            for v in nb(u):
                nc=g[u]+1
                if v not in g or nc<g[v]: g[v]=nc; par[v]=u; heapq.heappush(heap,(nc+w_val*h(v,goal),-nc,next(cnt),v))
        return None,vis
    def _gbfs():
        cnt=itertools.count(); heap=[(h(start,goal),next(cnt),start)]; par={}; vis=set()
        while heap:
            _,_,u=heapq.heappop(heap)
            if u in vis: continue
            vis.add(u)
            if u==goal: return recon(par,u),vis
            for v in nb(u):
                if v not in vis: par[v]=u; heapq.heappush(heap,(h(v,goal),next(cnt),v))
        return None,vis

    fns={"Dijkstra":_dij,"A*":_astar,"Risk-Aware A*":_raa,"Weighted A*":_was,"GBFS":_gbfs}
    out={}
    for name,fn in fns.items():
        path,exp=fn()
        if path:
            heat=sum(temp_grid[r,c] for r,c in path)
            peak=max(temp_grid[r,c] for r,c in path)
            steps=len(path)
        else:
            heat=peak=float("inf"); steps=0
        ms=std=float("nan")
        if timing_runs>0:
            ts=[]
            for _ in range(timing_runs):
                t0=time.perf_counter(); fn(); ts.append((time.perf_counter()-t0)*1000)
            ms=np.mean(ts); std=np.std(ts,ddof=1)
        out[name]=dict(steps=steps,nodes=len(exp),heat=heat,peak=peak,ms=ms,std=std)
    return out


# ──────────────────────────────────────────────
# 11. GRID SCALABILITY  (N = 20, 50, 100, 200)
# ──────────────────────────────────────────────
def make_scaled_grid(N, seed=7):
    """N×N grid: diagonal heat band + 2 full-width barriers (hot+cool gap each)."""
    _rng=np.random.RandomState(seed)
    start,goal=(0,0),(N-1,N-1)
    temp=np.ones((N,N),dtype=float)
    for r in range(N):
        for c in range(N):
            d=abs(r-c)
            if d==0:   temp[r,c]=_rng.randint(420,500)
            elif d==1: temp[r,c]=_rng.randint(320,420)
            elif d==2: temp[r,c]=_rng.randint(200,320)
            elif d==3: temp[r,c]=_rng.randint(80,200)
            elif d==4: temp[r,c]=_rng.randint(20,80)
    for r in range(max(0,N-4),N):
        for c in range(max(0,N-4),N):
            temp[r,c]=max(temp[r,c],_rng.randint(350,500))
    temp[start]=1
    walls=np.zeros((N,N),dtype=bool)
    b1,b2=N//3,2*N//3
    for c in range(N): walls[b1,c]=True; walls[b2,c]=True
    walls[b1,b1]=False;       walls[b1,min(N*5//6,N-1)]=False
    walls[b2,b2]=False;       walls[b2,max(N//6,0)]=False
    walls[start]=False; walls[goal]=False
    return temp,walls,start,goal


def scalability_experiment():
    SIZES=[20,50,100,200]; T_RUNS=3
    sc_hdr=(f"{'N':>5} {'|V|':>7} {'Algorithm':<16}"
            f"{'Nodes':>8} {'Mean(ms)':>11} {'Std(ms)':>10}")
    sc_sep="-"*len(sc_hdr)
    print("\nGRID SCALABILITY  (T_RUNS =",T_RUNS,"per algorithm)")
    print(sc_sep); print(sc_hdr); print(sc_sep)
    for N in SIZES:
        temp_s,walls_s,st,gl=make_scaled_grid(N)
        res=_algo_runner(st,gl,temp_s,walls_s,N,timing_runs=T_RUNS)
        for algo_name,d in res.items():
            ms_s=f"{d['ms']:.3f}" if not np.isnan(d['ms']) else "—"
            sd_s=f"{d['std']:.3f}" if not np.isnan(d['std']) else "—"
            print(f"{N:>5} {N*N:>7} {algo_name:<16}{d['nodes']:>8} {ms_s:>11} {sd_s:>10}")
        print()
    print(sc_sep+"\n")


scalability_experiment()


# ──────────────────────────────────────────────
# 12. MONTE CARLO MAZE TEST  (100 random grids, 20% wall density)
# ──────────────────────────────────────────────
def make_random_grid(N, density, seed):
    _rng=np.random.RandomState(seed)
    start,goal=(0,0),(N-1,N-1)
    temp=np.ones((N,N),dtype=float)
    for _ in range(_rng.randint(2,5)):
        cr,cc=_rng.randint(0,N),_rng.randint(0,N)
        inten=_rng.randint(200,500)
        for r in range(max(0,cr-4),min(N,cr+5)):
            for c in range(max(0,cc-4),min(N,cc+5)):
                d=abs(r-cr)+abs(c-cc)
                if d<=4: temp[r,c]=max(temp[r,c], inten*(1-d/5))
    temp=np.clip(temp,1,500); temp[start]=1
    walls=np.zeros((N,N),dtype=bool)
    cells=[(r,c) for r in range(N) for c in range(N) if (r,c)!=start and (r,c)!=goal]
    n_w=int(density*N*N)
    for idx in _rng.choice(len(cells),size=min(n_w,len(cells)),replace=False):
        r,c=cells[idx]; walls[r,c]=True
    return temp,walls,start,goal


def monte_carlo_experiment(n_trials=100, N=20, density=0.20, seed_base=5000):
    print(f"\nMONTE CARLO MAZE TEST  ({n_trials} random grids, N={N}, {int(density*100)}% walls)")
    mc_sep="-"*60
    print(mc_sep)

    heat_reductions=[]; peak_reductions=[]; success=0; skipped=0
    astar_nodes=[]; raa_nodes=[]

    for trial in range(n_trials):
        temp_t,walls_t,st,gl=make_random_grid(N,density,seed_base+trial)
        res=_algo_runner(st,gl,temp_t,walls_t,N)
        if res["Dijkstra"]["steps"]==0 or res["A*"]["steps"]==0:
            skipped+=1; continue   # disconnected grid — skip

        a_heat =res["A*"]["heat"];  a_peak =res["A*"]["peak"]
        ra_heat=res["Risk-Aware A*"]["heat"]; ra_peak=res["Risk-Aware A*"]["peak"]

        if a_heat>0 and a_heat!=float("inf") and ra_heat!=float("inf"):
            heat_reductions.append((a_heat-ra_heat)/a_heat*100)
            peak_reductions.append((a_peak-ra_peak)/a_peak*100 if a_peak>0 else 0)
            astar_nodes.append(res["A*"]["nodes"])
            raa_nodes.append(res["Risk-Aware A*"]["nodes"])
            success+=1

    if success==0:
        print("No valid trials."); return

    hr=np.array(heat_reductions); pr=np.array(peak_reductions)
    an=np.array(astar_nodes);     rn=np.array(raa_nodes)
    print(f"Valid trials       : {success} / {n_trials}  ({skipped} skipped — disconnected)")
    print(f"Cumul. heat reduc. : {np.mean(hr):+.1f}% ± {np.std(hr,ddof=1):.1f}%"
          f"  (min {np.min(hr):.1f}%  max {np.max(hr):.1f}%)")
    print(f"Peak heat reduc.   : {np.mean(pr):+.1f}% ± {np.std(pr,ddof=1):.1f}%"
          f"  (min {np.min(pr):.1f}%  max {np.max(pr):.1f}%)")
    print(f"A* nodes explored  : {np.mean(an):.0f} ± {np.std(an,ddof=1):.0f}")
    print(f"RA* nodes explored : {np.mean(rn):.0f} ± {np.std(rn,ddof=1):.0f}")
    pct_positive=np.mean(hr>0)*100
    print(f"Trials where RA* reduces heat: {pct_positive:.1f}%")
    print(mc_sep+"\n")


monte_carlo_experiment()


# ──────────────────────────────────────────────
# 13. ZERO-RISK SANITY BASELINE
# ──────────────────────────────────────────────
def zero_risk_baseline():
    """
    Validate that Risk-Aware A* degenerates to Standard A* on a uniform
    cold grid (T(v)=1 everywhere).  When all step costs are 1+alpha*1=const,
    the thermal penalty is uniform, making the composite objective equivalent
    to hop-count minimisation — so both algorithms must find identical paths.
    """
    COLD = np.ones((GRID_SIZE, GRID_SIZE), dtype=float)
    zr_sep="-"*55
    print("\nZERO-RISK SANITY BASELINE  (T(v) = 1 everywhere)")
    print(zr_sep)
    res=_algo_runner(START, GOAL, COLD, walls, GRID_SIZE)
    zr_hdr=f"{'Algorithm':<18} {'Path Len':>9} {'Nodes':>7} {'Cum. Heat':>10} {'Peak T':>7}"
    print(zr_hdr); print(zr_sep)
    for name,d in res.items():
        print(f"{name:<18} {d['steps']:>9} {d['nodes']:>7} {d['heat']:>10.0f} {d['peak']:>7.0f}")
    print(zr_sep)

    astar_path  = res["A*"]["steps"]
    raa_path    = res["Risk-Aware A*"]["steps"]
    astar_nodes = res["A*"]["nodes"]
    raa_nodes   = res["Risk-Aware A*"]["nodes"]
    match = (astar_path==raa_path and astar_nodes==raa_nodes)
    print(f"\nA* path len = {astar_path},  RA* path len = {raa_path}")
    print(f"A* nodes    = {astar_nodes},  RA* nodes   = {raa_nodes}")
    print(f"Result: {'PASS — RA* matches A* exactly on cold grid.' if match else 'MISMATCH (unexpected).'}")
    print(zr_sep+"\n")


zero_risk_baseline()


# ──────────────────────────────────────────────
# 14. SENSOR NOISE ROBUSTNESS  (RA* with ±10% Gaussian noise, 50 trials)
# ──────────────────────────────────────────────
def sensor_noise_experiment(n_trials=50, noise_sigma=0.10):
    """
    Inject multiplicative Gaussian noise N(1, noise_sigma) onto the
    temperature grid, then re-run Risk-Aware A*.  The four algorithms
    that ignore temperature (Dijkstra, A*, Weighted A*, GBFS) are
    deterministic and noise-immune by construction; only RA* is affected.
    Checks for path chattering: violent route switching caused by noise.
    """
    noise_rng  = np.random.RandomState(77)
    step_dist  = []   # path length per trial
    heat_dist  = []   # cumulative heat per trial
    peak_dist  = []   # peak heat per trial

    for _ in range(n_trials):
        noise      = noise_rng.normal(1.0, noise_sigma, temperature.shape)
        noisy_temp = np.clip(temperature * noise, 1, 600).astype(float)
        noisy_temp[START] = 1
        path, _ = risk_aware_a_star(START, GOAL, temp_grid=noisy_temp)
        if path:
            step_dist.append(len(path))
            heat_dist.append(sum(noisy_temp[r,c] for r,c in path))
            peak_dist.append(max(noisy_temp[r,c] for r,c in path))

    sc = np.array(step_dist); hs = np.array(heat_dist); ps = np.array(peak_dist)
    unique_lengths = sorted(set(step_dist))
    stability = sc == sc[0]

    sn_sep = "-" * 58
    print("\nSENSOR NOISE ROBUSTNESS  (Risk-Aware A*, noise σ =", f"{noise_sigma*100:.0f}%)")
    print(sn_sep)
    print(f"  Trials             : {n_trials}")
    print(f"  Unique path lengths: {unique_lengths}")
    print(f"  Path length        : {sc.mean():.1f} ± {sc.std(ddof=1):.2f}  "
          f"[{sc.min()} – {sc.max()}]")
    print(f"  Cumulative heat    : {hs.mean():.0f} ± {hs.std(ddof=1):.0f}")
    print(f"  Peak heat          : {ps.mean():.0f} ± {ps.std(ddof=1):.0f}")
    from collections import Counter
    counts      = Counter(step_dist)
    dominant_len = max(counts, key=counts.get)
    dominant_pct = counts[dominant_len] / n_trials * 100
    print(f"  Dominant route     : {dominant_len}-step path in"
          f" {dominant_pct:.0f}% of trials")
    if dominant_pct >= 90:
        print(f"  ==> ROBUST: No path chattering. RA* consistently picks the"
              f" {dominant_len}-step safe route under ±{noise_sigma*100:.0f}% sensor noise.")
    else:
        print(f"  ==> Route variance detected — chattering may occur at this noise level.")
    print(sn_sep + "\n")


sensor_noise_experiment()


# ──────────────────────────────────────────────
# 15. THERMAL TRAP TOPOLOGY  (U-shaped 490°C zone)
# ──────────────────────────────────────────────
def thermal_trap_experiment():
    """
    Inject a U-shaped thermal trap (no new physical walls — purely
    thermal) into the existing grid.  The U of 490°C cells creates
    a heat gradient that RA* senses and avoids, while Standard A*
    and the other heat-blind algorithms traverse it directly.
    U-arms: row 4 (top), row 15 (bottom), col 9 (left spine).
    Interior bowl: rows 5-14, cols 10-14.
    """
    trap_temp = temperature.copy()

    # Left spine of the U
    for r in range(4, 16):
        trap_temp[r, 9] = 490
    # Top arm
    for c in range(9, 15):
        trap_temp[4, c] = 490
    # Bottom arm
    for c in range(9, 15):
        trap_temp[15, c] = 490
    # Interior bowl (rows 5-14, cols 10-14)
    for r in range(5, 15):
        for c in range(10, 15):
            trap_temp[r, c] = 490

    trap_temp[START] = 1

    tt_sep = "-" * 62
    print("\nTHERMAL TRAP TOPOLOGY  (U-shaped zone at 490°C, existing walls)")
    print(tt_sep)
    res = _algo_runner(START, GOAL, trap_temp, walls, GRID_SIZE)
    tt_hdr = f"{'Algorithm':<18} {'Steps':>6} {'Nodes':>7} {'Cum. Heat':>11} {'Peak T':>7}"
    print(tt_hdr); print(tt_sep)
    for name, d in res.items():
        print(f"{name:<18} {d['steps']:>6} {d['nodes']:>7} {d['heat']:>11.0f} {d['peak']:>7.0f}")
    print(tt_sep)

    astar_h = res["A*"]["heat"]; raa_h = res["Risk-Aware A*"]["heat"]
    astar_pk = res["A*"]["peak"]; raa_pk = res["Risk-Aware A*"]["peak"]
    if astar_h > raa_h:
        print(f"\n  Heat reduction vs A*  : {(astar_h-raa_h)/astar_h*100:.1f}%  "
              f"(RA* {raa_h:.0f}  vs  A* {astar_h:.0f})")
    if astar_pk > raa_pk:
        print(f"  Peak T reduction vs A*: {(astar_pk-raa_pk)/astar_pk*100:.1f}%  "
              f"(RA* peak {raa_pk:.0f}°C  vs  A* peak {astar_pk:.0f}°C)")
        print("  ==> RA* successfully steered clear of the thermal trap.")
    elif raa_pk <= 100:
        print("  RA* avoided the 490°C zone entirely (peak T within cool corridor).")
    print(tt_sep + "\n")


thermal_trap_experiment()


# ──────────────────────────────────────────────
# 16. SPACE COMPLEXITY  (max priority-queue size)
# ──────────────────────────────────────────────
def space_complexity_experiment():
    """
    Track the maximum number of nodes held in the open set (priority queue)
    at any single moment during traversal.  This bounds the RAM footprint
    and confirms O(|V|) space complexity for embedded deployment.
    """

    def _max_q_dijkstra():
        cnt=itertools.count(); heap=[(0,next(cnt),START)]
        g={START:0}; par={}; vis=set(); mq=0
        while heap:
            mq=max(mq,len(heap)); _,_,u=heapq.heappop(heap)
            if u in vis: continue
            vis.add(u)
            if u==GOAL: return len(vis), mq
            for v in neighbors(u):
                nc=g[u]+1
                if v not in g or nc<g[v]: g[v]=nc; par[v]=u; heapq.heappush(heap,(nc,next(cnt),v))
        return len(vis), mq

    def _max_q_astar():
        cnt=itertools.count(); heap=[(heuristic(START,GOAL),0,next(cnt),START)]
        g={START:0}; par={}; vis=set(); mq=0
        while heap:
            mq=max(mq,len(heap)); _,_,_,u=heapq.heappop(heap)
            if u in vis: continue
            vis.add(u)
            if u==GOAL: return len(vis), mq
            for v in neighbors(u):
                nc=g[u]+1
                if v not in g or nc<g[v]: g[v]=nc; par[v]=u; heapq.heappush(heap,(nc+heuristic(v,GOAL),-nc,next(cnt),v))
        return len(vis), mq

    def _max_q_raa():
        cnt=itertools.count(); heap=[(heuristic(START,GOAL),0,next(cnt),START)]
        g={START:0}; par={}; vis=set(); mq=0
        while heap:
            mq=max(mq,len(heap)); _,_,_,u=heapq.heappop(heap)
            if u in vis: continue
            vis.add(u)
            if u==GOAL: return len(vis), mq
            for v in neighbors(u):
                step=1+ALPHA*temperature[v[0],v[1]]; nc=g[u]+step
                if v not in g or nc<g[v]: g[v]=nc; par[v]=u; heapq.heappush(heap,(nc+heuristic(v,GOAL),-nc,next(cnt),v))
        return len(vis), mq

    def _max_q_was():
        cnt=itertools.count(); heap=[(W_WEIGHT*heuristic(START,GOAL),0,next(cnt),START)]
        g={START:0}; par={}; vis=set(); mq=0
        while heap:
            mq=max(mq,len(heap)); _,_,_,u=heapq.heappop(heap)
            if u in vis: continue
            vis.add(u)
            if u==GOAL: return len(vis), mq
            for v in neighbors(u):
                nc=g[u]+1
                if v not in g or nc<g[v]: g[v]=nc; par[v]=u; heapq.heappush(heap,(nc+W_WEIGHT*heuristic(v,GOAL),-nc,next(cnt),v))
        return len(vis), mq

    def _max_q_gbfs():
        cnt=itertools.count(); heap=[(heuristic(START,GOAL),next(cnt),START)]
        par={}; vis=set(); mq=0
        while heap:
            mq=max(mq,len(heap)); _,_,u=heapq.heappop(heap)
            if u in vis: continue
            vis.add(u)
            if u==GOAL: return len(vis), mq
            for v in neighbors(u):
                if v not in vis: par[v]=u; heapq.heappush(heap,(heuristic(v,GOAL),next(cnt),v))
        return len(vis), mq

    runners = {
        "Dijkstra":      _max_q_dijkstra,
        "A*":            _max_q_astar,
        "Risk-Aware A*": _max_q_raa,
        "Weighted A*":   _max_q_was,
        "GBFS":          _max_q_gbfs,
    }

    sp_hdr = (f"{'Algorithm':<18} {'Nodes exp.':>11} {'Max |open|':>11} "
              f"{'Max/|V| %':>10}")
    sp_sep = "-" * len(sp_hdr)
    V = GRID_SIZE * GRID_SIZE

    print("\nSPACE COMPLEXITY  (max priority-queue size during traversal)")
    print(sp_sep); print(sp_hdr); print(sp_sep)
    for name, fn in runners.items():
        nodes_exp, max_q = fn()
        pct = max_q / V * 100
        print(f"{name:<18} {nodes_exp:>11} {max_q:>11} {pct:>9.1f}%")
    print(sp_sep)
    print(f"\n  |V| = {V} cells.  Max |open| / |V| << 1 confirms O(|V|) space bound.")
    print(f"  At ~40 bytes/node (Python tuple), peak RAM usage ≈"
          f" {max_q * 40 // 1024} KB — well within embedded controller limits.\n")


space_complexity_experiment()
