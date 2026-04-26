# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
FINAL COMBINED VALIDATION — Indoor Pathfinding Stress Test
============================================================
50x50 realistic partitioned office building with a 'No Free Lunch'
multi-hazard thermal map.

ARCHITECTURE
  Outer perimeter walls (all four sides).
  Continuous perimeter hallway (cols 1-2 left, cols 47-48 right,
  row 1-2 top, row 47-48 bottom).
  Inner building shell with doorways connecting hallway to offices.
  Enclosed office rooms, narrow doorway chokepoints.
  Central lobby zone (rows 17-32) with Class A fire.

HEAT CLASSES
  Class A (450-500 degC): Central lobby — organic radial fire.
                          Blocks the geometrically shortest path.
  Class B (200-300 degC): Blob in the LEFT perimeter stairwell (rows 17-32)
                          and a secondary blob in the upper-right back offices.
                          Blocks the 'obvious' perimeter corridor detour.
  Spot fires (80-120 degC): Scattered in lower-left and upper-right rooms.

NO FREE LUNCH: Every structural route carries some thermal load.

EXPECTED ALGORITHM BEHAVIOUR
  Dijkstra, A*        ->  straight through Class A fire (shortest hop-count)
  Weighted A*, GBFS   ->  enter Class A lobby via heuristic pull
  Risk-Aware A*       ->  calculates 1+alpha*T composite cost and navigates
                          the winding office corridor route, accepting spot-fire
                          exposure to avoid Class A and Class B zones entirely.
"""

import heapq, itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — CONSTANTS  (sourced from complex_env)
# ══════════════════════════════════════════════════════════════════════════════
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from complex_env import (
    N, START, GOAL, ALPHA, W_WEIGHT,
    LOBBY_R_LO, LOBBY_R_HI, LOBBY_C_LO, LOBBY_C_HI,
    CLASS_B, SPOTS,
    get_environment,
)

OUT_DIR = Path(__file__).parent

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — BUILD
# ══════════════════════════════════════════════════════════════════════════════
print('Building complex 50x50 environment...')
temperature, walls = get_environment()
print(f'  Walls: {walls.sum()} cells  |  Passable: {(~walls).sum()} cells')
print(f'  Rooms separated by {walls[4:46, :].sum()} internal wall cells')
print()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def neighbors(node):
    r, c = node
    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
        nr, nc = r+dr, c+dc
        if 0 <= nr < N and 0 <= nc < N and not walls[nr, nc]:
            yield (nr, nc)

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def reconstruct(came_from, node):
    path = [node]
    while node in came_from:
        node = came_from[node]
        path.append(node)
    path.reverse();  return path

def path_metrics(path, tgrid):
    if path is None: return 0, float('inf'), float('inf')
    heats = [tgrid[r, c] for r, c in path]
    return len(path), sum(heats), max(heats)

def passes_through_lobby(path):
    if path is None: return False
    return any(LOBBY_R_LO <= r <= LOBBY_R_HI and
               LOBBY_C_LO <= c <= LOBBY_C_HI for r, c in path)

def classify_route(path):
    if path is None: return 'NO PATH'
    
    mid_cols = [c for r, c in path if 18 <= r <= 32]
    avg_c = sum(mid_cols) / len(mid_cols) if mid_cols else START[1]
    
    if avg_c < 5:
        return 'FAR LEFT STAIR (Class C ambient)'
    elif avg_c <= 18:
        return 'INNER LEFT HALL (Class B core)'
    elif avg_c <= 30:
        return 'MAIN LOBBY (Class A core)'
    elif avg_c <= 35:
        return 'INNER RIGHT HALL (Class B spread)'
    else:
        return 'EAST CONFERENCE (Class C ambient)'


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ALGORITHMS  (self-contained, unmodified logic)
# ══════════════════════════════════════════════════════════════════════════════
def dijkstra(start, goal):
    cnt = itertools.count()
    heap = [(0, next(cnt), start)]
    g = {start: 0};  par = {};  vis = set()
    while heap:
        cost, _, u = heapq.heappop(heap)
        if u in vis: continue
        vis.add(u)
        if u == goal: return reconstruct(par, u), vis
        for v in neighbors(u):
            nc = g[u] + 1
            if v not in g or nc < g[v]:
                g[v] = nc;  par[v] = u
                heapq.heappush(heap, (nc, next(cnt), v))
    return None, vis

def a_star(start, goal):
    cnt = itertools.count()
    heap = [(manhattan(start, goal), 0, next(cnt), start)]
    g = {start: 0};  par = {};  vis = set()
    while heap:
        _, ng, _, u = heapq.heappop(heap)
        if u in vis: continue
        vis.add(u)
        if u == goal: return reconstruct(par, u), vis
        for v in neighbors(u):
            nc = g[u] + 1
            if v not in g or nc < g[v]:
                g[v] = nc;  par[v] = u
                heapq.heappush(heap, (nc + manhattan(v, goal), -nc, next(cnt), v))
    return None, vis

def weighted_a_star(start, goal, w=W_WEIGHT):
    cnt = itertools.count()
    heap = [(w*manhattan(start, goal), 0, next(cnt), start)]
    g = {start: 0};  par = {};  vis = set()
    while heap:
        _, ng, _, u = heapq.heappop(heap)
        if u in vis: continue
        vis.add(u)
        if u == goal: return reconstruct(par, u), vis
        for v in neighbors(u):
            nc = g[u] + 1
            if v not in g or nc < g[v]:
                g[v] = nc;  par[v] = u
                heapq.heappush(heap, (nc + w*manhattan(v, goal), -nc, next(cnt), v))
    return None, vis

def gbfs(start, goal):
    cnt = itertools.count()
    heap = [(manhattan(start, goal), next(cnt), start)]
    par = {};  vis = set()
    while heap:
        _, _, u = heapq.heappop(heap)
        if u in vis: continue
        vis.add(u)
        if u == goal: return reconstruct(par, u), vis
        for v in neighbors(u):
            if v not in vis:
                par[v] = u
                heapq.heappush(heap, (manhattan(v, goal), next(cnt), v))
    return None, vis

def risk_aware_a_star(start, goal, alpha=ALPHA, temp_grid=None):
    if temp_grid is None: temp_grid = temperature
    cnt = itertools.count()
    heap = [(manhattan(start, goal), 0, next(cnt), start)]
    g = {start: 0};  par = {};  vis = set()
    while heap:
        _, ng, _, u = heapq.heappop(heap)
        if u in vis: continue
        vis.add(u)
        if u == goal: return reconstruct(par, u), vis
        for v in neighbors(u):
            step = 1 + alpha * temp_grid[v[0], v[1]]
            nc   = g[u] + step
            if v not in g or nc < g[v]:
                g[v] = nc;  par[v] = u
                heapq.heappush(heap, (nc + manhattan(v, goal), -nc, next(cnt), v))
    return None, vis


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — RUN
# ══════════════════════════════════════════════════════════════════════════════
ALGORITHMS = {
    'Dijkstra':          dijkstra,
    'A*':                a_star,
    'Weighted A* (W=2)': weighted_a_star,
    'GBFS':              gbfs,
    'Risk-Aware A*':     risk_aware_a_star,
}

results = {}
for name, fn in ALGORITHMS.items():
    path, explored = fn(START, GOAL)
    steps, cum, peak = path_metrics(path, temperature)
    results[name] = dict(path=path, explored=explored, steps=steps,
                         nodes=len(explored), cum_heat=cum,
                         peak_heat=peak, in_lobby=passes_through_lobby(path),
                         route=classify_route(path))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
SEP = '=' * 105

print(f'\n{SEP}')
print('  COMPLEX INDOOR FLOORPLAN — No Free Lunch Multi-Hazard Pathfinding')
print(f'  Start: {START}   Goal: {GOAL}   Grid: {N}x{N}   alpha={ALPHA}')
print(f'  Class A (lobby)  : rows {LOBBY_R_LO}-{LOBBY_R_HI}, cols {LOBBY_C_LO}-{LOBBY_C_HI}  [450-500 degC]')
print(f'  Class B (blobs)  : {len(CLASS_B)} blobs — left stairwell (rows 16-33) + upper-right offices  [200-280 degC]')
print(f'  Spot fires       : {len(SPOTS)} organic blooms scattered through partitioned rooms            [ 70-130 degC]')
print(SEP)
HDR = (f"{'Algorithm':<22} {'Steps':>6} {'Explored':>9} "
       f"{'Cum.Heat(degC)':>15} {'Peak(degC)':>11}  Route")
print(HDR)
print('-' * 105)
for name, d in results.items():
    print(f"{name:<22} {d['steps']:>6} {d['nodes']:>9} "
          f"{d['cum_heat']:>15.1f} {d['peak_heat']:>11.0f}  {d['route']}")
print(SEP)

print('\n  ROUTING VERDICT')
print('-' * 105)
raa   = results['Risk-Aware A*']
astar = results['A*']
for name, d in results.items():
    if d['in_lobby']:
        s = f"  [FAIL] {name:<22} Peak {d['peak_heat']:.0f} degC -- Class A traversal!"
    else:
        s = f"  [PASS] {name:<22} Peak {d['peak_heat']:.0f} degC -- {d['steps']} steps"
    print(s)

print()
if not raa['in_lobby'] and astar['in_lobby']:
    saved = astar['cum_heat'] - raa['cum_heat']
    pct   = saved / astar['cum_heat'] * 100
    extra = raa['steps'] - astar['steps']
    print(f"  Risk-Aware A* avoids all Class A fire.  Route: {raa['route']}")
    print(f"  >> Thermal saving vs A*  : {saved:,.0f} degC ({pct:.1f}% reduction)")
    print(f"  >> Path length overhead  : {extra:+d} steps")
elif raa['in_lobby']:
    print('  WARNING: Risk-Aware A* entered Class A zone. Increase ALPHA.')
print(SEP + '\n')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — VISUALIZATION  (2x3 subplot grid)
# ══════════════════════════════════════════════════════════════════════════════
HEAT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'fire_final', ['#070712', '#151530', '#8b1212', '#d44000', '#f0d800'])

PATH_COLORS = {
    'Dijkstra':          '#00e5ff',
    'A*':                '#76ff03',
    'Weighted A* (W=2)': '#e040fb',
    'GBFS':              '#ffd740',
    'Risk-Aware A*':     '#ff9100',
}

fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='#06060e')
fig.suptitle(
    'Final Combined Validation  |  No Free Lunch Multi-Hazard  |  50x50 Office Building',
    fontsize=12, fontweight='bold', color='white', y=0.998)

algo_list = list(results.items())
for idx, (name, d) in enumerate(algo_list):
    ri, ci = divmod(idx, 3)
    ax = axes[ri, ci]
    ax.set_facecolor('#06060e')

    # Heat map
    tmp = temperature.astype(float).copy();  tmp[walls] = np.nan
    im = ax.imshow(tmp, cmap=HEAT_CMAP, origin='upper',
                   vmin=0, vmax=500, interpolation='bilinear', alpha=0.84)

    # Walls
    wr, wc = np.where(walls)
    ax.scatter(wc, wr, marker='s', s=26, color='#181828',
               edgecolors='#303055', linewidths=0.4, zorder=3)

    # Class A lobby outline
    ax.add_patch(plt.Rectangle(
        (LOBBY_C_LO-0.5, LOBBY_R_LO-0.5),
        LOBBY_C_HI-LOBBY_C_LO+1, LOBBY_R_HI-LOBBY_R_LO+1,
        lw=1.5, edgecolor='#ff1133', facecolor='none',
        ls='--', zorder=4, alpha=0.9))

    # Class B blobs — soft amber ellipses
    for (br, bc, bpk, bhr, bhc) in CLASS_B:
        ax.add_patch(mpatches.Ellipse(
            (bc, br), width=bhc*3, height=bhr*2.5,
            lw=1.2, edgecolor='#ffaa00', facecolor='none',
            ls=':', zorder=4, alpha=0.75))

    # Explored nodes
    if d['explored']:
        e = np.array(list(d['explored']))
        ax.scatter(e[:,1], e[:,0], marker='.', s=4,
                   color='white', alpha=0.13, zorder=4)

    # Path
    color = PATH_COLORS[name]
    if d['path']:
        p = np.array(d['path'])
        ax.plot(p[:,1], p[:,0], color=color, lw=2.2,
                marker='o', markersize=1.7, markerfacecolor=color,
                markeredgecolor='white', markeredgewidth=0.1, zorder=6)

    # Start / goal markers
    ax.plot(START[1], START[0], marker='^', ms=10, color='#00e676',
            markeredgecolor='white', mew=0.7, zorder=7)
    ax.plot(GOAL[1],  GOAL[0],  marker='*', ms=13, color='#ff1744',
            markeredgecolor='white', mew=0.7, zorder=7)

    verdict = '[CLASS A]' if d['in_lobby'] else '[DETOUR]'
    ax.set_title(f'{name}  {verdict}', fontsize=9, fontweight='bold',
                 color=color, pad=5)
    ax.text(0.5, -0.075,
            f"Steps:{d['steps']}  Explored:{d['nodes']}  "
            f"Cum:{d['cum_heat']:.0f}C  Peak:{d['peak_heat']:.0f}C",
            transform=ax.transAxes, ha='center', fontsize=6.8, color='#8090a8')

    ax.set_xlim(-0.5, N-0.5);  ax.set_ylim(N-0.5, -0.5)
    ax.set_xticks(range(0, N, 10));  ax.set_yticks(range(0, N, 10))
    ax.tick_params(colors='#2a2a50', labelsize=5.5)
    for sp in ax.spines.values(): sp.set_color('#12122a')

# ── Legend panel (axes[1,2]) ──────────────────────────────────────────────────
ax_leg = axes[1, 2]
ax_leg.set_facecolor('#06060e');  ax_leg.axis('off')

handles = [
    mpatches.Patch(fc='#181828', ec='#303055',            label='Wall'),
    mpatches.Patch(fc='white',   alpha=0.15,              label='Explored'),
    mpatches.Patch(fc='#00e676',                          label=f'Start {START}'),
    mpatches.Patch(fc='#ff1744',                          label=f'Goal  {GOAL}'),
    mpatches.Patch(fc='none', ec='#ff1133', ls='--',     label='Class A lobby'),
    mpatches.Patch(fc='none', ec='#ffaa00', ls=':',      label='Class B blob'),
]
for n, c in PATH_COLORS.items():
    handles.append(mpatches.Patch(fc=c, label=n))

ax_leg.legend(handles=handles, loc='upper center', fontsize=7.8,
              frameon=True, facecolor='#0c0c1e', edgecolor='#222244',
              labelcolor='white', handlelength=1.5, handleheight=1.1)

rows_txt = []
for name, d in results.items():
    tag = '[HOT]' if d['in_lobby'] else '[ok] '
    rows_txt.append(
        f"{name:<22} {d['steps']:>4}  {d['cum_heat']:>8.0f}C  "
        f"{d['peak_heat']:>5.0f}C  {tag}")
ax_leg.text(
    0.5, 0.10,
    'Algorithm              Steps  Cum.Heat   Peak   Route\n'
    + '-'*55 + '\n' + '\n'.join(rows_txt),
    transform=ax_leg.transAxes, ha='center', va='center',
    fontsize=5.6, color='#8090a8', family='monospace',
    bbox=dict(boxstyle='round,pad=0.5', fc='#08081a', ec='#181835'))

# Colorbar
cax = fig.add_axes([0.936, 0.13, 0.010, 0.75])
cb  = fig.colorbar(im, cax=cax)
cb.set_label('Temperature (degC)', color='white', fontsize=8)
cb.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=6)

plt.subplots_adjust(left=0.03, right=0.93, top=0.95, bottom=0.07,
                    wspace=0.22, hspace=0.33)

out_path = OUT_DIR / 'indoor_floorplan_comparison.png'
plt.savefig(out_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
print(f'Plot saved: {out_path}')
plt.show()
