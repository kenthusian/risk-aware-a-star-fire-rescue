# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
robustness_validation.py
========================
Three robustness experiments for the Modified Risk-Aware A* algorithm:

  EXP-1  Sensor Noise Injection
         Simulate smoke-occluded IR sensors by applying +/-10% Gaussian
         noise to the perceived temperature map.  RA* plans on the noisy
         percept but performance is measured on the ground-truth grid.
         Proves the planner does not catastrophically reroute through
         Class A fire when a cell reads 430 instead of 460 degC.

  EXP-2  Grid Resolution Scaling
         Re-run the full 5-algorithm comparison at N=50, 100, and 200.
         Confirms that the O(1) per-step thermal penalty keeps planning
         time within the 50-200 ms real-time replanning budget.

  EXP-3  Variable Alpha (Mission Priority)
         Dynamic alpha ranging from 0.2 (life-rescue, fast / hot) to
         1.0 (recon, slow / cold).  Visualises how the routing decision
         and thermal exposure change continuously with alpha.
"""

import heapq, itertools, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from complex_env import (
    N as N_BASE, START as START_50, GOAL as GOAL_50, ALPHA, W_WEIGHT,
    LOBBY_R_LO, LOBBY_R_HI, LOBBY_C_LO, LOBBY_C_HI,
    CLASS_B, SPOTS,
    get_environment, build_env_scaled,
)

OUT_DIR = Path(__file__).parent
HEAT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'fire', ['#070712','#151530','#8b1212','#d44000','#f0d800'])


# (Environment builders moved to complex_env.py)

# ══════════════════════════════════════════════════════════════════════════════
# ALGORITHMS  (generic — work on any grid by receiving walls/N from closure)
# ══════════════════════════════════════════════════════════════════════════════
def _nbrs(node, N, walls):
    r, c = node
    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
        nr, nc = r+dr, c+dc
        if 0 <= nr < N and 0 <= nc < N and not walls[nr,nc]:
            yield (nr, nc)

def _mh(a, b): return abs(a[0]-b[0])+abs(a[1]-b[1])

def _recon(cf, node):
    p=[node]
    while node in cf: node=cf[node]; p.append(node)
    p.reverse(); return p

def run_dijkstra(start, goal, N, walls):
    cnt=itertools.count(); heap=[(0,next(cnt),start)]
    g={start:0}; par={}; vis=set()
    while heap:
        cost,_,u=heapq.heappop(heap)
        if u in vis: continue
        vis.add(u)
        if u==goal: return _recon(par,u), vis
        for v in _nbrs(u,N,walls):
            nc=g[u]+1
            if v not in g or nc<g[v]:
                g[v]=nc; par[v]=u; heapq.heappush(heap,(nc,next(cnt),v))
    return None, vis

def run_a_star(start, goal, N, walls):
    cnt=itertools.count(); heap=[((_mh(start,goal)),0,next(cnt),start)]
    g={start:0}; par={}; vis=set()
    while heap:
        _,ng,_,u=heapq.heappop(heap)
        if u in vis: continue
        vis.add(u)
        if u==goal: return _recon(par,u), vis
        for v in _nbrs(u,N,walls):
            nc=g[u]+1
            if v not in g or nc<g[v]:
                g[v]=nc; par[v]=u
                heapq.heappush(heap,(nc+_mh(v,goal),-nc,next(cnt),v))
    return None, vis

def run_wstar(start, goal, N, walls, w=W_WEIGHT):
    cnt=itertools.count(); heap=[(w*_mh(start,goal),0,next(cnt),start)]
    g={start:0}; par={}; vis=set()
    while heap:
        _,ng,_,u=heapq.heappop(heap)
        if u in vis: continue
        vis.add(u)
        if u==goal: return _recon(par,u), vis
        for v in _nbrs(u,N,walls):
            nc=g[u]+1
            if v not in g or nc<g[v]:
                g[v]=nc; par[v]=u
                heapq.heappush(heap,(nc+w*_mh(v,goal),-nc,next(cnt),v))
    return None, vis

def run_gbfs(start, goal, N, walls):
    cnt=itertools.count(); heap=[(_mh(start,goal),next(cnt),start)]
    par={}; vis=set()
    while heap:
        _,_,u=heapq.heappop(heap)
        if u in vis: continue
        vis.add(u)
        if u==goal: return _recon(par,u), vis
        for v in _nbrs(u,N,walls):
            if v not in vis:
                par[v]=u; heapq.heappush(heap,(_mh(v,goal),next(cnt),v))
    return None, vis

def run_raa(start, goal, N, walls, temp_grid, alpha=ALPHA):
    cnt=itertools.count(); heap=[(_mh(start,goal),0,next(cnt),start)]
    g={start:0}; par={}; vis=set()
    while heap:
        _,ng,_,u=heapq.heappop(heap)
        if u in vis: continue
        vis.add(u)
        if u==goal: return _recon(par,u), vis
        for v in _nbrs(u,N,walls):
            step=1+alpha*temp_grid[v[0],v[1]]
            nc=g[u]+step
            if v not in g or nc<g[v]:
                g[v]=nc; par[v]=u
                heapq.heappush(heap,(nc+_mh(v,goal),-nc,next(cnt),v))
    return None, vis

def path_metrics(path, tgrid):
    if path is None: return 0, float('inf'), float('inf')
    heats=[tgrid[r,c] for r,c in path]
    return len(path), sum(heats), max(heats)

def in_lobby(path, lr_lo, lr_hi, lc_lo, lc_hi):
    # Only map fails if it crosses the actual destructive Class A zone (center cols)
    # The inner corridors (cols < 20 or cols > 30) are successful detours.
    # Note: lc_lo, lc_hi passed in are now the strict bounds, see EXP-2 scaling.
    return path is not None and any(lr_lo<=r<=lr_hi and lc_lo<=c<=lc_hi for r,c in path)


# ══════════════════════════════════════════════════════════════════════════════
# BUILD BASE 50x50 ENV  (complex partitioned office from complex_env.py)
# ══════════════════════════════════════════════════════════════════════════════
print('Building complex 50x50 environment...')
temperature_50, walls_50 = get_environment()
# Alias constants to match 50-suffix naming used in experiments below
LOBBY_R_LO_50, LOBBY_R_HI_50 = 16, 33
LOBBY_C_LO_50, LOBBY_C_HI_50 = 20, 30
print('Done.\n')


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — SENSOR NOISE INJECTION
# ══════════════════════════════════════════════════════════════════════════════
print("="*75)
print("  EXP-1: SENSOR NOISE INJECTION  (RA* alpha=0.8, +/-10% Gaussian)")
print("="*75)
print(f"{'Trial':>6} {'Noise RNG':>10} {'Steps':>7} {'Cum.Heat':>10} "
      f"{'Peak':>7} {'Lobby?':>8} {'Status':>8}")
print("-"*75)

NOISE_SEEDS = [0, 7, 13, 21, 37, 42, 55, 63, 77, 99]
NOISE_LEVEL  = 0.10
noise_records = []

for seed in NOISE_SEEDS:
    rng_n  = np.random.RandomState(seed)
    noise  = 1.0 + NOISE_LEVEL * rng_n.randn(*temperature_50.shape)
    t_noisy = np.clip(temperature_50 * noise, 1, 700)   # allow brief spikes

    path, _ = run_raa(START_50, GOAL_50, N_BASE, walls_50,
                      temp_grid=t_noisy, alpha=ALPHA)
    steps, cum, peak = path_metrics(path, temperature_50)   # ground-truth cost
    lobby = in_lobby(path, LOBBY_R_LO_50, LOBBY_R_HI_50,
                     LOBBY_C_LO_50, LOBBY_C_HI_50)
    status = "FAIL" if lobby else "PASS"
    noise_records.append(dict(seed=seed, path=path, steps=steps,
                              cum_heat=cum, peak_heat=peak, lobby=lobby))
    print(f"{seed:>6}       {seed:>6}  {steps:>7}   {cum:>9.1f}  "
          f"{peak:>7.0f}  {'YES' if lobby else 'no':>8}  {status:>8}")

passed = sum(1 for r in noise_records if not r['lobby'])
print("-"*75)
print(f"  Result: {passed}/{len(NOISE_SEEDS)} trials routed AWAY from Class A fire")
print(f"  Max peak temp on ground-truth map across all trials: "
      f"{max(r['peak_heat'] for r in noise_records):.0f} degC")
print()

# ── Figure 1: Noise trials ────────────────────────────────────────────────────
fig1, axes1 = plt.subplots(2, 5, figsize=(20, 8), facecolor='#06060e')
fig1.suptitle(
    f'EXP-1: Sensor Noise Robustness  |  RA* alpha={ALPHA}  '
    f'|  +/-{int(NOISE_LEVEL*100)}% Gaussian noise  (10 trials)',
    fontsize=11, fontweight='bold', color='white', y=0.998)

for idx, rec in enumerate(noise_records):
    ax = axes1.flat[idx]
    ax.set_facecolor('#06060e')
    tmp = temperature_50.astype(float).copy();  tmp[walls_50] = np.nan
    ax.imshow(tmp, cmap=HEAT_CMAP, origin='upper', vmin=0, vmax=500,
              interpolation='bilinear', alpha=0.82)
    wr, wc = np.where(walls_50)
    ax.scatter(wc, wr, marker='s', s=18, color='#181828',
               edgecolors='#303055', lw=0.4, zorder=3)
    ax.add_patch(plt.Rectangle(
        (LOBBY_C_LO_50-0.5, LOBBY_R_LO_50-0.5),
        LOBBY_C_HI_50-LOBBY_C_LO_50+1, LOBBY_R_HI_50-LOBBY_R_LO_50+1,
        lw=1.3, edgecolor='#ff1133', facecolor='none', ls='--', zorder=4))
    if rec['path']:
        p = np.array(rec['path'])
        c = '#00ff88' if not rec['lobby'] else '#ff4444'
        ax.plot(p[:,1], p[:,0], color=c, lw=2.0, zorder=6)
    ax.plot(START_50[1], START_50[0], marker='^', ms=8,
            color='#00e676', mew=0.6, markeredgecolor='white', zorder=7)
    ax.plot(GOAL_50[1],  GOAL_50[0],  marker='*', ms=11,
            color='#ff1744', mew=0.6, markeredgecolor='white', zorder=7)
    tag = 'PASS' if not rec['lobby'] else 'FAIL'
    ax.set_title(f"Seed {rec['seed']}  [{tag}]  Peak:{rec['peak_heat']:.0f}C",
                 fontsize=7.5, color='#00ff88' if not rec['lobby'] else '#ff4444',
                 fontweight='bold', pad=3)
    ax.set_xlim(-0.5,49.5);  ax.set_ylim(49.5,-0.5)
    ax.set_xticks([]);  ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color('#12122a')

plt.tight_layout(rect=[0,0,1,0.97])
fig1.savefig(OUT_DIR / 'exp1_sensor_noise.png', dpi=160,
             facecolor=fig1.get_facecolor(), bbox_inches='tight')
print(f"  Figure saved: exp1_sensor_noise.png")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — GRID RESOLUTION SCALING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*75)
print("  EXP-2: GRID RESOLUTION SCALING  (timing in ms, 5 runs each)")
print("="*75)

ALGO_NAMES = ['Dijkstra', 'A*', 'Weighted A* (W=2)', 'GBFS', 'Risk-Aware A*']
ALGO_FNS   = [run_dijkstra, run_a_star, run_wstar, run_gbfs, run_raa]
SCALES     = [50, 100, 200]
N_RUNS     = 5

scale_results = {}   # {N: {algo_name: {steps, time_ms}}}

for res in SCALES:
    print(f"  Building {res}x{res} environment... ", end='', flush=True)
    if res == 50:
        w_scaled, t_scaled = walls_50, temperature_50
        st, go = START_50, GOAL_50
    else:
        st, go, w_scaled, t_scaled = build_env_scaled(res)
    lr_lo = int(16*(res/50));  lr_hi = int(33*(res/50))
    lc_lo = int(20*(res/50));  lc_hi = int(30*(res/50))
    print(f"done  (lobby rows {lr_lo}-{lr_hi}, cols {lc_lo}-{lc_hi})")
    scale_results[res] = {}

    HDR = f"  {'Algorithm':<22} {'Steps':>6} {'Lobby?':>8} {'Time (ms)':>11}"
    print(HDR);  print("  " + "-"*56)
    algos = list(zip(ALGO_NAMES, ALGO_FNS))
    for algo_name, f in algos:
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            if algo_name == 'Risk-Aware A*':
                path, _ = f(st, go, res, w_scaled, temp_grid=t_scaled, alpha=ALPHA)
            else:
                path, _ = f(st, go, res, w_scaled)
            t1 = time.perf_counter()
            times.append((t1-t0)*1000)
        
        steps = len(path)-1 if path else 0
        lobby = in_lobby(path, lr_lo, lr_hi, lc_lo, lc_hi)
        med_ms = float(np.median(times))
        scale_results[res][algo_name] = dict(steps=steps, time_ms=med_ms, lobby=lobby)
        print(f"  {algo_name:<22} {steps:>6}  {'YES' if lobby else 'no':>8} "
              f"  {med_ms:>8.2f} ms")

# ── Figure 2: Timing comparison ───────────────────────────────────────────────
fig2, (ax_bar, ax_tbl) = plt.subplots(1, 2, figsize=(14, 6), facecolor='#06060e')
fig2.suptitle('EXP-2: Grid Resolution Scaling  |  Median planning time (5 runs)',
              fontsize=11, fontweight='bold', color='white', y=1.01)

algo_colors = ['#00e5ff','#76ff03','#e040fb','#ffd740','#ff9100']
x  = np.arange(len(SCALES))
bw = 0.14

ax_bar.set_facecolor('#06060e')
for i, (name, col) in enumerate(zip(ALGO_NAMES, algo_colors)):
    vals = [scale_results[N_sc][name]['time_ms'] for N_sc in SCALES]
    bars = ax_bar.bar(x + (i-2)*bw, vals, bw*0.92, label=name,
                      color=col, alpha=0.88, edgecolor='#0a0a1a', lw=0.5)

ax_bar.axhline(50,  color='#ff6666', lw=1.0, ls='--', alpha=0.6)
ax_bar.axhline(200, color='#ff3333', lw=1.4, ls='--', alpha=0.9)
ax_bar.text(2.6, 205, '200 ms budget', color='#ff3333', fontsize=7.5)
ax_bar.text(2.6,  55,  '50 ms target',  color='#ff6666', fontsize=7.5)
ax_bar.set_xticks(x);  ax_bar.set_xticklabels([f'{N}x{N}' for N in SCALES])
ax_bar.set_xlabel('Grid Resolution', color='white', fontsize=9)
ax_bar.set_ylabel('Median Planning Time (ms)', color='white', fontsize=9)
ax_bar.tick_params(colors='white', labelsize=8)
ax_bar.set_facecolor('#0a0a18')
for sp in ax_bar.spines.values(): sp.set_color('#202040')
ax_bar.legend(fontsize=7.5, frameon=True, facecolor='#0c0c20',
              edgecolor='#222244', labelcolor='white', loc='upper left')
ax_bar.yaxis.label.set_color('white')
ax_bar.xaxis.label.set_color('white')

# Table panel
ax_tbl.set_facecolor('#06060e');  ax_tbl.axis('off')
rows_t = []
for N_sc in SCALES:
    for name in ALGO_NAMES:
        r = scale_results[N_sc][name]
        lobby_tag = '[HOT]' if r['lobby'] else '[ok] '
        rows_t.append(f"{N_sc:>5}x{N_sc:<5} {name:<22} "
                      f"{r['steps']:>5}  {r['time_ms']:>8.2f} ms  {lobby_tag}")

ax_tbl.text(0.5, 0.98,
    f"{'Grid':<11} {'Algorithm':<22} {'Steps':>5}  {'Time':>9}   Lobby\n"
    + '-'*60 + '\n' + '\n'.join(rows_t),
    transform=ax_tbl.transAxes, ha='center', va='top',
    fontsize=6.8, color='#9ab0c0', family='monospace',
    bbox=dict(boxstyle='round,pad=0.5', fc='#08081a', ec='#181835'))

plt.tight_layout()
fig2.savefig(OUT_DIR / 'exp2_scaling.png', dpi=160,
             facecolor=fig2.get_facecolor(), bbox_inches='tight')
print(f"\n  Figure saved: exp2_scaling.png")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — VARIABLE ALPHA (MISSION PRIORITY)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*75)
print("  EXP-3: VARIABLE ALPHA — Mission Priority Spectrum")
print("="*75)

ALPHA_CONFIGS = [
    (0.000, "Pure Geometric [CLASS A]",  '#ff007f'),
    (0.003, "Aggressive/Fast [CLASS B]", '#ff6600'),
    (0.009, "Medium Rescue [CLASS C]",   '#ffcc00'),
    (0.012, "Standard Balanced [SAFE]",  '#00ccff'),
    (0.069, "High Risk-Aversion [SAFE]", '#00ff88'),
    (0.150, "Max Preservation [SAFE]",   '#ccff00'),
]

print(f"\n  {'alpha':>6}  {'Mission Mode':<26} {'Steps':>6} "
      f"{'Cum.Heat':>10} {'Peak':>7} {'Route'}")
print("  " + "-"*72)
alpha_records = []
for alpha_val, label, _ in ALPHA_CONFIGS:
    path, exp = run_raa(START_50, GOAL_50, N_BASE, walls_50,
                        temp_grid=temperature_50, alpha=alpha_val)
    steps, cum, peak = path_metrics(path, temperature_50)
    # Route classification based on columns traversed in the middle rows (15-30)
    mid_cols = [c for r, c in path if 18 <= r <= 32]
    avg_col = sum(mid_cols) / len(mid_cols) if mid_cols else START_50[1]
    
    if avg_col < 5:
        route = 'FAR LEFT STAIR (Class C)'
    elif avg_col <= 18:
        route = 'INNER LEFT HALL (Class B)'
    elif avg_col <= 30:
        route = 'MAIN LOBBY (Class A)'
    else:
        route = 'EAST CONFERENCE (Safe)'
        
    alpha_records.append(dict(alpha=alpha_val, label=label, path=path,
                              explored=exp, steps=steps, cum_heat=cum,
                              peak_heat=peak, lobby=(20<=avg_col<=30)))
    print(f"  {alpha_val:>6.3f}  {label:<26} {steps:>6}   "
          f"{cum:>9.1f}  {peak:>7.0f}  {route}")

print()

# ── Figure 3: Alpha spectrum ───────────────────────────────────────────────────
fig3, axes3 = plt.subplots(2, 3, figsize=(18, 12), facecolor='#06060e')
fig3.suptitle(
    'EXP-3: Variable Alpha — Mission Priority  |  Risk-Aware A* routing decisions',
    fontsize=11, fontweight='bold', color='white', y=0.998)

for idx, (rec, (_, _, col)) in enumerate(zip(alpha_records, ALPHA_CONFIGS)):
    ri, ci = divmod(idx, 3)
    ax = axes3[ri, ci]
    ax.set_facecolor('#06060e')

    tmp = temperature_50.astype(float).copy();  tmp[walls_50] = np.nan
    im = ax.imshow(tmp, cmap=HEAT_CMAP, origin='upper', vmin=0, vmax=500,
                   interpolation='bilinear', alpha=0.82)
    wr, wc = np.where(walls_50)
    ax.scatter(wc, wr, marker='s', s=26, color='#181828',
               edgecolors='#303055', lw=0.4, zorder=3)
    ax.add_patch(plt.Rectangle(
        (LOBBY_C_LO_50-0.5, LOBBY_R_LO_50-0.5),
        LOBBY_C_HI_50-LOBBY_C_LO_50+1, LOBBY_R_HI_50-LOBBY_R_LO_50+1,
        lw=1.3, edgecolor='#ff1133', facecolor='none', ls='--', zorder=4))
    if rec['explored']:
        e = np.array(list(rec['explored']))
        ax.scatter(e[:,1], e[:,0], marker='.', s=3,
                   color='white', alpha=0.10, zorder=4)
    if rec['path']:
        p = np.array(rec['path'])
        ax.plot(p[:,1], p[:,0], color=col, lw=2.3,
                marker='o', markersize=1.5, markerfacecolor=col,
                markeredgecolor='white', markeredgewidth=0.1, zorder=6)
    ax.plot(START_50[1], START_50[0], marker='^', ms=10,
            color='#00e676', mew=0.7, markeredgecolor='white', zorder=7)
    ax.plot(GOAL_50[1],  GOAL_50[0],  marker='*', ms=13,
            color='#ff1744', mew=0.7, markeredgecolor='white', zorder=7)

    route_tag = '[CLASS A]' if rec['lobby'] else '[DETOUR]'
    ax.set_title(
        f"alpha={rec['alpha']:.3f}  |  {rec['label']}  {route_tag}",
        fontsize=8.5, fontweight='bold', color=col, pad=5)
    ax.text(0.5, -0.075,
            f"Steps:{rec['steps']}  Cum:{rec['cum_heat']:.0f}C  "
            f"Peak:{rec['peak_heat']:.0f}C",
            transform=ax.transAxes, ha='center', fontsize=7.2, color='#8090a8')
    ax.set_xlim(-0.5,49.5);  ax.set_ylim(49.5,-0.5)
    ax.set_xticks(range(0,50,10));  ax.set_yticks(range(0,50,10))
    ax.tick_params(colors='#2a2a50', labelsize=5.5)
    for sp in ax.spines.values(): sp.set_color('#12122a')

# Colorbar
cax3 = fig3.add_axes([0.936, 0.10, 0.010, 0.82])
cb3  = fig3.colorbar(im, cax=cax3)
cb3.set_label('Temperature (degC)', color='white', fontsize=8)
cb3.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=6)

plt.subplots_adjust(left=0.03, right=0.93, top=0.95, bottom=0.07,
                    wspace=0.22, hspace=0.33)
fig3.savefig(OUT_DIR / 'exp3_variable_alpha.png', dpi=160,
             facecolor=fig3.get_facecolor(), bbox_inches='tight')
print(f"  Figure saved: exp3_variable_alpha.png")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*75)
print("  VALIDATION SUMMARY")
print("="*75)

noise_pass = sum(1 for r in noise_records if not r['lobby'])
print(f"  EXP-1 Noise : {noise_pass}/{len(noise_records)} trials avoided Class A fire  "
      f"({'ROBUST' if noise_pass == len(noise_records) else 'PARTIAL'})")

for N_sc in SCALES:
    raa_t  = scale_results[N_sc]['Risk-Aware A*']['time_ms']
    astar_t = scale_results[N_sc]['A*']['time_ms']
    overhead = raa_t - astar_t
    within   = raa_t <= 200
    budget_tag = 'within 200ms budget' if raa_t <= 200 else 'EXCEEDS 200ms budget'
    print(f"  EXP-2 {N_sc:>3}x{N_sc:<3}: RA* {raa_t:>7.2f} ms  "
          f"A* {astar_t:>7.2f} ms  overhead {overhead:+.2f} ms  [{budget_tag}]")

switch_alpha = None
for i in range(len(alpha_records)-1):
    if alpha_records[i]['lobby'] != alpha_records[i+1]['lobby']:
        switch_alpha = (alpha_records[i]['alpha'], alpha_records[i+1]['alpha'])
        break
if switch_alpha:
    print(f"  EXP-3 Alpha : Routing switch from DIRECT -> DETOUR "
          f"between alpha={switch_alpha[0]} and alpha={switch_alpha[1]}")
else:
    print(f"  EXP-3 Alpha : alpha=0.0 (lobby:{alpha_records[0]['lobby']}) "
          f"-> alpha=2.0 (lobby:{alpha_records[-1]['lobby']})")

print("="*75)
print("  All three figures saved to:", OUT_DIR)
print()

plt.show()
