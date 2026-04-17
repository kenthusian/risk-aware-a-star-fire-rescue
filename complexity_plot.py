"""
complexity_plot.py
==================
Figure 3 – Empirical Wall-Clock Time vs. Search-Space Expansion
             with O((|V|+|E|) log |V|) theoretical bound overlay.

Run:  python complexity_plot.py
Saves: complexity_comparison.png  (in the same directory)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

# ──────────────────────────────────────────────────────────────────
# 0.  STYLE / THEME
# ──────────────────────────────────────────────────────────────────
BG       = "#0f0f1a"
PANEL_BG = "#14142a"
GRID_COL = "#252540"
TEXT_COL = "#cfd8dc"
ACCENT   = "#e0e0ff"
DIM      = "#607d8b"

COLORS = {
    "GBFS":          "#ffd740",
    "Weighted A*":   "#e040fb",
    "Standard A*":   "#76ff03",
    "Dijkstra":      "#00e5ff",
    "Risk-Aware A*": "#ff9100",
}

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   PANEL_BG,
    "axes.edgecolor":   GRID_COL,
    "axes.labelcolor":  TEXT_COL,
    "xtick.color":      TEXT_COL,
    "ytick.color":      TEXT_COL,
    "text.color":       TEXT_COL,
    "grid.color":       GRID_COL,
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
    "font.size":        10,
})

# ──────────────────────────────────────────────────────────────────
# 1.  DATA  (mean ± σ of 50 runs — updated from fire_rescue_output.txt)
# ──────────────────────────────────────────────────────────────────
algorithms     = ["GBFS", "Weighted A*", "Standard A*", "Dijkstra", "Risk-Aware A*"]

# Mean wall-clock times (ms) — latest run (fire_rescue_improved.py)
empirical_ms   = [0.127,  0.338,         0.438,         0.512,      0.998 ]
# Sample std dev (ddof=1, n=50 runs) — from prior 50-run capture
std_ms         = [0.105,  0.091,         0.089,         0.239,      0.198 ]
# Nodes expanded (unchanged — determined by wall topology, not temperature)
nodes_explored = [63,     94,            198,           334,        332   ]

# Theoretical O((|V|+|E|) log |V|) ceiling, scaled to ms via
# calibration constant k = max_empirical / max_nodes
V     = 400
E     = 1_600
log_V = np.log2(V)          # ≈ 8.64
k     = max(empirical_ms) / max(nodes_explored)
theoretical_ceil = [n * log_V * k for n in nodes_explored]

# Budget utilisation = how much of the theoretical ceiling is used (%)
utilisation_pct = [e / t * 100 for e, t in zip(empirical_ms, theoretical_ceil)]

x     = np.arange(len(algorithms))
bar_w = 0.45
bar_colors = [COLORS[a] for a in algorithms]

# ──────────────────────────────────────────────────────────────────
# 2.  FIGURE — 2 rows, generous height, wider for breathing room
# ──────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(12, 10),
    gridspec_kw={"hspace": 0.62, "top": 0.88, "bottom": 0.07}
)
fig.suptitle(
    "Time Complexity Analysis — Fire-Rescue Pathfinding Algorithms",
    fontsize=15, fontweight="bold", color=ACCENT, y=1.00
)

# ──────────────────────────────────────────────────────────────────
# 3.  PANEL A — bars (wall-clock time) + dashed line (nodes explored)
# ──────────────────────────────────────────────────────────────────
bars = ax1.bar(
    x, empirical_ms,
    width=bar_w, color=bar_colors, alpha=0.90,
    edgecolor="white", linewidth=0.7, zorder=3
)

# ── ms labels drawn INSIDE bars (60 % height) — never touch the node line ──
# y_top must clear the tallest errorbar tip: max_mean + max_std
y_top = (max(empirical_ms) + max(std_ms)) * 1.30
for bar, val in zip(bars, empirical_ms):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        val * 0.55,
        f"{val:.3f} ms",
        ha="center", va="center",
        fontsize=9.5, fontweight="bold", color="white", zorder=6
    )

# ── σ error bars — white caps show ±1 std dev ──
ax1.errorbar(
    x, empirical_ms, yerr=std_ms,
    fmt="none", color="white", alpha=0.85,
    capsize=6, capthick=2.0, elinewidth=1.8, zorder=7
)

ax1.set_ylabel("Mean Execution Time (ms)", color=TEXT_COL, fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(algorithms, fontsize=11)
ax1.set_ylim(0, y_top)
ax1.grid(axis="y")
ax1.set_title(
    "Panel A — Empirical Wall-Clock Time  ±1σ   ·   Nodes Explored",
    fontsize=12, fontweight="bold", pad=12, color=ACCENT
)
for sp in ax1.spines.values():
    sp.set_color(GRID_COL)

# ── Secondary axis: nodes explored ──
NODE_COLOR = "#ff4081"
ax1b = ax1.twinx()
ax1b.set_facecolor(PANEL_BG)
ax1b.plot(
    x, nodes_explored,
    color=NODE_COLOR, marker="D", markersize=9,
    linewidth=2.5, linestyle="--", zorder=7
)
ax1b.set_ylabel("Nodes Explored", color=NODE_COLOR, fontsize=11, labelpad=10)
ax1b.tick_params(axis="y", labelcolor=NODE_COLOR, labelsize=10)
ax1b.set_ylim(0, max(nodes_explored) * 1.40)
for sp in ax1b.spines.values():
    sp.set_color(GRID_COL)

# Node-count labels: always ABOVE the diamond so they never sink into bars.
# Dijkstra (i=3, n=334) and Risk-Aware A* (i=4, n=332) are nearly identical,
# so push i=4 higher to prevent stacking.
# A dark bbox keeps the label readable over any bar colour.
for i, (xi, n) in enumerate(zip(x, nodes_explored)):
    y_off = 35 if i == 4 else 12   # extra lift for Risk-Aware A* label
    ax1b.text(
        xi, n + y_off, str(n),
        ha="center", va="bottom",
        fontsize=9, color=NODE_COLOR, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#0f0f1a",
                  edgecolor="none", alpha=0.75)
    )

# ── Legend (use matching bar colours) ──
legend_patches = [
    mpatches.Patch(facecolor=COLORS["GBFS"],         edgecolor="white", label="GBFS"),
    mpatches.Patch(facecolor=COLORS["Weighted A*"],  edgecolor="white", label="Weighted A*"),
    mpatches.Patch(facecolor=COLORS["Standard A*"],  edgecolor="white", label="Standard A*"),
    mpatches.Patch(facecolor=COLORS["Dijkstra"],     edgecolor="white", label="Dijkstra"),
    mpatches.Patch(facecolor=COLORS["Risk-Aware A*"],edgecolor="white", label="Risk-Aware A*"),
    plt.Line2D([0], [0], color=NODE_COLOR, marker="D", markersize=7,
               linewidth=2, linestyle="--", label="Nodes Explored"),
]
ax1.legend(
    handles=legend_patches,
    loc="upper left", fontsize=9,
    ncol=2,
    facecolor="#1a1a2e", edgecolor=GRID_COL, labelcolor=TEXT_COL,
    framealpha=0.85
)

# ──────────────────────────────────────────────────────────────────
# 4.  PANEL B — Budget Utilisation (empirical / theoretical ceiling)
#     Much more readable than paired bars at wildly different scales.
# ──────────────────────────────────────────────────────────────────
util_bars = ax2.barh(
    x, utilisation_pct,
    height=0.50, color=bar_colors, alpha=0.88,
    edgecolor="white", linewidth=0.7, zorder=3
)

# 100% reference line
ax2.axvline(100, color="#546e7a", linewidth=1.5, linestyle=":", zorder=4,
            label="100% = Theoretical Ceiling")

# Shade the "slack" region
ax2.axvspan(max(utilisation_pct) + 1, 105, color="#1a1a2e", alpha=0.0)

# Value labels inside / at end of each bar
for i, (bar, pct, emp, th) in enumerate(
        zip(util_bars, utilisation_pct, empirical_ms, theoretical_ceil)):
    label = f"{pct:.1f}%   ({emp:.3f} ms  vs  {th:.2f} ms bound)"
    ax2.text(
        pct + 0.5, bar.get_y() + bar.get_height() / 2,
        label,
        ha="left", va="center",
        fontsize=9.5, fontweight="bold", color="white"
    )

ax2.set_xlabel(
    "Budget Utilisation  (%)  =  Empirical Time  /  O((|V|+|E|) log|V|) Ceiling",
    color=TEXT_COL, fontsize=11, labelpad=8
)
ax2.set_xlim(0, 105)
ax2.set_yticks(x)
ax2.set_yticklabels(algorithms, fontsize=11)
ax2.invert_yaxis()   # fastest at top
ax2.grid(axis="x")
ax2.set_title(
    r"Panel B — Budget Utilisation vs. $O\!\left((|V|+|E|)\,\log|V|\right)$ Theoretical Bound",
    fontsize=12, fontweight="bold", pad=12, color=ACCENT
)
for sp in ax2.spines.values():
    sp.set_color(GRID_COL)

ax2.legend(
    loc="lower right", fontsize=9.5,
    facecolor="#1a1a2e", edgecolor=GRID_COL, labelcolor=TEXT_COL
)




# ──────────────────────────────────────────────────────────────────
# 5.  SAVE & SHOW
# ──────────────────────────────────────────────────────────────────
out = "complexity_comparison.png"
plt.savefig(out, dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"[complexity_plot.py]  Figure saved → {out}")
plt.show()
