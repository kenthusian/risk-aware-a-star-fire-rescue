import numpy as np
import io
from complex_env import get_environment, START, GOAL
from robustness_validation import run_raa, path_metrics

with io.open("test_out.txt", "w", encoding="utf-8") as f:
    t_50, w_50 = get_environment()
    alphas = [0.0, 0.003, 0.012, 0.050, 0.150, 0.40, 0.60, 1.0, 5.0, 10.0]
    for a in alphas:
        p, _ = run_raa(START, GOAL, 50, w_50, t_50, alpha=a)
        steps, c, maxh = path_metrics(p, t_50)
        mid_cols = [col for r, col in p if 18 <= r <= 32]
        ac = sum(mid_cols) / len(mid_cols) if mid_cols else START[1]
        f.write(f"Alpha {a:5.3f} -> Steps: {steps:3d}, Cum: {c:6.0f}, Peak: {maxh:3.0f}, AvgCol: {ac:.1f}\n")
