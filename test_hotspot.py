import numpy as np
from complex_env import get_environment, START, GOAL
from robustness_validation import run_raa, path_metrics

t_50, w_50 = get_environment()

alphas = [0.4, 0.6]
for a in alphas:
    p, _ = run_raa(START, GOAL, 50, w_50, t_50, alpha=a)
    hlist = [(r, c, t_50[r, c]) for r, c in p]
    mh = max(hlist, key=lambda x: x[2])
    print(f"Alpha {a} -> Max heat {mh[2]:.1f} at {mh[0]},{mh[1]}")
