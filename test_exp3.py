import time
import numpy as np
from complex_env import get_environment, START, GOAL, ALPHA
from robustness_validation import run_raa, path_metrics, ALPHA_CONFIGS

print("Building environment...")
t_50, w_50 = get_environment()
print("Done. Running tests...")

for alpha_val, label, _ in ALPHA_CONFIGS:
    p, exp = run_raa(START, GOAL, 50, w_50, t_50, alpha=alpha_val)
    steps, c, maxh = path_metrics(p, t_50)
    
    mid_cols = [col for r, col in p if 18 <= r <= 32]
    ac = sum(mid_cols) / len(mid_cols) if mid_cols else START[1]
    
    if ac < 5:
        route = 'FAR LEFT STAIR'
    elif ac <= 18:
        route = 'INNER LEFT HALL'
    elif ac <= 30:
        route = 'MAIN LOBBY'
    else:
        route = 'EAST CONFERENCE'
        
    print(f"Alpha {alpha_val:5.3f} -> Steps: {steps:3d}, Cum: {c:6.0f}, Peak: {maxh:3.0f}, Route: {route}")
print("FINISH")
