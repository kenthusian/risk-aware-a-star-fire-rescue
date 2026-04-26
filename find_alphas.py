import numpy as np
from complex_env import get_environment, START, GOAL
from robustness_validation import run_raa, path_metrics

t_50, w_50 = get_environment()

unique_routes = {}
# Sweep alpha 
alphas = np.linspace(0.00, 0.60, 200)

for a in alphas:
    p, _ = run_raa(START, GOAL, 50, w_50, t_50, alpha=a)
    steps, c, maxh = path_metrics(p, t_50)
    
    route_key = (steps, f"{maxh:.0f}")
    if route_key not in unique_routes:
        unique_routes[route_key] = a

print(f"Found {len(unique_routes)} unique configurations:")
for k, a in unique_routes.items():
    print(f"Alpha {a:.3f} -> Steps: {k[0]}, Peak: {k[1]}")
