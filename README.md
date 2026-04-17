# Risk-Aware A* for Autonomous Fire-Rescue Agents

This repository contains the simulation environment, algorithm implementations, and visualization scripts for the paper: **"Optimizing Pathfinding for Autonomous Fire-Rescue Agents: A Risk-Aware A* Approach."**

Standard pathfinding algorithms (like Dijkstra or A*) optimize exclusively for distance, which can route autonomous agents through fatal thermal hazards. This project introduces a **Modified Risk-Aware A*** algorithm that augments the traversal cost function with a raw-temperature penalty term, enabling robots to balance path efficiency with hardware survivability in real-time.

## Key Results
* **Thermal Safety:** Achieved a **31.9% reduction** in cumulative thermal exposure and a **3.3% reduction** in peak temperature compared to Standard A*, at a minimal path-length overhead of 12.8%.
* **Computational Tractability:** The thermal penalty requires only an `O(1)` array lookup per edge evaluation. The algorithm preserves the `O((|V| + |E|) log |V|)` asymptotic complexity of Standard A*, executing in under 1 millisecond on a 20x20 grid, well within the 50–200 ms real-time replanning budget.
* **Space Complexity:** Peak priority queue usage remained under 27% of `|V|` (less than 5 KB RAM), confirming strict suitability for embedded microcontrollers (e.g., Raspberry Pi 4, STM32).

## Algorithms Evaluated
The simulation compares five distinct pathfinding strategies:
1. **Dijkstra's Algorithm** (Distance-optimal, uninformed)
2. **Standard A*** (Distance-optimal, heuristic-guided)
3. **Weighted A* (W=2.0)** (Bounded sub-optimal, exploration-minimizing)
4. **Greedy Best-First Search** (GBFS) (Pure heuristic, no cost guarantees)
5. **Modified Risk-Aware A*** (Proposed: distance + thermal penalty objective)

## Repository Structure
* `grid_env.py`: Generates the 20x20 discrete thermal map with radial temperature decay and dual-gap barriers.
* `pathfinders.py`: Contains the implementations for all five search algorithms.
* `simulation_runner.py`: Executes the 50-run timing tests and Monte Carlo topological validations.
* `visualize.py`: Generates the 2x3 grid pathing overlays and the time-complexity performance charts using `matplotlib`.
* `requirements.txt`: Python dependencies.

## Getting Started

**1. Clone the repository:**
```bash
git clone [https://github.com/YourUsername/RiskAware-Astar-FireRescue.git](https://github.com/YourUsername/RiskAware-Astar-FireRescue.git)
cd RiskAware-Astar-FireRescue
2. Install dependencies:

Bash
pip install -r requirements.txt
3. Run the primary simulation & generate visualizations:

Bash
python simulation_runner.py
Extended Validations
The codebase includes scripts to reproduce the six supplementary experiments discussed in the paper:

Grid Scalability: Testing N scaling up to 200x200 grids.

Monte Carlo: 100 random obstacle topologies (20% wall density).

Sensor Noise: Robustness testing under ±10% Gaussian measurement noise.

Thermal Traps: Concave U-shaped hazard avoidance.

Citation
If you use this code or simulation environment in your research, please cite our paper:

Code snippet
@inproceedings{kilak2026riskaware,
  title={Optimizing Pathfinding for Autonomous Fire-Rescue Agents: A Risk-Aware A* Approach},
  author={Kilak, Arav and Shukla, Aniruddh and Vidhani, Vedant and Mohanty, Dev Kumar and Sundaresan, Mithul Ram and Sulthana H, Parveen},
  booktitle={IEEE [Insert Conference Name Here]},
  year={2026},
  organization={Vellore Institute of Technology (VIT)}
}
