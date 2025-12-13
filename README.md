# Järntorget Multimodal Traffic Simulator (JMTS)

A Python-based agent-based simulation framework for studying how construction events and vehicle congestion impact travel times in urban networks. The simulator uses real GIS network data (GeoPackage format) and supports dynamic routing, pedestrian-aware vehicle behavior, and construction event tracking.

## Overview

JMTS models the interaction between vehicles (cars), pedestrians, and infrastructure (trams) on a driving network. Key features:

- **Dynamic routing**: Vehicles plan and replan routes using A* pathfinding with environment-aware travel time weights
- **Congestion modeling**: Travel times increase with density and decrease with pedestrian presence via realistic multipliers
- **Construction events**: Infrastructure disruptions applied to edges trigger replanning and increase travel time multipliers
- **Batch experiments**: Run hundreds of simulations across parameter sweeps (fleet size, construction counts, seeds)
- **Travel time statistics**: Aggregate statistics collected per simulation (mean, median, stdev travel times and speeds)

## Project Structure

```
model/
  agents.py              - CarAgent and PedestrianAgent behavior
  env.py                 - Environment class: occupancy tracking, travel time weights, construction events
  network_utils.py       - Graph loading from GeoPackage; travel time inference from speed columns
  experiment_runner.py   - Batch simulation runner; CSV output of results
  __pycache__/

tools/
  plot_experiment_results.py  - Generate publication-ready plots from experiment CSV
  analyze_experiment_csv.py   - Summarize CSV results by vehicle count and construction count

data/
  processed/networks/
    drive_edges_clean.gpkg    - Driving network edges (required)
    walk_edges_clean.gpkg     - Pedestrian network edges (optional)
    tram_features_clean.gpkg  - Tram network (optional)

archive/
  model/                 - Archived demo and diagnostic scripts (not used in active pipeline)

results/
  exp_factor3/           - Example experiment output (plots and report)

experiment_*.csv         - CSV outputs from batch runs

requirements.txt         - Python package dependencies
```

## Core Model Components

### CarAgent (`model/agents.py`)

- **State**: position, speed, route, arrival time, distance travelled
- **Behavior**:
  - Plans A* route using environment-aware edge weights (via `env.get_edge_travel_time()`)
  - Accelerates/brakes based on leader distance and pedestrian presence
  - Lanes support with lane-change evaluation (cooldown enforced)
  - Re-plans if construction event modifies route travel time significantly

### Environment (`model/env.py`)

- **Occupancy tracking**: Records which vehicles/pedestrians occupy edges and lanes
- **Travel time calculation**: `get_edge_travel_time(edge)` returns `base * multiplier * density_factor * ped_factor`
  - `base`: edge attribute `travel_time_s` (inferred from speed if absent)
  - `multiplier`: construction multiplier (>1 if construction active)
  - `density_factor`: increases when edge occupancy > 0.1 vehicles/m
  - `ped_factor`: penalty if pedestrians present on edge
- **Construction events**: `apply_construction(edge, factor, until)` sets multiplier and triggers replanning
- **Pedestrian detection**: Identifies pedestrians within lookahead distance; agents yield/brake

### Network Graph (`model/network_utils.py`)

- Loads GeoPackage edge layer into NetworkX DiGraph
- Infers `travel_time_s` from:
  - Explicit `travel_time_s` column (preferred)
  - `speed_kmph` or `maxspeed` or `speed` columns (computed as `length_m / speed_m_s`)
  - Fallback: 50 km/h default speed
- Node keys: `(x, y)` coordinate tuples
- Edge attributes: `length_m`, `travel_time_s`, `lanes`

### Experiment Runner (`model/experiment_runner.py`)

- **`batch_run()`**: Execute multiple simulations across parameter sweeps
  - `vehicle_counts`: list of fleet sizes
  - `construction_counts`: list of construction event counts
  - `seeds`: random seeds for reproducibility
  - `route_only_constructions`: if True, place constructions only on edges along agents' planned routes (reduces noise)
  - Auto-computes `sim_time` from predicted route times
- **`run_single_sim()`**: Run one experiment and return statistics
  - Spawns vehicles with random O/D pairs
  - Applies construction events at t=0
  - Steps simulation, collects arrival times and average speeds
  - Returns: `mean_travel_time`, `median_travel_time`, `stdev_travel_time`, `mean_avg_speed`, etc.
- **CSV output**: One row per simulation with all statistics

## Usage

### Quick Start: Run a Small Batch Experiment

```python
from model.experiment_runner import batch_run

batch_run(
    vehicle_counts=[10, 50, 100],
    construction_counts=[0, 10, 50],
    seeds=[0, 1, 2],
    out_csv='my_results.csv'
)
```

This runs 3 × 3 × 3 = 27 simulations and writes results to `my_results.csv`.

### Generate Plots

```bash
python tools/plot_experiment_results.py my_results.csv --out-dir results/my_exp --prefix my_exp
```

Produces publication-ready PNG/PDF plots with LaTeX-style fonts and a Markdown report.

### Analyze Results

```bash
python tools/analyze_experiment_csv.py
```

Summarizes CSV by vehicle count and construction count, printing mean/median travel times and speeds.

## Key Assumptions & Tuning

- **Density threshold**: Slowdown kicks in when edge occupancy > 0.1 veh/m. Adjust in `env.py` for higher/lower sensitivity.
- **Construction multiplier**: Default factor=3.0 increases travel time 3× on affected edges. Tune via `factor` parameter in `batch_run()`.
- **Route-only constructions**: Restricting construction placement to edges along planned routes (`route_only_constructions=True`) reduces overlap noise and produces clearer trends.
- **Simulation time**: Auto-computed from predicted route times + 60 s buffer. Override with `sim_time` parameter if needed.

## Dependencies

See `requirements.txt`. Main packages:
- `osmnx` — network dara extraction
- `networkx` — graph operations
- `geopandas`, `shapely`, `fiona` — GIS data loading
- `pandas`, `numpy` — data handling
- `matplotlib` — plotting (with optional LaTeX rendering)

## Results & Interpretation

Recent experiments (e.g., `exp_factor3.csv`, `experiment_wide_range.csv`) show:
- **Travel time increases with vehicle count** (fleet congestion effect)
- **Travel time increases with construction count** (especially when `factor` is high)
- **Average speeds decrease with higher vehicle counts** and construction events
- **Route-only construction placement produces clearer signals** than random placement

Plots in `results/` visualize these trends with error bands (stddev across seeds).
