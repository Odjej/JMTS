[![DOI](https://zenodo.org/badge/1099278845.svg)](https://doi.org/10.5281/zenodo.18040915)
# Järntorget Multimodal Traffic Simulator (JMTS)

A Python-based agent-based simulation framework for studying how construction events and vehicle congestion impact travel times and routing behavior in urban networks. The simulator uses real GIS network data from OpenStreetMap (GeoPackage format) and supports dynamic routing, pedestrian-aware vehicle behavior, construction event tracking, and detailed per-vehicle detour analysis.

## Overview

JMTS models the interaction between vehicles (cars), pedestrians, and infrastructure on a driving network centered around Järntorget in Gothenburg, Sweden. Key features:

- **Dynamic routing**: Vehicles plan and replan routes using A* pathfinding with environment-aware travel time weights
- **Congestion modeling**: Travel times increase with density and pedestrian presence via realistic multipliers
- **Construction events**: Infrastructure disruptions (slow zones, lane reductions, closures) applied to edges trigger automatic replanning and increase travel time multipliers
- **Replanning tracking**: Monitors how many times each vehicle recalculates its route due to construction events
- **Detour analysis**: Calculates detour percentages by comparing actual distance traveled to initial planned route length
- **Batch experiments**: Run hundreds of simulations across parameter sweeps (fleet size, construction counts, seeds)
- **Per-vehicle data**: Track individual vehicle travel times, distances, replans, and detour behavior
- **Comprehensive statistics**: Aggregate and per-vehicle metrics including mean/median travel times, speeds, detour rates, and replanning frequency

## Project Structure

```
model/
  agents.py              - CarAgent and PedestrianAgent classes with GFM dynamics
  env.py                 - Environment: occupancy, travel times, construction, replanning
  network_utils.py       - Graph loading from GeoPackage; travel time inference
  experiment_runner.py   - Batch simulation runner with per-vehicle tracking
  __pycache__/

tools/
  plot_essential_individual.py    - Generate 4 key publication plots (paradox, impact, effectiveness, distribution)
  plot_experiment_results.py      - General experiment visualization
  analyze_experiment_csv.py       - Summarize aggregate CSV results
  analyze_wide_range.py           - Analyze wide parameter sweep results
  generate_detour_summary.py      - Generate comprehensive detour statistics summary
  analyze_detours.py              - Detailed detour analysis utilities
  plot_detours_batch.py           - Batch plotting for detour metrics
  plot_detours_comprehensive.py   - Comprehensive detour visualization
  visualize_car_agent.py          - Individual agent trajectory visualization
  __pycache__/

data/
  processed/networks/
    drive_edges_clean.gpkg    - Driving network edges (Järntorget area, ~474 edges)
    drive_nodes_clean.gpkg    - Driving network nodes (~239 nodes)
    walk_edges_clean.gpkg     - Pedestrian network edges
    walk_nodes_clean.gpkg     - Pedestrian network nodes
    gota_alv.gpkg             - Göta Älv river features for visualization
    process_data.ipynb        - Notebook for network data processing
  raw/osm/
    get_osm_data.ipynb        - Notebook for extracting OSM data
    cache/                    - Cached OSM query results

archive/
  model/                 - Archived demo and diagnostic scripts (not used in active pipeline)

results/
  data/
    experiment_wide_range_v2.csv          - Aggregate results (90 simulations)
    experiment_wide_range_v2_per_vehicle.csv  - Per-vehicle data (27,900 vehicles)
    detours_summary_stats_new.csv         - Summary statistics with replanning data
  batch_plots/           - Batch visualization outputs
  detour_plots/          - Detour analysis plots
  wide_range/            - Wide parameter sweep results

cache/                   - Simulation cache files

test_replans.py          - Test script for verifying replanning counter

requirements.txt         - Python package dependencies
```

## Core Model Components

### CarAgent (`model/agents.py`)

- **State**: position, speed, route, arrival time, distance travelled, lane, replanning counter
- **Behavior**:
  - Plans A* route using environment-aware edge weights (via `env.get_edge_travel_time()`)
  - Simplified Generalized Force Model (GFM) dynamics: relaxation to desired speed with leader-distance braking
  - Pedestrian-aware yielding (full stop when pedestrians cross path)
  - Lane support with utility-based lane-change evaluation (cooldown enforced to prevent oscillation)
  - Automatically re-plans when construction events affect remaining route (increments `num_replans` counter)
  - Tracks initial route length for detour percentage calculation

### PedestrianAgent (`model/agents.py`)

- **State**: position, speed, direction, route
- **Behavior**:
  - Fujii-like pedestrian dynamics with social force model
  - Avoids other pedestrians and obstacles
  - Fixed population (100 pedestrians in standard setup)
  - Interacts with car agents at crossings

### Environment (`model/env.py`)

- **Occupancy tracking**: Records which vehicles/pedestrians occupy edges and lanes (sorted by fractional position)
- **Travel time calculation**: `get_edge_travel_time(edge)` returns `base * multiplier * density_factor * ped_factor`
  - `base`: edge attribute `travel_time_s` (inferred from speed if absent)
  - `multiplier`: construction multiplier (>1 for slow zones, ∞ for closures)
  - `density_factor`: increases when edge occupancy > 0.1 vehicles/m (1 + 0.5*(density/0.1))
  - `ped_factor`: penalty if pedestrians present on edge (up to 2.0×)
- **Construction events**: `apply_construction(edge, kind, factor, until)` sets multiplier and triggers replanning
  - Three types: 'slow' (3× slowdown), 'lane_reduction' (1.5× slowdown), 'closure' (infinite travel time)
  - Automatically triggers `_trigger_replanning()` for affected vehicles
  - Increments vehicle `num_replans` counter when replanning occurs
- **Pedestrian detection**: Identifies pedestrians within lookahead distance; agents yield/brake
- **Lane changing**: Vehicles evaluate adjacent lanes and change if speed gain > threshold with cooldown enforcement

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
  - `vehicle_counts`: list of fleet sizes (default: [10, 50, 100, 200, 500, 1000])
  - `construction_counts`: list of construction event counts (default: [0, 5, 10, 20, 40])
  - `seeds`: random seeds for reproducibility (default: [0, 1, 2])
  - `route_only_constructions`: if True, place constructions only on edges along agents' planned routes (reduces noise, **recommended**)
  - `track_per_vehicle`: if True, output per-vehicle data including replans and detours
  - `slow_prob`, `lane_reduction_prob`, `closure_prob`: probabilities for construction types (default: 0.7, 0.2, 0.1)
  - `slow_factor`, `lane_factor`: travel time multipliers for construction types (default: 3.0, 1.5)
  - Auto-computes `sim_time` from predicted route times (120% of max predicted time + 60s buffer)
- **`run_single_sim()`**: Run one experiment and return statistics
  - Spawns vehicles with random O/D pairs
  - Plans initial routes accounting for construction (if constructions placed before planning)
  - Applies construction events at t=0, triggering replanning for affected vehicles
  - Steps simulation, collects arrival times, distances, speeds, and replan counts
  - Returns aggregate stats: `mean_travel_time`, `mean_num_replans`, `pct_vehicles_rerouted`, `mean_detour_ratio`
  - If `track_per_vehicle=True`, also returns list with individual vehicle data
- **CSV output**: 
  - Aggregate CSV: one row per simulation with fleet-wide statistics
  - Per-vehicle CSV: one row per vehicle with individual metrics (travel_time, distance_travelled, num_replans, detour_pct)

## Usage

### Quick Start: Run a Small Batch Experiment

```python
from model.experiment_runner import batch_run

batch_run(
    vehicle_counts=[10, 50, 100],
    construction_counts=[0, 10, 20],
    seeds=[0, 1, 2],
    out_csv='my_results.csv',
    route_only_constructions=True,
    track_per_vehicle=True
)
```

This runs 3 × 3 × 3 = 27 simulations and writes:
- `my_results.csv`: aggregate statistics per simulation
- `my_results_per_vehicle.csv`: individual vehicle data

### Run Full Parameter Sweep (90 simulations)

```bash
python -m model.experiment_runner
```

Runs the default configuration (6 fleet sizes × 5 construction counts × 3 seeds = 90 simulations) and outputs to `experiment_wide_range_v2.csv` and `experiment_wide_range_v2_per_vehicle.csv`.

### Generate Detour Summary Statistics

```bash
python tools/generate_detour_summary.py experiment_wide_range_v2 --out=results/data/detours_summary_stats.csv
```

Analyzes per-vehicle data and generates comprehensive summary including:
- Overall detour statistics (mean, median, std, min, max)
- Detour categories (any, major >10%, severe >20%, extreme >50%)
- Replanning statistics (mean replans, max replans, % vehicles rerouted)
- Breakdown by construction count and fleet size

### Generate Publication Plots

```bash
cd results/data
python ../../tools/plot_essential_individual.py
```

Generates 4 key publication-ready plots:
1. `plot_01_paradox.png/pdf`: Mean detour % and travel time vs fleet size (shows the routing paradox)
2. `plot_02_impact.png/pdf`: Percentage of vehicles affected by detours
3. `plot_03_effectiveness.png/pdf`: Heatmap of mean detour cost by fleet size and constructions
4. `plot_04_distribution.png/pdf`: Detour distribution comparison (low vs high density)

### Test Replanning Counter

```bash
python test_replans.py
```

Runs a quick test (20 vehicles, 5 constructions) to verify the replanning counter is working correctly.

## Key Assumptions & Parameters

- **Density threshold**: Slowdown kicks in when edge occupancy > 0.1 veh/m. Adjust in `env.py::get_edge_travel_time()` for different sensitivity.
- **Construction types & probabilities**:
  - 'slow' (70% probability): 3.0× travel time multiplier
  - 'lane_reduction' (20% probability): 1.5× travel time multiplier  
  - 'closure' (10% probability): infinite travel time (forces rerouting)
- **Route-only constructions**: Setting `route_only_constructions=True` places constructions only on edges that appear in at least one vehicle's initial route. This **significantly improves signal clarity** by ensuring constructions actually affect traffic.
- **Simulation time**: Auto-computed as 120% of maximum predicted route time + 60s buffer. Can be overridden with `sim_time` parameter.
- **Network size**: Järntorget area with ~239 nodes, ~474 edges, 1 km radius
- **Pedestrian population**: Fixed at 100 pedestrians in standard setup
- **GFM parameters**:
  - Desired speed: 13.89 m/s (50 km/h)
  - Max acceleration: 2.0 m/s²
  - Comfortable deceleration: 3.0 m/s²
  - Safe time headway: 1.0 s
  - Minimum gap: 2.0 m

## Dependencies

See `requirements.txt`. Main packages:
- `osmnx>=1.2` — OpenStreetMap network data extraction
- `networkx>=2.6` — graph operations and A* pathfinding
- `geopandas>=0.12`, `shapely>=1.8`, `fiona>=1.8` — GIS data loading
- `pandas>=1.3`, `numpy>=1.21` — data handling and numerical operations
- `matplotlib>=3.4` — plotting (with optional LaTeX rendering)
- `matplotlib-scalebar` — scale bars for map visualizations

Install all dependencies:
```bash
pip install -r requirements.txt
```

Or use the provided installation in the project setup.

## Key Findings & Results

Analysis of 90 simulations (27,900 total vehicles) across varying fleet densities and construction scenarios reveals:

### The Routing Paradox
- **At low density (10-100 vehicles)**: Mean detours reach 7.5% as vehicles actively seek alternatives to avoid construction
- **At high density (>500 vehicles)**: Mean detours collapse to <1% despite 20% longer travel times
- **Mechanism**: Network saturation eliminates viable alternative routes faster than congestion motivates detour-seeking
- **Result**: Distributed routing systems paradoxically converge to uniform behavior under capacity constraints

### Replanning Behavior
- **36.2% of vehicles** replan their routes at least once during simulation
- **Up to 14 replans** observed for individual vehicles in high-construction scenarios
- Mean replans per vehicle: 0.64 across all simulations
- Replanning occurs when construction events affect remaining route segments

### Detour Statistics
- **57.4% of vehicles** take any detour (>0% extra distance)
- **3.0% of vehicles** take major detours (>10% extra distance)
- **1.1% of vehicles** take severe detours (>20% extra distance)
- Mean detour: 1.24% (median: 0.00%, indicating bimodal distribution)
- Max observed detour: 403.55% (extreme case with multiple construction blockages)

### Construction Impact Independence
- Number of construction sites has **minimal impact** compared to fleet density
- Fleet size dominates detour behavior across all construction scenarios
- Suggests network saturation effects overwhelm localized disruptions

### Implications
- Traffic management should prioritize **maintaining route diversity** over minimizing point congestion
- High-density scenarios force convergence to shortest paths regardless of construction impacts
- Resembles inverse Braess's paradox: removing capacity (via congestion) reduces route exploration when it would be most beneficial

Detailed plots and analysis available in `results/` directory.
