"""
Analyze detour behavior in the wide range experiment.
Detours are detected by comparing actual route distance to theoretical minimum.
"""
import pandas as pd
import numpy as np
from model.network_utils import load_edges, build_graph_from_edges
import networkx as nx

# Load the network
print("Loading network edges...")
edges_gdf = load_edges('data/processed/networks/drive_edges_clean.gpkg')
G = build_graph_from_edges(edges_gdf)

print(f"Network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Load experiment results
df = pd.read_csv('experiment_wide_range.csv')

print("\n" + "=" * 80)
print("DETOUR ANALYSIS")
print("=" * 80)

print(f"""
NOTE: Detour detection requires:
1. Recording individual vehicle routes during simulation
2. Computing shortest path distance for each vehicle's origin-destination pair
3. Comparing actual distance to shortest path

Current CSV only stores aggregate statistics (mean_travel_time, mean_avg_speed).
To enable detour tracking, we need to modify the simulation to log:
  - Each vehicle's start/end coordinates
  - Actual route distance traveled
  - Number of times vehicle replanned route
  
""")

print("=" * 80)
print("PROPOSED SOLUTION")
print("=" * 80)

print("""
Option 1: Enhance experiment_runner.py to track per-vehicle data
  - Store vehicle_id, start_coord, end_coord, distance_traveled, num_replans
  - Output as experiment_wide_range_vehicles.csv (one row per vehicle)
  - Allows post-hoc analysis of detours and replanning behavior

Option 2: Lightweight proxy metric
  - Use ratio of actual_distance / expected_distance based on baseline travel time
  - If actual_distance > expected_distance, vehicle took detour
  - Can be computed from existing metrics with some assumptions

Option 3: Add to next experiment run
  - Modify experiment_runner.py NOW to log vehicle details
  - Re-run experiment (only 144 simulations, ~30-60 min depending on network)
  - Get complete detour dataset

RECOMMENDATION: Implement Option 3 - modify experiment_runner.py to capture:
  - n_replans: Number of times each vehicle replanned its route
  - distance_traveled: Actual distance covered (already in agent)
  - took_detour: Boolean flag if distance > baseline shortest path
""")

print("\n" + "=" * 80)
print("CURRENT DATA AVAILABLE IN CSV")
print("=" * 80)
print(f"Columns: {df.columns.tolist()}")
print(f"\nColumns that could indicate detours (indirect):")
print("  - stdev_travel_time: High variance suggests some vehicles took longer routes")
print("  - median_travel_time vs mean_travel_time: Gap indicates outliers (possible detours)")

# Compute proxy metrics
print("\n" + "=" * 80)
print("PROXY DETOUR METRIC (from existing data)")
print("=" * 80)

# Vehicles with travel time > median may have taken detours
df['likely_detour_ratio'] = (df['mean_travel_time'] - df['median_travel_time']) / df['median_travel_time']

print("\nVehicles that likely took detours (mean > median):")
print(df[['num_vehicles', 'num_constructions', 'seed', 'mean_travel_time', 'median_travel_time', 'likely_detour_ratio']].head(20))

print(f"\nAverage detour indicator by fleet size:")
print(df.groupby('num_vehicles')['likely_detour_ratio'].mean())

print(f"\nAverage detour indicator by construction count:")
print(df.groupby('num_constructions')['likely_detour_ratio'].mean())

print("\n" + "=" * 80)
print("RECOMMENDATION FOR POSTER")
print("=" * 80)
print("""
Instead of detour count, you could show:
1. "Vehicle heterogeneity": stdev_travel_time (higher = more diverse routes)
2. "Outlier rate": percentage of vehicles > 1.5x median time
3. "Route adaptation": compare baseline vs. with constructions

Would you like me to:
A) Modify experiment_runner.py to track per-vehicle detours (requires re-run)
B) Create poster-ready plots using the proxy metrics above
C) Both?
""")
