import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load experiment data
df = pd.read_csv('experiment_wide_range.csv')

print("=" * 80)
print("WIDE RANGE EXPERIMENT ANALYSIS")
print("=" * 80)
print(f"\nExperiment shape: {df.shape}")
print(f"Fleet sizes: {sorted(df['num_vehicles'].unique())}")
print(f"Construction counts: {sorted(df['num_constructions'].unique())}")
print(f"Random seeds: {sorted(df['seed'].unique())}")
print(f"Total runs: {len(df)}")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print(f"\nMean Travel Time (all runs): {df['mean_travel_time'].mean():.2f} ± {df['mean_travel_time'].std():.2f} s")
print(f"Range: {df['mean_travel_time'].min():.2f} - {df['mean_travel_time'].max():.2f} s")
print(f"\nMean Speed (all runs): {df['mean_avg_speed'].mean():.2f} ± {df['mean_avg_speed'].std():.2f} m/s")

# Impact of fleet size on travel time
print("\n" + "=" * 80)
print("IMPACT OF FLEET SIZE ON TRAVEL TIME")
print("=" * 80)
fleet_impact = df.groupby('num_vehicles')[['mean_travel_time', 'mean_avg_speed']].agg(['mean', 'std'])
print(fleet_impact)

# Impact of construction events
print("\n" + "=" * 80)
print("IMPACT OF CONSTRUCTION EVENTS ON TRAVEL TIME")
print("=" * 80)
construction_impact = df.groupby('num_constructions')[['mean_travel_time', 'mean_avg_speed']].agg(['mean', 'std'])
print(construction_impact)

# Cross-tabulation: fleet size vs constructions
print("\n" + "=" * 80)
print("TRAVEL TIME: FLEET SIZE vs CONSTRUCTIONS (averaged over seeds)")
print("=" * 80)
pivot_travel = df.pivot_table(values='mean_travel_time', 
                               index='num_vehicles', 
                               columns='num_constructions', 
                               aggfunc='mean')
print(pivot_travel)

print("\n" + "=" * 80)
print("MEAN SPEED: FLEET SIZE vs CONSTRUCTIONS (averaged over seeds)")
print("=" * 80)
pivot_speed = df.pivot_table(values='mean_avg_speed', 
                              index='num_vehicles', 
                              columns='num_constructions', 
                              aggfunc='mean')
print(pivot_speed)

# Quantify construction impact
print("\n" + "=" * 80)
print("QUANTIFIED CONSTRUCTION IMPACT (% increase from baseline)")
print("=" * 80)
for fleet in sorted(df['num_vehicles'].unique()):
    baseline = df[(df['num_vehicles'] == fleet) & (df['num_constructions'] == 0)]['mean_travel_time'].mean()
    print(f"\nFleet size: {fleet} vehicles (baseline: {baseline:.1f} s)")
    for const in sorted(df['num_constructions'].unique()):
        if const == 0:
            continue
        time = df[(df['num_vehicles'] == fleet) & (df['num_constructions'] == const)]['mean_travel_time'].mean()
        increase = ((time - baseline) / baseline) * 100
        print(f"  Constructions={const}: {time:.1f} s (+{increase:.1f}%)")

# Variance analysis
print("\n" + "=" * 80)
print("VARIABILITY ACROSS SEEDS")
print("=" * 80)
print("\nTravel time std dev by fleet size:")
print(df.groupby('num_vehicles')['mean_travel_time'].std())
print("\nTravel time std dev by construction count:")
print(df.groupby('num_constructions')['mean_travel_time'].std())

# All vehicles arrived?
print("\n" + "=" * 80)
print("ARRIVAL RATE CHECK")
print("=" * 80)
print(f"All simulations had 100% arrival rate: {(df['arrival_rate'] == 1.0).all()}")
print(f"Min arrival rate: {df['arrival_rate'].min()}")
print(f"Max arrival rate: {df['arrival_rate'].max()}")
