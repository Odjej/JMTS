"""
Analysis of per-vehicle detour behavior from experiment_wide_range_v2
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df_agg = pd.read_csv('experiment_wide_range_v2.csv')
df_pv = pd.read_csv('experiment_wide_range_v2_per_vehicle.csv')

# Filter to arrived vehicles
arrived_df = df_pv[df_pv['arrived'] == 1].copy()

# Calculate detour percentage
arrived_df['detour_pct'] = ((arrived_df['distance_travelled'] / arrived_df['initial_route_length']) - 1) * 100

print("=" * 90)
print("DETOUR ANALYSIS SUMMARY")
print("=" * 90)

print("\n### HEADLINE METRICS ###\n")
print(f"Total simulations: {len(df_agg)}")
print(f"Total vehicles analyzed: {len(arrived_df)}")
print(f"\nDetour Statistics (all arrived vehicles):")
print(f"  Mean detour: {arrived_df['detour_pct'].mean():.2f}% extra distance")
print(f"  Median detour: {arrived_df['detour_pct'].median():.2f}%")
print(f"  Std dev: {arrived_df['detour_pct'].std():.2f}%")
print(f"  Range: {arrived_df['detour_pct'].min():.2f}% to {arrived_df['detour_pct'].max():.2f}%")

print(f"\nVehicles with route extensions:")
vehicles_with_detour = (arrived_df['detour_pct'] > 0).sum()
print(f"  Count: {vehicles_with_detour} / {len(arrived_df)} ({vehicles_with_detour/len(arrived_df)*100:.1f}%)")

vehicles_major_detour = (arrived_df['detour_pct'] > 10).sum()
print(f"  With >10% extension: {vehicles_major_detour} ({vehicles_major_detour/len(arrived_df)*100:.1f}%)")

print("\n### BY FLEET SIZE ###\n")
fleet_analysis = arrived_df.groupby('num_vehicles').agg({
    'detour_pct': ['mean', 'median', 'std', 'max'],
    'vehicle_id': 'count'
}).round(2)
fleet_analysis.columns = ['mean_detour_%', 'median_detour_%', 'stdev_%', 'max_detour_%', 'n_vehicles']
print(fleet_analysis)

print("\n### BY CONSTRUCTION COUNT ###\n")
const_analysis = arrived_df.groupby('num_constructions').agg({
    'detour_pct': ['mean', 'median', 'std', 'max'],
    'vehicle_id': 'count'
}).round(2)
const_analysis.columns = ['mean_detour_%', 'median_detour_%', 'stdev_%', 'max_detour_%', 'n_vehicles']
print(const_analysis)


# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Detour % by fleet size
ax1 = axes[0, 0]
fleet_data = arrived_df.groupby('num_vehicles')['detour_pct'].apply(list)
ax1.boxplot([fleet_data[v] for v in sorted(fleet_data.index)], 
            labels=[str(v) for v in sorted(fleet_data.index)])
ax1.set_xlabel('Fleet Size (vehicles)')
ax1.set_ylabel('Detour %')
ax1.set_title('Route Extension Distribution by Fleet Size')
ax1.grid(True, alpha=0.3)
ax1.axhline(0, color='red', linestyle='--', alpha=0.5, label='No detour')
ax1.legend()

# 2. Vehicles with detours by scenario
ax2 = axes[0, 1]
pivot_detours = df_agg.pivot_table(
    values='vehicles_with_detours',
    index='num_vehicles',
    columns='num_constructions',
    aggfunc='mean'
)
im2 = ax2.imshow(pivot_detours, cmap='YlOrRd', aspect='auto')
ax2.set_xticks(range(len(pivot_detours.columns)))
ax2.set_yticks(range(len(pivot_detours.index)))
ax2.set_xticklabels(pivot_detours.columns)
ax2.set_yticklabels(pivot_detours.index)
ax2.set_xlabel('Number of Constructions')
ax2.set_ylabel('Fleet Size')
ax2.set_title('Avg Vehicles with >10% Longer Routes')
for i in range(len(pivot_detours.index)):
    for j in range(len(pivot_detours.columns)):
        text = ax2.text(j, i, f'{pivot_detours.iloc[i, j]:.0f}',
                       ha="center", va="center", color="black", fontsize=10)
plt.colorbar(im2, ax=ax2)

# 3. Mean detour % by scenario
ax3 = axes[1, 0]
pivot_mean_detour = df_agg.pivot_table(
    values='mean_detour_ratio',
    index='num_vehicles',
    columns='num_constructions',
    aggfunc='mean'
)
# Convert ratio to percentage
pivot_mean_detour = (pivot_mean_detour - 1) * 100
im3 = ax3.imshow(pivot_mean_detour, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=3)
ax3.set_xticks(range(len(pivot_mean_detour.columns)))
ax3.set_yticks(range(len(pivot_mean_detour.index)))
ax3.set_xticklabels(pivot_mean_detour.columns)
ax3.set_yticklabels(pivot_mean_detour.index)
ax3.set_xlabel('Number of Constructions')
ax3.set_ylabel('Fleet Size')
ax3.set_title('Mean Detour % (Ratio - 1)')
for i in range(len(pivot_mean_detour.index)):
    for j in range(len(pivot_mean_detour.columns)):
        text = ax3.text(j, i, f'{pivot_mean_detour.iloc[i, j]:.2f}%',
                       ha="center", va="center", color="black", fontsize=9)
plt.colorbar(im3, ax=ax3, label='%')

# 4. Detour vs Travel Time correlation
ax4 = axes[1, 1]
scatter_data = df_agg.copy()
scatter = ax4.scatter(scatter_data['mean_travel_time'], 
                     scatter_data['mean_detour_ratio']*100,
                     c=scatter_data['num_vehicles'],
                     s=scatter_data['vehicles_with_detours']*2,
                     alpha=0.6, cmap='viridis')
ax4.set_xlabel('Mean Travel Time (s)')
ax4.set_ylabel('Mean Detour % (Ratio - 1)')
ax4.set_title('Detour vs Travel Time (point size = detour count)')
ax4.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('Fleet Size')

plt.tight_layout()
plt.savefig('detours_analysis.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved to: detours_analysis.png")

# Export detailed statistics
stats_export = pd.DataFrame({
    'metric': [
        'Total simulations',
        'Total vehicles',
        'Mean detour %',
        'Median detour %',
        'Vehicles with detour (>0%)',
        'Vehicles with major detour (>10%)',
    ],
    'value': [
        len(df_agg),
        len(arrived_df),
        f"{arrived_df['detour_pct'].mean():.2f}%",
        f"{arrived_df['detour_pct'].median():.2f}%",
        f"{vehicles_with_detour} ({vehicles_with_detour/len(arrived_df)*100:.1f}%)",
        f"{vehicles_major_detour} ({vehicles_major_detour/len(arrived_df)*100:.1f}%)",
    ]
})

print("\n" + "=" * 90)
stats_export.to_csv('detours_summary_stats.csv', index=False)
print("Summary statistics exported to: detours_summary_stats.csv")
