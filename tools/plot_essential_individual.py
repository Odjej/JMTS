"""
Four essential plots - created individually, one per file
Publication-ready in same style as plot_experiment_results.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

# Set style to match existing report plots
try:
    rcParams['text.usetex'] = True
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Computer Modern']
except Exception:
    rcParams['font.family'] = 'serif'
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['font.serif'] = ['Times New Roman']

# Load data
df_agg = pd.read_csv('experiment_wide_range_v2.csv')
df_pv = pd.read_csv('experiment_wide_range_v2_per_vehicle.csv')

# Filter to arrived vehicles
arrived_df = df_pv[df_pv['arrived'] == 1].copy()
arrived_df['detour_pct'] = ((arrived_df['distance_travelled'] / arrived_df['initial_route_length']) - 1) * 100

# ============================================================================
# PLOT 1: Mean Detour % vs Fleet Size (showing the paradox)
# ============================================================================
print("Creating Plot 1")
fig1, ax1 = plt.subplots(figsize=(12, 7))

fleet_stats = arrived_df.groupby('num_vehicles').agg({
    'detour_pct': ['mean', 'std'],
    'vehicle_id': 'count'
}).reset_index()
fleet_stats.columns = ['fleet_size', 'mean_detour', 'std_detour', 'n_vehicles']

# Also get travel time by fleet
travel_time_by_fleet = df_agg.groupby('num_vehicles')['mean_travel_time'].mean()

# Create twin axis for travel time
ax1_twin = ax1.twinx()

# Plot 1: Mean detour with error bars
line1 = ax1.plot(fleet_stats['fleet_size'], fleet_stats['mean_detour'], 
                 'o-', linewidth=3, markersize=12, label='Mean Detour %', 
                 color='steelblue')
ax1.fill_between(fleet_stats['fleet_size'], 
                 fleet_stats['mean_detour'] - fleet_stats['std_detour'],
                 fleet_stats['mean_detour'] + fleet_stats['std_detour'],
                 alpha=0.2, color='steelblue', label='±1 Std Dev')

# Plot 2: Travel time on twin axis
line2 = ax1_twin.plot(travel_time_by_fleet.index, travel_time_by_fleet.values, 
                      's-', linewidth=3, markersize=12, label='Mean Travel Time', 
                      color='darkred')

ax1.set_xlabel('Fleet Size (vehicles)', fontsize=13, fontweight='bold')
ax1.set_ylabel(r'Mean Detour Distance (\%)', fontsize=13, fontweight='bold', color='steelblue')
ax1_twin.set_ylabel('Mean Travel Time (s)', fontsize=13, fontweight='bold', color='darkred')
ax1.tick_params(axis='y', labelcolor='steelblue', labelsize=11)
ax1_twin.tick_params(axis='y', labelcolor='darkred', labelsize=11)
ax1.set_title('Mean Detour Distance and Travel Time vs Fleet Size', 
              fontsize=14, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')
ax1.tick_params(axis='x', labelsize=11)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12, framealpha=0.95)

# Add annotations
ax1.annotate('Low density:\nLarge detours\n(more options)', xy=(10, 7.5), xytext=(20, 6.5),
            arrowprops=dict(arrowstyle='->', color='green', lw=2.5),
            fontsize=11, fontweight='bold', color='green',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='lightgreen', alpha=0.8, edgecolor='green', linewidth=2))
ax1.annotate('High density:\nSmall detours\n(no alternatives)', xy=(1000, 0.47), xytext=(200, -3.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2.5),
            fontsize=11, fontweight='bold', color='red',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='lightcoral', alpha=0.8, edgecolor='red', linewidth=2))

plt.tight_layout()
plt.savefig('plot_01_paradox.png', dpi=600, bbox_inches='tight')
plt.savefig('plot_01_paradox.pdf', bbox_inches='tight')
print("✓ Saved")
plt.close()

# ============================================================================
# PLOT 2: Percentage of vehicles affected by detours
# ============================================================================
print("Creating Plot 2")
fig2, ax2 = plt.subplots(figsize=(12, 7))

detour_impact = []
for fleet_size in sorted(arrived_df['num_vehicles'].unique()):
    fleet_data = arrived_df[arrived_df['num_vehicles'] == fleet_size]
    pct_any_detour = (fleet_data['detour_pct'] > 0).sum() / len(fleet_data) * 100
    pct_major_detour = (fleet_data['detour_pct'] > 10).sum() / len(fleet_data) * 100
    detour_impact.append({
        'fleet_size': fleet_size,
        'any_detour': pct_any_detour,
        'major_detour': pct_major_detour
    })

detour_impact_df = pd.DataFrame(detour_impact)

x = np.arange(len(detour_impact_df))
width = 0.35

bars1 = ax2.bar(x - width/2, detour_impact_df['any_detour'], width, 
                label=r'Any Detour $(>0\%)$', color='skyblue', edgecolor='black', linewidth=2)
bars2 = ax2.bar(x + width/2, detour_impact_df['major_detour'], width,
                label=r'Major Detour $(>10\%)$', color='indianred', edgecolor='black', linewidth=2)

ax2.set_xlabel('Fleet Size (vehicles)', fontsize=13, fontweight='bold')
ax2.set_ylabel('% of Vehicles', fontsize=13, fontweight='bold')
ax2.set_title('Percentage of Vehicles Affected by Detours vs Fleet Size', 
              fontsize=14, fontweight='bold', pad=20)
ax2.set_xticks(x)
ax2.set_xticklabels([f'{int(fs)}' for fs in detour_impact_df['fleet_size']], fontsize=11)
ax2.legend(fontsize=12, loc='upper right', framealpha=0.95)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, 80)
ax2.tick_params(axis='y', labelsize=11)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('plot_02_impact.png', dpi=600, bbox_inches='tight')
plt.savefig('plot_02_impact.pdf', bbox_inches='tight')
print("✓ Saved")
plt.close()

# ============================================================================
# PLOT 3: Heatmap showing effectiveness of detours
# ============================================================================
print("Creating Plot 3")
fig3, ax3 = plt.subplots(figsize=(12, 7))

# Create a matrix: fleet size vs constructions, showing if detours help
effectiveness = []
for nv in sorted(arrived_df['num_vehicles'].unique()):
    row = []
    for nc in sorted(arrived_df['num_constructions'].unique()):
        subset = arrived_df[(arrived_df['num_vehicles'] == nv) & 
                           (arrived_df['num_constructions'] == nc)]
        # Calculate: mean detour as % of route
        if len(subset) > 0:
            mean_detour = subset['detour_pct'].mean()
            row.append(mean_detour)
        else:
            row.append(0)
    effectiveness.append(row)

effectiveness_arr = np.array(effectiveness)
fleet_sizes_unique = sorted(arrived_df['num_vehicles'].unique())
const_counts_unique = sorted(arrived_df['num_constructions'].unique())

im = ax3.imshow(effectiveness_arr, cmap='RdYlGn_r', aspect='auto', vmin=-0.01, vmax=25)
ax3.set_xticks(range(len(const_counts_unique)))
ax3.set_yticks(range(len(fleet_sizes_unique)))
ax3.set_xticklabels([f'{int(c)}' for c in const_counts_unique], fontsize=11)
ax3.set_yticklabels([f'{int(f)}' for f in fleet_sizes_unique], fontsize=11)
ax3.set_xlabel('Number of Constructions', fontsize=13, fontweight='bold')
ax3.set_ylabel('Fleet Size (vehicles)', fontsize=13, fontweight='bold')
ax3.set_title(r'Mean Detour Cost (\% extra distance) by Fleet Size and Constructions', 
              fontsize=14, fontweight='bold', pad=20)

# Add values to cells
for i in range(len(fleet_sizes_unique)):
    for j in range(len(const_counts_unique)):
        text = ax3.text(j, i, f'{effectiveness_arr[i, j]:.2f}%',
                       ha="center", va="center", color="black", fontsize=11, fontweight='bold')

cbar = plt.colorbar(im, ax=ax3, label='Mean Detour %')
cbar.ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig('plot_03_effectiveness.png', dpi=600, bbox_inches='tight')
plt.savefig('plot_03_effectiveness.pdf', bbox_inches='tight')
print("✓ Saved")
plt.close()

# ============================================================================
# PLOT 4: Distribution comparison - low density vs high density
# ============================================================================
print("Creating Plot 4: Distribution Comparison...")
fig4, ax4 = plt.subplots(figsize=(12, 7))

low_density_data = arrived_df[arrived_df['num_vehicles'] <= 100]['detour_pct']
high_density_data = arrived_df[arrived_df['num_vehicles'] >= 500]['detour_pct']

# Create histograms
ax4.hist(low_density_data, bins=50, alpha=0.6, label='Low Density (10-100 vehicles)', 
         color='green', edgecolor='darkgreen', linewidth=1.5)
ax4.hist(high_density_data, bins=50, alpha=0.6, label='High Density (500-1000 vehicles)', 
         color='red', edgecolor='darkred', linewidth=1.5)

# Add mean lines
ax4.axvline(low_density_data.mean(), color='darkgreen', linestyle='--', linewidth=3, 
            label=f'Mean Low Density: {low_density_data.mean():.2f}%')
ax4.axvline(high_density_data.mean(), color='darkred', linestyle='--', linewidth=3,
            label=f'Mean High Density: {high_density_data.mean():.2f}%')

ax4.set_xlabel(r'Detour Distance (\%)', fontsize=13, fontweight='bold')
ax4.set_ylabel('Number of Vehicles', fontsize=13, fontweight='bold')
ax4.set_title('Detour Distance vs Vehicle Count Distribution\nLow Density vs High Density', 
              fontsize=14, fontweight='bold', pad=20)
ax4.legend(fontsize=11, loc='lower right', framealpha=0.95)
ax4.grid(True, alpha=0.3, axis='y')
ax4.tick_params(axis='both', labelsize=11)
ax4.set_xlim(-30, 150)

# Add text box with interpretation
textstr = 'At low density: Detours are\nEFFECTIVE - vehicles choose\nthem to avoid congestion\n\nAt high density: Detours are\nFORCED - network saturation\nleaves no good alternatives'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=2)
ax4.text(0.98, 0.97, textstr, transform=ax4.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right', bbox=props, fontweight='bold')

plt.tight_layout()
plt.savefig('plot_04_distribution.png', dpi=600, bbox_inches='tight')
plt.savefig('plot_04_distribution.pdf', bbox_inches='tight')
print("✓ Saved")
plt.close()

