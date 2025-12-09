"""
Comprehensive detour analysis plots
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

# Load data
df_agg = pd.read_csv('experiment_wide_range_v2.csv')
df_pv = pd.read_csv('experiment_wide_range_v2_per_vehicle.csv')

# Filter to arrived vehicles
arrived_df = df_pv[df_pv['arrived'] == 1].copy()
arrived_df['detour_pct'] = ((arrived_df['distance_travelled'] / arrived_df['initial_route_length']) - 1) * 100

# Create main figure with 3x3 subplots
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Overall detour distribution (histogram)
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(arrived_df['detour_pct'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax1.axvline(arrived_df['detour_pct'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {arrived_df["detour_pct"].mean():.2f}%')
ax1.axvline(arrived_df['detour_pct'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {arrived_df["detour_pct"].median():.2f}%')
ax1.set_xlabel('Detour Distance %')
ax1.set_ylabel('Number of Vehicles')
ax1.set_title('Overall Detour Distribution\n(All 16,740 vehicles)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Detour by fleet size (violin plot)
ax2 = fig.add_subplot(gs[0, 1])
fleet_data = [arrived_df[arrived_df['num_vehicles'] == v]['detour_pct'].values for v in sorted(arrived_df['num_vehicles'].unique())]
parts = ax2.violinplot(fleet_data, positions=range(len(fleet_data)), showmeans=True, showmedians=True)
ax2.set_xticks(range(len(fleet_data)))
ax2.set_xticklabels([str(v) for v in sorted(arrived_df['num_vehicles'].unique())])
ax2.set_xlabel('Fleet Size (vehicles)')
ax2.set_ylabel('Detour %')
ax2.set_title('Detour Distribution by Fleet Size')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(-30, 150)

# 3. Detour by construction count (violin plot)
ax3 = fig.add_subplot(gs[0, 2])
const_data = [arrived_df[arrived_df['num_constructions'] == c]['detour_pct'].values for c in sorted(arrived_df['num_constructions'].unique())]
parts = ax3.violinplot(const_data, positions=range(len(const_data)), showmeans=True, showmedians=True)
ax3.set_xticks(range(len(const_data)))
ax3.set_xticklabels([str(c) for c in sorted(arrived_df['num_constructions'].unique())])
ax3.set_xlabel('Number of Constructions')
ax3.set_ylabel('Detour %')
ax3.set_title('Detour Distribution by Construction Count')
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(-30, 150)

# 4. Heatmap: Vehicles with detours (>0%)
ax4 = fig.add_subplot(gs[1, 0])
vehicles_detoured = df_agg.pivot_table(
    values='vehicles_with_detours',
    index='num_vehicles',
    columns='num_constructions',
    aggfunc='mean'
)
sns.heatmap(vehicles_detoured, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax4, 
            cbar_kws={'label': 'Avg Count'})
ax4.set_title('Vehicles with >10% Longer Routes\n(By Fleet Size & Constructions)')
ax4.set_xlabel('Number of Constructions')
ax4.set_ylabel('Fleet Size')

# 5. Heatmap: Mean detour ratio
ax5 = fig.add_subplot(gs[1, 1])
mean_detour_pct = df_agg.pivot_table(
    values='mean_detour_ratio',
    index='num_vehicles',
    columns='num_constructions',
    aggfunc='mean'
) * 100 - 100  # Convert ratio to percentage
sns.heatmap(mean_detour_pct, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax5,
            cbar_kws={'label': 'Detour %'}, vmin=-0.5, vmax=3)
ax5.set_title('Mean Detour % by Scenario\n(Ratio - 1, converted to %)')
ax5.set_xlabel('Number of Constructions')
ax5.set_ylabel('Fleet Size')

# 6. Heatmap: % of vehicles with any detour
ax6 = fig.add_subplot(gs[1, 2])
detour_pct_by_scenario = arrived_df.groupby(['num_vehicles', 'num_constructions']).apply(
    lambda x: (x['detour_pct'] > 0).sum() / len(x) * 100
).unstack()
sns.heatmap(detour_pct_by_scenario, annot=True, fmt='.1f', cmap='Blues', ax=ax6,
            cbar_kws={'label': '%'})
ax6.set_title('% Vehicles with Any Detour (>0%)\n(By Fleet Size & Constructions)')
ax6.set_xlabel('Number of Constructions')
ax6.set_ylabel('Fleet Size')

# 7. Box plot: Fleet size comparison
ax7 = fig.add_subplot(gs[2, 0])
arrived_df_sorted = arrived_df.sort_values('num_vehicles')
bp = ax7.boxplot([arrived_df_sorted[arrived_df_sorted['num_vehicles'] == v]['detour_pct'].values 
                   for v in sorted(arrived_df_sorted['num_vehicles'].unique())],
                  labels=[str(v) for v in sorted(arrived_df_sorted['num_vehicles'].unique())],
                  patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax7.set_xlabel('Fleet Size (vehicles)')
ax7.set_ylabel('Detour %')
ax7.set_title('Detour Box Plot by Fleet Size')
ax7.grid(True, alpha=0.3, axis='y')
ax7.axhline(0, color='red', linestyle='--', alpha=0.5)

# 8. Box plot: Construction count comparison
ax8 = fig.add_subplot(gs[2, 1])
arrived_df_sorted = arrived_df.sort_values('num_constructions')
bp = ax8.boxplot([arrived_df_sorted[arrived_df_sorted['num_constructions'] == c]['detour_pct'].values 
                   for c in sorted(arrived_df_sorted['num_constructions'].unique())],
                  labels=[str(c) for c in sorted(arrived_df_sorted['num_constructions'].unique())],
                  patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightcoral')
ax8.set_xlabel('Number of Constructions')
ax8.set_ylabel('Detour %')
ax8.set_title('Detour Box Plot by Construction Count')
ax8.grid(True, alpha=0.3, axis='y')
ax8.axhline(0, color='red', linestyle='--', alpha=0.5)

# 9. Scatter: Detour vs Travel Time (colored by fleet size)
ax9 = fig.add_subplot(gs[2, 2])
for fleet_size in sorted(df_agg['num_vehicles'].unique()):
    data = df_agg[df_agg['num_vehicles'] == fleet_size]
    ax9.scatter(data['mean_travel_time'], 
               (data['mean_detour_ratio']-1)*100,
               s=data['vehicles_with_detours']*3 + 10,
               alpha=0.6, label=f'{fleet_size} veh')
ax9.set_xlabel('Mean Travel Time (s)')
ax9.set_ylabel('Mean Detour %')
ax9.set_title('Detour vs Travel Time\n(Point size = detour count)')
ax9.legend(loc='best', fontsize=8)
ax9.grid(True, alpha=0.3)

# Main title
fig.suptitle('Traffic Simulation Detour Analysis - Experiment Wide Range v2', 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('detours_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
print("Saved: detours_analysis_comprehensive.png (3x3 comprehensive grid)")

# Create a second figure: Focused detour metrics
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Mean detour by scenario
ax = axes[0, 0]
scenario_stats = arrived_df.groupby(['num_vehicles', 'num_constructions'])['detour_pct'].mean().reset_index()
pivot = scenario_stats.pivot(index='num_vehicles', columns='num_constructions', values='detour_pct')
pivot.plot(kind='line', ax=ax, marker='o', linewidth=2, markersize=8)
ax.set_xlabel('Fleet Size (vehicles)')
ax.set_ylabel('Mean Detour %')
ax.set_title('Mean Detour % by Fleet Size\n(Lines = construction counts)')
ax.legend(title='Constructions', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# 2. Percentage of vehicles with detours
ax = axes[0, 1]
pct_detoured = arrived_df.groupby(['num_vehicles', 'num_constructions']).apply(
    lambda x: (x['detour_pct'] > 0).sum() / len(x) * 100
).reset_index(name='pct_detoured')
pivot_pct = pct_detoured.pivot(index='num_vehicles', columns='num_constructions', values='pct_detoured')
pivot_pct.plot(kind='line', ax=ax, marker='s', linewidth=2, markersize=8)
ax.set_xlabel('Fleet Size (vehicles)')
ax.set_ylabel('% Vehicles with Detours')
ax.set_title('% of Vehicles Taking Any Detour\n(Lines = construction counts)')
ax.legend(title='Constructions', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# 3. Distribution comparison: 0 vs 10 constructions
ax = axes[1, 0]
data_no_const = arrived_df[arrived_df['num_constructions'] == 0]['detour_pct']
data_with_const = arrived_df[arrived_df['num_constructions'] == 10]['detour_pct']
ax.hist([data_no_const, data_with_const], bins=40, label=['0 Constructions', '10 Constructions'], 
        color=['green', 'red'], alpha=0.6, edgecolor='black')
ax.set_xlabel('Detour %')
ax.set_ylabel('Frequency')
ax.set_title('Detour Distribution: No Constructions vs 10 Constructions')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-10, 50)

# 4. Summary statistics table
ax = axes[1, 1]
ax.axis('off')
summary_data = [
    ['Metric', 'Value'],
    ['Total Vehicles', f'{len(arrived_df):,}'],
    ['Mean Detour', f'{arrived_df["detour_pct"].mean():.2f}%'],
    ['Median Detour', f'{arrived_df["detour_pct"].median():.2f}%'],
    ['Std Dev', f'{arrived_df["detour_pct"].std():.2f}%'],
    ['Min Detour', f'{arrived_df["detour_pct"].min():.2f}%'],
    ['Max Detour', f'{arrived_df["detour_pct"].max():.2f}%'],
    ['', ''],
    ['Vehicles with detour (>0%)', f'{(arrived_df["detour_pct"] > 0).sum():,} ({(arrived_df["detour_pct"] > 0).sum()/len(arrived_df)*100:.1f}%)'],
    ['Vehicles with major detour (>10%)', f'{(arrived_df["detour_pct"] > 10).sum():,} ({(arrived_df["detour_pct"] > 10).sum()/len(arrived_df)*100:.1f}%)'],
    ['Vehicles with severe detour (>20%)', f'{(arrived_df["detour_pct"] > 20).sum():,} ({(arrived_df["detour_pct"] > 20).sum()/len(arrived_df)*100:.1f}%)'],
]

table = ax.table(cellText=summary_data, cellLoc='left', loc='center', 
                colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color header row
for i in range(2):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color alternating rows
for i in range(1, len(summary_data)):
    for j in range(2):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')

ax.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

fig2.suptitle('Detour Metrics Summary', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('detours_summary_plots.png', dpi=300, bbox_inches='tight')
print("Saved: detours_summary_plots.png (summary statistics)")

# Create a third figure: Scenario deep dive
fig3 = plt.figure(figsize=(16, 10))
gs3 = fig3.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Get top scenarios
top_detour_scenarios = df_agg.nlargest(6, 'vehicles_with_detours')

for idx, (_, scenario) in enumerate(top_detour_scenarios.iterrows()):
    ax = fig3.add_subplot(gs3[idx // 3, idx % 3])
    
    # Filter data for this scenario
    scenario_data = arrived_df[
        (arrived_df['num_vehicles'] == scenario['num_vehicles']) &
        (arrived_df['num_constructions'] == scenario['num_constructions']) &
        (arrived_df['seed'] == scenario['seed'])
    ]['detour_pct']
    
    ax.hist(scenario_data, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(scenario_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {scenario_data.mean():.2f}%')
    ax.set_xlabel('Detour %')
    ax.set_ylabel('Number of Vehicles')
    ax.set_title(f'Scenario: {int(scenario["num_vehicles"])} vehicles, {int(scenario["num_constructions"])} constructions\n' +
                f'Mean: {scenario_data.mean():.2f}%, Detoured (>10%): {(scenario_data > 10).sum()}')
    ax.legend()
    ax.grid(True, alpha=0.3)

fig3.suptitle('Top 6 Highest-Detour Scenarios', fontsize=14, fontweight='bold')
plt.savefig('detours_top_scenarios.png', dpi=300, bbox_inches='tight')
print("Saved: detours_top_scenarios.png (top scenarios)")

