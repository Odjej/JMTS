"""
Generate detour summary statistics from batch experiment data.

Usage:
    python tools/generate_detour_summary.py experiment_wide_range_v2 --out=results/data/detours_summary_stats.csv

Reads per-vehicle data and generates CSV summary statistics.
"""
import sys
import os
import argparse
import pandas as pd


def generate_summary(pv_csv, out_csv):
    """Load per-vehicle data and generate comprehensive summary statistics."""
    print(f'Loading {pv_csv}...')
    df_pv = pd.read_csv(pv_csv)
    print(f'  -> {len(df_pv)} total vehicle records')
    
    # Filter to arrived vehicles
    arrived_df = df_pv[df_pv['arrived'] == 1].copy()
    print(f'  -> {len(arrived_df)} arrived vehicles')
    
    # Calculate detour percentage
    arrived_df['detour_pct'] = ((arrived_df['distance_travelled'] / arrived_df['initial_route_length']) - 1) * 100
    
    # Get unique simulation scenarios
    simulations = df_pv.groupby(['num_vehicles', 'num_constructions', 'seed']).size().reset_index(name='count')
    num_simulations = len(simulations)
    
    # Calculate statistics
    stats = {
        'Total simulations': num_simulations,
        'Total vehicles (all)': len(df_pv),
        'Total vehicles (arrived)': len(arrived_df),
        'Mean detour %': f'{arrived_df["detour_pct"].mean():.2f}%',
        'Median detour %': f'{arrived_df["detour_pct"].median():.2f}%',
        'Std dev detour %': f'{arrived_df["detour_pct"].std():.2f}%',
        'Min detour %': f'{arrived_df["detour_pct"].min():.2f}%',
        'Max detour %': f'{arrived_df["detour_pct"].max():.2f}%',
    }
    
    # Vehicle detour categories
    any_detour = (arrived_df['detour_pct'] > 0).sum()
    any_detour_pct = any_detour / len(arrived_df) * 100
    stats['Vehicles with any detour (>0%)'] = f'{any_detour:,} ({any_detour_pct:.1f}%)'
    
    major_detour = (arrived_df['detour_pct'] > 10).sum()
    major_detour_pct = major_detour / len(arrived_df) * 100
    stats['Vehicles with major detour (>10%)'] = f'{major_detour:,} ({major_detour_pct:.1f}%)'
    
    severe_detour = (arrived_df['detour_pct'] > 20).sum()
    severe_detour_pct = severe_detour / len(arrived_df) * 100
    stats['Vehicles with severe detour (>20%)'] = f'{severe_detour:,} ({severe_detour_pct:.1f}%)'
    
    extreme_detour = (arrived_df['detour_pct'] > 50).sum()
    extreme_detour_pct = extreme_detour / len(arrived_df) * 100
    stats['Vehicles with extreme detour (>50%)'] = f'{extreme_detour:,} ({extreme_detour_pct:.1f}%)'
    
    # Rerouting statistics
    mean_replans = arrived_df['num_replans'].mean()
    max_replans = arrived_df['num_replans'].max()
    vehicles_rerouted = (arrived_df['num_replans'] > 0).sum()
    vehicles_rerouted_pct = vehicles_rerouted / len(arrived_df) * 100
    
    stats['Mean replans per vehicle'] = f'{mean_replans:.2f}'
    stats['Max replans observed'] = f'{int(max_replans)}'
    stats['Vehicles with at least 1 replan'] = f'{vehicles_rerouted:,} ({vehicles_rerouted_pct:.1f}%)'
    
    # By construction count
    stats[''] = ''  # Separator
    stats['Detour metrics by construction count:'] = ''
    for const_count in sorted(arrived_df['num_constructions'].unique()):
        subset = arrived_df[arrived_df['num_constructions'] == const_count]
        mean_det = subset['detour_pct'].mean()
        any_det = (subset['detour_pct'] > 0).sum() / len(subset) * 100 if len(subset) > 0 else 0
        stats[f'  {int(const_count)} constructions: mean detour'] = f'{mean_det:.2f}%'
        stats[f'  {int(const_count)} constructions: % with detour'] = f'{any_det:.1f}%'
    
    # By fleet size
    stats['  '] = ''  # Separator
    stats['Detour metrics by fleet size:'] = ''
    for fleet_size in sorted(arrived_df['num_vehicles'].unique()):
        subset = arrived_df[arrived_df['num_vehicles'] == fleet_size]
        mean_det = subset['detour_pct'].mean()
        any_det = (subset['detour_pct'] > 0).sum() / len(subset) * 100 if len(subset) > 0 else 0
        stats[f'  {int(fleet_size)} vehicles: mean detour'] = f'{mean_det:.2f}%'
        stats[f'  {int(fleet_size)} vehicles: % with detour'] = f'{any_det:.1f}%'
    
    # Write CSV
    os.makedirs(os.path.dirname(out_csv) if os.path.dirname(out_csv) else '.', exist_ok=True)
    
    with open(out_csv, 'w', newline='') as f:
        f.write('metric,value\n')
        for metric, value in stats.items():
            # Escape quotes in values
            value_str = str(value).replace('"', '""')
            f.write(f'{metric},{value_str}\n')
    
    print(f'\nSummary statistics:')
    for metric, value in stats.items():
        if metric and not metric.startswith('  '):
            print(f'  {metric}: {value}')
    
    print(f'\nWritten to {out_csv}')
    return out_csv


def main():
    parser = argparse.ArgumentParser(
        description='Generate detour summary statistics from batch per-vehicle data.'
    )
    parser.add_argument('base_name',
                       help='Base name of per-vehicle CSV (e.g., "experiment_wide_range_v2")')
    parser.add_argument('--out', default='results/data/detours_summary_stats.csv',
                       help='Output CSV file path')
    args = parser.parse_args()
    
    pv_csv = f'{args.base_name}_per_vehicle.csv'
    
    if not os.path.exists(pv_csv):
        print(f'Error: Per-vehicle CSV not found: {pv_csv}')
        sys.exit(1)
    
    generate_summary(pv_csv, args.out)


if __name__ == '__main__':
    main()
