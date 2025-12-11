"""
Plot experiment batch results with mixed construction types.

Usage:
    python tools/plot_batch_results.py experiment_wide_range_v2.csv --out-dir=results/batch

Produces:
 - mean_travel_time_vs_constructions.png/pdf
 - mean_avg_speed_vs_constructions.png/pdf
"""
import os
import sys
import argparse

try:
    import pandas as pd
except Exception:
    pd = None

import csv
import math
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


def load_csv(path):
    if pd is not None:
        return pd.read_csv(path)
    # fallback to csv module -> convert to simple dict-of-lists
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        rows = list(r)
    # try to coerce numeric columns
    cols = rows[0].keys() if rows else []
    data = {c: [] for c in cols}
    for row in rows:
        for c in cols:
            v = row[c]
            # numeric?
            try:
                if v == '' or v is None:
                    data[c].append(np.nan)
                else:
                    if '.' in v or 'e' in v.lower():
                        data[c].append(float(v))
                    else:
                        data[c].append(int(v))
            except Exception:
                data[c].append(v)
    import pandas as _pd
    return _pd.DataFrame(data)


def ensure_out_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def plot_travel_time(df, out_dir, prefix='batch'):
    """Plot mean travel time vs construction count by vehicle count with error bands."""
    try:
        rcParams['text.usetex'] = True
        rcParams['font.family'] = 'serif'
        rcParams['font.serif'] = ['Computer Modern']
    except Exception:
        rcParams['font.family'] = 'serif'
        rcParams['mathtext.fontset'] = 'stix'
        rcParams['font.serif'] = ['Times New Roman']
    
    # Group by vehicles and constructions: compute mean and std
    grouped = df.groupby(['num_vehicles', 'num_constructions']).agg(
        mean_time=('mean_travel_time', 'mean'),
        std_time=('mean_travel_time', 'std'),
        count=('mean_travel_time', 'count')
    ).reset_index()
    
    vehicles = sorted(df['num_vehicles'].unique())
    constructions = sorted(df['num_constructions'].unique())

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap('tab10')
    
    for i, v in enumerate(vehicles):
        sub = grouped[grouped['num_vehicles'] == v]
        x = sub['num_constructions'].values
        y = sub['mean_time'].values
        yerr = sub['std_time'].fillna(0).values
        
        # plot line with markers
        ax.plot(x, y, marker='o', label=f'{v} vehicles', color=cmap(i % 10), linewidth=2)
        # shaded error band
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=cmap(i % 10))

    ax.set_xlabel('Number of construction events')
    ax.set_ylabel('Mean travel time (s)')
    ax.set_title('Travel time vs construction events\n(mixed construction types with probabilistic selection)')
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(title='Fleet size', loc='best')
    
    png = os.path.join(out_dir, f'{prefix}_mean_travel_time_vs_constructions.png')
    pdf = os.path.join(out_dir, f'{prefix}_mean_travel_time_vs_constructions.pdf')
    fig.tight_layout()
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    return png, pdf


def plot_avg_speed(df, out_dir, prefix='batch'):
    """Plot mean average speed vs construction count by vehicle count with error bands."""
    try:
        rcParams['text.usetex'] = True
        rcParams['font.family'] = 'serif'
        rcParams['font.serif'] = ['Computer Modern']
    except Exception:
        rcParams['font.family'] = 'serif'
        rcParams['mathtext.fontset'] = 'stix'
        rcParams['font.serif'] = ['Times New Roman']
    
    grouped = df.groupby(['num_vehicles', 'num_constructions']).agg(
        mean_speed=('mean_avg_speed', 'mean'),
        std_speed=('mean_avg_speed', 'std'),
        count=('mean_avg_speed', 'count')
    ).reset_index()
    
    vehicles = sorted(df['num_vehicles'].unique())
    constructions = sorted(df['num_constructions'].unique())

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap('tab10')
    
    for i, v in enumerate(vehicles):
        sub = grouped[grouped['num_vehicles'] == v]
        x = sub['num_constructions'].values
        y = sub['mean_speed'].values
        yerr = sub['std_speed'].fillna(0).values
        
        # plot line with markers
        ax.plot(x, y, marker='s', label=f'{v} vehicles', color=cmap(i % 10), linewidth=2)
        # shaded error band
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=cmap(i % 10))

    ax.set_xlabel('Number of construction events')
    ax.set_ylabel('Mean average speed (m/s)')
    ax.set_title('Average speed vs construction events\n(mixed construction types with probabilistic selection)')
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(title='Fleet size', loc='best')
    
    png = os.path.join(out_dir, f'{prefix}_mean_avg_speed_vs_constructions.png')
    pdf = os.path.join(out_dir, f'{prefix}_mean_avg_speed_vs_constructions.pdf')
    fig.tight_layout()
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    return png, pdf


def main():
    p = argparse.ArgumentParser(description='Plot batch experiment results with mixed construction types.')
    p.add_argument('csv', help='experiment CSV file (e.g., experiment_wide_range_v2.csv)')
    p.add_argument('--out-dir', default='results/batch_plots', help='output directory for plots')
    p.add_argument('--prefix', default='batch', help='file prefix for output images')
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print(f'Error: CSV file not found: {args.csv}')
        sys.exit(1)

    print(f'Loading {args.csv}...')
    df = load_csv(args.csv)
    print(f'Loaded {len(df)} rows')
    
    out_dir = ensure_out_dir(args.out_dir)
    
    print(f'Plotting travel time...')
    t1 = plot_travel_time(df, out_dir, prefix=args.prefix)
    print(f'  -> {t1[0]}')
    print(f'  -> {t1[1]}')
    
    print(f'Plotting average speed...')
    t2 = plot_avg_speed(df, out_dir, prefix=args.prefix)
    print(f'  -> {t2[0]}')
    print(f'  -> {t2[1]}')
    
    print('Done.')


if __name__ == '__main__':
    main()
