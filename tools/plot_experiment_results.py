"""
Plot experiment CSV results in report-ready format.

Usage:
    python tools/plot_experiment_results.py exp_factor3.csv --out-dir=results/exp_factor3

Produces:
 - mean_travel_time_vs_constructions.png/pdf
 - mean_avg_speed_vs_constructions.png/pdf
 - report.md (summary with embedded images)

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


def plot_travel_time(df, out_dir, prefix='results'):
    # try to use LaTeX for text if available; otherwise use serif math fonts
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
        std_time=('mean_travel_time', 'std'))
    grouped = grouped.reset_index()
    vehicles = sorted(df['num_vehicles'].unique())
    constructions = sorted(df['num_constructions'].unique())

    plt.figure(figsize=(9,6))
    cmap = plt.get_cmap('tab10')
    for i, v in enumerate(vehicles):
        sub = grouped[grouped['num_vehicles'] == v]
        # fill missing construction points with nan
        x = sub['num_constructions']
        y = sub['mean_time']
        yerr = sub['std_time'].fillna(0)
        plt.plot(x, y, marker='o', label=f'{v} vehicles', color=cmap(i%10))
        # shaded error
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=cmap(i%10))

    plt.xlabel('Number of constructions')
    plt.ylabel('Mean travel time (s)')
    plt.title('Mean travel time vs constructions by vehicle count')
    plt.grid(alpha=0.3)
    plt.legend(title='Fleet size')
    png = os.path.join(out_dir, f'{prefix}_mean_travel_time_vs_constructions.png')
    pdf = os.path.join(out_dir, f'{prefix}_mean_travel_time_vs_constructions.pdf')
    plt.tight_layout()
    plt.savefig(png, dpi=300)
    plt.savefig(pdf)
    plt.close()
    return png, pdf


def plot_avg_speed(df, out_dir, prefix='results'):
    # ensure LaTeX-like fonts consistent with travel-time plot
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
        std_speed=('mean_avg_speed', 'std'))
    grouped = grouped.reset_index()
    vehicles = sorted(df['num_vehicles'].unique())
    constructions = sorted(df['num_constructions'].unique())

    plt.figure(figsize=(9,6))
    cmap = plt.get_cmap('tab10')
    for i, v in enumerate(vehicles):
        sub = grouped[grouped['num_vehicles'] == v]
        x = sub['num_constructions']
        y = sub['mean_speed']
        yerr = sub['std_speed'].fillna(0)
        plt.plot(x, y, marker='o', label=f'{v} vehicles', color=cmap(i%10))
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=cmap(i%10))

    plt.xlabel('Number of constructions')
    plt.ylabel('Mean average speed (m/s)')
    plt.title('Mean average speed vs constructions by vehicle count')
    plt.grid(alpha=0.3)
    plt.legend(title='Fleet size')
    png = os.path.join(out_dir, f'{prefix}_mean_avg_speed_vs_constructions.png')
    pdf = os.path.join(out_dir, f'{prefix}_mean_avg_speed_vs_constructions.pdf')
    plt.tight_layout()
    plt.savefig(png, dpi=300)
    plt.savefig(pdf)
    plt.close()
    return png, pdf


def create_report(out_dir, images, csv_path):
    md = []
    md.append('# Experiment Results')
    md.append('')
    md.append(f'Source CSV: `{csv_path}`')
    md.append('')
    md.append('Generated figures:')
    md.append('')
    for img in images:
        fname = os.path.basename(img)
        md.append(f'![{fname}]({fname})')
        md.append('')
    md.append('')
    md.append('## Notes')
    md.append('- Plots show mean values aggregated across seeds. Error shading shows stddev when available.')

    md_path = os.path.join(out_dir, 'report.md')
    with open(md_path, 'w', encoding='utf8') as f:
        f.write('\n'.join(md))
    return md_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv', help='experiment CSV file')
    p.add_argument('--out-dir', default='results/exp_plots', help='output directory')
    p.add_argument('--prefix', default='results', help='file prefix')
    args = p.parse_args()

    df = load_csv(args.csv)
    out_dir = ensure_out_dir(args.out_dir)
    t1 = plot_travel_time(df, out_dir, prefix=args.prefix)
    t2 = plot_avg_speed(df, out_dir, prefix=args.prefix)
    imgs = [t1[0], t1[1], t2[0], t2[1]]
    md = create_report(out_dir, imgs, args.csv)
    print('Wrote images:', imgs)
    print('Wrote report:', md)

if __name__ == '__main__':
    main()
