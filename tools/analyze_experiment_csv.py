import csv
import statistics
from collections import defaultdict

def analyze(fn):
    rows=[]
    with open(fn, newline='') as f:
        r=csv.DictReader(f)
        for row in r:
            row['num_vehicles']=int(row['num_vehicles'])
            row['num_constructions']=int(row['num_constructions'])
            row['mean_travel_time']=float(row['mean_travel_time'])
            # handle possible missing speed columns
            try:
                row['mean_avg_speed']=float(row.get('mean_avg_speed', 'nan'))
            except Exception:
                row['mean_avg_speed']=float('nan')
            rows.append(row)
    by_v=defaultdict(list)
    by_c=defaultdict(list)
    for row in rows:
        by_v[row['num_vehicles']].append(row)
        by_c[row['num_constructions']].append(row)

    print('Summary by num_vehicles:')
    for v in sorted(by_v):
        times=[r['mean_travel_time'] for r in by_v[v]]
        speeds=[r['mean_avg_speed'] for r in by_v[v] if not (r['mean_avg_speed']!=r['mean_avg_speed'])]
        print(f'  vehicles={v}: mean_time={statistics.mean(times):.2f}s, median_time={statistics.median(times):.2f}s, mean_speed={statistics.mean(speeds):.3f} m/s')

    print('\nSummary by num_constructions:')
    for c in sorted(by_c):
        times=[r['mean_travel_time'] for r in by_c[c]]
        speeds=[r['mean_avg_speed'] for r in by_c[c] if not (r['mean_avg_speed']!=r['mean_avg_speed'])]
        print(f'  constructions={c}: mean_time={statistics.mean(times):.2f}s, median_time={statistics.median(times):.2f}s, mean_speed={statistics.mean(speeds):.3f} m/s')

if __name__=='__main__':
    analyze('experiment_area_reduced.csv')
