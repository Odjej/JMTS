import random
import csv
import time
import statistics
from typing import List, Tuple

try:
    from model.network_utils import load_edges, build_graph_from_edges
except Exception:
    from .network_utils import load_edges, build_graph_from_edges

try:
    from model.env import Environment, random_graph_od
except Exception:
    from .env import Environment, random_graph_od

try:
    from model.agents import CarAgent
except Exception:
    from .agents import CarAgent

import os

DEFAULT_GPKG = os.path.join('data', 'processed', 'networks', 'drive_edges_clean.gpkg')


def spawn_cars(G, n: int, seed: int = 0) -> List[CarAgent]:
    random.seed(seed)
    cars = []
    for i in range(n):
        s, t = random_graph_od(G)
        car = CarAgent(f'car_{seed}_{i}', s, t)
        cars.append(car)
    return cars


def pick_random_edges(G, k: int, seed: int = 0):
    edges = list(G.edges())
    random.seed(seed)
    if k >= len(edges):
        return edges
    return random.sample(edges, k)


def run_single_sim(G, num_vehicles: int, num_constructions: int, construction_kind: str = 'slow', factor: float = 1.5,
                   sim_time: float = 300.0, dt: float = 1.0, seed: int = 0) -> dict:
    """Run a single simulation and return travel time statistics."""
    env = Environment(G)

    cars = spawn_cars(G, num_vehicles, seed=seed)
    for c in cars:
        # plan initial route using environment travel times
        try:
            c.plan_route(G, env=env)
        except Exception:
            c.plan_route(G)
        env.register(c)

    # apply construction events immediately at t=0 on random edges
    edges = pick_random_edges(G, num_constructions, seed=seed)
    for e in edges:
        env.apply_construction(e, kind=construction_kind, factor=factor, until=None)

    steps = max(1, int(sim_time // dt))
    for _ in range(steps):
        env.step(dt)

    # collect travel times and arrival stats
    travel_times = []
    arrived = 0
    for c in cars:
        # judge arrival: position_index at or beyond last node OR distance to end < 5 m
        try:
            if getattr(c, 'position_index', 0) >= max(0, len(getattr(c, 'route', [])) - 1):
                arrived = arrived + 1
                travel_times.append(c.travel_time)
                continue
            # fallback distance check
            end = c.end_coord
            pos = getattr(c, 'current_coord', c.start_coord)
            dx = pos[0] - end[0]
            dy = pos[1] - end[1]
            if (dx*dx + dy*dy)**0.5 < 5.0:
                arrived = arrived + 1
                travel_times.append(c.travel_time)
            else:
                # censored: did not arrive in sim_time
                travel_times.append(float('nan'))
        except Exception:
            travel_times.append(float('nan'))

    # compute stats for arrived vehicles
    arrived_times = [t for t in travel_times if not (t is None or (isinstance(t, float) and (t != t)))]
    result = {
        'num_vehicles': num_vehicles,
        'num_constructions': num_constructions,
        'seed': seed,
        'n_arrived': len(arrived_times),
        'n_total': num_vehicles,
        'arrival_rate': len(arrived_times) / max(1, num_vehicles),
    }
    if arrived_times:
        result.update({
            'mean_travel_time': statistics.mean(arrived_times),
            'median_travel_time': statistics.median(arrived_times),
            'stdev_travel_time': statistics.stdev(arrived_times) if len(arrived_times) > 1 else 0.0,
        })
    else:
        result.update({'mean_travel_time': float('nan'), 'median_travel_time': float('nan'), 'stdev_travel_time': float('nan')})

    return result


def batch_run(graph_path: str = DEFAULT_GPKG,
              vehicle_counts: List[int] = [10, 25, 50, 100, 200, 500, 1000, 2000],
              construction_counts: List[int] = [0, 1, 3, 5, 10, 20, 50, 100, 200],
              seeds: List[int] = [0, 1, 2],
              out_csv: str = 'experiment_results.csv',
              sim_time: float = 4000.0,
              dt: float = 1.0):
    print(f'Loading edges from: {graph_path}')
    edges = load_edges(graph_path)
    G = build_graph_from_edges(edges)
    rows = []
    total = len(vehicle_counts) * len(construction_counts) * len(seeds)
    i = 0
    start_time = time.time()
    for nv in vehicle_counts:
        for nc in construction_counts:
            for s in seeds:
                i += 1
                print(f'Running {i}/{total}: vehicles={nv}, constructions={nc}, seed={s}')
                res = run_single_sim(G, nv, nc, sim_time=sim_time, dt=dt, seed=s)
                rows.append(res)
    # write CSV
    keys = ['num_vehicles', 'num_constructions', 'seed', 'n_arrived', 'n_total', 'arrival_rate',
            'mean_travel_time', 'median_travel_time', 'stdev_travel_time']
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in keys})
    elapsed = time.time() - start_time
    print(f'Done. Results written to {out_csv} (elapsed {elapsed:.1f}s)')
    return out_csv


if __name__ == '__main__':
    # quick smoke run with small values
    batch_run(vehicle_counts=[10], construction_counts=[0, 1], seeds=[0, 1], out_csv='experiment_quick.csv', sim_time=400)
