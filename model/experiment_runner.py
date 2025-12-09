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


def pick_edges_on_routes(G, cars, k: int, seed: int = 0):
    """Return up to k edges sampled from the union of edges along provided agents' planned routes.

    cars: iterable of agents with a `route` attribute (list of nodes).
    Edges are returned in the same (u, v) tuple format used by the graph.
    """
    # gather route edges
    route_edges = set()
    for c in cars:
        r = getattr(c, 'route', None) or []
        for i in range(max(0, len(r) - 1)):
            u = r[i]
            v = r[i + 1]
            if G.has_edge(u, v):
                route_edges.add((u, v))
            elif G.has_edge(v, u):
                # account for undirected/bidirectional storage
                route_edges.add((v, u))

    edges = list(route_edges)
    random.seed(seed)
    if not edges:
        # fallback to global random edges
        return pick_random_edges(G, k, seed=seed)
    if k >= len(edges):
        return edges
    return random.sample(edges, k)


def run_single_sim(G, num_vehicles: int, num_constructions: int, construction_kind: str = 'slow', factor: float = 3,
                   sim_time: float = 300.0, dt: float = 1.0, seed: int = 0,
                   route_only_constructions: bool = False, track_per_vehicle: bool = False) -> dict:
    """Run a single simulation and return travel time statistics (optionally with per-vehicle logging)."""
    env = Environment(G)

    cars = spawn_cars(G, num_vehicles, seed=seed)
    
    # Initialize per-vehicle tracking if requested
    if track_per_vehicle:
        for c in cars:
            c.num_replans = 0  # Track route replans (detours)
            c.initial_route_length = None
    
    for c in cars:
        # plan initial route using environment travel times
        try:
            c.plan_route(G, env=env)
        except Exception:
            c.plan_route(G)
        
        # Store initial route length for detour detection
        if track_per_vehicle and c.route:
            try:
                from model.network_utils import nearest_node
            except Exception:
                from .network_utils import nearest_node
            route_length = 0.0
            for i in range(len(c.route) - 1):
                u, v = c.route[i], c.route[i + 1]
                try:
                    ux, uy = G.nodes[u]['x'], G.nodes[u]['y']
                    vx, vy = G.nodes[v]['x'], G.nodes[v]['y']
                    route_length += ((ux - vx)**2 + (uy - vy)**2)**0.5
                except:
                    pass
            c.initial_route_length = route_length
        
        env.register(c)

    # apply construction events immediately at t=0 on random edges
    if route_only_constructions:
        edges = pick_edges_on_routes(G, cars, num_constructions, seed=seed)
    else:
        edges = pick_random_edges(G, num_constructions, seed=seed)
    for e in edges:
        env.apply_construction(e, kind=construction_kind, factor=factor, until=None)

    # --- auto-compute required sim_time from predicted route travel times ---
    try:
        predicted_times = []
        for c in cars:
            # ensure route planned
            if not getattr(c, 'route', None):
                try:
                    c.plan_route(G, env=env)
                except Exception:
                    c.plan_route(G)
            # sum env edge travel times along the route
            rt = 0.0
            r = getattr(c, 'route', []) or []
            for i in range(max(0, len(r) - 1)):
                u = r[i]
                v = r[i + 1]
                try:
                    rt += float(env.get_edge_travel_time((u, v)))
                except Exception:
                    # fallback: use euclidean length / desired speed
                    try:
                        ux, uy = G.nodes[u]['x'], G.nodes[u]['y']
                        vx, vy = G.nodes[v]['x'], G.nodes[v]['y']
                        length = ((ux - vx) ** 2 + (uy - vy) ** 2) ** 0.5
                        rt += length / max(1e-3, getattr(c, 'v0', 13.89))
                    except Exception:
                        rt += 0.0
            predicted_times.append(rt)
        if predicted_times:
            max_pred = max(predicted_times)
            # choose sim_time as provided or larger of predicted * 1.2 + buffer
            computed = max_pred * 1.2 + 60.0
            if sim_time is None or sim_time < computed:
                sim_time = computed
    except Exception:
        # if anything goes wrong, keep provided sim_time
        pass

    steps = max(1, int(sim_time // dt))
    for _ in range(steps):
        env.step(dt)

    # collect travel times and arrival stats
    travel_times = []
    avg_speeds = []
    arrived = 0
    per_vehicle_data = []  # for detailed per-vehicle logging
    
    for c in cars:
        try:
            # prefer explicit arrival_time recorded by agents
            if hasattr(c, 'arrival_time'):
                arrived += 1
                t = float(c.arrival_time)
                travel_times.append(t)
                # average speed = distance_travelled / time
                d = float(getattr(c, 'distance_travelled', 0.0))
                avg_speeds.append(d / t if t > 0 else 0.0)
                
                if track_per_vehicle:
                    per_vehicle_data.append({
                        'vehicle_id': c.agent_id,
                        'travel_time': t,
                        'distance_travelled': d,
                        'avg_speed': d / t if t > 0 else 0.0,
                        'num_replans': getattr(c, 'num_replans', 0),
                        'initial_route_length': getattr(c, 'initial_route_length', 0),
                        'arrived': 1
                    })
                continue

            # judge arrival by position index or proximity
            if getattr(c, 'position_index', 0) >= max(0, len(getattr(c, 'route', [])) - 1):
                arrived += 1
                travel_times.append(c.travel_time)
                d = float(getattr(c, 'distance_travelled', 0.0))
                avg_speeds.append(d / c.travel_time if c.travel_time > 0 else 0.0)
                if track_per_vehicle:
                    per_vehicle_data.append({
                        'vehicle_id': c.agent_id,
                        'travel_time': c.travel_time,
                        'distance_travelled': d,
                        'avg_speed': d / c.travel_time if c.travel_time > 0 else 0.0,
                        'num_replans': getattr(c, 'num_replans', 0),
                        'initial_route_length': getattr(c, 'initial_route_length', 0),
                        'arrived': 1
                    })
                continue
            end = c.end_coord
            pos = getattr(c, 'current_coord', c.start_coord)
            dx = pos[0] - end[0]
            dy = pos[1] - end[1]
            if (dx*dx + dy*dy)**0.5 < 5.0:
                arrived += 1
                travel_times.append(c.travel_time)
                d = float(getattr(c, 'distance_travelled', 0.0))
                avg_speeds.append(d / c.travel_time if c.travel_time > 0 else 0.0)
                if track_per_vehicle:
                    per_vehicle_data.append({
                        'vehicle_id': c.agent_id,
                        'travel_time': c.travel_time,
                        'distance_travelled': d,
                        'avg_speed': d / c.travel_time if c.travel_time > 0 else 0.0,
                        'num_replans': getattr(c, 'num_replans', 0),
                        'initial_route_length': getattr(c, 'initial_route_length', 0),
                        'arrived': 1
                    })
            else:
                travel_times.append(float('nan'))
                if track_per_vehicle:
                    per_vehicle_data.append({
                        'vehicle_id': c.agent_id,
                        'travel_time': float('nan'),
                        'distance_travelled': 0,
                        'avg_speed': 0,
                        'num_replans': getattr(c, 'num_replans', 0),
                        'initial_route_length': getattr(c, 'initial_route_length', 0),
                        'arrived': 0
                    })
        except Exception:
            travel_times.append(float('nan'))
            if track_per_vehicle:
                per_vehicle_data.append({
                    'vehicle_id': c.agent_id,
                    'travel_time': float('nan'),
                    'distance_travelled': 0,
                    'avg_speed': 0,
                    'num_replans': getattr(c, 'num_replans', 0),
                    'initial_route_length': getattr(c, 'initial_route_length', 0),
                    'arrived': 0
                })

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

    # aggregate average speeds for arrived vehicles
    arrived_speeds = [s for s in avg_speeds if not (s is None or (isinstance(s, float) and (s != s)))]
    if arrived_speeds:
        result.update({
            'mean_avg_speed': statistics.mean(arrived_speeds),
            'median_avg_speed': statistics.median(arrived_speeds),
            'stdev_avg_speed': statistics.stdev(arrived_speeds) if len(arrived_speeds) > 1 else 0.0,
        })
    else:
        result.update({'mean_avg_speed': float('nan'), 'median_avg_speed': float('nan'), 'stdev_avg_speed': float('nan')})

    # Add per-vehicle detour metrics to result
    if track_per_vehicle and per_vehicle_data:
        arrived_vehicles = [v for v in per_vehicle_data if v['arrived'] == 1]
        if arrived_vehicles:
            # Calculate detour rates
            replans = [v['num_replans'] for v in arrived_vehicles]
            detour_distances = []
            for v in arrived_vehicles:
                if v['distance_travelled'] > 0 and v['initial_route_length'] > 0:
                    detour_ratio = v['distance_travelled'] / v['initial_route_length']
                    detour_distances.append(detour_ratio)
            
            result['mean_num_replans'] = statistics.mean(replans) if replans else 0.0
            result['max_num_replans'] = max(replans) if replans else 0
            result['pct_vehicles_rerouted'] = (sum(1 for r in replans if r > 0) / len(replans) * 100) if replans else 0.0
            result['mean_detour_ratio'] = statistics.mean(detour_distances) if detour_distances else 1.0
            result['vehicles_with_detours'] = sum(1 for d in detour_distances if d > 1.1)  # >10% longer route
        
        result['per_vehicle_data'] = per_vehicle_data

    return result


def batch_run(graph_path: str = DEFAULT_GPKG,
              vehicle_counts: List[int] = [10, 25, 50, 100],
              construction_counts: List[int] = [0, 1, 3, 5, 10, 20],
              seeds: List[int] = [0, 1, 2],
              out_csv: str = 'experiment_results.csv',
              sim_time: float = 5000.0,
              dt: float = 1.0,
              route_only_constructions: bool = False,
              track_per_vehicle: bool = False,
              per_vehicle_csv: str = None):
    print(f'Loading edges from: {graph_path}')
    edges = load_edges(graph_path)
    G = build_graph_from_edges(edges)
    rows = []
    all_per_vehicle_data = []  # Collect all per-vehicle data
    total = len(vehicle_counts) * len(construction_counts) * len(seeds)
    i = 0
    start_time = time.time()
    for nv in vehicle_counts:
        for nc in construction_counts:
            for s in seeds:
                i += 1
                print(f'Running {i}/{total}: vehicles={nv}, constructions={nc}, seed={s}')
                res = run_single_sim(G, nv, nc, sim_time=sim_time, dt=dt, seed=s,
                                     route_only_constructions=route_only_constructions,
                                     track_per_vehicle=track_per_vehicle)
                
                # Extract per-vehicle data if present
                if track_per_vehicle and 'per_vehicle_data' in res:
                    pvd = res.pop('per_vehicle_data')
                    for v in pvd:
                        v['num_vehicles'] = nv
                        v['num_constructions'] = nc
                        v['seed'] = s
                    all_per_vehicle_data.extend(pvd)
                
                rows.append(res)
    
    # write aggregated CSV
    keys = ['num_vehicles', 'num_constructions', 'seed', 'n_arrived', 'n_total', 'arrival_rate',
            'mean_travel_time', 'median_travel_time', 'stdev_travel_time',
            'mean_avg_speed', 'median_avg_speed', 'stdev_avg_speed']
    
    # Add detour metrics if present
    if track_per_vehicle and rows and any('mean_num_replans' in r for r in rows):
        keys.extend(['mean_num_replans', 'max_num_replans', 'pct_vehicles_rerouted', 
                     'mean_detour_ratio', 'vehicles_with_detours'])
    
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in keys})
    
    # Write per-vehicle data if collected
    if all_per_vehicle_data:
        if per_vehicle_csv is None:
            per_vehicle_csv = out_csv.replace('.csv', '_per_vehicle.csv')
        pv_keys = ['num_vehicles', 'num_constructions', 'seed', 'vehicle_id', 'travel_time', 
                   'distance_travelled', 'avg_speed', 'num_replans', 'initial_route_length', 'arrived']
        with open(per_vehicle_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=pv_keys)
            writer.writeheader()
            for v in all_per_vehicle_data:
                writer.writerow({k: v.get(k, '') for k in pv_keys})
        print(f'Per-vehicle data written to {per_vehicle_csv}')
    
    elapsed = time.time() - start_time
    print(f'Done. Results written to {out_csv} (elapsed {elapsed:.1f}s)')
    return out_csv


if __name__ == '__main__':
    batch_run(vehicle_counts=[10, 50, 100, 200, 500, 1000], 
              construction_counts=[0, 5, 10], 
              seeds=[0, 1, 2], 
              out_csv='experiment_wide_range_v2.csv', 
              sim_time=5000, 
              route_only_constructions=True,
              track_per_vehicle=True)
