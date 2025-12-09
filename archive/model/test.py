from model.network_utils import load_edges, build_graph_from_edges
from model.env import Environment, random_graph_od
from model.experiment_runner import spawn_cars
G = build_graph_from_edges(load_edges('data/processed/networks/drive_edges_clean.gpkg'))
env = Environment(G)
cars = spawn_cars(G, 100, seed=0)
preds = []
lengths = []
for c in cars:
    c.plan_route(G, env=env)
    r = getattr(c, 'route', []) or []
    total_t = 0.0
    total_len = 0.0
    for i in range(max(0, len(r)-1)):
        u, v = r[i], r[i+1]
        try:
            total_t += env.get_edge_travel_time((u, v))
            total_len += float(G.get_edge_data(u, v, default={}).get('length_m', 0.0))
        except Exception:
            pass
    preds.append(total_t)
    lengths.append(total_len)
print('median_pred_s', sorted(preds)[len(preds)//2], 'median_len_m', sorted(lengths)[len(lengths)//2])


def fraction_route_affected(G, cars, construction_edges_set):
    fracs = []
    for c in cars:
        r = getattr(c, 'route', []) or []
        total = 0.0
        affected = 0.0
        for i in range(max(0, len(r)-1)):
            u, v = r[i], r[i+1]
            try:
                ux, uy = G.nodes[u]['x'], G.nodes[u]['y']
                vx, vy = G.nodes[v]['x'], G.nodes[v]['y']
                length = ((ux-vx)**2 + (uy-vy)**2)**0.5
            except Exception:
                length = 0.0
            total += length
            if (u, v) in construction_edges_set or (v, u) in construction_edges_set:
                affected += length
        fracs.append(affected/total if total>0 else 0.0)
    return fracs


# Quick diagnostic: sample some random construction sets and report fractions
import random
edges = list(G.edges())
for k in [1, 3, 5, 10, 20, 100, 200, 500, 1000]:
    random.seed(0)
    sampled = set(random.sample(edges, min(k, len(edges))))
    fracs = fraction_route_affected(G, cars, sampled)
    fracs_sorted = sorted(fracs)
    import statistics
    print(f'constructions={k}: median_frac={statistics.median(fracs_sorted):.6f}, mean_frac={statistics.mean(fracs_sorted):.6f}, max_frac={max(fracs_sorted):.6f}')
