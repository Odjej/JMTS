"""Demo runner: build driving graph from repository data, compute an A* route
for a `CarAgent`, print travel time and save a plotted route image.

Usage examples:
  python -m model.run_routing
  python -m model.run_routing --start-node 0 --end-node 10
  python -m model.run_routing --start-x 18.06 --start-y 59.33 --end-x 18.07 --end-y 59.34
"""
from pathlib import Path
import argparse
import math
import sys
import random

import matplotlib.pyplot as plt

from model.network_utils import load_edges, build_graph_from_edges, nearest_node
from model.agents import CarAgent


def plot_graph_and_route(G, route_nodes, out_path='route.png'):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot all edges
    for u, v, data in G.edges(data=True):
        x0, y0 = G.nodes[u]['x'], G.nodes[u]['y']
        x1, y1 = G.nodes[v]['x'], G.nodes[v]['y']
        ax.plot([x0, x1], [y0, y1], color='lightgray', linewidth=0.5)

    # Plot route
    if route_nodes:
        xs = [n[0] for n in route_nodes]
        ys = [n[1] for n in route_nodes]
        ax.plot(xs, ys, color='red', linewidth=2, marker='o')

    ax.set_title('A* Route on Driving Network')
    ax.set_aspect('equal')
    plt.axis('off')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f'Saved route figure to {out_path}')


def compute_route_travel_time(G, route):
    total = 0.0
    if not route or len(route) < 2:
        return total
    for u, v in zip(route, route[1:]):
        data = G.get_edge_data(u, v)
        if data is None:
            # No direct edge; skip or treat as infinite
            continue
        # For DiGraph, get_edge_data returns dict of attributes
        # or for MultiDiGraph a nested dict - we built DiGraph earlier
        travel_time = data.get('travel_time_s', data.get('travel_time', 0))
        total += float(travel_time)
    return total


def resolve_repo_path(p: Path) -> Path:
    # If p is absolute or exists, return; otherwise resolve relative to repo root
    if p.exists():
        return p
    repo_root = Path(__file__).resolve().parents[1]
    candidate = repo_root / p
    return candidate


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run A* routing demo on drive network')
    parser.add_argument('--start-node', type=int, help='Start node index into graph node list')
    parser.add_argument('--end-node', type=int, help='End node index into graph node list')
    parser.add_argument('--start-x', type=float, help='Start coordinate x (projected)')
    parser.add_argument('--start-y', type=float, help='Start coordinate y (projected)')
    parser.add_argument('--end-x', type=float, help='End coordinate x (projected)')
    parser.add_argument('--end-y', type=float, help='End coordinate y (projected)')
    parser.add_argument('--edges', default='data/processed/networks/drive_edges_clean.gpkg', help='Path to drive edges gpkg')
    parser.add_argument('--out', default='route.png', help='Output image path')
    args = parser.parse_args(argv)

    edges_path = resolve_repo_path(Path(args.edges))
    if not edges_path.exists():
        print(f'ERROR: edges file not found: {edges_path}', file=sys.stderr)
        return 2

    edges = load_edges(str(edges_path))
    G = build_graph_from_edges(edges)

    nodes = list(G.nodes())
    if len(nodes) < 2:
        print('Graph has insufficient nodes')
        return 1

    start_coord = None
    end_coord = None

    if args.start_x is not None and args.start_y is not None and args.end_x is not None and args.end_y is not None:
        start_coord = (args.start_x, args.start_y)
        end_coord = (args.end_x, args.end_y)
    elif args.start_node is not None and args.end_node is not None:
        try:
            start_coord = nodes[args.start_node]
            end_coord = nodes[args.end_node]
        except Exception:
            print('Invalid node indices provided', file=sys.stderr)
            return 3
    else:
        start_coord = random.choice(nodes)
        end_coord = random.choice(nodes)
        while end_coord == start_coord:
            end_coord = random.choice(nodes)

    car = CarAgent('car_1', start_coord=start_coord, end_coord=end_coord)
    car.plan_route(G)
    if not car.route:
        print('No path found between selected points')
        return 4

    travel_time_s = compute_route_travel_time(G, car.route)
    print(f'Planned route with {len(car.route)} nodes; estimated travel time = {travel_time_s:.1f} s')

    plot_graph_and_route(G, car.route, out_path=args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
