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
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from model.network_utils import load_edges, build_graph_from_edges, nearest_node
from model.agents import CarAgent


def plot_graph_and_route(G, route_nodes, ped_route=None, out_path='route.png'):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot all edges
    edge_count = 0
    minx = miny = float('inf')
    maxx = maxy = float('-inf')
    for u, v, data in G.edges(data=True):
        try:
            x0, y0 = G.nodes[u]['x'], G.nodes[u]['y']
            x1, y1 = G.nodes[v]['x'], G.nodes[v]['y']
        except Exception:
            continue
        ax.plot([x0, x1], [y0, y1], color='lightgray', linewidth=0.5)
        edge_count += 1
        minx = min(minx, x0, x1)
        miny = min(miny, y0, y1)
        maxx = max(maxx, x0, x1)
        maxy = max(maxy, y0, y1)

    # Plot car route
    if route_nodes:
        xs = [n[0] for n in route_nodes]
        ys = [n[1] for n in route_nodes]
        ax.plot(xs, ys, color='red', linewidth=2, marker='o', label='car')
        minx = min(minx, min(xs)) if xs else minx
        miny = min(miny, min(ys)) if ys else miny
        maxx = max(maxx, max(xs)) if xs else maxx
        maxy = max(maxy, max(ys)) if ys else maxy

    # Plot pedestrian route if present
    if ped_route:
        xs = [n[0] for n in ped_route]
        ys = [n[1] for n in ped_route]
        ax.plot(xs, ys, color='blue', linewidth=1.5, linestyle='--', marker='o', label='ped')

    ax.set_title('A* Route on Driving Network')
    # If edges were plotted, set reasonable axis limits, otherwise show nodes
    if edge_count > 0 and minx < maxx and miny < maxy:
        dx = maxx - minx
        dy = maxy - miny
        pad = max(dx, dy) * 0.02 if max(dx, dy) > 0 else 1.0
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)
    else:
        # Fallback: plot nodes as points so something is visible
        node_coords = [(d.get('x', 0), d.get('y', 0)) for (_n, d) in G.nodes(data=True)]
        if node_coords:
            xs = [c[0] for c in node_coords]
            ys = [c[1] for c in node_coords]
            ax.scatter(xs, ys, s=2, color='gray')
            try:
                dx = max(xs) - min(xs)
                dy = max(ys) - min(ys)
                pad = max(dx, dy) * 0.02 if max(dx, dy) > 0 else 1.0
                ax.set_xlim(min(xs) - pad, max(xs) + pad)
                ax.set_ylim(min(ys) - pad, max(ys) + pad)
            except Exception:
                pass

    ax.set_aspect('equal')
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f'Saved route figure to {out_path} (edges plotted: {edge_count}, nodes: {len(G.nodes())})')


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
    parser.add_argument('--print-route', action='store_true', help='Print route node coordinates to stdout')
    parser.add_argument('--save-route', help='Save route to file: .csv, .gpkg or .geojson supported')
    parser.add_argument('--basemap', action='store_true', help='Overlay route on a web basemap (requires geopandas & contextily)')
    parser.add_argument('--plot-ped', action='store_true', help='Also plan and plot one pedestrian route on the same graph')
    args = parser.parse_args(argv)

    edges_path = resolve_repo_path(Path(args.edges))
    if not edges_path.exists():
        print(f'ERROR: edges file not found: {edges_path}', file=sys.stderr)
        return 2

    edges = load_edges(str(edges_path))
    G = build_graph_from_edges(edges)

    # Attempt to load pedestrian (walk) network for pedestrian routing
    walk_edges = None
    G_walk = None
    walk_path = resolve_repo_path(Path('data/processed/networks/walk_edges_clean.gpkg'))
    if walk_path.exists():
        try:
            walk_edges = load_edges(str(walk_path))
            G_walk = build_graph_from_edges(walk_edges)
            print(f'Loaded pedestrian graph: {len(G_walk.nodes())} nodes')
        except Exception as e:
            print('Failed to load pedestrian network, falling back to driving network for pedestrians:', e, file=sys.stderr)

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

    # Optionally plan a pedestrian route (use same graph for simplicity)
    ped_route = None
    if getattr(args, 'plot_ped', False):
        # pick two different nodes
        if G_walk is not None:
            nodes2 = list(G_walk.nodes())
            pstart = random.choice(nodes2)
            pend = random.choice(nodes2)
            while pend == pstart:
                pend = random.choice(nodes2)
            try:
                ped_path = nx.shortest_path(G_walk, pstart, pend, weight='travel_time_s')
                ped_route = ped_path
                print(f'Planned pedestrian route on walking graph with {len(ped_route)} nodes')
            except nx.NetworkXNoPath:
                ped_route = None
                print('No pedestrian path found on walking graph')
        else:
            nodes2 = list(G.nodes())
            pstart = random.choice(nodes2)
            pend = random.choice(nodes2)
            while pend == pstart:
                pend = random.choice(nodes2)
            try:
                ped_path = nx.shortest_path(G, pstart, pend, weight='travel_time_s')
                ped_route = ped_path
                print(f'Planned pedestrian route on driving graph with {len(ped_route)} nodes')
            except nx.NetworkXNoPath:
                ped_route = None
                print('No pedestrian path found on driving graph')

    travel_time_s = compute_route_travel_time(G, car.route)
    print(f'Planned route with {len(car.route)} nodes; estimated travel time = {travel_time_s:.1f} s')
    # Optionally print route coords
    if getattr(args, 'print_route', False):
        for i, n in enumerate(car.route):
            print(f'{i}: {n[0]:.3f}, {n[1]:.3f}')
        if ped_route:
            print('\nPedestrian route:')
            for i, n in enumerate(ped_route):
                print(f'{i}: {n[0]:.3f}, {n[1]:.3f}')

    # Optionally save route to CSV or GeoPackage/GeoJSON
    if args.save_route:
        outp = Path(args.save_route)
        suffix = outp.suffix.lower()
        try:
            if suffix == '.csv':
                import csv
                with open(outp, 'w', newline='', encoding='utf-8') as fh:
                    writer = csv.writer(fh)
                    writer.writerow(['order', 'x', 'y'])
                    for i, (x, y) in enumerate(car.route):
                        writer.writerow([i, x, y])
                print(f'Wrote route CSV to {outp}')
            elif suffix in ('.gpkg', '.geojson', '.json'):
                try:
                    import geopandas as gpd
                    from shapely.geometry import LineString
                except Exception as e:
                    print('Saving route as GeoPackage requires geopandas and shapely:', e, file=sys.stderr)
                else:
                    line = LineString(car.route)
                    gdf = gpd.GeoDataFrame({'id':[0], 'length_m':[line.length]}, geometry=[line], crs=None)
                    gdf.to_file(outp, driver='GPKG' if suffix == '.gpkg' else 'GeoJSON')
                    print(f'Wrote route geometry to {outp}')
            else:
                print('Unsupported save extension. Use .csv, .gpkg or .geojson', file=sys.stderr)
        except Exception as e:
            print('Failed to save route:', e, file=sys.stderr)

    # If basemap requested, attempt geopandas/contextily plotting
    if getattr(args, 'basemap', False):
        try:
            import geopandas as gpd
            import contextily as ctx
            from shapely.geometry import LineString
        except Exception as e:
            print('Basemap plotting requires geopandas, shapely and contextily:', e, file=sys.stderr)
            print('Falling back to simple matplotlib plot.')
            plot_graph_and_route(G, car.route, out_path=args.out)
        else:
            # edges is the GeoDataFrame loaded earlier from the gpkg; project to web mercator
            try:
                edges_gdf = edges.to_crs(epsg=3857)
            except Exception:
                edges_gdf = edges

            # Build route LineString in the same CRS as edges_gdf
            try:
                route_line = LineString(car.route)
                route_gdf = gpd.GeoDataFrame({'id':[0]}, geometry=[route_line], crs=edges.crs)
                route_gdf = route_gdf.to_crs(epsg=3857)
            except Exception as e:
                print('Failed to build route geometry for basemap:', e, file=sys.stderr)
                plot_graph_and_route(G, car.route, ped_route=ped_route, out_path=args.out)
            else:
                fig, ax = plt.subplots(figsize=(10, 10))
                # plot network lightly
                try:
                    edges_gdf.plot(ax=ax, linewidth=0.5, color='lightgray')
                except Exception:
                    # fallback: draw edges from G
                    for u, v, data in G.edges(data=True):
                        x0, y0 = G.nodes[u]['x'], G.nodes[u]['y']
                        x1, y1 = G.nodes[v]['x'], G.nodes[v]['y']
                        ax.plot([x0, x1], [y0, y1], color='lightgray', linewidth=0.5)

                # Plot route and pedestrian route on top of network (reprojected to WebMercator)
                try:
                    route_gdf.plot(ax=ax, color='red', linewidth=2)
                except Exception as e:
                    print('Failed to plot car route on basemap:', e, file=sys.stderr)

                if ped_route:
                    try:
                        ped_line = LineString(ped_route)
                        ped_crs = walk_edges.crs if walk_edges is not None else edges.crs
                        ped_gdf = gpd.GeoDataFrame({'id':[0]}, geometry=[ped_line], crs=ped_crs).to_crs(epsg=3857)
                        ped_gdf.plot(ax=ax, color='blue', linewidth=1.5, linestyle='--')
                    except Exception as e:
                        print('Failed to plot pedestrian route on basemap:', e, file=sys.stderr)
                # add basemap
                try:
                    ctx.add_basemap(ax, crs='EPSG:3857')
                except Exception as e:
                    print('contextily basemap failed:', e, file=sys.stderr)

                # Title
                ax.set_title('A* Route on Network')

                # Legend (create custom handles)
                handles = []
                handles.append(Line2D([0], [0], color='red', lw=2, label='Car'))
                if ped_route:
                    handles.append(Line2D([0], [0], color='blue', lw=1.5, linestyle='--', label='Pedestrian'))
                ax.legend(handles=handles, loc='upper right')

                # Scalebar: in WebMercator units (meters)
                try:
                    x0, x1 = ax.get_xlim()
                    y0, y1 = ax.get_ylim()
                    dx = abs(x1 - x0)
                    dy = abs(y1 - y0)
                    # choose a nice scale length ~ 10% of map width
                    ideal = dx * 0.10
                    candidates = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
                    scale_len = min(candidates, key=lambda c: abs(c - ideal))
                    # scalebar position in data coords 
                    sb_x = x1 - 0.1 * dx
                    sb_y = y0 + 0.02 * dy
                    ax.plot([sb_x, sb_x + scale_len], [sb_y, sb_y], color='k', linewidth=3)
                    ax.plot([sb_x, sb_x], [sb_y - 0.005 * dy, sb_y + 0.005 * dy], color='k', linewidth=2)
                    ax.plot([sb_x + scale_len, sb_x + scale_len], [sb_y - 0.005 * dy, sb_y + 0.005 * dy], color='k', linewidth=2)
                    label = f"{scale_len} m" if scale_len < 1000 else f"{scale_len//1000} km"
                    ax.text(sb_x + scale_len / 2, sb_y + 0.01 * dy, label, ha='center', va='bottom', fontsize=9, color='k')
                except Exception:
                    pass

                # North arrow (axes fraction coordinates)
                try:
                    na_x = 0.08
                    na_y = 0.92
                    arrow = FancyArrowPatch((na_x, na_y - 0.08), (na_x, na_y), transform=ax.transAxes,
                                            arrowstyle='-|>', mutation_scale=20, color='k')
                    ax.add_patch(arrow)
                    ax.text(na_x, na_y - 0.10, 'N', transform=ax.transAxes, ha='center', va='top', fontsize=12, weight='bold')
                except Exception:
                    pass

                ax.set_axis_off()
                plt.tight_layout()
                fig.savefig(args.out, dpi=150)
                print(f'Saved basemap route figure to {args.out}')
    else:
        plot_graph_and_route(G, car.route, ped_route=ped_route, out_path=args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
