"""Robust simulate helper: loads driving network from the repo's data folder
and builds a NetworkX graph. This avoids brittle relative paths and missing
GraphML files by using the available `.gpkg` files in `data/processed/networks`.
"""
from pathlib import Path
import sys

from model.network_utils import load_edges, build_graph_from_edges


def load_drive_graph():
    repo_root = Path(__file__).resolve().parents[1]
    edges_path = repo_root / 'data' / 'processed' / 'networks' / 'drive_edges_clean.gpkg'
    if not edges_path.exists():
        print(f'ERROR: expected drive edges file not found: {edges_path}', file=sys.stderr)
        return None

    print(f'Loading edges from: {edges_path}')
    edges = load_edges(str(edges_path))
    G = build_graph_from_edges(edges)
    print(f'Loaded graph: {len(G.nodes)} nodes, {len(G.edges)} edges')
    return G


def main():
    G = load_drive_graph()
    if G is None:
        return
    # Example: print first 5 nodes
    print('Sample nodes:')
    for i, n in enumerate(G.nodes(data=True)):
        print(n)
        if i >= 4:
            break


if __name__ == '__main__':
    main()
