import geopandas as gpd
import networkx as nx
import numpy as np
import geopandas as gpd

def load_edges(gpkg_path, layer=None):
    """Load edge layer from a GeoPackage and ensure a metric CRS and length."""
    gdf = gpd.read_file(gpkg_path, layer=layer)
    # Project to a metric CRS if possible
    try:
        utm = gdf.estimate_utm_crs()
        gdf = gdf.to_crs(utm)
    except Exception:
        # Fallback to Web Mercator
        gdf = gdf.to_crs(epsg=3857)

    gdf['length_m'] = gdf.geometry.length
    return gdf

def build_graph_from_edges(edges_gdf, default_speed_kmph=50):
    """
    Build a directed NetworkX graph from an edges GeoDataFrame.

    Nodes are created from geometry endpoints (coordinate tuples). Edges get
    attributes: length_m, travel_time_s, edge_id.
    """
    G = nx.DiGraph()

    default_speed_m_s = (default_speed_kmph * 1000) / 3600.0

    for idx, row in edges_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        try:
            coords = list(geom.coords)
        except Exception:
            # For multilinestrings, take first line
            coords = list(list(geom.geoms)[0].coords)

        u = (float(coords[0][0]), float(coords[0][1]))
        v = (float(coords[-1][0]), float(coords[-1][1]))

        length_m = float(row.get('length_m', geom.length))

        # Prefer an explicit travel_time if present. Otherwise try common speed fields
        # (speed_kmph, maxspeed) before falling back to the default_speed_kmph.
        travel_time_s = None
        if 'travel_time_s' in row and row.get('travel_time_s') is not None:
            try:
                travel_time_s = float(row.get('travel_time_s'))
            except Exception:
                travel_time_s = None
        if travel_time_s is None:
            # helper to parse speed values that might be strings like '50' or '50 km/h'
            def _parse_speed(val):
                if val is None:
                    return None
                try:
                    # try numeric first
                    return float(val)
                except Exception:
                    try:
                        # take leading numeric token
                        s = str(val).strip().split()[0]
                        return float(s)
                    except Exception:
                        return None

            speed_kmph = None
            for key in ('speed_kmph', 'speed', 'maxspeed'):
                if key in row and row.get(key) is not None:
                    speed_kmph = _parse_speed(row.get(key))
                    if speed_kmph is not None:
                        break

            if speed_kmph is not None:
                speed_m_s = (speed_kmph * 1000.0) / 3600.0
                # avoid division by zero
                if speed_m_s <= 0:
                    travel_time_s = length_m / default_speed_m_s
                else:
                    travel_time_s = length_m / speed_m_s
            else:
                travel_time_s = length_m / default_speed_m_s
        travel_time_s = float(travel_time_s)
        edge_id = row.get('edge_id', idx)

        # Add nodes with position attribute
        if u not in G:
            G.add_node(u, x=u[0], y=u[1])
        if v not in G:
            G.add_node(v, x=v[0], y=v[1])

        G.add_edge(u, v, edge_id=edge_id, length_m=length_m, travel_time_s=travel_time_s)

    return G

def nearest_node(G, point):
    """Find nearest graph node to a given (x,y) point."""
    px, py = point
    nodes = list(G.nodes(data=True))
    coords = np.array([(data['x'], data['y']) for (_n, data) in nodes])
    dists = np.hypot(coords[:,0] - px, coords[:,1] - py)
    idx = int(np.argmin(dists))
    return nodes[idx][0]









