import math
from collections import defaultdict
import random
from typing import Tuple, Dict, Any, List

try:
    from shapely.geometry import LineString, Point
    from shapely.strtree import STRtree
    _HAS_SHAPELY = True
except Exception:
    _HAS_SHAPELY = False

class Environment:
    """
    Minimal environment for multi-modal simulation.
    Supports:
      - get_leader(agent)
      - get_pedestrians_in_view(ped)
      - optional lane statistics (stubs)
    """

    def __init__(self, G):
        self.G = G
        self.agents = []
        self.time = 0.0
        # default cooldown (s) between successive lane changes per agent
        self.lane_change_cooldown_default = 3.0
        # Build edge geometries and spatial index for quick lookup
        self._edge_geoms = []  # list of shapely LineString objects (if available)
        self._edge_keys = []   # parallel list of edge keys (u,v)
        self.edge_index = None
        # occupancy: (edge_key, lane) -> list of occupant dicts {agent, frac, pos, speed}
        # edge_key is (u,v) where u/v are node coordinate tuples
        self.edge_occupancy: Dict[Tuple[Tuple[Any, Any], int], List[Dict[str, Any]]] = defaultdict(list)
        # dynamic travel time multipliers and construction events
        self.edge_multiplier: Dict[Tuple[Any, Any], float] = {}
        self.construction_events: Dict[Tuple[Any, Any], Dict[str, Any]] = {}

        if _HAS_SHAPELY:
            for u, v, data in self.G.edges(data=True):
                # construct straight LineString between node coordinates
                try:
                    x0, y0 = self.G.nodes[u]['x'], self.G.nodes[u]['y']
                    x1, y1 = self.G.nodes[v]['x'], self.G.nodes[v]['y']
                except Exception:
                    continue
                geom = LineString([(x0, y0), (x1, y1)])
                self._edge_geoms.append(geom)
                self._edge_keys.append((u, v))
            if self._edge_geoms:
                try:
                    self.edge_index = STRtree(self._edge_geoms)
                except Exception:
                    self.edge_index = None

    def register(self, agent):
        """Should be called after creating each agent."""
        self.agents.append(agent)
        # give agent a backref to env
        try:
            agent.env = self
        except Exception:
            pass
        # ensure agent has lane attribute (default 0)
        if not hasattr(agent, 'lane'):
            try:
                agent.lane = 0
            except Exception:
                pass
        # initialize lane-change bookkeeping on the agent
        try:
            # timestamp of last lane change -> very negative so agent may change immediately
            agent.last_lane_change_time = -1e9
            if not hasattr(agent, 'lane_change_cooldown'):
                agent.lane_change_cooldown = self.lane_change_cooldown_default
        except Exception:
            pass

    @staticmethod
    def dist(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])
    
    def get_leader(self, agent):
        """
        Returns (distance_m, leader_speed) of the nearest agent
        in front of the given agent along its route.
        """
        # New spatial-aware leader lookup using edge occupancy and route traversal
        if not hasattr(agent, "route") or len(agent.route) < 2:
            return (None, 0.0)

        # Identify agent's current edge and fractional position
        idx = getattr(agent, 'position_index', 0)
        if idx >= len(agent.route) - 1:
            return (None, 0.0)

        # helper to compute distance along remaining route from a given edge and frac
        def distance_along_route(start_idx: int, start_frac: float) -> float:
            d = 0.0
            # remaining length on first segment
            a = agent.route[start_idx]
            b = agent.route[start_idx + 1]
            seg_len = math.hypot(b[0]-a[0], b[1]-a[1])
            d += seg_len * (1.0 - start_frac)
            # subsequent segments
            for i in range(start_idx + 1, len(agent.route) - 1):
                a = agent.route[i]
                b = agent.route[i+1]
                d += math.hypot(b[0]-a[0], b[1]-a[1])
            return d

        # agent fractional position on current segment (0..1)
        a_pos = getattr(agent, 'current_coord', None)
        a_idx = idx
        a_frac = 0.0
        try:
            p0 = agent.route[a_idx]
            p1 = agent.route[a_idx + 1]
            seg_len = math.hypot(p1[0]-p0[0], p1[1]-p0[1])
            if seg_len > 1e-9:
                # project position onto segment
                vx = a_pos[0] - p0[0]
                vy = a_pos[1] - p0[1]
                dot = vx*(p1[0]-p0[0]) + vy*(p1[1]-p0[1])
                a_frac = max(0.0, min(1.0, dot / (seg_len*seg_len)))
        except Exception:
            return (None, 0.0)

        best_dist = None
        best_speed = 0.0

        # Search occupancy starting from current edge onwards
        # First check same edge in same lane
        cur_edge = (agent.route[a_idx], agent.route[a_idx + 1])
        lane = getattr(agent, 'lane', 0)
        occ_list = self.edge_occupancy.get((cur_edge, lane), [])
        for occ in occ_list:
            if occ.get('agent') is agent:
                continue
            frac = occ.get('frac', 0.0)
            if frac <= a_frac + 1e-6:
                continue
            # distance along current segment
            d = (frac - a_frac) * math.hypot(cur_edge[1][0]-cur_edge[0][0], cur_edge[1][1]-cur_edge[0][1])
            if best_dist is None or d < best_dist:
                best_dist = d
                best_speed = occ.get('speed', 0.0)

        # Then check subsequent edges along route (same lane)
        if best_dist is None:
            cum = 0.0
            # remaining length in current segment
            a = agent.route[a_idx]
            b = agent.route[a_idx + 1]
            seg_len = math.hypot(b[0]-a[0], b[1]-a[1])
            cum += seg_len * (1.0 - a_frac)
            for i in range(a_idx + 1, len(agent.route) - 1):
                e = (agent.route[i], agent.route[i+1])
                occs = self.edge_occupancy.get((e, lane), [])
                if occs:
                    # find nearest occupant on this edge (smallest frac)
                    nearest = min([o for o in occs if o.get('agent') is not agent], key=lambda o: o.get('frac', 0.0), default=None)
                    if nearest:
                        d = cum + nearest.get('frac', 0.0) * math.hypot(e[1][0]-e[0][0], e[1][1]-e[0][1])
                        best_dist = d
                        best_speed = nearest.get('speed', 0.0)
                        break
                # add full edge length and continue
                a = agent.route[i]
                b = agent.route[i+1]
                cum += math.hypot(b[0]-a[0], b[1]-a[1])

        return (best_dist, best_speed)


    def get_pedestrians_in_view(self, ped_agent):
        """
        Return list of (pos, speed, dir) of pedestrians in the view cone.
        """
        results = []
        cx, cy = ped_agent.current_coord

        for other in self.agents:
            if other is ped_agent:
                continue
            if getattr(other, "mode", None) != "walk":
                continue

            ox, oy = other.current_coord
            dx = ox - cx
            dy = oy - cy
            dist = math.hypot(dx, dy)

            if dist < 1e-6 or dist > ped_agent.view_depth:
                continue

            # angle check
            dot = dx * ped_agent.direction[0] + dy * ped_agent.direction[1]
            cosang = dot / (dist + 1e-9)
            ang = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))

            if ang <= ped_agent.view_angle_deg / 2:
                results.append(
                    ((ox, oy),
                     getattr(other, "v", ped_agent.desired_speed),
                     getattr(other, "direction", ped_agent.direction))
                )

        return results
 

    def avg_speed_on_lane(self, agent):
        # Simple lane avg speed: compute average speed of occupants on same edge
        try:
            idx = agent.position_index
            if idx >= len(agent.route) - 1:
                return getattr(agent, 'v', 0.0)
            edge = (agent.route[idx], agent.route[idx + 1])
            lane = getattr(agent, 'lane', 0)
            occ = self.edge_occupancy.get((edge, lane), [])
            if not occ:
                return getattr(agent, 'v', 0.0)
            return sum(o.get('speed', 0.0) for o in occ) / max(1, len(occ))
        except Exception:
            return getattr(agent, 'v', 0.0)

    def avg_speed_on_adjacent_lane(self, agent):
        # Very simple adjacent-lane evaluation: check left and right lane averages and return the best
        try:
            lane = getattr(agent, 'lane', 0)
            u, v = agent.route[agent.position_index], agent.route[agent.position_index + 1]
            max_lane = int(self.G.get_edge_data(u, v, {}).get('lanes', 1)) - 1
            candidates = []
            for nl in (lane - 1, lane + 1):
                if nl < 0 or nl > max_lane:
                    continue
                occ = self.edge_occupancy.get(((u, v), nl), [])
                if not occ:
                    candidates.append(getattr(agent, 'v', 0.0))
                else:
                    candidates.append(sum(o.get('speed', 0.0) for o in occ) / max(1, len(occ)))
            if not candidates:
                return self.avg_speed_on_lane(agent)
            return max(candidates)
        except Exception:
            return self.avg_speed_on_lane(agent)

    def can_change_lane(self, agent, min_speed_gain: float = 1.0):
        """Decide if agent can change lane to overtake.

        Returns the target lane index if lane change is advised and available, else None.
        Simple rule: if adjacent lane avg speed > current lane avg speed + min_speed_gain
        and adjacent lane density is not higher than current lane, allow change.
        """
        try:
            # enforce per-agent cooldown to avoid rapid back-and-forth lane oscillation
            last = getattr(agent, 'last_lane_change_time', -1e9)
            cooldown = getattr(agent, 'lane_change_cooldown', self.lane_change_cooldown_default)
            if (self.time - last) < float(cooldown):
                return None
            lane = getattr(agent, 'lane', 0)
            idx = agent.position_index
            if idx >= len(agent.route) - 1:
                return None
            edge = (agent.route[idx], agent.route[idx + 1])
            u, v = edge
            num_lanes = int(self.G.get_edge_data(u, v, {}).get('lanes', 1))
            cur_speed = self.avg_speed_on_lane(agent)
            cur_density = self.get_edge_density(edge)
            best_gain = 0.0
            best_lane = None
            for nl in (lane - 1, lane + 1):
                if nl < 0 or nl >= num_lanes:
                    continue
                occ = self.edge_occupancy.get((edge, nl), [])
                if not occ:
                    adj_speed = getattr(agent, 'v', 0.0)
                    adj_density = 0.0
                else:
                    adj_speed = sum(o.get('speed',0.0) for o in occ) / max(1, len(occ))
                    # compute density for that lane
                    adj_density = len(occ) / max(1e-6, math.hypot(edge[1][0]-edge[0][0], edge[1][1]-edge[0][1]))
                gain = adj_speed - cur_speed
                # prefer lanes with higher speed and not much higher density
                if gain > min_speed_gain and adj_density <= max(cur_density * 1.2, cur_density + 0.1):
                    if gain > best_gain:
                        best_gain = gain
                        best_lane = nl
            return best_lane
        except Exception:
            return None

    def change_lane(self, agent, target_lane: int) -> bool:
        """Perform lane change for agent when allowed. Returns True if changed."""
        try:
            if target_lane is None:
                return False
            agent.lane = int(target_lane)
            # record lane change time to enforce cooldown/hysteresis
            try:
                agent.last_lane_change_time = float(self.time)
            except Exception:
                pass
            return True
        except Exception:
            return False


    # ------------------- Occupancy & Edge Dynamics -------------------
    def _project_frac_on_segment(self, p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
        """Return (frac, dist_to_segment) where frac is 0..1 along segment a->b."""
        ax, ay = a
        bx, by = b
        vx = bx - ax
        vy = by - ay
        wx = p[0] - ax
        wy = p[1] - ay
        seg_len2 = vx*vx + vy*vy
        if seg_len2 < 1e-9:
            return 0.0, math.hypot(wx, wy)
        dot = wx*vx + wy*vy
        frac = max(0.0, min(1.0, dot / seg_len2))
        projx = ax + frac * vx
        projy = ay + frac * vy
        dist = math.hypot(p[0]-projx, p[1]-projy)
        return frac, dist

    def _distance_along_route_to_point(self, agent, point: Tuple[float, float], lateral_threshold: float = 3.0):
        """Return distance (m) along agent.route from current agent position to projection of `point` onto route.
        If projection is not within lateral_threshold on any route segment or the projected location is behind
        the agent, return None.
        """
        if not hasattr(agent, 'route') or not agent.route:
            return None
        # compute agent current index and frac
        a_idx = getattr(agent, 'position_index', 0)
        try:
            a_pos = getattr(agent, 'current_coord', agent.route[a_idx])
        except Exception:
            return None

        # compute agent fractional position on current segment
        try:
            p0 = agent.route[a_idx]
            p1 = agent.route[a_idx + 1]
            seg_len = math.hypot(p1[0]-p0[0], p1[1]-p0[1])
            if seg_len > 1e-9:
                vx = a_pos[0] - p0[0]
                vy = a_pos[1] - p0[1]
                dot = vx*(p1[0]-p0[0]) + vy*(p1[1]-p0[1])
                a_frac = max(0.0, min(1.0, dot / (seg_len*seg_len)))
            else:
                a_frac = 0.0
        except Exception:
            return None

        # helper to compute cumulative length from agent position to a given (seg_idx, frac)
        def cum_len_to(seg_idx:int, frac:float) -> float:
            d = 0.0
            # remaining on current segment
            p0 = agent.route[a_idx]
            p1 = agent.route[a_idx + 1]
            first_seg_len = math.hypot(p1[0]-p0[0], p1[1]-p0[1])
            d += max(0.0, first_seg_len * (1.0 - a_frac))
            # full segments between
            for i in range(a_idx+1, seg_idx):
                pa = agent.route[i]
                pb = agent.route[i+1]
                d += math.hypot(pb[0]-pa[0], pb[1]-pa[1])
            # partial of target segment
            if seg_idx < len(agent.route)-1:
                ta = agent.route[seg_idx]
                tb = agent.route[seg_idx+1]
                segl = math.hypot(tb[0]-ta[0], tb[1]-ta[1])
                d += segl * frac
            return d

        best_dist = None
        for i in range(0, len(agent.route)-1):
            a = agent.route[i]
            b = agent.route[i+1]
            frac, latdist = self._project_frac_on_segment(point, a, b)
            if latdist > lateral_threshold:
                continue
            # compute distance along route from agent pos to this projected point
            # if projection is behind agent (i < a_idx or equal but frac <= a_frac) skip
            if i < a_idx:
                continue
            if i == a_idx and frac <= a_frac + 1e-9:
                continue
            d = cum_len_to(i, frac)
            if d < 0:
                continue
            if best_dist is None or d < best_dist:
                best_dist = d
        return best_dist

    def get_pedestrians_ahead(self, agent, look_ahead_m: float = 30.0, lateral_threshold: float = 3.0):
        """Return (distance_m, pedestrian_speed) for the nearest pedestrian ahead on the agent's route within look_ahead_m.
        Returns (None, 0.0) if none found.
        """
        best = None
        best_speed = 0.0
        for other in self.agents:
            if other is agent:
                continue
            if getattr(other, 'mode', None) != 'walk':
                continue
            pt = getattr(other, 'current_coord', None)
            if pt is None:
                continue
            d = self._distance_along_route_to_point(agent, pt, lateral_threshold=lateral_threshold)
            if d is None:
                continue
            if d <= look_ahead_m:
                if best is None or d < best:
                    best = d
                    best_speed = getattr(other, 'v', 0.0)
        if best is None:
            return (None, 0.0)
        return (best, best_speed)

    def get_pedestrians_on_edge(self, edge_key, lateral_threshold: float = 3.0):
        """Return number of pedestrians currently on the geometric edge (u,v) within lateral_threshold.
        edge_key is (u,v) node tuple pair (coords).
        """
        cnt = 0
        u, v = edge_key
        a = u
        b = v
        for other in self.agents:
            if getattr(other, 'mode', None) != 'walk':
                continue
            pt = getattr(other, 'current_coord', None)
            if pt is None:
                continue
            frac, latdist = self._project_frac_on_segment(pt, a, b)
            if latdist <= lateral_threshold and 0.0 <= frac <= 1.0:
                cnt += 1
        return cnt

    def update_occupancy(self):
        """Rebuild edge occupancy from current agent positions."""
        # clear
        self.edge_occupancy = defaultdict(list)
        for agent in self.agents:
            if not hasattr(agent, 'route') or not agent.route:
                continue
            idx = getattr(agent, 'position_index', 0)
            if idx >= len(agent.route) - 1:
                continue
            a = agent.route[idx]
            b = agent.route[idx+1]
            frac, dist = self._project_frac_on_segment(getattr(agent, 'current_coord', a), a, b)
            edge_key = (a, b)
            lane = getattr(agent, 'lane', 0)
            occ = {
                'agent': agent,
                'frac': frac,
                'pos': getattr(agent, 'current_coord', a),
                'speed': getattr(agent, 'v', 0.0)
            }
            self.edge_occupancy[(edge_key, lane)].append(occ)

        # sort occupancy lists by fraction ascending
        for k in list(self.edge_occupancy.keys()):
            self.edge_occupancy[k].sort(key=lambda o: o.get('frac', 0.0))

    def get_edge_occupancy(self, edge_key):
        """Return occupancy list for an edge key (u,v)."""
        # Support both old (edge_key) and new ((edge_key, lane)) queries
        if isinstance(edge_key, tuple) and len(edge_key) == 2 and isinstance(edge_key[0], tuple) and isinstance(edge_key[1], int):
            return list(self.edge_occupancy.get(edge_key, []))
        # return combined occupancy across all lanes for backward compatibility
        results = []
        for (e, lane), occ in self.edge_occupancy.items():
            if e == edge_key:
                results.extend(occ)
        return results

    def get_edge_density(self, edge_key):
        """Return simple density: vehicles per meter on edge."""
        occ = self.edge_occupancy.get(edge_key, [])
        try:
            u, v = edge_key
            a = u
            b = v
            L = math.hypot(b[0]-a[0], b[1]-a[1])
            if L <= 0:
                return 0.0
            return len(occ) / L
        except Exception:
            return 0.0

    def apply_construction(self, edge_key, kind='slow', factor=3.0, lanes_reduced=0, until=None):
        """Apply a construction event modifying edge travel times.

        - kind: 'slow'|'closure'|'lane_reduction'
        - factor: multiplier for travel time (e.g., 1.5 slows by 50%)
        - lanes_reduced: integer, not used for lane topology yet
        - until: simulation time until which event applies (None = persistent)
        """
        # Ensure the provided edge_key exists in the graph; if orientation is reversed
        # try to canonicalize to the graph's edge orientation. If the edge is not
        # present in the graph, warn and do not apply the construction.
        try:
            u, v = edge_key
        except Exception:
            print(f"apply_construction: invalid edge_key {edge_key}")
            return False

        if not self.G.has_edge(u, v):
            if self.G.has_edge(v, u):
                # use the actual stored orientation
                edge_key = (v, u)
                u, v = edge_key
            else:
                # edge not present in graph: ignore and warn
                print(f"apply_construction: edge {edge_key} not found in graph; skipping")
                return False

        self.construction_events[edge_key] = {
            'kind': kind,
            'factor': float(factor),
            'lanes_reduced': int(lanes_reduced),
            'until': until
        }
        # set multiplier
        if kind == 'closure':
            self.edge_multiplier[edge_key] = float('inf')
        else:
            self.edge_multiplier[edge_key] = float(factor)
        # trigger replanning for agents whose remaining route uses this edge
        try:
            self._trigger_replanning(edge_key, kind=kind)
        except Exception:
            pass

    def clear_expired_construction(self):
        now = self.time
        to_remove = []
        for e, ev in self.construction_events.items():
            if ev.get('until') is not None and ev.get('until') <= now:
                to_remove.append(e)
        for e in to_remove:
            del self.construction_events[e]
            if e in self.edge_multiplier:
                del self.edge_multiplier[e]

    def get_edge_travel_time(self, edge_key):
        """Return current travel time for edge (u,v) taking multipliers into account."""
        u, v = edge_key
        data = self.G.get_edge_data(u, v) or {}
        base = float(data.get('travel_time_s', data.get('length_m', 0) / 13.89))
        mult = self.edge_multiplier.get(edge_key, 1.0)
        if math.isinf(mult):
            return float('inf')
        # also incorporate density-based slowdown (simple model)
        density = self.get_edge_density(edge_key)
        # example: for density > 0.1 veh/m apply additional slowdown: 1 + 0.5*(density/0.1)
        density_factor = 1.0
        if density > 0.1:
            density_factor += 0.5 * (density / 0.1)
        # pedestrian presence penalty: increase travel time if pedestrians are on edge
        try:
            ped_cnt = self.get_pedestrians_on_edge(edge_key)
        except Exception:
            ped_cnt = 0
        ped_factor = 1.0
        if ped_cnt > 0:
            # each pedestrian on the edge increases travel time modestly (cap at factor 2.0)
            ped_factor += min(1.0, 0.5 * ped_cnt)

        return base * mult * density_factor * ped_factor

    # ------------------- Replanning helpers -------------------
    def _align_agent_to_route(self, agent):
        """Set agent.position_index to the best matching segment on its current route
        based on agent.current_coord. If no good match is found, leave index at 0.
        """
        try:
            if not hasattr(agent, 'route') or not agent.route:
                return
            best_i = None
            best_dist = None
            for i in range(0, len(agent.route) - 1):
                a = agent.route[i]
                b = agent.route[i + 1]
                frac, latdist = self._project_frac_on_segment(getattr(agent, 'current_coord', a), a, b)
                if best_dist is None or latdist < best_dist:
                    best_dist = latdist
                    best_i = i
            if best_i is not None:
                agent.position_index = best_i
        except Exception:
            pass

    def _trigger_replanning(self, edge_key, kind='slow'):
        """Force agents whose remaining route includes `edge_key` to re-plan.
        This is a lightweight replanning: agents will compute a new route from
        their current position to their `end_coord` using the current environment
        travel times.
        """
        affected = []
        for agent in list(self.agents):
            try:
                if not hasattr(agent, 'route') or not agent.route:
                    continue
                idx = getattr(agent, 'position_index', 0)
                for i in range(idx, len(agent.route) - 1):
                    e = (agent.route[i], agent.route[i + 1])
                    if e == edge_key:
                        affected.append(agent)
                        break
            except Exception:
                continue

        for agent in affected:
            try:
                # re-plan from the agent's current coordinate
                start = getattr(agent, 'current_coord', None)
                agent.plan_route(self.G, env=self, from_coord=start)
                # align agent's position index with new route
                self._align_agent_to_route(agent)
                # if the edge was closed, stop the agent to avoid driving through it
                if kind == 'closure':
                    try:
                        agent.v = 0.0
                    except Exception:
                        pass
            except Exception:
                # ignore failures in replanning for robustness
                pass

    def step(self, dt: float):
        """Advance the environment clock and step all registered agents.

        This method attempts to call agent.step with signatures:
          - step(dt, sim_time, env)
          - step(dt, env)
          - step(dt)
        After agents have been stepped, occupancy is updated and expired
        construction events are cleared.
        """
        self.time += dt
        for agent in list(self.agents):
            # Call agent.step with the most complete signature first
            try:
                agent.step(dt, self.time, self)
                continue
            except TypeError:
                pass
            try:
                agent.step(dt, self)
                continue
            except TypeError:
                pass
            try:
                agent.step(dt)
            except Exception:
                # ignore agents without step or if step fails
                pass

        # After agents moved, update occupancy and handle construction expiry
        try:
            self.update_occupancy()
        except Exception:
            # best-effort: don't crash simulation if occupancy update fails
            pass
        try:
            self.clear_expired_construction()
        except Exception:
            pass


def random_graph_od(G):
    """Return two random (different) node coords from the graph."""
    nodes = list(G.nodes())
    start = random.choice(nodes)
    end = random.choice(nodes)
    while end == start:
        end = random.choice(nodes)
    return start, end
 