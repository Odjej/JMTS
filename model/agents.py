import math
import random
import networkx as nx
from typing import Tuple, List, Optional

# --- Car agent implementing a simplified Generalized Force Model (GFM) ---
class CarAgent:
    def __init__(self, agent_id: str, start_coord: Tuple[float, float], end_coord: Tuple[float, float],
                 priority: int = 1, desired_speed_m_s: float = 13.89):
        self.agent_id = agent_id
        self.start_coord = start_coord
        self.current_coord = start_coord
        self.end_coord = end_coord
        self.mode = 'drive'
        self.priority = priority

        # kinematic state (along graph nodes)
        self.route: List[Tuple[float, float]] = []   # list of node coordinates 
        self._start_node = None
        self._end_node = None

        # dynamic variables
        self.v = desired_speed_m_s * 0.6  # initial speed (m/s), conservative
        self.v0 = desired_speed_m_s       # desired (free) speed m/s
        self.a_max = 2.0                  # m/s^2
        self.tau = 1.0                    # relaxation time (s) for desired speed term
        self.length_m = 4.5               # vehicle length
        self.position_index = 0           # index into route of current node (simple discrete position)
        self.travel_time = 0.0

    def plan_route(self, G: nx.Graph, default_speed_m_s: float = 13.89):
        """A* route planning from start_coord to end_coord on graph G."""
        try:
            from model.network_utils import nearest_node
        except Exception:
            from .network_utils import nearest_node

        self._start_node = nearest_node(G, self.start_coord)
        self._end_node = nearest_node(G, self.end_coord)

        def heuristic(u, v):
            ux, uy = G.nodes[u]['x'], G.nodes[u]['y']
            vx, vy = G.nodes[v]['x'], G.nodes[v]['y']
            euclid = math.hypot(ux - vx, uy - vy)
            return euclid / default_speed_m_s

        try:
            path = nx.astar_path(G, self._start_node, self._end_node, heuristic=heuristic, weight='travel_time_s')
        except nx.NetworkXNoPath:
            self.route = []
            return

        # store as node coordinate tuples 
        self.route = path
        self.position_index = 0

    def _desired_accel(self):
        """Relaxation to desired speed: (v0 - v) / tau, clipped by a_max"""
        a = (self.v0 - self.v) / max(self.tau, 1e-6)
        if a > self.a_max:
            a = self.a_max
        return a

    def _repulsive_from_leader(self, leader_dist, leader_speed):
        """
        Simple repulsive term: if close to leader, brake harder.
        This is a light-weight implementation of the GFM interaction term.
        """
        if leader_dist is None:
            return 0.0
        # desired gap (simple): s0 + v*T + v*(v - v_leader)/(2*sqrt(a*b))
        s0 = 2.0  # m (minimum gap)
        T = 1.0   # s (safe time headway)
        b = 3.0   # comfortable decel
        desired_gap = s0 + max(0.0, self.v * T + (self.v * (self.v - leader_speed)) / (2.0 * math.sqrt(self.a_max * b)))
        gap = max(1e-3, leader_dist - self.length_m)
        # if gap < desired_gap -> braking proportional to difference
        if gap < desired_gap:
            # braking acceleration (negative)
            return - self.a_max * (desired_gap - gap) / (desired_gap + 0.1)
        return 0.0

    def _find_virtual_preceding(self, env):
        """
        env is expected to expose:
          - a function to query nearest vehicle/tram/pedestrian ahead on same subsection
          - or we can pass a simple list of agents and a graph to compute spatial distances.
        Here we provide a thin interface: env.get_leader(self) -> (distance_m, leader_speed)
        If not available, returns (None, 0.0)
        """
        if env is None:
            return (None, 0.0)
        try:
            dist, spd = env.get_leader(self)
            return (dist, spd)
        except Exception:
            return (None, 0.0)

    def step(self, dt: float, env=None):
        """
        Advance vehicle state over timestep dt (seconds).
        env: simulation environment helper that can provide leader info, traffic light state,
             virtual preceding pedestrians, etc. This keeps the agent logic separate.
        """
        # desired acceleration toward v0
        a_des = self._desired_accel()

        # virtual preceding vehicle/pedestrian/tram
        leader_dist, leader_speed = self._find_virtual_preceding(env)
        a_rep = self._repulsive_from_leader(leader_dist, leader_speed)

        # simple stochastic fluctuation xi(t)
        xi = random.gauss(0, 0.05)

        a_total = a_des + a_rep + xi
        # clip by comfortable braking/accel limits
        if a_total > self.a_max:
            a_total = self.a_max
        if a_total < -5.0:  # limit strong braking
            a_total = -5.0

        # integrate
        self.v = max(0.0, self.v + a_total * dt)
        # advance position along route: we store only nodes; to keep simple we pop nodes if we pass them.
        if not self.route:
            return
        # approximate travel along straight segment between nodes using current speed
        # compute distance that would be covered:
        dist_to_cover = self.v * dt
        while dist_to_cover > 0 and self.position_index < len(self.route) - 1:
            p0 = self.route[self.position_index]
            p1 = self.route[self.position_index + 1]
            seg_len = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            if seg_len <= 1e-6:
                self.position_index += 1
                continue
            if dist_to_cover >= seg_len:
                # move to next node
                dist_to_cover -= seg_len
                self.position_index += 1
                self.current_coord = self.route[self.position_index]
            else:
                # move partway: linear interpolate between p0 and p1
                ratio = dist_to_cover / seg_len
                self.current_coord = (p0[0] + (p1[0] - p0[0]) * ratio,
                                      p0[1] + (p1[1] - p0[1]) * ratio)
                dist_to_cover = 0

        self.travel_time += dt

    @property
    def position(self):
        return getattr(self, 'current_coord', None)

    #  a simple lane shift evaluator 
    def evaluate_lane_shift(self, env):
        """
        Return True if shifting lane is beneficial.
        env should provide lane traffic speeds / densities for candidate lanes.
        For now, this is a stub implementing the idea: check neighboring lane avg speed.
        """
        try:
            current_lane_speed = env.avg_speed_on_lane(self)
            adjacent_lane_speed = env.avg_speed_on_adjacent_lane(self)
            return adjacent_lane_speed > current_lane_speed + 1.0
        except Exception:
            return False


# --- Pedestrian agent  ---
class PedestrianAgent:
    def __init__(self, agent_id: str, start_coord: Tuple[float, float], end_coord: Tuple[float, float]):
        self.agent_id = agent_id
        self.current_coord = start_coord
        self.end_coord = end_coord
        self.mode = 'walk'

        # body and view parameters from Fujii et al.
        self.body_diameter = 0.49  # m
        self.view_depth = 3.0      # R (m)
        self.view_angle_deg = 160.0
        self.desired_speed = max(0.8, random.gauss(1.30, 0.20))  # m/s
        self.v = self.desired_speed
        self.direction = self._compute_direction_to_target()

    def _compute_direction_to_target(self):
        tx, ty = self.end_coord
        cx, cy = self.current_coord
        dx = tx - cx
        dy = ty - cy
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            return (0.0, 0.0)
        return (dx / norm, dy / norm)

    def _in_view_cone(self, other_pos):
        cx, cy = self.current_coord
        ox, oy = other_pos
        vx = ox - cx
        vy = oy - cy
        dist = math.hypot(vx, vy)
        if dist > self.view_depth or dist < 1e-6:
            return False
        # angle test
        dot = vx * self.direction[0] + vy * self.direction[1]
        cosang = dot / (dist * max(1e-9, math.hypot(*self.direction)))
        ang = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
        return ang <= (self.view_angle_deg / 2.0)

    def step(self, dt: float, env=None):
        """
        env must provide a function: env.get_pedestrians_in_view(self) -> list of (pos, speed, direction)
        The decision flow follows the discrete rules in Fujii et al. (free walk -> collision avoidance -> stop -> overtaking -> following)
        """
        # recompute desired direction to target
        self.direction = self._compute_direction_to_target()

        # get nearest pedestrian in view
        nearest = None
        if env is not None:
            try:
                nearby = env.get_pedestrians_in_view(self)
                if nearby:
                    # pick nearest
                    nearest = min(nearby, key=lambda p: math.hypot(p[0][0] - self.current_coord[0], p[0][1] - self.current_coord[1]))
            except Exception:
                nearest = None

        if nearest is None:
            # Free walk: go toward target at desired speed
            speed = self.desired_speed
            dir_vec = self.direction
        else:
            other_pos, other_speed, other_dir = nearest
            # compute relative geometry
            rx = other_pos[0] - self.current_coord[0]
            ry = other_pos[1] - self.current_coord[1]
            dist = math.hypot(rx, ry)
            # personal space coefficient kps and overtaking accel kac from paper
            kps = 1.2
            kac = 1.3
            # angle between other and my dir
            dot = (other_dir[0] * self.direction[0] + other_dir[1] * self.direction[1])
            # if opposite direction => collision avoidance
            if dot < -0.5:
                # collision avoidance: slow (momentary stop if too close)
                if dist < (kps * self.body_diameter):
                    speed = 0.0  # momentary stop
                    dir_vec = (-self.direction[1], self.direction[0])  # try sidestep 
                else:
                    speed = 0.0
                    # shift direction slightly left/right to avoid
                    dir_vec = (self.direction[0] + 0.3 * (-rx / (dist + 1e-6)), self.direction[1] + 0.3 * (-ry / (dist + 1e-6)))
                    nd = math.hypot(dir_vec[0], dir_vec[1])
                    if nd > 1e-6:
                        dir_vec = (dir_vec[0] / nd, dir_vec[1] / nd)
            else:
                # same direction: follow or overtake depending on leader speed
                if other_speed < self.desired_speed - 0.2:
                    # overtake: accelerate (up to kac * v0), and shift direction slightly
                    speed = min(self.desired_speed * kac, self.v + 1.0)
                    dir_vec = (self.direction[0] + 0.2 * (-rx / (dist + 1e-6)), self.direction[1] + 0.2 * (-ry / (dist + 1e-6)))
                    nd = math.hypot(dir_vec[0], dir_vec[1])
                    if nd > 1e-6:
                        dir_vec = (dir_vec[0] / nd, dir_vec[1] / nd)
                else:
                    # follow: match leader speed
                    speed = min(self.desired_speed, other_speed)
                    dir_vec = (other_dir[0], other_dir[1])

        # update position
        self.v = speed
        dx = dir_vec[0] * self.v * dt
        dy = dir_vec[1] * self.v * dt
        self.current_coord = (self.current_coord[0] + dx, self.current_coord[1] + dy)

        # if reached target (within small radius) -> finish
        if math.hypot(self.current_coord[0] - self.end_coord[0], self.current_coord[1] - self.end_coord[1]) < 0.5:
            self.mode = 'arrived'

    @property
    def position(self):
        return getattr(self, 'current_coord', None)


