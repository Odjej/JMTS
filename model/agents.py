"""
Agent behaviors for vehicle and pedestrian simulation.

Implements:
  - CarAgent: vehicles with kinematic dynamics, route planning, and congestion-aware braking
  - PedestrianAgent: pedestrians with Fujii-like behavior for walking and avoidance
"""
import math
import random
import networkx as nx
from typing import Tuple, List


class CarAgent:
    """Simulates a car agent using simplified Generalized Force Model (GFM) dynamics.
    
    Features:
      - A* route planning with environment-aware edge weights
      - Relaxation toward desired speed with leader-distance braking
      - Pedestrian-aware yielding/braking
      - Lane changes with cooldown
      - Travel time and distance tracking for statistics
    """
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
        # track distance traveled for average speed calculations
        self.distance_travelled = 0.0
        # lane index on current edge (0 = leftmost/default)
        self.lane = 0

    def plan_route(self, G: nx.Graph, default_speed_m_s: float = 13.89, env=None, from_coord=None):
        """
        Plan A* route from current position to destination using environment-aware edge weights.
        
        Uses A* search on the network graph with heuristic based on Euclidean distance.
        Edge weights include historical travel time and construction penalties if enabled.
        
        Args:
            G: NetworkX graph with 'x' and 'y' node attributes
            default_speed_m_s: Fallback speed for weight calculation (m/s), default 13.89
            env: Environment object with get_edge_travel_time() method for dynamic weights
            from_coord: Optional start coordinate; uses self.start_coord if None
        """
        try:
            from model.network_utils import nearest_node
        except Exception:
            from .network_utils import nearest_node

        start_point = from_coord if from_coord is not None else self.start_coord
        self._start_node = nearest_node(G, start_point)
        self._end_node = nearest_node(G, self.end_coord)

        def heuristic(u, v):
            ux, uy = G.nodes[u]['x'], G.nodes[u]['y']
            vx, vy = G.nodes[v]['x'], G.nodes[v]['y']
            euclid = math.hypot(ux - vx, uy - vy)
            return euclid / default_speed_m_s

        # choose weight function: if env provided, prefer dynamic edge travel time
        if env is not None and hasattr(env, 'get_edge_travel_time'):
            def weight(u, v, data=None):
                try:
                    return env.get_edge_travel_time((u, v))
                except Exception:
                    return float(data.get('travel_time_s', data.get('length_m', 0) / default_speed_m_s)) if data is not None else 0.0
        else:
            def weight(u, v, data=None):
                if data is None:
                    data = G.get_edge_data(u, v) or {}
                return float(data.get('travel_time_s', data.get('length_m', 0) / default_speed_m_s))

        try:
            path = nx.astar_path(G, self._start_node, self._end_node, heuristic=heuristic, weight=weight)
        except nx.NetworkXNoPath:
            self.route = []
            return

        # store as node coordinate tuples
        self.route = path
        self.position_index = 0

    def _desired_accel(self):
        """Calculate relaxation acceleration toward desired speed.
        
        Implements relaxation term: a = (v0 - v) / tau, clipped by maximum acceleration.
        
        Returns:
            float: Acceleration in m/s^2, bounded by [-a_max, a_max]
        """
        a = (self.v0 - self.v) / max(self.tau, 1e-6)
        if a > self.a_max:
            a = self.a_max
        return a

    def _repulsive_from_leader(self, leader_dist, leader_speed):
        """
        Calculate braking acceleration due to leading vehicle.
        
        Implements Intelligent Driver Model (IDM)-style deceleration when gap is less
        than desired safe following distance. If no leader, returns 0.
        
        Args:
            leader_dist: Distance to leader vehicle (m) or None if no leader
            leader_speed: Speed of leader vehicle (m/s)
        
        Returns:
            float: Braking acceleration (negative), 0 if safe distance maintained
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
        if env is None:
            return (None, 0.0)
        try:
            dist, spd = env.get_leader(self)
            return (dist, spd)
        except Exception:
            return (None, 0.0)

    def attempt_lane_change(self, env):
        if env is None:
            return False
        try:
            try:
                ped_dist, ped_speed = env.get_pedestrians_ahead(self, look_ahead_m=10.0)
            except Exception:
                ped_dist = None
            if ped_dist is not None and ped_dist <= 8.0:
                return False
            target = env.can_change_lane(self)
            if target is None:
                return False
            return env.change_lane(self, target)
        except Exception:
            return False

    def step(self, dt: float, env=None):
        # decide on lane change (overtaking) before acceleration/movement
        try:
            if self.evaluate_lane_shift(env):
                changed = self.attempt_lane_change(env)
                if changed:
                    self.v = min(self.v * 1.05 + 0.1, self.v0)
        except Exception:
            pass

        a_des = self._desired_accel()
        leader_dist, leader_speed = self._find_virtual_preceding(env)
        ped_dist, ped_speed = (None, 0.0)
        try:
            if env is not None:
                ped_dist, ped_speed = env.get_pedestrians_ahead(self, look_ahead_m=30.0)
        except Exception:
            ped_dist, ped_speed = (None, 0.0)
        a_ped = 0.0
        if ped_dist is not None:
            if ped_dist <= 2.0:
                a_ped = -5.0
            elif ped_dist <= 8.0:
                a_ped = -3.0 * (1.0 - (ped_dist - 2.0) / 6.0)
        a_rep = self._repulsive_from_leader(leader_dist, leader_speed)

        xi = random.gauss(0, 0.05)
        a_total = a_des + a_rep + a_ped + xi
        if a_total > self.a_max:
            a_total = self.a_max
        if a_total < -5.0:
            a_total = -5.0

        self.v = max(0.0, self.v + a_total * dt)

        if not self.route:
            return
        dist_to_cover = self.v * dt
        moved = 0.0
        while dist_to_cover > 0 and self.position_index < len(self.route) - 1:
            p0 = self.route[self.position_index]
            p1 = self.route[self.position_index + 1]
            cur = getattr(self, 'current_coord', p0)
            remain = math.hypot(p1[0] - cur[0], p1[1] - cur[1])
            if remain <= 1e-6:
                self.position_index += 1
                self.current_coord = self.route[self.position_index]
                continue
            if dist_to_cover >= remain:
                # we travel the remainder of this segment
                moved += remain
                dist_to_cover -= remain
                self.position_index += 1
                self.current_coord = self.route[self.position_index]
            else:
                # partial traversal
                moved += dist_to_cover
                ratio = dist_to_cover / remain
                self.current_coord = (cur[0] + (p1[0] - cur[0]) * ratio,
                                      cur[1] + (p1[1] - cur[1]) * ratio)
                dist_to_cover = 0

        # increment travel time
        self.travel_time += dt

        # accumulate distance traveled this timestep
        try:
            self.distance_travelled += moved
        except Exception:
            pass

        # record arrival time if reached
        try:
            at_end = (self.position_index >= len(self.route) - 1)
        except Exception:
            at_end = False
        if at_end and not hasattr(self, 'arrival_time'):
            try:
                if hasattr(self, 'env') and getattr(self.env, 'time', None) is not None:
                    self.arrival_time = float(self.env.time)
                else:
                    self.arrival_time = float(self.travel_time)
            except Exception:
                self.arrival_time = float(self.travel_time)

    @property
    def position(self):
        return getattr(self, 'current_coord', None)

    def evaluate_lane_shift(self, env):
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
        # recompute desired direction to target
        self.direction = self._compute_direction_to_target()

        # get nearest pedestrian in view
        nearest = None
        if env is not None:
            try:
                nearby = env.get_pedestrians_in_view(self)
                if nearby:
                    nearest = min(nearby, key=lambda p: math.hypot(p[0][0] - self.current_coord[0], p[0][1] - self.current_coord[1]))
            except Exception:
                nearest = None

        if nearest is None:
            speed = self.desired_speed
            dir_vec = self.direction
        else:
            other_pos, other_speed, other_dir = nearest
            rx = other_pos[0] - self.current_coord[0]
            ry = other_pos[1] - self.current_coord[1]
            dist = math.hypot(rx, ry)
            kps = 1.2
            kac = 1.3
            dot = (other_dir[0] * self.direction[0] + other_dir[1] * self.direction[1])
            if dot < -0.5:
                if dist < (kps * self.body_diameter):
                    speed = 0.0
                    dir_vec = (-self.direction[1], self.direction[0])
                else:
                    speed = 0.0
                    dir_vec = (self.direction[0] + 0.3 * (-rx / (dist + 1e-6)), self.direction[1] + 0.3 * (-ry / (dist + 1e-6)))
                    nd = math.hypot(dir_vec[0], dir_vec[1])
                    if nd > 1e-6:
                        dir_vec = (dir_vec[0] / nd, dir_vec[1] / nd)
            else:
                if other_speed < self.desired_speed - 0.2:
                    speed = min(self.desired_speed * kac, self.v + 1.0)
                    dir_vec = (self.direction[0] + 0.2 * (-rx / (dist + 1e-6)), self.direction[1] + 0.2 * (-ry / (dist + 1e-6)))
                    nd = math.hypot(dir_vec[0], dir_vec[1])
                    if nd > 1e-6:
                        dir_vec = (dir_vec[0] / nd, dir_vec[1] / nd)
                else:
                    speed = min(self.desired_speed, other_speed)
                    dir_vec = (other_dir[0], other_dir[1])

        self.v = speed
        dx = dir_vec[0] * self.v * dt
        dy = dir_vec[1] * self.v * dt
        self.current_coord = (self.current_coord[0] + dx, self.current_coord[1] + dy)
        if math.hypot(self.current_coord[0] - self.end_coord[0], self.current_coord[1] - self.end_coord[1]) < 0.5:
            self.mode = 'arrived'

    @property
    def position(self):
        return getattr(self, 'current_coord', None)


