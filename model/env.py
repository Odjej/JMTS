import math
from collections import defaultdict
import random

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

    def register(self, agent):
        """Should be called after creating each agent."""
        self.agents.append(agent)

    @staticmethod
    def dist(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])
    
    def get_leader(self, agent):
        """
        Returns (distance_m, leader_speed) of the nearest agent
        in front of the given agent along its route.
        """
        # we assume each agent has:
        #   - route (list of node coords)
        #   - position_index
        #   - current_coord
        if not hasattr(agent, "route") or len(agent.route) < 2:
            return (None, 0.0)

        a_pos = agent.current_coord
        a_idx = agent.position_index

        best_dist = None
        best_speed = 0.0

        for other in self.agents:
            if other is agent:
                continue
            if not hasattr(other, "route"):
                continue

            # Only consider others on the SAME route segment
            if other.route != agent.route:
                continue

            o_idx = other.position_index
            o_pos = other.current_coord

            # Other must be "ahead"
            if o_idx < a_idx:
                continue
            if o_idx == a_idx:
                # same segment => test projection direction
                # check if other is in front (positive dot product)
                seg = (
                    agent.route[a_idx+1][0] - agent.route[a_idx][0],
                    agent.route[a_idx+1][1] - agent.route[a_idx][1]
                )
                v = (o_pos[0] - a_pos[0], o_pos[1] - a_pos[1])
                dot = seg[0]*v[0] + seg[1]*v[1]
                if dot <= 0:
                    continue

            d = self.dist(a_pos, o_pos)
            if best_dist is None or d < best_dist:
                best_dist = d
                best_speed = getattr(other, "v", 0.0)

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
        return getattr(agent, "v", 0.0)

    def avg_speed_on_adjacent_lane(self, agent):
        return getattr(agent, "v", 0.0)


def random_graph_od(G):
    """Return two random (different) node coords from the graph."""
    nodes = list(G.nodes())
    start = random.choice(nodes)
    end = random.choice(nodes)
    while end == start:
        end = random.choice(nodes)
    return start, end

    def step(self, dt: float):
        """Advance the environment clock and step all registered agents.

        This method attempts to call agent.step with signatures:
          - step(dt, sim_time, env)
          - step(dt, env)
          - step(dt)
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