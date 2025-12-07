import math
import networkx as nx
from typing import Tuple, List


class PedestrianAgent:
	def __init__(self, agent_id: str, start_coord: Tuple[float, float], end_coord: Tuple[float, float]):
		self.agent_id = agent_id
		self.current_coord = start_coord
		self.end_coord = end_coord
		self.mode = 'walk'

	def move(self):
		# Placeholder for pedestrian movement
		pass


class CarAgent:
	def __init__(self, agent_id: str, start_coord: Tuple[float, float], end_coord: Tuple[float, float], priority: int = 1):
		self.agent_id = agent_id
		self.start_coord = start_coord
		self.current_coord = start_coord
		self.end_coord = end_coord
		self.mode = 'drive'
		self.priority = priority
		self.travel_time = 0.0
		self.route: List[Tuple[float, float]] = []
		self._start_node = None
		self._end_node = None

	def plan_route(self, G: nx.Graph, default_speed_m_s: float = 13.89):
		"""Plan a route using A* on graph G, storing node-coordinate path in `self.route`.

		Heuristic uses straight-line distance divided by `default_speed_m_s` to
		keep units consistent with the edge weight `travel_time_s`.
		"""
		# Find nearest graph nodes to start/end coordinates
		try:
			from model.network_utils import nearest_node
		except Exception:
			# If imported as a script, try relative import
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

		self.route = path

	def get_route_coords(self) -> List[Tuple[float, float]]:
		return [(n[0], n[1]) for n in self.route]

	def move_along_route(self):
		# Simple step: pop next node and update current_coord
		if not self.route:
			return
		if (self.current_coord[0], self.current_coord[1]) == self.route[0]:
			# already at first node
			self.route.pop(0)
		if self.route:
			self.current_coord = self.route.pop(0)


class ConstructionEvent:
	def __init__(self, affected_edge_id, delay):
		self.affected_edge_id = affected_edge_id
		self.delay = delay

	def affects_edge(self, edge_id):
		return edge_id == self.affected_edge_id


class TramAgent:
	def __init__(self, agent_id: str, start_feature, end_feature):
		self.agent_id = agent_id
		self.current_feature = start_feature
		self.end_feature = end_feature
		self.mode = 'tram'

	def move(self):
		pass

		pass
