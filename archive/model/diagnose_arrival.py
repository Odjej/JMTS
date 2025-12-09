"""(archived) Diagnostic: inspect agent planned routes and predicted travel times.

Run this from the repo root: `python -m model.diagnose_arrival`
This is an archived copy of the original `model/diagnose_arrival.py`.
"""
import math
import random
from model.network_utils import load_edges, build_graph_from_edges, nearest_node
from model.env import Environment, random_graph_od
from model.agents import CarAgent
import os

G = build_graph_from_edges(load_edges(os.path.join('data','processed','networks','drive_edges_clean.gpkg')))
print(f'Loaded graph: {len(G.nodes())} nodes, {len(G.edges())} edges')

# spawn a few agents
agents = []
for i in range(5):
    s, t = random_graph_od(G)
    car = CarAgent(f'diag_{i}', s, t)
    agents.append(car)

env = Environment(G)

for car in agents:
    car.plan_route(G, env=env)
    env.register(car)
    route = getattr(car, 'route', [])
    if not route:
        print(car.agent_id, 'no route found')
        continue
    # compute route geometric length and predicted travel time via env.get_edge_travel_time
    total_length = 0.0
    predicted_time = 0.0
    for i in range(0, len(route)-1):
        e = (route[i], route[i+1])
        L = math.hypot(e[1][0]-e[0][0], e[1][1]-e[0][1])
        total_length += L
        predicted_time += env.get_edge_travel_time(e)
    print(f'{car.agent_id}: route nodes={len(route)} length_m={total_length:.1f} predicted_time_s={predicted_time:.1f}')

# show a sample agent route and nodes
sample = agents[0]
print('\nSample agent route snippet:')
for i, n in enumerate(sample.route[:10]):
    print(i, n)

# run a short sim for this small set and report final positions
sim_time = 5000
dt = 1.0
steps = int(sim_time // dt)
speed_samples = {c.agent_id: [] for c in agents}
for t in range(steps):
    # record a sample of speeds at start of step
    for c in agents:
        speed_samples[c.agent_id].append(getattr(c, 'v', 0.0))
    env.step(dt)

print('\nAfter sim:')
for car in agents:
    pos = getattr(car, 'current_coord', None)
    idx = getattr(car, 'position_index', None)
    route_len = len(getattr(car, 'route', []))
    arrived = idx >= max(0, route_len - 1)
    print(car.agent_id, 'pos=', pos, 'index=', idx, 'route_len=', route_len, 'arrived=', arrived, 'travel_time=', car.travel_time, 'v=', getattr(car, 'v', None))
    # print a few speed samples
    samp = speed_samples.get(car.agent_id, [])
    print('  speed samples (first 5):', [round(s,2) for s in samp[:5]], '... last:', round(samp[-1],2) if samp else None)
