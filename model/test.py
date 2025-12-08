from model.simulate import load_drive_graph
from model.env import Environment
from model.agents import CarAgent
G = load_drive_graph()
# set some edges to have 2 lanes for the test (optional)
for u,v,data in list(G.edges(data=True))[:10]:
    data['lanes'] = 2
env = Environment(G)
# create a slow car ahead and a faster car behind on the same route
start, end = list(G.nodes())[0], list(G.nodes())[3]
slow = CarAgent('slow', start, end, desired_speed_m_s=6.0)
fast = CarAgent('fast', start, end, desired_speed_m_s=16.0)
slow.plan_route(G, env=env)
fast.plan_route(G, env=env)
env.register(slow)
env.register(fast)
# place slow ahead a bit
slow.position_index = 1
slow.current_coord = slow.route[1]
fast.position_index = 0
fast.current_coord = fast.route[0]
# simulate
for t in range(30):
    env.step(1.0)
    print(f"t={t} fast lane={fast.lane} fast pos={fast.position} fast v={fast.v:.2f} slow lane={slow.lane} slow v={slow.v:.2f}")