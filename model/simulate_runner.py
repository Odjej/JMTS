from model.env import Environment, random_graph_od
from model.simulate import load_drive_graph
from model.agents import CarAgent, PedestrianAgent


def main():
    G = load_drive_graph()
    env = Environment(G)

    # -----------------------
    # CAR
    # -----------------------
    start, end = random_graph_od(G)
    car = CarAgent("car_1", start, end)
    car.plan_route(G)
    env.register(car)

    # -----------------------
    # PEDESTRIAN
    # -----------------------
    p_start, p_end = random_graph_od(G)
    ped = PedestrianAgent("ped_1", p_start, p_end)
    env.register(ped)


    # -----------------------
    # RUN SIM
    # -----------------------
    dt = 1.0
    T  = 300

    for step in range(int(T/dt)):
        t = step * dt
        env.step(dt)

        if step % 10 == 0:
            print(
                f"t={t:.1f}s  "
                f"car={car.position}  "
                f"ped={ped.position}"
            )