"""09. 質量一定の着陸機を、高度0・速度0で月面に降ろす。上向きが正。"""

import numpy as np

from OpenGoddard import Dynamics, Problem

PARAMETERS = {
    "initial_altitude": 100.0,  # m
    "initial_velocity": -10.0,  # m/s
    "gravity": 1.62,  # m/s^2
    "mass": 1000.0,  # kg
    "max_thrust": 4000.0,  # N
}


def dynamics(problem, p, section):
    dx = Dynamics(problem, section)
    dx[0] = problem.states(1, section)
    dx[1] = problem.controls(0, section) / p["mass"] - p["gravity"]
    return dx()


def equality(problem, p):
    h = problem.states_all_section(0)
    v = problem.states_all_section(1)
    c = problem.condition()
    c.equal(h[0], p["initial_altitude"], unit=p["initial_altitude"])
    c.equal(v[0], p["initial_velocity"], unit=problem.unit_states[0][1])
    c.equal(h[-1], 0, unit=p["initial_altitude"])
    c.equal(v[-1], 0, unit=problem.unit_states[0][1])
    return c()


def running_cost(problem, p):
    return problem.controls_all_section(0) / (p["max_thrust"] * problem.unit_time)


def make_problem(parameters=PARAMETERS, nodes=12):
    p = parameters
    g = p["gravity"]
    braking = p["max_thrust"] / p["mass"] - g
    switch_speed = np.sqrt(
        (p["initial_velocity"] ** 2 + 2 * g * p["initial_altitude"])
        * braking
        / (braking + g)
    )
    coast = (switch_speed + p["initial_velocity"]) / g
    burn = switch_speed / braking
    problem = Problem([0, coast, coast + burn], [nodes, nodes], [2, 2], [1, 1])
    problem.set_unit_time(np.sqrt(p["initial_altitude"] / g))
    problem.set_unit_states_all_section(0, p["initial_altitude"])
    problem.set_unit_states_all_section(1, np.sqrt(p["initial_altitude"] * g))
    problem.set_unit_controls_all_section(0, p["max_thrust"])
    h0, v0 = p["initial_altitude"], p["initial_velocity"]
    thrusts = [0, p["max_thrust"]]  # 自由落下と減速の初期値
    for section in range(2):
        t = problem.time[section] - problem.time_start(section)
        thrust = thrusts[section]
        acceleration = thrust / p["mass"] - g
        h = h0 + v0 * t + acceleration * t**2 / 2
        v = v0 + acceleration * t
        problem.set_states(0, section, h)
        problem.set_states(1, section, v)
        problem.set_controls(0, section, np.full(nodes, thrust))
        problem.set_states_bounds(0, section, 0, p["initial_altitude"])
        problem.set_states_bounds(1, section, None, 0)
        problem.set_controls_bounds(0, section, 0, p["max_thrust"])
        h0, v0 = h[-1], v[-1]
    problem.dynamics = [dynamics, dynamics]
    problem.equality = equality
    problem.running_cost = running_cost
    return problem


def solve(parameters=PARAMETERS, nodes=12):
    problem = make_problem(parameters, nodes)
    problem.solve(parameters, maxiter=300, ftol=1e-9)
    return problem


def plot(problem):
    import matplotlib.pyplot as plt

    time = problem.time_all_section
    fig, axes = plt.subplots(3, 1, sharex=True, layout="constrained")
    axes[0].plot(time, problem.states_all_section(0), "o-")
    axes[0].set_ylabel("altitude [m]")
    axes[1].plot(time, problem.states_all_section(1), "o-")
    axes[1].set_ylabel("velocity [m/s]")
    axes[2].plot(time, problem.controls_all_section(0), "o-")
    axes[2].set(xlabel="time [s]", ylabel="thrust [N]")
    return fig


def main():
    import matplotlib.pyplot as plt

    problem = solve()
    print("Time:", problem.time_final(-1))
    plot(problem)
    plt.show()
    return problem


if __name__ == "__main__":
    main()
