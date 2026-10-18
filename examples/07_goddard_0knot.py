"""07. Goddard問題・単一区間。

高度、速度、質量を状態にし、空気抵抗と燃料消費を考慮する。
"""

import numpy as np

from OpenGoddard import Dynamics, Guess, Problem

PARAMETERS = {
    "g0": 1.0,
    "H0": 1.0,
    "V0": 0.0,
    "M0": 1.0,
    "Hc": 500,
    "Dc": 310.0,
    "c": 0.5,
    "Mf": 0.6,
    "T_max": 3.5,
}


# 状態は高度・速度・質量。制御は推力。いずれも無次元量。
def dynamics(problem, p, section):
    h = problem.states(0, section)
    v = problem.states(1, section)
    mass = problem.states(2, section)
    thrust = problem.controls(0, section)
    drag = p["Dc"] * v**2 * np.exp(-p["Hc"] * (h - p["H0"]) / p["H0"])
    gravity = p["g0"] * (p["H0"] / h) ** 2
    dx = Dynamics(problem, section)
    dx[0] = v
    dx[1] = (thrust - drag) / mass - gravity
    dx[2] = -thrust / p["c"]
    return dx()


def equality(problem, p):
    h = problem.states_all_section(0)
    v = problem.states_all_section(1)
    m = problem.states_all_section(2)
    result = problem.condition()
    result.equal(h[0], p["H0"])
    result.equal(v[0], p["V0"])
    result.equal(m[0], p["M0"])
    result.equal(v[-1], 0.0)
    result.equal(m[-1], p["Mf"])
    return result()


def inequality(problem, p):
    h = problem.states_all_section(0)
    v = problem.states_all_section(1)
    m = problem.states_all_section(2)
    T = problem.controls_all_section(0)
    tf = problem.time_final(-1)
    result = problem.condition()
    result.lower_bound(h[1:], p["H0"])
    result.lower_bound(v[1:-1], 0.0)
    result.lower_bound(m[:-1], p["Mf"])
    result.lower_bound(T, 0.0)
    result.lower_bound(tf, 0.1)
    result.upper_bound(m[1:], p["M0"])
    result.upper_bound(T, p["T_max"])
    return result()


def cost(problem, p):
    h = problem.states_all_section(0)
    return -h[-1]


def make_problem(parameters=PARAMETERS):
    problem = Problem([0.0, 0.3], [50], [3], [1], maxIterator=30)
    problem.set_states_all_section(
        0, Guess.cubic(problem.time_all_section, 1.0, 0.0, 1.01, 0.0)
    )
    problem.set_states_all_section(1, Guess.linear(problem.time_all_section, 0.0, 0.0))
    problem.set_states_all_section(
        2, Guess.cubic(problem.time_all_section, 1.0, -0.6, 0.6, 0.0)
    )
    problem.set_controls_all_section(
        0, Guess.cubic(problem.time_all_section, 3.5, 0.0, 0.0, 0.0)
    )
    problem.dynamics = [dynamics]
    problem.knot_states_smooth = []
    problem.cost = cost
    problem.equality = equality
    problem.inequality = inequality
    return problem


def solve(parameters=PARAMETERS):
    problem = make_problem(parameters)
    problem.solve(parameters, ftol=1e-10)
    return problem


def plot(problem):
    import matplotlib.pyplot as plt

    time = problem.time_all_section
    labels = ["height [1]", "velocity [1]", "mass [1]", "thrust [1]"]
    values = [problem.states_all_section(i) for i in range(3)]
    values += [problem.controls_all_section(i) for i in range(1)]
    fig, axes = plt.subplots(4, 1, sharex=True, layout="constrained", figsize=(8, 6))
    for ax, values_i, label in zip(axes, values, labels):
        ax.plot(time, values_i, "o-")
        ax.set_ylabel(label)
        ax.grid()
    axes[-1].set_xlabel("time")
    return fig


def main():
    import matplotlib.pyplot as plt

    problem = solve()
    plot(problem)
    plt.show()
    return problem


if __name__ == "__main__":
    main()
