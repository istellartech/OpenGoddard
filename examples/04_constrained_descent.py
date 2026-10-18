"""04. 最速降下の経路に、深さの上限を加える。深さは下向きが正。"""

import numpy as np

from OpenGoddard import Dynamics, Guess, Problem

PARAMETERS = {"distance": 1.0, "final_depth": 0.2, "gravity": 9.81, "max_depth": 0.3}


def dynamics(problem, p, section):
    v = problem.states(2, section)
    theta = problem.controls(0, section)
    dx = Dynamics(problem, section)
    dx[0] = v * np.sin(theta)  # 水平位置
    dx[1] = v * np.cos(theta)  # 深さ
    dx[2] = p["gravity"] * np.cos(theta)  # 速さ
    return dx()


def equality(problem, p):
    x = problem.states(0, 0)
    depth = problem.states(1, 0)
    v = problem.states(2, 0)
    c = problem.condition()
    c.equal(x[0], 0)
    c.equal(depth[0], 0)
    c.equal(v[0], 0)
    c.equal(x[-1], p["distance"])
    c.equal(depth[-1], p["final_depth"])
    return c()


def cost(problem, p):
    return problem.time_final(0)


def make_problem(parameters=PARAMETERS, nodes=40):
    p = parameters
    problem = Problem([0, 0.7], [nodes], [3], [1])
    time = problem.time[0]
    fraction = time / 0.7
    depth = p["final_depth"] * fraction + 0.2 * np.sin(np.pi * fraction)
    problem.set_states(0, 0, p["distance"] * fraction)
    problem.set_states(1, 0, depth)
    problem.set_states(2, 0, np.sqrt(2 * p["gravity"] * depth))
    problem.set_controls(0, 0, Guess.linear(time, 0, 2))
    # 03に加える経路制約。どのノードでも床より下へ降りない。
    problem.set_states_bounds(1, 0, 0, p["max_depth"])
    problem.set_states_bounds(2, 0, 0, None)
    problem.set_controls_bounds(0, 0, 0, np.pi)
    problem.set_time_final_bounds(0, 0.01, 5)
    problem.dynamics = [dynamics]
    problem.equality = equality
    problem.cost = cost
    return problem


def solve(parameters=PARAMETERS, nodes=40):
    problem = make_problem(parameters, nodes)
    problem.solve(parameters, maxiter=500, ftol=1e-10)
    return problem


def plot(problem, parameters=PARAMETERS):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(problem.states(0, 0), problem.states(1, 0), "o-")
    ax.axhline(parameters["max_depth"], linestyle="--", color="k")
    ax.set(xlabel="x [m]", ylabel="depth [m]")
    ax.invert_yaxis()
    return fig


def main():
    import matplotlib.pyplot as plt

    problem = solve()
    print("Time:", problem.time_final(0))
    plot(problem)
    plt.show()
    return problem


if __name__ == "__main__":
    main()
