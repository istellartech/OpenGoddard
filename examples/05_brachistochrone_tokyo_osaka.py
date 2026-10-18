"""05. 東京・大阪の最速降下。

03のモデルを実距離・重力に変え、単位スケールを導入する。平面内、摩擦なしの近似。
"""

import numpy as np

from OpenGoddard import Dynamics, Guess, Problem

PARAMETERS = {"g": 9.8, "l": 600000, "h": 300000}  # 重力 [m/s^2]、距離 [m]


# 状態は水平距離・深さ・速さ。制御は軌道の角度。
def dynamics(problem, p, section):
    v = problem.states(2, section)
    theta = problem.controls(0, section)
    g = p["g"]
    dx = Dynamics(problem, section)
    dx[0] = v * np.sin(theta)
    dx[1] = v * np.cos(theta)
    dx[2] = g * np.cos(theta)
    return dx()


def equality(problem, p):
    x = problem.states_all_section(0)
    y = problem.states_all_section(1)
    v = problem.states_all_section(2)
    result = problem.condition()
    result.equal(x[0], 0.0)
    result.equal(y[0], 0.0)
    result.equal(v[0], 0.0)
    result.equal(x[-1], p["l"])
    result.equal(y[-1], 0.0)
    return result()


def inequality(problem, p):
    x = problem.states_all_section(0)
    y = problem.states_all_section(1)
    theta = problem.controls_all_section(0)
    result = problem.condition()
    result.lower_bound(x, 0)
    result.lower_bound(y, 0)
    result.lower_bound(theta, 0)
    result.upper_bound(theta, np.pi)
    result.upper_bound(x, p["l"])
    return result()


def cost(problem, p):
    tf = problem.time_final(-1)
    return tf


def make_problem(parameters=PARAMETERS):
    problem = Problem([0.0, 600.0], [30], [3], [1], maxIterator=10)
    p = parameters
    unit_x = 300000
    unit_y = 100000
    unit_t = 100
    unit_v = unit_x / unit_t
    problem.set_unit_states_all_section(0, unit_x)
    problem.set_unit_states_all_section(1, unit_y)
    problem.set_unit_states_all_section(2, unit_v)
    problem.set_unit_controls_all_section(0, 1.0)
    problem.set_unit_time(unit_t)
    half_nodes = int(problem.nodes[0] / 2)
    y_init0 = Guess.linear(problem.time_all_section[:half_nodes], 0, p["h"])
    y_init1 = Guess.linear(problem.time_all_section[half_nodes:], p["h"], 0)
    problem.set_states_all_section(
        0, Guess.linear(problem.time_all_section, 0.0, p["l"])
    )
    problem.set_states_all_section(1, np.hstack((y_init0, y_init1)))
    problem.set_controls_all_section(
        0, Guess.linear(problem.time_all_section, 0.0, np.pi)
    )
    problem.dynamics = [dynamics]
    problem.knot_states_smooth = []
    problem.cost = cost
    problem.equality = equality
    problem.inequality = inequality
    return problem


def solve(parameters=PARAMETERS):
    problem = make_problem(parameters)
    problem.solve(parameters, maxiter=2000)
    return problem


def plot(problem):
    import matplotlib.pyplot as plt

    time = problem.time_all_section
    x = problem.states_all_section(0)
    depth = problem.states_all_section(1)
    speed = problem.states_all_section(2)
    theta = problem.controls_all_section(0)
    fig, axes = plt.subplots(1, 3, layout="constrained", figsize=(12, 4))
    axes[0].plot(x, depth, "o-")
    axes[0].set(xlabel="x [m]", ylabel="depth [m]")
    axes[0].invert_yaxis()
    axes[1].plot(time, speed, "o-")
    axes[1].set(xlabel="time [s]", ylabel="speed [m/s]")
    axes[2].plot(time, theta, "o-")
    axes[2].set(xlabel="time [s]", ylabel="theta [rad]")
    return fig


def main():
    import matplotlib.pyplot as plt

    problem = solve()
    plot(problem)
    plt.show()
    return problem


if __name__ == "__main__":
    main()
