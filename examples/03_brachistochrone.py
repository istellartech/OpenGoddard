"""03. 最速降下。

位置 x、深さ y、速さ v を状態、下向きからの角度を制御とする。
"""

import numpy as np

from OpenGoddard import Dynamics, Guess, Problem

PARAMETERS = {"g": 1.0, "l": 1.0}  # 重力 [m/s^2]、終点の水平距離 [m]


# 状態は x・深さ・速さ。制御は下向きから測った角度。
def dynamics(problem, p, section):
    v = problem.states(2, section)
    theta = problem.controls(0, section)
    dx = Dynamics(problem, section)
    dx[0] = v * np.sin(theta)
    dx[1] = v * np.cos(theta)
    dx[2] = p["g"] * np.cos(theta)
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
    return result()


def inequality(problem, p):
    y = problem.states_all_section(1)
    theta = problem.controls_all_section(0)
    tf = problem.time_final(-1)
    result = problem.condition()
    result.lower_bound(tf, 0.1)
    result.lower_bound(y[1:], 0)
    result.lower_bound(theta, 0)
    return result()


def cost(problem, p):
    tf = problem.time_final(-1)
    return tf


def make_problem(parameters=PARAMETERS):
    problem = Problem([0.0, 2.0], [20], [3], [1], maxIterator=30)
    p = parameters
    problem.set_states_all_section(
        0, Guess.linear(problem.time_all_section, 0.0, p["l"])
    )
    problem.set_states_all_section(
        1, Guess.linear(problem.time_all_section, 0.0, p["l"] / np.sqrt(3))
    )
    problem.set_controls_all_section(
        0, Guess.linear(problem.time_all_section, np.deg2rad(30), np.deg2rad(30))
    )
    problem.dynamics = [dynamics]
    problem.knot_states_smooth = []
    problem.cost = cost
    problem.equality = equality
    problem.inequality = inequality
    return problem


def solve(parameters=PARAMETERS):
    problem = make_problem(parameters)
    problem.solve(parameters)
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
