"""06. 単純な2次元ロケット。

水平・鉛直の速度と位置を状態にし、加速度の方向を操作する。
"""

import numpy as np

from OpenGoddard import Dynamics, Problem

PARAMETERS = {"a": 1.0}  # 加速度の大きさ [m/s^2]


# 状態は水平速度・鉛直速度・水平位置・鉛直位置。制御は加速度の角度。
def dynamics(problem, p, section):
    vx = problem.states(0, section)
    vy = problem.states(1, section)
    angle = problem.controls(0, section)
    dx = Dynamics(problem, section)
    dx[0] = p["a"] * np.cos(angle)
    dx[1] = p["a"] * np.sin(angle)
    dx[2] = vx
    dx[3] = vy
    return dx()


def equality(problem, p):
    u = problem.states_all_section(0)
    v = problem.states_all_section(1)
    x = problem.states_all_section(2)
    y = problem.states_all_section(3)
    result = problem.condition()
    result.equal(u[0], 0.0)
    result.equal(v[0], 0.0)
    result.equal(x[0], 0.0)
    result.equal(y[0], 0.0)
    result.equal(u[-1], 1.0)
    result.equal(v[-1], 0.0)
    result.equal(y[-1], 1.0)
    return result()


def inequality(problem, p):
    beta = problem.controls_all_section(0)
    result = problem.condition()
    result.lower_bound(beta, -np.pi / 2)
    result.upper_bound(beta, np.pi / 2)
    return result()


def cost(problem, p):
    tf = problem.time_final(-1)
    return tf


def make_problem(parameters=PARAMETERS):
    problem = Problem([0.0, 2.0], [20], [4], [1], maxIterator=50)
    duration = problem.time_final(0) - problem.time_start(0)
    fraction = (problem.time[0] - problem.time_start(0)) / duration
    problem.set_states(0, 0, fraction)
    problem.set_states(1, 0, 6 / duration * fraction * (1 - fraction))
    problem.set_states(2, 0, duration / 2 * fraction**2)
    problem.set_states(3, 0, 3 * fraction**2 - 2 * fraction**3)
    problem.set_controls(
        0, 0, np.arctan2(6 / duration**2 * (1 - 2 * fraction), 1 / duration)
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

    fig, axes = plt.subplots(1, 2, layout="constrained", figsize=(10, 4))
    axes[0].plot(problem.states(2, 0), problem.states(3, 0), "o-")
    axes[0].set(xlabel="x [m]", ylabel="y [m]")
    axes[1].plot(problem.time_all_section, problem.controls(0, 0), "o-")
    axes[1].set(xlabel="time [s]", ylabel="angle [rad]")
    return fig


def main():
    import matplotlib.pyplot as plt

    problem = solve()
    plot(problem)
    plt.show()
    return problem


if __name__ == "__main__":
    main()
