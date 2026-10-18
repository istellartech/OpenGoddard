"""02. 加速度の上限を守り、最短時間で移動して停止する。"""

import numpy as np

from OpenGoddard import Dynamics, Problem

PARAMETERS = {"distance": 10.0, "max_acceleration": 2.0}  # m、m/s^2


def dynamics(problem, p, section):
    dx = Dynamics(problem, section)
    dx[0] = problem.states(1, section)  # 位置の変化率は速度
    dx[1] = problem.controls(0, section)  # 速度の変化率は加速度
    return dx()


def equality(problem, p):
    x = problem.states_all_section(0)
    v = problem.states_all_section(1)
    c = problem.condition()
    c.equal(x[0], 0)
    c.equal(v[0], 0)
    c.equal(x[-1], p["distance"])
    c.equal(v[-1], 0)
    return c()


def cost(problem, p):
    return problem.time_final(-1)


def make_problem(parameters=PARAMETERS, nodes=12):
    p = parameters
    a = p["max_acceleration"]
    # 加速・減速の2区間。位置と速度は区間の境界で自動的につながる。
    # 区間の時刻は答えを知らない前提で 0 -> 3 -> 6 s と大まかに置く。
    problem = Problem([0, 3, 6], [nodes, nodes], [2, 2], [1, 1])
    # 初期推定：6 s かけて一定の速さで進む（加速度 0）。始点・終点で止まる条件は
    # まだ満たしていない。
    t = problem.time_all_section
    speed = p["distance"] / t[-1]
    problem.set_states_all_section(0, speed * t)
    problem.set_states_all_section(1, np.full(len(t), speed))
    problem.set_controls_all_section(0, np.zeros(len(t)))
    for section in range(2):
        problem.set_controls_bounds(0, section, -a, a)
    problem.dynamics = [dynamics, dynamics]
    problem.equality = equality
    problem.cost = cost
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
    axes[0].set_ylabel("position [m]")
    axes[1].plot(time, problem.states_all_section(1), "o-")
    axes[1].set_ylabel("velocity [m/s]")
    axes[2].plot(time, problem.controls_all_section(0), "o-")
    axes[2].set(xlabel="time [s]", ylabel="acceleration [m/s^2]")
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
