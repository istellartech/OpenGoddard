"""15. 低推力軌道遷移。

燃料使用量に対応する推力成分の絶対値を積分して最小化する。無次元モデル。
"""

import numpy as np

from OpenGoddard import Condition, Dynamics, Guess, Problem

PARAMETERS = {
    "u_max": 0.01,
    "r0": 1.0,
    "vr0": 0.0,
    "vt0": 1.0,
    "rf": 4.0,
    "vrf": 0.0,
    "vtf": 0.5,
    "tf_max": 55,
}


# 状態は軌道半径・半径方向速度・接線方向速度。推力は各方向の正負を別々に持つ。
def dynamics(problem, p, section):
    r = problem.states(0, section)
    vr = problem.states(1, section)
    vt = problem.states(2, section)
    ur1 = problem.controls(0, section)
    ur2 = problem.controls(1, section)
    ut1 = problem.controls(2, section)
    ut2 = problem.controls(3, section)
    dx = Dynamics(problem, section)
    dx[0] = vr
    dx[1] = vt**2 / r - 1 / r**2 + (ur1 - ur2)
    dx[2] = -vr * vt / r + (ut1 - ut2)
    return dx()


def equality(problem, p):
    r = problem.states_all_section(0)
    vr = problem.states_all_section(1)
    vt = problem.states_all_section(2)
    result = Condition()
    result.equal(r[0], p["r0"])
    result.equal(vr[0], p["vr0"])
    result.equal(vt[0], p["vt0"])
    result.equal(r[-1], p["rf"])
    result.equal(vr[-1], p["vrf"])
    result.equal(vt[-1], p["vtf"])
    return result()


def inequality(problem, p):
    r = problem.states_all_section(0)
    result = Condition()
    result.lower_bound(r[1:-1], p["r0"])
    result.upper_bound(r[1:-1], p["rf"])
    return result()


def running_cost(problem, p):
    ur1 = problem.controls_all_section(0)
    ur2 = problem.controls_all_section(1)
    ut1 = problem.controls_all_section(2)
    ut2 = problem.controls_all_section(3)
    return ur1 + ur2 + (ut1 + ut2)


def make_problem(parameters=PARAMETERS):
    problem = Problem([0.0, 50.0], [60], [3], [4], maxIterator=10)
    p = parameters
    r_init = Guess.linear(problem.time_all_section, p["r0"], p["rf"])
    problem.set_states_all_section(
        0, Guess.linear(problem.time_all_section, p["r0"], p["rf"])
    )
    problem.set_states_all_section(
        1, Guess.linear(problem.time_all_section, p["vr0"], p["vrf"])
    )
    problem.set_states_all_section(2, 1 / np.sqrt(r_init))
    problem.set_controls_all_section(0, Guess.constant(problem.time_all_section, 0.0))
    problem.set_controls_all_section(
        2, Guess.constant(problem.time_all_section, p["u_max"] / 2)
    )
    for control in range(4):
        problem.set_controls_bounds(control, 0, 0.0, p["u_max"])
    problem.set_time_final_bounds(0, 1.0, p["tf_max"])
    problem.dynamics = [dynamics]
    problem.knot_states_smooth = []
    problem.running_cost = running_cost
    problem.equality = equality
    problem.inequality = inequality
    return problem


def solve(parameters=PARAMETERS):
    problem = make_problem(parameters)
    problem.solve(parameters, ftol=1e-10, maxiter=300)
    return problem


def plot(problem):
    import matplotlib.pyplot as plt

    time = problem.time_all_section
    labels = [
        "radius [1]",
        "vr [1]",
        "vt [1]",
        "ur+ [1]",
        "ur- [1]",
        "ut+ [1]",
        "ut- [1]",
    ]
    values = [problem.states_all_section(i) for i in range(3)]
    values += [problem.controls_all_section(i) for i in range(4)]
    fig, axes = plt.subplots(7, 1, sharex=True, layout="constrained", figsize=(8, 10.5))
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
