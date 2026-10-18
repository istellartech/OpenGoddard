"""11. 2段ロケット上昇。

10に段分離を追加する。位置と速度を接続し、質量は分離時に不連続にする。
"""

import numpy as np

from OpenGoddard import Condition, Dynamics, Guess, Problem


class Rocket:
    GMe = 3.986004418 * 10**14
    Re = 6371.0 * 1000
    g0 = 9.80665

    def __init__(self):
        self.M0 = 5000  # Initial total mass [kg]
        self.M0_1st = 3000
        self.M0_2nd = 2000
        self.Mc = 0.4  # Initial Propellant mass over total mass
        self.Cd = 0.2  # Drag Coefficient [-]
        self.area = 10  # area [m2]
        self.Isp = 300.0  # Isp [s]
        self.max_thrust = 2  # maximum thrust to initial weight ratio

    def air_density(self, h):
        beta = 1 / 8500.0
        rho0 = 1.225
        return rho0 * np.exp(-beta * h)


# 状態は地心距離・鉛直速度・質量。section で段を指定する。
def dynamics(problem, obj, section):
    R = problem.states(0, section)
    v = problem.states(1, section)
    m = problem.states(2, section)
    T = problem.controls(0, section)
    rho = obj.air_density(R - obj.Re)
    drag = 0.5 * rho * v**2 * obj.Cd * obj.area
    g = obj.GMe / R**2
    g0 = obj.g0
    Isp = obj.Isp
    dx = Dynamics(problem, section)
    dx[0] = v
    dx[1] = (T - drag) / m - g
    dx[2] = -T / g0 / Isp
    return dx()


def equality(problem, obj):
    R = problem.states_all_section(0)
    v = problem.states_all_section(1)
    m = problem.states_all_section(2)
    result = Condition()
    result.equal(R[0], obj.Re)
    result.equal(v[0], 0.0)
    result.equal(m[0], obj.M0)
    result.equal(v[-1], 0.0)
    result.equal(m[-1], obj.M0_2nd * obj.Mc)
    R1 = problem.states(0, 0)
    v1 = problem.states(1, 0)
    m1 = problem.states(2, 0)
    R2 = problem.states(0, 1)
    v2 = problem.states(1, 1)
    m2 = problem.states(2, 1)
    result.equal(R1[-1], R2[0])
    result.equal(v1[-1], v2[0])
    result.equal(m1[-1], m2[0] + 1200)
    return result()


def inequality(problem, obj):
    result = Condition()
    return result()


def cost(problem, obj):
    R = problem.states_all_section(0)
    return -R[-1] / obj.Re


def make_problem(parameters):
    problem = Problem([0.0, 300, 600], [25, 25], [3, 3], [1, 1], maxIterator=30)
    obj = parameters
    unit_R = obj.Re
    unit_v = np.sqrt(obj.GMe / obj.Re)
    unit_m = obj.M0
    unit_t = unit_R / unit_v
    unit_T = unit_m * unit_R / unit_t**2
    problem.set_unit_states_all_section(0, unit_R)
    problem.set_unit_states_all_section(1, unit_v)
    problem.set_unit_states_all_section(2, unit_m)
    problem.set_unit_controls_all_section(0, unit_T)
    problem.set_unit_time(unit_t)
    problem.set_states_all_section(
        0, Guess.cubic(problem.time_all_section, obj.Re, 0.0, obj.Re + 50 * 1000, 0.0)
    )
    problem.set_states_all_section(1, Guess.linear(problem.time_all_section, 0.0, 0.0))
    problem.set_states_all_section(
        2, Guess.cubic(problem.time_all_section, obj.M0, -0.6, obj.M0 * obj.Mc, 0.0)
    )
    problem.set_controls_all_section(
        0,
        Guess.cubic(
            problem.time_all_section, obj.max_thrust * obj.M0 * obj.g0, 0.0, 0.0, 0.0
        ),
    )
    problem.set_states_bounds_all_section(0, obj.Re, None)
    problem.set_states_bounds_all_section(1, 0.0, None)
    problem.set_states_bounds_all_section(2, obj.M0_2nd * obj.Mc, obj.M0)
    problem.set_controls_bounds_all_section(0, 0.0, obj.max_thrust * obj.M0 * obj.g0)
    problem.set_time_final_bounds(0, 50.0, None)
    problem.set_time_final_bounds(1, 50.0, None)
    problem.dynamics = [dynamics, dynamics]
    problem.knot_states_smooth = [False]
    problem.cost = cost
    problem.equality = equality
    problem.inequality = inequality
    return problem


def solve():
    parameters = Rocket()
    problem = make_problem(parameters)
    problem.solve(parameters, ftol=1e-10, maxiter=2000)
    return problem


def plot(problem):
    import matplotlib.pyplot as plt

    time = problem.time_all_section
    labels = ["radius [m]", "velocity [m/s]", "mass [kg]", "thrust [N]"]
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
