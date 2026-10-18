"""13. 極座標・2段軌道投入。

12の極座標モデルに段分離と区間ごとの機体パラメータを追加する。
"""

import numpy as np

from OpenGoddard import Condition, Dynamics, Guess, Problem


class Rocket:
    GMe = 3.986004418 * 10**14
    Re = 6371.0 * 1000
    g0 = 9.80665

    def __init__(self):
        self.Vr = np.sqrt(self.GMe / self.Re)  # m/s
        self.H0 = 10.0  # m
        self.V0 = 0.0
        self.M0 = [20000.0, 1000.0]  # kg
        self.Mdry = [2000, 200]
        self.Mp = [self.M0[0] - self.Mdry[0], self.M0[1] - self.Mdry[1]]
        self.Minit = self.M0[0] + self.M0[1]
        self.Cd = [0.2, 0.2]
        self.A = [3.14, 3.14]  # m2
        self.Isp = [300.0, 350.0]  # s
        self.Tmax = [self.M0[0] * self.g0 * 1.5, self.M0[1] * self.g0 * 1.5]
        self.MaxQ = 14000.0  # Pa
        self.MaxG = 8.0  # G
        self.Htarget = 500.0 * 1000  # m
        self.Rtarget = self.Re + self.Htarget  # m/s
        self.Vtarget = np.sqrt(self.GMe / self.Rtarget)  # m/s

    def air_density(self, h):
        beta = 1 / 8500.0
        rho0 = 1.225
        return rho0 * np.exp(-beta * np.maximum(h, -100.0))


# 状態は半径・角度・半径方向速度・接線方向速度・質量。section で段を指定する。
def dynamics(problem, obj, section):
    R = problem.states(0, section)
    Vr = problem.states(2, section)
    Vt = problem.states(3, section)
    m = problem.states(4, section)
    Tr = problem.controls(0, section)
    Tt = problem.controls(1, section)
    rho = obj.air_density(R - obj.Re)
    Dr = 0.5 * rho * Vr * np.sqrt(Vr**2 + Vt**2) * obj.Cd[section] * obj.A[section]
    Dt = 0.5 * rho * Vt * np.sqrt(Vr**2 + Vt**2) * obj.Cd[section] * obj.A[section]
    g = obj.g0 * (obj.Re / R) ** 2
    g0 = obj.g0
    Isp = obj.Isp[section]
    dx = Dynamics(problem, section)
    dx[0] = Vr
    dx[1] = Vt / R
    dx[2] = Tr / m - Dr / m - g + Vt**2 / R
    dx[3] = Tt / m - Dt / m - Vr * Vt / R
    dx[4] = -np.sqrt(Tr**2 + Tt**2) / g0 / Isp
    return dx()


def equality(problem, obj):
    Vr = problem.states_all_section(2)
    Vt = problem.states_all_section(3)
    R0 = problem.states(0, 0)
    R1 = problem.states(0, 1)
    theta0 = problem.states(1, 0)
    theta1 = problem.states(1, 1)
    Vr0 = problem.states(2, 0)
    Vr1 = problem.states(2, 1)
    Vt0 = problem.states(3, 0)
    Vt1 = problem.states(3, 1)
    m0 = problem.states(4, 0)
    m1 = problem.states(4, 1)
    unit_R = problem.unit_states[0][0]
    unit_V = problem.unit_states[0][2]
    unit_m = problem.unit_states[0][4]
    result = Condition()
    result.equal(R0[0], obj.Re, unit=unit_R)
    result.equal(theta0[0], 0.0)
    result.equal(Vr0[0], 0.0, unit=unit_V)
    result.equal(Vt0[0], 0.0, unit=unit_V)
    result.equal(m0[0], obj.M0[0], unit=unit_m)
    result.equal(m1[0], obj.M0[1], unit=unit_m)
    result.equal(R1[-1], obj.Rtarget, unit=unit_R)
    result.equal(Vr[-1], 0.0, unit=unit_V)
    result.equal(Vt[-1], obj.Vtarget, unit=unit_V)
    result.equal(R1[0], R0[-1], unit=unit_R)
    result.equal(theta1[0], theta0[-1])
    result.equal(Vr1[0], Vr0[-1], unit=unit_V)
    result.equal(Vt1[0], Vt0[-1], unit=unit_V)
    return result()


def inequality(problem, obj):
    R = problem.states_all_section(0)
    Vr = problem.states_all_section(2)
    Vt = problem.states_all_section(3)
    m = problem.states_all_section(4)
    Tr = problem.controls_all_section(0)
    Tt = problem.controls_all_section(1)
    Tr0 = problem.controls(0, 0)
    Tr1 = problem.controls(0, 1)
    Tt0 = problem.controls(1, 0)
    Tt1 = problem.controls(1, 1)
    rho = obj.air_density(R - obj.Re)
    Dr0 = 0.5 * rho * Vr * np.sqrt(Vr**2 + Vt**2) * obj.Cd[0] * obj.A[0]
    Dt0 = 0.5 * rho * Vt * np.sqrt(Vr**2 + Vt**2) * obj.Cd[0] * obj.A[0]
    Dr1 = 0.5 * rho * Vr * np.sqrt(Vr**2 + Vt**2) * obj.Cd[1] * obj.A[1]
    Dt1 = 0.5 * rho * Vt * np.sqrt(Vr**2 + Vt**2) * obj.Cd[1] * obj.A[1]
    a_r0 = (Tr - Dr0) / m
    a_t0 = (Tt - Dt0) / m
    a_mag0 = np.sqrt(a_r0**2 + a_t0**2)
    a_r1 = (Tr - Dr1) / m
    a_t1 = (Tt - Dt1) / m
    a_mag1 = np.sqrt(a_r1**2 + a_t1**2)
    T0 = np.sqrt(Tr0**2 + Tt0**2)
    T1 = np.sqrt(Tr1**2 + Tt1**2)
    result = Condition()
    result.lower_bound(R, obj.Re, unit=problem.unit_states[0][0])
    result.upper_bound(T0, obj.Tmax[0], unit=problem.unit_controls[0][0])
    result.upper_bound(T1, obj.Tmax[1], unit=problem.unit_controls[0][0])
    result.upper_bound(a_mag0, obj.MaxG * obj.g0)
    result.upper_bound(a_mag1, obj.MaxG * obj.g0)
    return result()


def cost(problem, obj):
    m1 = problem.states(4, 1)
    return -m1[-1] / problem.unit_states[1][4]


def make_problem(parameters):
    problem = Problem([0.0, 100, 200], [20, 20], [5, 5], [2, 2], maxIterator=40)
    obj = parameters
    unit_R = obj.Re
    unit_theta = 1
    unit_V = np.sqrt(obj.GMe / obj.Re)
    unit_m = obj.M0[0]
    unit_t = unit_R / unit_V
    unit_T = unit_m * unit_R / unit_t**2
    problem.set_unit_states_all_section(0, unit_R)
    problem.set_unit_states_all_section(1, unit_theta)
    problem.set_unit_states_all_section(2, unit_V)
    problem.set_unit_states_all_section(3, unit_V)
    problem.set_unit_states_all_section(4, unit_m)
    problem.set_unit_controls_all_section(0, unit_T)
    problem.set_unit_controls_all_section(1, unit_T)
    problem.set_unit_time(unit_t)
    M_init0 = Guess.cubic(problem.time[0], obj.M0[0], -0.6, obj.Mdry[0], 0.0)
    M_init1 = Guess.cubic(problem.time[1], obj.M0[1], -0.6, obj.Mdry[1], 0.0)
    Tr_init0 = Guess.cubic(problem.time[0], obj.Tmax[0] * 9 / 10, 0.0, 0.0, 0.0)
    Tr_init1 = Guess.cubic(problem.time[1], obj.Tmax[1] * 9 / 10, 0.0, 0.0, 0.0)
    Tt_init0 = Guess.cubic(problem.time[0], obj.Tmax[0] / 10, 0.0, 0.0, 0.0)
    Tt_init1 = Guess.cubic(problem.time[1], obj.Tmax[1] / 10, 0.0, 0.0, 0.0)
    problem.set_states_all_section(
        0, Guess.cubic(problem.time_all_section, obj.Re, 0.0, obj.Rtarget, 0.0)
    )
    problem.set_states_all_section(
        1, Guess.cubic(problem.time_all_section, 0.0, 0.0, np.deg2rad(25.0), 0.0)
    )
    problem.set_states_all_section(2, Guess.linear(problem.time_all_section, 0.0, 0.0))
    problem.set_states_all_section(
        3, Guess.linear(problem.time_all_section, 0.0, obj.Vtarget)
    )
    problem.set_states_all_section(4, np.hstack((M_init0, M_init1)))
    problem.set_controls_all_section(0, np.hstack((Tr_init0, Tr_init1)))
    problem.set_controls_all_section(1, np.hstack((Tt_init0, Tt_init1)))
    problem.set_states_bounds_all_section(0, obj.Re, None)
    problem.set_states_bounds(4, 0, obj.Mdry[0], obj.M0[0])
    problem.set_states_bounds(4, 1, 1.0, obj.M0[1])
    problem.set_controls_bounds(0, 0, -obj.Tmax[1], obj.Tmax[0])
    problem.set_controls_bounds(1, 0, -obj.Tmax[1], obj.Tmax[0])
    problem.set_controls_bounds(0, 1, -obj.Tmax[1], obj.Tmax[1])
    problem.set_controls_bounds(1, 1, -obj.Tmax[1], obj.Tmax[1])
    problem.dynamics = [dynamics, dynamics]
    problem.knot_states_smooth = [False]
    problem.cost = cost
    problem.equality = equality
    problem.inequality = inequality
    return problem


def solve():
    parameters = Rocket()
    problem = make_problem(parameters)
    problem.solve(parameters, ftol=1e-06, maxiter=3000)
    return problem


def plot(problem):
    import matplotlib.pyplot as plt

    time = problem.time_all_section
    labels = [
        "radius [m]",
        "theta [rad]",
        "vr [m/s]",
        "vt [m/s]",
        "mass [kg]",
        "Tr [N]",
        "Tt [N]",
    ]
    values = [problem.states_all_section(i) for i in range(5)]
    values += [problem.controls_all_section(i) for i in range(2)]
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
