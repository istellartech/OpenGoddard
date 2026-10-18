"""12. 極座標・単段軌道投入。

状態は半径、角度、半径方向速度、接線方向速度、質量。推力は2成分。
"""

import numpy as np

from OpenGoddard import Condition, Dynamics, Guess, Problem


class Rocket:
    GMe = 3.986004418 * 10**14
    Re = 6371.0 * 1000
    g0 = 9.80665

    def __init__(self):
        self.M0 = 100000.0  # kg
        self.Mp = self.M0 * 0.99
        self.Cd = 0.6
        self.A = 4.0  # m2
        self.Isp = 300.0  # s
        self.g0 = 9.80665  # m/s2
        self.Tmax = self.M0 * self.g0 * 1.5
        self.MaxG = 8.0  # G
        self.Htarget = 400.0 * 1000  # m
        self.Rtarget = self.Re + self.Htarget  # m
        self.Vtarget = np.sqrt(self.GMe / self.Rtarget)  # m/s

    def air_density(self, h):
        beta = 1 / 8500.0
        rho0 = 1.225
        return rho0 * np.exp(-beta * h)


# 状態は半径・角度・半径方向速度・接線方向速度・質量。推力は2成分。
def dynamics(problem, obj, section):
    R = problem.states(0, section)
    Vr = problem.states(2, section)
    Vt = problem.states(3, section)
    m = problem.states(4, section)
    Tr = problem.controls(0, section)
    Tt = problem.controls(1, section)
    rho = obj.air_density(R - obj.Re)
    Dr = 0.5 * rho * Vr * np.sqrt(Vr**2 + Vt**2) * obj.Cd * obj.A
    Dt = 0.5 * rho * Vt * np.sqrt(Vr**2 + Vt**2) * obj.Cd * obj.A
    g = obj.GMe / R**2
    g0 = obj.g0
    Isp = obj.Isp
    dx = Dynamics(problem, section)
    dx[0] = Vr
    dx[1] = Vt / R
    dx[2] = Tr / m - Dr / m - g + Vt**2 / R
    dx[3] = Tt / m - Dt / m - Vr * Vt / R
    dx[4] = -np.sqrt(Tr**2 + Tt**2) / g0 / Isp
    return dx()


def equality(problem, obj):
    R = problem.states_all_section(0)
    theta = problem.states_all_section(1)
    Vr = problem.states_all_section(2)
    Vt = problem.states_all_section(3)
    m = problem.states_all_section(4)
    result = Condition()
    result.equal(R[0], obj.Re, unit=problem.unit_states[0][0])
    result.equal(theta[0], 0.0, unit=problem.unit_states[0][1])
    result.equal(Vr[0], 0.0, unit=problem.unit_states[0][2])
    result.equal(Vt[0], 0.0, unit=problem.unit_states[0][3])
    result.equal(m[0], obj.M0, unit=problem.unit_states[0][4])
    result.equal(R[-1], obj.Rtarget, unit=problem.unit_states[0][0])
    result.equal(Vr[-1], 0.0, unit=problem.unit_states[0][2])
    result.equal(Vt[-1], obj.Vtarget, unit=problem.unit_states[0][3])
    return result()


def inequality(problem, obj):
    R = problem.states_all_section(0)
    Vr = problem.states_all_section(2)
    Vt = problem.states_all_section(3)
    m = problem.states_all_section(4)
    Tr = problem.controls_all_section(0)
    Tt = problem.controls_all_section(1)
    rho = obj.air_density(R - obj.Re)
    Dr = 0.5 * rho * Vr * np.sqrt(Vr**2 + Vt**2) * obj.Cd * obj.A
    Dt = 0.5 * rho * Vt * np.sqrt(Vr**2 + Vt**2) * obj.Cd * obj.A
    a_r = (Tr - Dr) / m
    a_t = (Tt - Dt) / m
    a_mag = np.sqrt(a_r**2 + a_t**2)
    T = np.sqrt(Tr**2 + Tt**2)
    result = Condition()
    result.lower_bound(m[1:], obj.M0 - obj.Mp, unit=problem.unit_states[0][4])
    result.lower_bound(Tt, 0.0, unit=problem.unit_controls[0][0])
    result.upper_bound(m, obj.M0, unit=problem.unit_states[0][4])
    result.upper_bound(T, obj.Tmax, unit=problem.unit_controls[0][0])
    result.upper_bound(a_mag, obj.MaxG * obj.g0, unit=obj.MaxG * obj.g0)
    return result()


def cost(problem, obj):
    m = problem.states_all_section(4)
    return -m[-1] / problem.unit_states[0][4]


def make_problem(parameters):
    problem = Problem([0.0, 200], [30], [5], [2], maxIterator=20)
    obj = parameters
    unit_R = obj.Re
    unit_theta = 1
    unit_V = np.sqrt(obj.GMe / obj.Re)
    unit_m = obj.M0
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
    problem.set_states_all_section(
        4, Guess.cubic(problem.time_all_section, obj.M0, -0.6, obj.M0 - obj.Mp, 0.0)
    )
    problem.set_controls_all_section(
        0, Guess.cubic(problem.time_all_section, obj.Tmax / 2, 0.0, 0.0, 0.0)
    )
    problem.set_controls_all_section(
        1, Guess.cubic(problem.time_all_section, obj.Tmax / 2, 0.0, 0.0, 0.0)
    )
    problem.set_states_bounds_all_section(0, obj.Re, None)
    problem.set_controls_bounds_all_section(0, 0.0, obj.Tmax)
    problem.set_controls_bounds_all_section(1, 0.0, obj.Tmax)
    problem.dynamics = [dynamics]
    problem.knot_states_smooth = []
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
