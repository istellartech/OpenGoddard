"""14. 大樹からの2段軌道投入。

13に表形式の大気・抗力係数、地球自転、圧力による比推力変化を追加する。
"""

from pathlib import Path

import numpy as np
from scipy import interpolate

from OpenGoddard import Condition, Dynamics, Guess, Problem


class Rocket:
    stdAtmo = np.loadtxt(
        str(Path(__file__).resolve().parent / "data/US_standard_atmosphere.csv"),
        delimiter=",",
        skiprows=2,
    )
    stdAltitude = stdAtmo[:, 0] * 1000.0
    stdPressure = stdAtmo[:, 2]
    stdDensity = stdAtmo[:, 3]
    stdSoundSpeed = stdAtmo[:, 4]
    airPressure = interpolate.interp1d(
        stdAltitude, stdPressure, bounds_error=False, fill_value=(stdPressure[0], 0.0)
    )
    airDensity = interpolate.interp1d(
        stdAltitude, stdDensity, bounds_error=False, fill_value=(stdDensity[0], 0.0)
    )
    airSound = interpolate.interp1d(
        stdAltitude,
        stdSoundSpeed,
        bounds_error=False,
        fill_value=(stdSoundSpeed[0], stdSoundSpeed[-1]),
    )
    CdLog = np.loadtxt(
        str(Path(__file__).resolve().parent / "data/Cd.csv"), delimiter=",", skiprows=1
    )
    Cd = interpolate.interp1d(CdLog[:, 0], CdLog[:, 1], fill_value="extrapolate")

    def __init__(self):
        self.GMe = 3.986004418 * 10**14  # Earth gravitational constant [m^3/s^2]
        self.Re = 6371.0 * 1000  # Earth Radius [m]
        self.g0 = 9.80665  # Gravitational acceleration on Earth surface [m/s^2]
        self.Htarget = 561.0 * 1000  # Altitude [m]
        self.Rtarget = self.Re + self.Htarget  # Orbit Radius [m]
        self.Vtarget = np.sqrt(self.GMe / self.Rtarget)  # [m/s]
        self.omega = 7.2921159e-5  # Earth rotation rate [rad/s]
        self.lat_taiki = 42.506167  # Launch site latitude [deg]
        self.inclination = 96.7  # Orbit inclination [deg]
        # The launch site moves east at omega * Re * cos(lat). The orbit plane
        # crosses the site at azimuth Az with sin(Az) = cos(inc) / cos(lat), so
        # the in-plane part is omega * Re * cos(lat) * sin(Az) = omega * Re * cos(inc).
        # For inc = 96.7 deg this is about -54 m/s (a retrograde orbit loses speed).
        # The cross-plane part is ignored in this planar model.
        self.V0 = self.omega * self.Re * np.cos(np.deg2rad(self.inclination))  # [m/s]
        self.H0 = 10.0  # Initial Altitude [m]
        self.Mdry = [1300.0, 220.0]  # Dry Mass [kg], [1st stage, 2nd stage]
        self.beta = [10.0, 15.0]  # Structure Efficienty [%], [1st stage, 2nd stage]
        self.Mpayload = 100.0  # Payload Mass [kg]
        self.M0 = [
            self.Mdry[0] / self.beta[0] * 100.0,
            self.Mdry[1] / self.beta[1] * 100.0,
        ]  # Initial Stage Mass [kg], [1st stage, 2nd stage]
        self.Mp = [
            self.M0[0] - self.Mdry[0],
            self.M0[1] - self.Mdry[1],
        ]  # Propellant Mass [kg], [1st stage, 2nd stage]
        self.M0[1] = self.M0[1] + self.Mpayload
        self.Minit = self.M0[0] + self.M0[1]  # Initial Total Mass [kg]
        self.d = [1.8, 1.8]  # Diameter [m], [1st stage, 2nd stage]
        self.A = [
            0.25 * self.d[0] ** 2 * np.pi,
            0.25 * self.d[1] ** 2 * np.pi,
        ]  # Projected Area [m^2], [1st stage, 2nd stage]
        self.Cluster = 9
        self.Isp = [
            261.0 + 0.0,
            322.0 + 0.0,
        ]  # Specific Impulse [s], [1st stage at SL, 2nd stage at vac]
        self.dth = [53.9, 53.9]  # Throat Diameter [mm], [1st stage, 2nd stage]
        self.Ath = [
            0.25 * (self.dth[0] / 1000.0) ** 2 * np.pi,
            0.25 * (self.dth[1] / 1000.0) ** 2 * np.pi,
        ]  # Throat Area [m^2], [1st stage, 2nd stage]
        self.AR = [20.0, 140.0]  # Area Ratio, [1st stage, 2nd stage]
        self.Ae = [
            self.Ath[0] * self.AR[0] * self.Cluster,
            self.Ath[1] * self.AR[1],
        ]  # Exit Area [m^2], [1st stage, 2nd stage]
        self.ThrustMax = [33.3, 4.2]  # [N]
        self.ThrustMax = [
            self.ThrustMax[0] * self.g0 * 1000.0,
            self.ThrustMax[1] * self.g0 * 1000.0,
        ]  # [N]
        self.refMdot = [
            self.ThrustMax[0] / (self.Isp[0] * self.g0),
            self.ThrustMax[1] / (self.Isp[1] * self.g0),
        ]  # Isp補正用参照値
        self.MaxQ = 500000.0  # Pa
        self.MaxG = 20.0  # G

    def delta_thrust(self, section, h):
        """Thrust change from the rated value at altitude h [N].

        Stage 1 is rated at the launch site and gains (p(H0) - p(h)) * Ae as the
        outside pressure falls. Stage 2 is rated in vacuum and loses p(h) * Ae.
        """
        if section == 0:
            return (self.airPressure(self.H0) - self.airPressure(h)) * self.Ae[0]
        return -self.airPressure(h) * self.Ae[1]


# 状態は半径・角度・半径方向速度・接線方向速度・質量。section で段を指定する。
def dynamics(problem, obj, section):
    R = problem.states(0, section)
    Vr = problem.states(2, section)
    Vt = problem.states(3, section)
    m = problem.states(4, section)
    Tr = problem.controls(0, section)
    Tt = problem.controls(1, section)
    g0 = obj.g0
    g = obj.GMe / R**2
    rho = obj.airDensity(R - obj.Re)
    Mach = np.sqrt(Vr**2 + Vt**2) / obj.airSound(R - obj.Re)
    Cd = obj.Cd(Mach)
    dThrust = obj.delta_thrust(section, R - obj.Re)
    Isp = obj.Isp[section] + dThrust / (obj.refMdot[section] * g0)
    Dr = 0.5 * rho * Vr * np.sqrt(Vr**2 + Vt**2) * Cd * obj.A[section]
    Dt = 0.5 * rho * Vt * np.sqrt(Vr**2 + Vt**2) * Cd * obj.A[section]
    dx = Dynamics(problem, section)
    dx[0] = Vr
    dx[1] = Vt / R
    dx[2] = Tr / m - Dr / m - g + Vt**2 / R
    dx[3] = Tt / m - Dt / m - Vr * Vt / R
    dx[4] = -np.sqrt(Tr**2 + Tt**2) / (Isp * g0)
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
    result.equal(R0[0], obj.Re + obj.H0, unit=unit_R)
    result.equal(theta0[0], 0.0)
    result.equal(Vr0[0], 0.0, unit=unit_V)
    result.equal(Vt0[0], obj.V0, unit=unit_V)
    result.equal(m0[0], obj.Minit, unit=unit_m)
    result.equal(m1[0], obj.M0[1], unit=unit_m)
    result.equal(R1[0], R0[-1], unit=unit_R)
    result.equal(theta1[0], theta0[-1])
    result.equal(Vr1[0], Vr0[-1], unit=unit_V)
    result.equal(Vt1[0], Vt0[-1], unit=unit_V)
    result.equal(R1[-1], obj.Rtarget, unit=unit_R)
    result.equal(Vr[-1], 0.0, unit=unit_V)
    result.equal(Vt[-1], obj.Vtarget, unit=unit_V)
    return result()


def inequality(problem, obj):
    unit_R = problem.unit_states[0][0]
    unit_m = problem.unit_states[0][4]
    unit_T = problem.unit_controls[0][0]
    # Stage 1 carries the whole stage 2; stage 2 keeps its dry mass and payload.
    m_min = [obj.Mdry[0] + obj.M0[1], obj.Mdry[1] + obj.Mpayload]
    m_max = [obj.Minit, obj.M0[1]]
    result = Condition()
    for section in range(2):
        R = problem.states(0, section)
        Vr = problem.states(2, section)
        Vt = problem.states(3, section)
        m = problem.states(4, section)
        Tr = problem.controls(0, section)
        Tt = problem.controls(1, section)
        h = R - obj.Re
        rho = obj.airDensity(h)
        V = np.sqrt(Vr**2 + Vt**2)
        Cd = obj.Cd(V / obj.airSound(h))
        Dr = 0.5 * rho * Vr * V * Cd * obj.A[section]
        Dt = 0.5 * rho * Vt * V * Cd * obj.A[section]
        q = 0.5 * rho * V**2
        a_mag = np.sqrt(((Tr - Dr) / m) ** 2 + ((Tt - Dt) / m) ** 2)
        T = np.sqrt(Tr**2 + Tt**2)
        T_limit = obj.ThrustMax[section] + obj.delta_thrust(section, h)
        result.lower_bound(R, obj.Re, unit=unit_R)
        result.lower_bound(m, m_min[section], unit=unit_m)
        result.upper_bound(m, m_max[section], unit=unit_m)
        result.upper_bound(T, T_limit, unit=unit_T)
        result.upper_bound(q, obj.MaxQ, unit=obj.MaxQ)
        result.upper_bound(a_mag, obj.MaxG * obj.g0, unit=obj.MaxG * obj.g0)
    return result()


def cost(problem, obj):
    m1 = problem.states(4, 1)
    return -m1[-1] / problem.unit_states[1][4]


def make_problem(parameters):
    problem = Problem([0.0, 200, 800], [20, 30], [5, 5], [2, 2], maxIterator=90)
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
    M_init0 = Guess.cubic(problem.time[0], obj.Minit, 0.0, obj.Mdry[0] + obj.M0[1], 0.0)
    M_init1 = Guess.cubic(
        problem.time[1], obj.M0[1], 0.0, obj.Mdry[1] + obj.Mpayload, 0.0
    )
    Tr_init0 = Guess.cubic(problem.time[0], obj.ThrustMax[0] * 9 / 10, 0.0, 0.0, 0.0)
    Tr_init1 = Guess.cubic(problem.time[1], obj.ThrustMax[1] * 9 / 10, 0.0, 0.0, 0.0)
    Tt_init0 = Guess.cubic(problem.time[0], obj.ThrustMax[0] / 10, 0.0, 0.0, 0.0)
    Tt_init1 = Guess.cubic(problem.time[1], obj.ThrustMax[1] / 10, 0.0, 0.0, 0.0)
    problem.set_states_all_section(
        0, Guess.cubic(problem.time_all_section, obj.Re, 0.0, obj.Rtarget, 0.0)
    )
    problem.set_states_all_section(
        1, Guess.cubic(problem.time_all_section, 0.0, 0.0, np.deg2rad(25.0), 0.0)
    )
    problem.set_states_all_section(2, Guess.linear(problem.time_all_section, 0.0, 0.0))
    problem.set_states_all_section(
        3, Guess.linear(problem.time_all_section, obj.V0, obj.Vtarget)
    )
    problem.set_states_all_section(4, np.hstack((M_init0, M_init1)))
    problem.set_controls_all_section(0, np.hstack((Tr_init0, Tr_init1)))
    problem.set_controls_all_section(1, np.hstack((Tt_init0, Tt_init1)))
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
