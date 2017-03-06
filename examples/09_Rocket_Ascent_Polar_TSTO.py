# -*- coding: utf-8 -*-
# Copyright 2017 Interstellar Technologies Inc. All Rights Reserved.

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from OpenGoddard.optimize import Problem, Guess, Condition, Dynamics


class Rocket:
    GMe = 3.986004418 * 10**14  # Earth gravitational constant [m^3/s^2]
    Re = 6371.0 * 1000  # Earth Radius [m]
    g0 = 9.80665  # Gravitational acceleration on Earth surface [m/s^2]

    def __init__(self):
        self.Vr = np.sqrt(self.GMe / self.Re)  # m/s
        self.H0 = 10.0  # m
        self.V0 = 0.0
        self.M0 = [20000.0, 1000.0]  # kg
        self.Mdry = [2000, 200]
        self.Mp = [self.M0[0]-self.Mdry[0], self.M0[1]-self.M0[1]]
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
        beta = 1/8500.0  # scale factor [1/m]
        rho0 = 1.225  # kg/m3
        return rho0*np.exp(-beta*h)


def dynamics(prob, obj, section):
    R     = prob.states(0, section)
    theta = prob.states(1, section)
    Vr    = prob.states(2, section)
    Vt    = prob.states(3, section)
    m     = prob.states(4, section)
    Tr    = prob.controls(0, section)
    Tt    = prob.controls(1, section)

    rho = obj.air_density(R - obj.Re)
    Dr = 0.5 * rho * Vr * np.sqrt(Vr**2 + Vt**2) \
        * obj.Cd[section] * obj.A[section]  # [N]
    Dt = 0.5 * rho * Vt * np.sqrt(Vr**2 + Vt**2) \
        * obj.Cd[section] * obj.A[section]  # [N]
    g = obj.g0 * (obj.Re / R)**2  # [m/s2]
    g0 = obj.g0
    Isp = obj.Isp[section]

    dx = Dynamics(prob, section)
    dx[0] = Vr
    dx[1] = Vt / R
    dx[2] = Tr / m - Dr / m - g + Vt**2 / R
    dx[3] = Tt / m - Dt / m - (Vr * Vt) / R
    dx[4] = - np.sqrt(Tr**2 + Tt**2) / g0 / Isp

    return dx()


def equality(prob, obj):
    R     = prob.states_all_section(0)
    theta = prob.states_all_section(1)
    Vr    = prob.states_all_section(2)
    Vt    = prob.states_all_section(3)
    m     = prob.states_all_section(4)
    Tr    = prob.controls_all_section(0)
    Tt    = prob.controls_all_section(1)
    tf    = prob.time_final(-1)

    R0     = prob.states(0, 0)
    R1     = prob.states(0, 1)
    theta0 = prob.states(1, 0)
    theta1 = prob.states(1, 1)
    Vr0    = prob.states(2, 0)
    Vr1    = prob.states(2, 1)
    Vt0    = prob.states(3, 0)
    Vt1    = prob.states(3, 1)
    m0     = prob.states(4, 0)
    m1     = prob.states(4, 1)
    Tr0    = prob.controls(0, 0)
    Tr1    = prob.controls(0, 1)
    Tt0    = prob.controls(1, 0)
    Tt1    = prob.controls(1, 1)

    unit_R = prob.unit_states[0][0]
    unit_V = prob.unit_states[0][2]
    unit_m = prob.unit_states[0][4]

    result = Condition()

    # event condition
    result.add(R0[0] - obj.Re, unit=unit_R)
    result.add(theta0[0] - 0.0)
    result.add(Vr0[0] - 0.0, unit=unit_V)
    result.add(Vt0[0] - 0.0, unit=unit_V)
    result.add(m0[0] - obj.M0[0], unit=unit_m)
    result.add(m1[0] - obj.M0[1], unit=unit_m)
    # result.add(m1[0] - 1, unit=unit_m)
    result.add(R1[-1] - obj.Rtarget, unit=unit_R)
    result.add(Vr[-1] - 0.0, unit=unit_V)
    result.add(Vt[-1] - obj.Vtarget, unit=unit_V)

    # knotting condition
    result.add(R1[0] - R0[-1], unit=unit_R)
    result.add(theta1[0] - theta0[-1])
    result.add(Vr1[0] - Vr0[-1], unit=unit_V)
    result.add(Vt1[0] - Vt0[-1], unit=unit_V)
    # result.add(m1[0] - m0[-1] - (obj.Mdry[0] - obj.M0[1]), unit = unit_m)

    return result()


def inequality(prob, obj):
    R     = prob.states_all_section(0)
    theta = prob.states_all_section(1)
    Vr    = prob.states_all_section(2)
    Vt    = prob.states_all_section(3)
    m     = prob.states_all_section(4)
    Tr    = prob.controls_all_section(0)
    Tt    = prob.controls_all_section(1)
    tf    = prob.time_final(-1)

    R0     = prob.states(0, 0)
    R1     = prob.states(0, 1)
    theta0 = prob.states(1, 0)
    theta1 = prob.states(1, 1)
    Vr0    = prob.states(2, 0)
    Vr1    = prob.states(2, 1)
    Vt0    = prob.states(3, 0)
    Vt1    = prob.states(3, 1)
    m0     = prob.states(4, 0)
    m1     = prob.states(4, 1)
    Tr0    = prob.controls(0, 0)
    Tr1    = prob.controls(0, 1)
    Tt0    = prob.controls(1, 0)
    Tt1    = prob.controls(1, 1)

    rho = obj.air_density(R - obj.Re)
    Dr0 = 0.5 * rho * Vr * np.sqrt(Vr**2 + Vt**2) \
        * obj.Cd[0] * obj.A[0]  # [N]
    Dt0 = 0.5 * rho * Vt * np.sqrt(Vr**2 + Vt**2) \
        * obj.Cd[0] * obj.A[0]  # [N]
    Dr1 = 0.5 * rho * Vr * np.sqrt(Vr**2 + Vt**2) \
        * obj.Cd[1] * obj.A[1]  # [N]
    Dt1 = 0.5 * rho * Vt * np.sqrt(Vr**2 + Vt**2) \
        * obj.Cd[1] * obj.A[1]  # [N]
    g = obj.g0 * (obj.Re / R)**2  # [m/s2]

    # dynamic pressure
    q = 0.5 * rho * (Vr**2 + Vt**2)  # [Pa]
    # accelaration
    a_r0 = (Tr - Dr0) / m
    a_t0 = (Tt - Dt0) / m
    a_mag0 = np.sqrt(a_r0**2 + a_t0**2)  # [m/s2]
    a_r1 = (Tr - Dr1) / m
    a_t1 = (Tt - Dt1) / m
    a_mag1 = np.sqrt(a_r1**2 + a_t1**2)  # [m/s2]
    # Thrust
    T0 = np.sqrt(Tr0**2 + Tt0**2)
    T1 = np.sqrt(Tr1**2 + Tt1**2)

    result = Condition()

    # lower bounds
    result.add(R - obj.Re, unit=prob.unit_states[0][0])
    # result.add(Vr - 0.0, unit=prob.unit_states[0][2])
    # result.add(Vt - 0.0, unit=prob.unit_states[0][3])
    result.add(m0 - obj.Mdry[0], unit=prob.unit_states[0][4])
    # result.add(m1 - obj.Mdry[1], unit=prob.unit_states[0][4])
    result.add(m1 - 1, unit=prob.unit_states[0][4])
    result.add(Tr + obj.Tmax[1], unit=prob.unit_controls[0][0])
    # result.add(Tt + obj.Tmax / obj.unit_T, unit=prob.unit_controls[0][0])
    result.add(Tt + obj.Tmax[1], unit=prob.unit_controls[0][0])

    # upper bounds
    result.add(obj.M0[0] - m0, unit=prob.unit_states[0][4])
    result.add(obj.M0[1] - m1, unit=prob.unit_states[0][4])
    result.add(obj.Tmax[0] - Tr0, unit=prob.unit_controls[0][0])
    result.add(obj.Tmax[0] - Tt0, unit=prob.unit_controls[0][0])
    result.add(obj.Tmax[0] - T0, unit=prob.unit_controls[0][0])
    result.add(obj.Tmax[1] - Tr1, unit=prob.unit_controls[0][0])
    result.add(obj.Tmax[1] - Tt1, unit=prob.unit_controls[0][0])
    result.add(obj.Tmax[1] - T1, unit=prob.unit_controls[0][0])
    # result.add(obj.MaxQ - q, unit = prob.unit_states[0][0])
    result.add(obj.MaxG * obj.g0 - a_mag0)
    result.add(obj.MaxG * obj.g0 - a_mag1)

    return result()


def cost(prob, obj):
    m1 = prob.states(4, 1)
    return -m1[-1] / prob.unit_states[1][4]


# ========================
plt.close("all")
plt.ion()
# Program Starting Point
time_init = [0.0, 100, 200]
n = [20, 20]
num_states = [5, 5]
num_controls = [2, 2]
max_iteration = 20

flag_savefig = True
savefig_file = "08_Rocket_Ascent_Polar/TSTO_"

# ------------------------
# set OpenGoddard class for algorithm determination
prob = Problem(time_init, n, num_states, num_controls, max_iteration)

# ------------------------
# create instance of operating object
obj = Rocket()

unit_R = obj.Re
unit_theta = 1
unit_V = np.sqrt(obj.GMe / obj.Re)
unit_m = obj.M0[0]
unit_t = unit_R / unit_V
unit_T = unit_m * unit_R / unit_t ** 2
prob.set_unit_states_all_section(0, unit_R)
prob.set_unit_states_all_section(1, unit_theta)
prob.set_unit_states_all_section(2, unit_V)
prob.set_unit_states_all_section(3, unit_V)
prob.set_unit_states_all_section(4, unit_m)
prob.set_unit_controls_all_section(0, unit_T)
prob.set_unit_controls_all_section(1, unit_T)
prob.set_unit_time(unit_t)

# ========================
# Initial parameter guess

# altitude profile
R_init = Guess.cubic(prob.time_all_section, obj.Re, 0.0, obj.Rtarget, 0.0)
# Guess.plot(prob.time_all_section, R_init, "Altitude", "time", "Altitude")
# if(flag_savefig):plt.savefig(savefig_file + "guess_alt" + ".png")
# theta
theta_init = Guess.cubic(prob.time_all_section, 0.0, 0.0, np.deg2rad(25.0), 0.0)

# velocity
Vr_init = Guess.linear(prob.time_all_section, 0.0, 0.0)
Vt_init = Guess.linear(prob.time_all_section, 0.0, obj.Vtarget)
# Guess.plot(prob.time_all_section, V_init, "Velocity", "time", "Velocity")

# mass profile
M_init0 = Guess.cubic(prob.time_all_section, obj.M0[0], -0.6, obj.Mdry[0], 0.0)
M_init1 = Guess.cubic(prob.time_all_section, obj.M0[1], -0.6, obj.Mdry[1], 0.0)
M_init = np.hstack((M_init0, M_init1))
# Guess.plot(prob.time_all_section, M_init, "Mass", "time", "Mass")
# if(flag_savefig):plt.savefig(savefig_file + "guess_mass" + ".png")

# thrust profile
# T_init = Guess.zeros(prob.time_all_section)
Tr_init0 = Guess.cubic(prob.time[0], obj.Tmax[0]*9/10, 0.0, 0.0, 0.0)
Tr_init1 = Guess.cubic(prob.time[1], obj.Tmax[1]*9/10, 0.0, 0.0, 0.0)
Tr_init = np.hstack((Tr_init0, Tr_init1))
# Tt_init = Guess.cubic(prob.time_all_section, 0.0, 0.0, 0.0, 0.0)
Tt_init0 = Guess.cubic(prob.time[0], obj.Tmax[0]/10, 0.0, 0.0, 0.0)
Tt_init1 = Guess.cubic(prob.time[1], obj.Tmax[1]/10, 0.0, 0.0, 0.0)
Tt_init = np.hstack((Tr_init0, Tr_init1))
# Guess.plot(prob.time_all_section, T_init, "Thrust Guess", "time", "Thrust")
# if(flag_savefig):plt.savefig(savefig_file + "guess_thrust" + ".png")

# plt.show()

# ========================
# Substitution initial value to parameter vector to be optimized
# non dimensional values (Divide by scale factor)
prob.set_states_all_section(0, R_init)
prob.set_states_all_section(1, theta_init)
prob.set_states_all_section(2, Vr_init)
prob.set_states_all_section(3, Vt_init)
prob.set_states_all_section(4, M_init)
prob.set_controls_all_section(0, Tr_init)
prob.set_controls_all_section(1, Tt_init)

# ========================
# Main Process
# Assign problem to SQP solver
prob.dynamics = [dynamics, dynamics]
prob.knot_states_smooth = [False]
prob.cost = cost
# prob.cost_derivative = cost_derivative
prob.equality = equality
prob.inequality = inequality

plt.figure(0)
plt.title("Flight trajectory")
plt.xlabel("Downrange [km]")
plt.ylabel("Altitude [km]")


def display_func():
    R = prob.states_all_section(0)
    theta = prob.states_all_section(1)
    m = prob.states_all_section(4)
    tf = prob.time_final(-1)
    print("m0          : {0:.5f}".format(m[0]))
    print("mf          : {0:.5f}".format(m[-1]))
    print("max altitude: {0:.5f}".format(R[-1]))
    print("final time  : {0:.3f}".format(tf))
    plt.figure(0)
    plt.plot(theta * obj.Re / 1000, (R - obj.Re) / 1000, marker="o", label="trajectory")
    plt.grid()
    plt.pause(0.2)

prob.solve(obj, display_func, ftol=1e-8)

# ========================
# Post Process
# ------------------------
# Convert parameter vector to variable
R     = prob.states_all_section(0)
theta = prob.states_all_section(1)
Vr    = prob.states_all_section(2)
Vt    = prob.states_all_section(3)
m     = prob.states_all_section(4)
Tr    = prob.controls_all_section(0)
Tt    = prob.controls_all_section(1)
time = prob.time_update()

# ------------------------
# Calculate necessary variables
rho = obj.air_density(R - obj.Re)
Dr = 0.5 * rho * Vr * np.sqrt(Vr**2 + Vt**2) \
    * obj.Cd[0] * obj.A[0]  # [N]
Dt = 0.5 * rho * Vt * np.sqrt(Vr**2 + Vt**2) \
    * obj.Cd[0] * obj.A[0]  # [N]
g = obj.g0 * (obj.Re / R)**2  # [m/s2]

# dynamic pressure
q = 0.5 * rho * (Vr**2 + Vt**2)  # [Pa]
# accelaration
a_r = (Tr - Dr) / m
a_t = (Tt - Dt) / m
a_mag = np.sqrt(a_r**2 + a_t**2)  # [m/s2]
# Thrust
T = np.sqrt(Tr**2 + Tt**2)

# ------------------------
# Visualizetion
plt.figure()
plt.title("Altitude profile")
plt.plot(time, (R - obj.Re)/1000, marker="o", label="Altitude")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Altitude [km]")
if(flag_savefig): plt.savefig(savefig_file + "altitude" + ".png")

plt.figure()
plt.title("Velocity")
plt.plot(time, Vr, marker="o", label="Vr")
plt.plot(time, Vt, marker="o", label="Vt")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Velocity [m/s]")
plt.legend(loc="best")
if(flag_savefig): plt.savefig(savefig_file + "velocity" + ".png")

plt.figure()
plt.title("Mass")
plt.plot(time, m, marker="o", label="Mass")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Mass [kg]")
if(flag_savefig): plt.savefig(savefig_file + "mass" + ".png")

plt.figure()
plt.title("Acceleration")
plt.plot(time, a_r, marker="o", label="Acc r")
plt.plot(time, a_t, marker="o", label="Acc t")
plt.plot(time, a_mag, marker="o", label="Acc")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Acceleration [m/s2]")
if(flag_savefig): plt.savefig(savefig_file + "acceleration" + ".png")

plt.figure()
plt.title("Thrust profile")
plt.plot(time, Tr / 1000, marker="o", label="Tr")
plt.plot(time, Tt / 1000, marker="o", label="Tt")
plt.plot(time, T / 1000, marker="o", label="Thrust")
plt.plot(time, Dr / 1000, marker="o", label="Dr")
plt.plot(time, Dt / 1000, marker="o", label="Dt")
plt.plot(time, m * g / 1000, marker="o", label="Gravity")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Thrust [kN]")
plt.legend(loc="best")
if(flag_savefig): plt.savefig(savefig_file + "force" + ".png")

plt.figure()
plt.title("Flight trajectory")
plt.plot(theta * obj.Re / 1000, (R - obj.Re) / 1000, marker="o", label="trajectory")
plt.grid()
plt.xlabel("Downrange [km]")
plt.ylabel("Altitude [km]")
plt.legend(loc="best")
if(flag_savefig): plt.savefig(savefig_file + "trajectory" + ".png")


plt.show()
