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
        self.M0 = 5000  # Initial total mass [kg]
        self.Mc = 0.4  # Initial Propellant mass over total mass
        self.Cd = 0.2  # Drag Coefficient [-]
        self.area = 10  # area [m2]
        self.Isp = 300.0  # Isp [s]
        self.max_thrust = 2  # maximum thrust to initial weight ratio

    def air_density(self, h):
        beta = 1/8500.0  # scale factor [1/m]
        rho0 = 1.225  # kg/m3
        return rho0*np.exp(-beta*h)


def dynamics(prob, obj, section):
    R = prob.states(0, section)
    v = prob.states(1, section)
    m = prob.states(2, section)
    T = prob.controls(0, section)

    rho = obj.air_density(R - obj.Re)
    drag = 0.5 * rho * v ** 2 * obj.Cd * obj.area
    g = obj.GMe / R**2
    g0 = obj.g0
    Isp = obj.Isp

    dx = Dynamics(prob, section)
    dx[0] = v
    dx[1] = (T - drag) / m - g
    dx[2] = - T / g0 / Isp
    return dx()


def equality(prob, obj):
    R = prob.states_all_section(0)
    v = prob.states_all_section(1)
    m = prob.states_all_section(2)
    T = prob.controls_all_section(0)
    tf = prob.time_final(-1)

    result = Condition()

    # event condition
    result.add(R[0] - obj.Re)
    result.add(v[0] - 0.0)
    result.add(m[0] - obj.M0)
    result.add(v[-1] - 0.0)
    result.add(m[-1] - obj.M0 * obj.Mc)

    return result()


def inequality(prob, obj):
    R = prob.states_all_section(0)
    v = prob.states_all_section(1)
    m = prob.states_all_section(2)
    T = prob.controls_all_section(0)
    tf = prob.time_final(-1)

    result = Condition()
    # lower bounds
    result.add(R - obj.Re)
    result.add(v - 0.0)
    result.add(m - obj.M0 * obj.Mc)
    result.add(T - 0.0)
    result.add(tf - 10)
    # upper bounds
    result.add(obj.M0 - m)
    result.add(obj.max_thrust * obj.M0 * obj.g0 - T)

    return result()


def cost(prob, obj):
    R = prob.states_all_section(0)
    return -R[-1] / obj.Re


def cost_derivative(prob, obj):
    jac = Condition(prob.number_of_variables)
    index_R_end = prob.index_states(0, 0, -1)
    jac.change_value(index_R_end, -1)
    return jac()

# ========================
plt.close("all")
plt.ion()
# Program Starting Point
time_init = [0.0, 600]
n = [50]
num_states = [3]
num_controls = [1]
max_iteration = 20

flag_savefig = True
savefig_file = "06_Rocket_Ascent/Single_"

# ------------------------
# set OpenGoddard class for algorithm determination
prob = Problem(time_init, n, num_states, num_controls, max_iteration)

# ------------------------
# create instance of operating object
obj = Rocket()

# ------------------------
# set designer unit
unit_R = obj.Re
unit_v = np.sqrt(obj.GMe / obj.Re)
unit_m = obj.M0
unit_t = unit_R / unit_v
unit_T = unit_m * unit_R / unit_t**2
prob.set_unit_states_all_section(0, unit_R)
prob.set_unit_states_all_section(1, unit_v)
prob.set_unit_states_all_section(2, unit_m)
prob.set_unit_controls_all_section(0, unit_T)
prob.set_unit_time(unit_t)

# ========================
# Initial parameter guess

# altitude profile
R_init = Guess.cubic(prob.time_all_section, obj.Re, 0.0, obj.Re+50*1000, 0.0)
# Guess.plot(prob.time_all_section, R_init, "Altitude", "time", "Altitude")
# if(flag_savefig):plt.savefig(savefig_file + "guess_alt" + ".png")

# velocity
V_init = Guess.linear(prob.time_all_section, 0.0, 0.0)
# Guess.plot(prob.time_all_section, V_init, "Velocity", "time", "Velocity")

# mass profile
M_init = Guess.cubic(prob.time_all_section, obj.M0, -0.6, obj.M0*obj.Mc, 0.0)
# Guess.plot(prob.time_all_section, M_init, "Mass", "time", "Mass")
# if(flag_savefig):plt.savefig(savefig_file + "guess_mass" + ".png")

# thrust profile
T_init = Guess.cubic(prob.time_all_section, obj.max_thrust * obj.M0 * obj.g0, 0.0, 0.0, 0.0)
# Guess.plot(prob.time_all_section, T_init, "Thrust Guess", "time", "Thrust")
# if(flag_savefig):plt.savefig(savefig_file + "guess_mass" + ".png")

plt.show()

# ========================
# Substitution initial value to parameter vector to be optimized
prob.set_states_all_section(0, R_init)
prob.set_states_all_section(1, V_init)
prob.set_states_all_section(2, M_init)
prob.set_controls_all_section(0, T_init)

# ========================
# Main Process
# Assign problem to SQP solver
prob.dynamics = [dynamics]
prob.knot_states_smooth = []
prob.cost = cost
# prob.cost_derivative = cost_derivative
prob.equality = equality
prob.inequality = inequality


def display_func():
    R = prob.states_all_section(0)
    print("max altitude: {0:.5f}".format(R[-1] - obj.Re))

prob.solve(obj, display_func, ftol=1e-12)

# ========================
# Post Process
# ------------------------
# Convert parameter vector to variable
R = prob.states_all_section(0)
v = prob.states_all_section(1)
m = prob.states_all_section(2)
T = prob.controls_all_section(0)
time = prob.time_update()

# ------------------------
# Calculate necessary variables
rho = obj.air_density(R - obj.Re)
drag = 0.5 * rho * v ** 2 * obj.Cd * obj.area
g = obj.GMe / R**2

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
plt.plot(time, v, marker="o", label="Velocity")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Velocity [m/s]")
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
plt.title("Thrust profile")
plt.plot(time, T / 1000, marker="o", label="Thrust")
plt.plot(time, drag / 1000, marker="o", label="Drag")
plt.plot(time, m * g / 1000, marker="o", label="Gravity")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Thrust [kN]")
plt.legend(loc="best")
if(flag_savefig): plt.savefig(savefig_file + "force" + ".png")

plt.show()
