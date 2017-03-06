# -*- coding: utf-8 -*-
# Copyright 2017 Interstellar Technologies Inc. All Rights Reserved.
# 東京−大阪間600kmを重力のみで最速で到達する経路を見つける
# 摩擦などはないと仮定している。
# スケールはDIDOのマニュアル参考

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from OpenGoddard.optimize import Problem, Guess, Condition, Dynamics


class Ball:
    def __init__(self):
        self.g = 9.8  # gravity [m/s2]
        self.l = 600000  # goal [m]
        self.h = 300000  # depth limit [m]


def dynamics(prob, obj, section):
    x = prob.states(0, section)
    y = prob.states(1, section)
    v = prob.states(2, section)
    theta = prob.controls(0, section)

    g = obj.g

    dx = Dynamics(prob, section)
    dx[0] = v * np.sin(theta)
    dx[1] = v * np.cos(theta)
    dx[2] = g * np.cos(theta)
    return dx()


def equality(prob, obj):
    x = prob.states_all_section(0)
    y = prob.states_all_section(1)
    v = prob.states_all_section(2)
    theta = prob.controls_all_section(0)
    tf = prob.time_final(-1)

    result = Condition()

    # event condition
    result.add(x[0] - 0.0)
    result.add(y[0] - 0.0)
    result.add(v[0] - 0.0)
    result.add(x[-1] - obj.l)
    result.add(y[-1] - 0.0)
    # result.add(v[-1] - 0.0)

    return result()


def inequality(prob, obj):
    x = prob.states_all_section(0)
    y = prob.states_all_section(1)
    v = prob.states_all_section(2)
    theta = prob.controls_all_section(0)
    tf = prob.time_final(-1)

    result = Condition()
    # lower bounds
    # result.add(tf - 500)
    result.add(x - 0)
    result.add(y - 0)
    result.add(theta - 0)
    # result.add(100000 / obj.unit_y - y)
    # upper bounds
    result.add(np.pi - theta)
    result.add(obj.l - x)
    # result.add(700 - tf)

    return result()


def cost(prob, obj):
    tf = prob.time_final(-1)
    return tf


def cost_derivative(prob, obj):
    jac = Condition(prob.number_of_variables)
    index_tf = prob.index_time_final(0)
    # index_tf = prob.index_time_final(-1)
    jac.change_value(index_tf, 1)
    return jac()

# ========================
plt.close("all")
plt.ion()
# Program Starting Point
time_init = [0.0, 600.0]
n = [30]
num_states = [3]
num_controls = [1]
max_iteration = 10

flag_savefig = True

savefig_dir = "01_Brachistochrone/Tokyo_"

# ------------------------
# set OpenGoddard class for algorithm determination
prob = Problem(time_init, n, num_states, num_controls, max_iteration)
obj = Ball()
# obj.make_scales()

unit_x = 300000
unit_y = 100000
unit_t = 100
unit_v = unit_x / unit_t
prob.set_unit_states_all_section(0, unit_x)
prob.set_unit_states_all_section(1, unit_y)
prob.set_unit_states_all_section(2, unit_v)
prob.set_unit_controls_all_section(0, 1.0)
prob.set_unit_time(unit_t)

# ========================
# Initial parameter guess
half_nodes = int(prob.nodes[0] / 2)
theta_init = Guess.linear(prob.time_all_section, 0.0, np.pi)
# Guess.plot(prob.time_all_section, theta_init, "gamma", "time", "gamma")
# if(flag_savefig):plt.savefig(savefig_dir + "guess_gamma" + ".png")

x_init = Guess.linear(prob.time_all_section, 0.0, obj.l)
# Guess.plot(prob.time_all_section, x_init, "x", "time", "x")
# if(flag_savefig):plt.savefig(savefig_dir + "guess_x" + ".png")

# y_init = Guess.linear(prob.time_all_section, 0.0, obj.l / np.sqrt(3))
y_init0 = Guess.linear(prob.time_all_section[:half_nodes], 0, obj.h)
y_init1 = Guess.linear(prob.time_all_section[half_nodes:], obj.h, 0)
y_init = np.hstack((y_init0, y_init1))
# Guess.plot(prob.time_all_section, y_init, "y", "time", "y")
# if(flag_savefig):plt.savefig(savefig_dir + "guess_y" + ".png")

v_init0 = Guess.linear(prob.time_all_section[:half_nodes], 0, obj.h)
v_init1 = Guess.linear(prob.time_all_section[half_nodes:], obj.h, 0)
v_init = np.hstack((v_init0, v_init1))
# Guess.plot(prob.time_all_section, v_init, "v", "time", "v")
# if(flag_savefig):plt.savefig(savefig_dir + "guess_v" + ".png")


prob.set_states_all_section(0, x_init)
prob.set_states_all_section(1, y_init)
prob.set_controls_all_section(0, theta_init)

# ========================
# Main Process
# Assign problem to SQP solver
prob.dynamics = [dynamics]
prob.knot_states_smooth = []
prob.cost = cost
prob.cost_derivative = cost_derivative
prob.equality = equality
prob.inequality = inequality


def display_func():
    tf = prob.time_final(-1)
    print("tf: {0:.5f}".format(tf))


prob.solve(obj, display_func)

# ========================
# Post Process
# ------------------------
# Convert parameter vector to variable
x = prob.states_all_section(0)
y = prob.states_all_section(1)
v = prob.states_all_section(2)
theta = prob.controls_all_section(0)
time = prob.time_update()

# ------------------------
# Visualizetion
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(time, x/1000, marker="o", label="x")
plt.plot(time, y/1000, marker="o", label="y")
plt.plot(time, v, marker="o", label="v")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.ylabel("position [km], velocity [m/s]")
plt.legend(loc="best")

plt.subplot(2, 1, 2)
plt.plot(time, theta, marker="o", label="gamma")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("angle [rad]")
plt.legend(loc="best")
if(flag_savefig): plt.savefig(savefig_dir + "plot" + ".png")

plt.figure()
plt.plot(x/1000, y/1000, marker="o", label="trajectry")
plt.axhline(0, color="k")
plt.axvline(0, color="k")
plt.axvline(obj.l/1000, color="k")
plt.grid()
plt.xlabel("x [km]")
plt.ylabel("y [km]")
plt.axis('equal')
plt.legend(loc="best")
plt.gca().invert_yaxis()
if(flag_savefig): plt.savefig(savefig_dir + "trajectry" + ".png")

plt.show()
