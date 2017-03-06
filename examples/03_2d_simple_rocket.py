# -*- coding: utf-8 -*-
# Copyright 2017 Interstellar Technologies Inc. All Rights Reserved.

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from OpenGoddard.optimize import Problem, Guess, Condition, Dynamics


class Rocket:
    def __init__(self):
        self.a = 1.0  # acceleration


def dynamics(prob, obj, section):
    u = prob.states(0, section)
    v = prob.states(1, section)
    x = prob.states(2, section)
    y = prob.states(3, section)
    beta = prob.controls(0, section)

    a = obj.a

    dx = Dynamics(prob, section)
    dx[0] = a * np.cos(beta)
    dx[1] = a * np.sin(beta)
    dx[2] = u
    dx[3] = v
    return dx()


def equality(prob, obj):
    u = prob.states_all_section(0)
    v = prob.states_all_section(1)
    x = prob.states_all_section(2)
    y = prob.states_all_section(3)
    beta = prob.controls_all_section(0)
    tf = prob.time_final(-1)

    result = Condition()

    # event condition
    result.add(u[0] - 0.0)
    result.add(v[0] - 0.0)
    result.add(x[0] - 0.0)
    result.add(y[0] - 0.0)
    result.add(u[-1] - 1.0)
    result.add(v[-1] - 0.0)
    result.add(y[-1] - 1.0)

    return result()


def inequality(prob, obj):
    u = prob.states_all_section(0)
    v = prob.states_all_section(1)
    x = prob.states_all_section(2)
    y = prob.states_all_section(3)
    beta = prob.controls_all_section(0)
    tf = prob.time_final(-1)

    result = Condition()

    # lower bounds
    result.add(beta + np.pi/2)

    # upper bounds
    result.add(np.pi/2 - beta)

    return result()


def cost(prob, obj):
    tf = prob.time_final(-1)
    return tf


# ========================
plt.close("all")
# Program Starting Point
time_init = [0.0, 2.0]
n = [20]
num_states = [4]
num_controls = [1]
max_iteration = 50

flag_savefig = True
savefig_dir = "03_2d_simple_rocket/"

# ------------------------
# set OpenGoddard class for algorithm determination
prob = Problem(time_init, n, num_states, num_controls, max_iteration)
obj = Rocket()

# ========================
# Main Process
# Assign problem to SQP solver
prob.dynamics = [dynamics]
prob.knot_states_smooth = []
prob.cost = cost
prob.cost_derivative = None
prob.equality = equality
prob.inequality = inequality

prob.solve(obj)

# ========================
# Post Process
# ------------------------
# Convert parameter vector to variable
u = prob.states_all_section(0)
v = prob.states_all_section(1)
x = prob.states_all_section(2)
y = prob.states_all_section(3)
beta = prob.controls_all_section(0)
time = prob.time_update()

# ------------------------
# Visualizetion
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(time, u, marker="o", label="u")
plt.plot(time, v, marker="o", label="v")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.ylabel("velocity [m/s]")
plt.legend(loc="best")

plt.subplot(3, 1, 2)
plt.plot(time, x, marker="o", label="x")
plt.plot(time, y, marker="o", label="y")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.ylabel("position [m]")
plt.legend(loc="best")

plt.subplot(3, 1, 3)
plt.plot(time, beta, marker="o", label="Mass")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("angle [rad]")
plt.legend(loc="best")
if(flag_savefig): plt.savefig(savefig_dir + "plot" + ".png")

plt.figure()
plt.plot(x, y, marker="o", label="trajectry")
plt.axhline(0, color="k")
plt.axhline(1, color="k")
plt.axvline(0, color="k")
plt.grid()
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.axis('equal')
plt.legend(loc="best")
if(flag_savefig): plt.savefig(savefig_dir + "trajectry" + ".png")

plt.show()
