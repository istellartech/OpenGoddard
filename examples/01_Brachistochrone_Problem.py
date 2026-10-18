# -*- coding: utf-8 -*-
# Copyright 2017 Interstellar Technologies Inc. All Rights Reserved.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from OpenGoddard import DynamicsModel, LinearCallback
from OpenGoddard.optimize import Condition, Guess, Problem


class Ball:
    def __init__(self):
        self.g = 1.0  # gravity
        self.l = 1.0  # goal
        self.h = 0.1  # limit
        self.theta0 = np.deg2rad(30)  # limit and initial angle


def pointwise_dynamics(x, u, time, obj, section, xp):
    theta = u[0]
    return xp.stack((x[2] * xp.sin(theta), x[2] * xp.cos(theta), obj.g * xp.cos(theta)))


dynamics = DynamicsModel(pointwise_dynamics)


@LinearCallback
def equality(prob, obj):
    x = prob.states_all_section(0)
    y = prob.states_all_section(1)
    v = prob.states_all_section(2)
    theta = prob.controls_all_section(0)
    tf = prob.time_final(-1)

    result = prob.condition()

    # event condition
    result.equal(x[0], 0.0)
    result.equal(y[0], 0.0)
    result.equal(v[0], 0.0)
    result.equal(x[-1], obj.l)

    return result()


@LinearCallback
def inequality(prob, obj):
    x = prob.states_all_section(0)
    y = prob.states_all_section(1)
    v = prob.states_all_section(2)
    theta = prob.controls_all_section(0)
    tf = prob.time_final(-1)

    result = prob.condition()

    # lower bounds
    result.lower_bound(tf, 0.1)
    result.lower_bound(y[1:], 0)
    result.lower_bound(theta, 0)

    # upper bounds
    # result.upper_bound(theta, np.pi/2)
    # result.upper_bound(y, x * np.tan(obj.theta0) + obj.h)

    return result()


@LinearCallback
def cost(prob, obj):
    tf = prob.time_final(-1)
    return tf


def cost_derivative(prob, obj):
    jac = Condition(prob.number_of_variables)
    # index_tf = prob.index_time_final(0)
    index_tf = prob.index_time_final(-1)
    jac.change_value(index_tf, prob.unit_time)
    return jac()


# ========================

if __name__ == "__main__":
    plt.close("all")
    plt.ion()
    # Program Starting Point
    time_init = [0.0, 2.0]
    n = [20]
    num_states = [3]
    num_controls = [1]
    max_iteration = 30

    flag_savefig = True

    savefig_dir = str(Path(__file__).resolve().parent / "01_Brachistochrone/normal_")
    Path(savefig_dir).parent.mkdir(parents=True, exist_ok=True)

    # ------------------------
    # set OpenGoddard class for algorithm determination
    prob = Problem(time_init, n, num_states, num_controls, max_iteration)
    obj = Ball()

    # ========================
    # Initial parameter guess
    theta_init = Guess.linear(prob.time_all_section, np.deg2rad(30), np.deg2rad(30))
    # Guess.plot(prob.time_all_section, theta_init, "gamma", "time", "gamma")
    # if(flag_savefig):plt.savefig(savefig_dir + "guess_gamma" + savefig_add + ".png")

    x_init = Guess.linear(prob.time_all_section, 0.0, obj.l)
    # Guess.plot(prob.time_all_section, x_init, "x", "time", "x")
    # if(flag_savefig):plt.savefig(savefig_dir + "guess_x" + savefig_add + ".png")

    y_init = Guess.linear(prob.time_all_section, 0.0, obj.l / np.sqrt(3))
    # Guess.plot(prob.time_all_section, theta_init, "y", "time", "y")
    # if(flag_savefig):plt.savefig(savefig_dir + "guess_y" + savefig_add + ".png")

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

    prob.solve(obj)

    # ========================
    # Post Process
    # ------------------------
    # Convert parameter vector to variable
    x = prob.states_all_section(0)
    y = prob.states_all_section(1)
    gamma = prob.controls_all_section(0)
    time = prob.time_update()

    # ------------------------
    # Visualizetion
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(time, x, marker="o", label="x")
    plt.plot(time, y, marker="o", label="y")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.ylabel("velocity [m/s]")
    plt.legend(loc="best")

    plt.subplot(2, 1, 2)
    plt.plot(time, gamma, marker="o", label="gamma")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("angle [rad]")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_dir + "plot" + ".png")

    x_wall = np.linspace(0, obj.l)
    y_wall = x_wall * np.tan(obj.theta0) + obj.h
    plt.figure()
    plt.plot(x, y, marker="o", label="trajectry")
    # plt.plot(x_wall, y_wall, color = "k", label = "wall")
    plt.axhline(0, color="k")
    plt.axvline(0, color="k")
    plt.axvline(obj.l, color="k")
    plt.grid()
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    # plt.ylim([-0.02, 0.6])
    plt.legend(loc="best")
    plt.gca().invert_yaxis()
    if flag_savefig:
        plt.savefig(savefig_dir + "trajectry" + ".png")

    plt.show()
