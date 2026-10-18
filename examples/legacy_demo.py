"""Historical demonstrations, formerly embedded in optimize.py."""

# Copyright (c) 2017 Interstellar Technologies Inc.
# SPDX-License-Identifier: MIT

import matplotlib.pyplot as plt
import numpy as np

from OpenGoddard.optimize import Condition, Dynamics, Guess, Problem

if __name__ == "__main__":
    print("==== OpenGoddard test program ====")
    plt.close("all")
    plt.ion()

    time_init = [0, 1]
    n = [10]
    num_states = [3]
    num_controls = [1]
    max_iteration = 8

    prob = Problem(time_init, n, num_states, num_controls, max_iteration)

    print("t0 = %.1f\nnodes = " % (prob.t0))
    print(n)
    print("number of states = ")
    print(num_states)
    print("number of controls = ")
    print(num_controls)
    print("tau = ")
    print(prob.tau)
    # print("D = ")
    # print(prob.D)
    print("div = ")
    print(prob.div)
    print("=====" * 10)

    time_init = [0.0, 0.10, 0.2]
    n = [20, 10]
    num_states = [3, 3]
    num_controls = [1, 1]
    max_iteration = 1

    prob = Problem(time_init, n, num_states, num_controls, max_iteration)

    print("t0 = %.1f\nnodes = " % (prob.t0))
    print(n)
    print("number of states = ")
    print(num_states)
    print("number of controls = ")
    print(num_controls)
    print("tau = ")
    print(prob.tau)
    print("time_init = ")
    print(prob.time)
    print("time_all_section = ")
    print(prob.time_all_section)
    # print("D = ")
    # print(prob.D)
    print("div = ")
    print(prob.div)
    print("=====" * 10)

    print("p = ")
    print(np.round(prob.p, 0))
    print("states #0 all section = ")
    print(prob.states_all_section(0))
    print("states #1 all section = ")
    print(prob.states_all_section(1))
    print("controls #0 section #0 = ")
    print(prob.controls(0, 0))
    print("controls #0 all section = ")
    print(prob.controls_all_section(0))
    print("time final #0 = ")
    print(prob.time_final(0))
    print("time final #1 = ")
    print(prob.time_final(1))
    print("=====" * 10)

    plt.close("all")
    tau = prob.tau
    time_init = prob.time_init
    # initial estimation
    H_init = Guess.cubic(prob.time_all_section, 1.0, 0.0, 1.01, 0.0)
    Guess.plot(prob.time_all_section, H_init, "Altitude", "time", "Altitude")
    V_init = Guess.linear(prob.time_all_section, 0.0, 0.0)
    M_init = Guess.cubic(prob.time_all_section, 1.0, -0.6, 0.6, 0.0)
    T_init1 = Guess.linear(prob.time[0], 3.5, 3.5)
    T_init2 = Guess.linear(prob.time[1], 0.0, 0.0)
    T_init = np.hstack((T_init1, T_init2))
    Guess.plot(prob.time_all_section, T_init, "Thrust Guess", "time", "Thrust")

    plt.show()

    prob.set_states_all_section(0, H_init)
    prob.set_states_all_section(1, V_init)
    prob.set_states_all_section(2, M_init)
    prob.set_controls_all_section(0, T_init)

    print("p =")
    print(np.round(prob.p, 2))
    print("=====" * 10)

    print("=====                       ====")
    print("===== Rocket Ascent Problem ====")
    print("=====                       ====")

    obj = 1.0

    def dynamics(prob, obj, section):
        h = prob.states(0, section)
        v = prob.states(1, section)
        m = prob.states(2, section)
        T = prob.controls(0, section)

        Dc = 0.5 * 620 * 1.0 / 1.0
        c = 0.5 * np.sqrt(1.0 * 1.0)
        drag = 1 * Dc * v**2 * np.exp(-500 * (h - 1.0) / 1.0)
        g = 1.0 * (1.0 / h) ** 2

        dx = np.zeros(0)
        dx0 = v
        dx1 = (T - drag) / m - g
        dx2 = -T / c
        dx = np.hstack((dx0, dx1, dx2))
        return dx

    def equality(prob, obj):
        h = prob.states_all_section(0)
        v = prob.states_all_section(1)
        m = prob.states_all_section(2)
        prob.controls_all_section(0)
        prob.time_start(0)
        prob.time_final(0)
        prob.time_final(1)

        result = Condition()

        # event condition
        result.add(h[0] - 1.0)
        result.add(v[0] - 0.0)
        result.add(m[0] - 1.0)
        result.add(v[-1] - 0.0)
        result.add(m[-1] - 0.6)

        return result()

    def inequality(prob, obj):
        h = prob.states_all_section(0)
        v = prob.states_all_section(1)
        m = prob.states_all_section(2)
        T = prob.controls_all_section(0)
        tf = prob.time_final(-1)

        result = Condition()
        # lower bounds
        result.add(h - 1.0)
        result.add(v - 0.0)
        result.add(m - 0.6)
        result.add(T - 0.0)
        result.add(tf - 0.1)
        # upper bounds
        result.add(1.0 - m)
        result.add(3.5 - T)

        return result()

    def cost(prob, obj):
        h = prob.states_all_section(0)
        return -h[-1]

    def cost_derivative(prob, obj):
        jac = Condition(prob.number_of_variables)
        index_h_end = prob.index_states(0, -1, -1)
        jac.change_value(index_h_end, -1)
        return jac()

    dx0 = dynamics(prob, obj, 0)
    dx1 = dynamics(prob, obj, 1)
    result_eq = equality(prob, obj)
    result_ineq = inequality(prob, obj)
    print("dx section #0")
    print(np.round(dx0, 2))
    print("dx section #0")
    print(np.round(dx1, 2))
    print("equality = ")
    print(np.round(result_eq, 2))
    print("inequality = ")
    print(np.round(result_ineq, 2))
    print("=====" * 10)

    prob.dynamics = [dynamics, dynamics]
    prob.knot_states_smooth = [True]
    prob.cost = cost
    prob.cost_derivative = cost_derivative
    prob.equality = equality
    prob.inequality = inequality

    def display_func():
        h = prob.states_all_section(0)
        print("max altitude: {0:.5f}".format(h[-1]))

    prob.solve(obj, display_func)

    h = prob.states_all_section(0)
    v = prob.states_all_section(1)
    m = prob.states_all_section(2)
    T = prob.controls_all_section(0)
    time = prob.time_update()

    Dc = 0.5 * 620 * 1.0 / 1.0
    drag = 1 * Dc * v**2 * np.exp(-500 * (h - 1.0) / 1.0)
    g = 1.0 * (1.0 / h) ** 2

    plt.figure()
    plt.title("Altitude profile")
    plt.plot(time, h, marker="o", label="Altitude")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("Altitude [-]")

    plt.figure()
    plt.title("Velocity")
    plt.plot(time, v, marker="o", label="Velocity")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("Velocity [-]")

    plt.figure()
    plt.title("Mass")
    plt.plot(time, m, marker="o", label="Mass")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("Mass [-]")

    plt.figure()
    plt.title("Thrust profile")
    plt.plot(time, T, marker="o", label="Thrust")
    plt.plot(time, drag, marker="o", label="Drag")
    plt.plot(time, g, marker="o", label="Gravity")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("Thrust [-]")
    plt.legend(loc="best")

    print("=====                       ====")
    print("===== Bryson-Denham Problem ====")
    print("=====                       ====")

    class BrysonDenham:
        def __init__(self):
            self.max_x = 1.0 / 9.0

    def dynamics_Bryson(prob, obj, section):
        prob.states(0, section)
        v = prob.states(1, section)
        u = prob.controls(0, section)

        dx = Dynamics(prob, section)
        dx[0] = v
        dx[1] = u
        return dx()

    def equality_Bryson(prob, obj):
        x = prob.states_all_section(0)
        v = prob.states_all_section(1)
        prob.controls_all_section(0)
        tf = prob.time_final(-1)

        result = Condition()

        result.add(x[0] - 0.0)
        result.add(v[0] - 1.0)
        result.add(x[-1] - 0.0)
        result.add(v[-1] + 1.0)
        result.add(tf - 1.0)

        return result()

    def inequality_Bryson(prob, obj):
        x = prob.states_all_section(0)
        prob.states_all_section(1)
        prob.controls_all_section(0)

        result = Condition()

        # lower bounds
        result.add(x - 0.0)

        # upper bounds
        result.add(obj.max_x - x)

        return result()

    def cost_Bryson(prob, obj):
        prob.controls_all_section(0)
        # integrated = 0.5 * sum(u**2 *prob.w[0])
        integrated = 0.0
        return integrated

    def running_cost_Bryson(prob, obj):
        u = prob.controls_all_section(0)
        return 0.5 * u**2

    time_init = [0, 1.0]
    nodes = [30]
    num_states = [2]
    num_controls = [1]
    max_iteration = 10

    prob = Problem(time_init, nodes, num_states, num_controls, max_iteration)
    obj = BrysonDenham()

    x_init = Guess.constant(prob.time_all_section, 0.1)
    prob.set_states_all_section(0, x_init)

    prob.dynamics = [dynamics_Bryson]
    prob.knot_states_smooth = []
    prob.cost = cost_Bryson
    prob.running_cost = running_cost_Bryson
    prob.equality = equality_Bryson
    prob.inequality = inequality_Bryson

    prob.solve(obj)
    prob.to_csv("test.csv")
    prob.plot()

    x = prob.states_all_section(0)
    v = prob.states_all_section(1)
    u = prob.controls_all_section(0)
    time = prob.time_update()

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(time, x, "C0-o", label="x")
    plt.ylabel("x(t)")
    plt.legend()
    plt.grid()
    plt.title("Bryson-Denham Problem")
    plt.subplot(3, 1, 2)
    plt.plot(time, v, "C1-o", label="v")
    plt.ylabel("v(t)")
    plt.legend()
    plt.grid()
    plt.subplot(3, 1, 3)
    plt.plot(time, u, "C2-o", label="u")
    plt.ylabel("u(t)")
    plt.legend()
    plt.grid()

    print("=====                   ====")
    print("===== Unit scaling Test ====")
    print("=====                   ====")
    section = 0
    x = prob.states(0, section)
    v = prob.states(1, section)
    u = prob.controls(0, section)
    dx = Dynamics(prob)
    dx[0] = v
    dx[1] = u
    dx()

    print("=====                             ====")
    print("===== Bad Brachistochrone Problem ====")
    print("=====                             ====")

    class Ball:
        def __init__(self):
            self.g = 9.8  # gravity [m/s2]
            self.l = 600000  # goal [m]
            self.h = 300000  # depth limit [m]

    def dynamics_Brachistochrone(prob, obj, section):
        prob.states(0, section)
        prob.states(1, section)
        v = prob.states(2, section)
        theta = prob.controls(0, section)

        dx = Dynamics(prob, section)
        dx[0] = v * np.sin(theta)
        dx[1] = v * np.cos(theta)
        dx[2] = obj.g * np.cos(theta)
        return dx()

    def equality_Brachistochrone(prob, obj):
        x = prob.states_all_section(0)
        y = prob.states_all_section(1)
        v = prob.states_all_section(2)
        prob.controls_all_section(0)
        prob.time_final(-1)

        result = Condition()
        # result.add(x[0] - 0.0)
        # result.add(y[0] - 0.0)
        # result.add(v[0] - 0.0)
        # result.add(x[-1] - obj.l)
        # result.add(y[-1] - 0.0)
        result.equal(x[0], 0.0)
        result.equal(y[0], 0.0)
        result.equal(v[0], 0.0)
        result.equal(x[-1], obj.l)
        result.equal(y[-1], 0.0)

        return result()

    def inequality_Brachistochrone(prob, obj):
        x = prob.states_all_section(0)
        y = prob.states_all_section(1)
        prob.states_all_section(2)
        theta = prob.controls_all_section(0)
        tf = prob.time_final(-1)

        result = Condition()
        # # lower bounds
        # result.add(tf - 500)
        # result.add(x - 0)
        # result.add(y - 0)
        # result.add(theta - 0)
        # # upper bounds
        # result.add(np.pi - theta)
        # result.add(obj.l - x)
        # result.add(700 - tf)
        # lower bounds
        result.lower_bound(tf, 500)
        result.lower_bound(x, 0)
        result.lower_bound(y, 0)
        result.lower_bound(theta, 0)
        # upper bounds
        result.upper_bound(theta, np.pi)
        result.upper_bound(x, obj.l)
        result.upper_bound(tf, 700)

        return result()

    def cost_Brachistochrone(prob, obj):
        tf = prob.time_final(-1)
        return tf

    def cost_derivative_Brachistochrone(prob, obj):
        jac = Condition(prob.number_of_variables)
        index_tf = prob.index_time_final(0)
        # index_tf = prob.index_time_final(-1)
        jac.change_value(index_tf, 1)
        return jac()

    time_init = [0.0, 700.0]
    n = [30]
    num_states = [3]
    num_controls = [1]
    max_iteration = 10
    prob = Problem(time_init, n, num_states, num_controls, max_iteration)
    obj = Ball()

    unit_x = 300000
    unit_y = 100000
    unit_time = 100
    prob.set_unit_states_all_section(0, unit_x)
    prob.set_unit_states_all_section(1, unit_y)
    prob.set_unit_states_all_section(2, unit_x / unit_time)
    prob.set_unit_controls_all_section(0, 1.0)
    prob.set_unit_time(unit_time)

    half_nodes = int(prob.nodes[0] / 2)
    theta_init = Guess.linear(prob.time_all_section, 0.0, np.pi)
    x_init = Guess.linear(prob.time_all_section, 0.0, obj.l)
    y_init0 = Guess.linear(prob.time_all_section[:half_nodes], 0, obj.h)
    y_init1 = Guess.linear(prob.time_all_section[half_nodes:], obj.h, 0)
    y_init = np.hstack((y_init0, y_init1))
    v_init0 = Guess.linear(prob.time_all_section[:half_nodes], 0, obj.h)
    v_init1 = Guess.linear(prob.time_all_section[half_nodes:], obj.h, 0)
    v_init = np.hstack((v_init0, v_init1))

    prob.set_states_all_section(0, x_init)
    prob.set_states_all_section(1, y_init)
    prob.set_controls_all_section(0, theta_init)

    prob.dynamics = [dynamics_Brachistochrone]
    prob.knot_states_smooth = []
    prob.cost = cost_Brachistochrone
    prob.cost_derivative = cost_derivative_Brachistochrone
    prob.equality = equality_Brachistochrone
    prob.inequality = inequality_Brachistochrone

    def display_func():
        tf = prob.time_final(-1)
        print("tf: {0:.5f}".format(tf))

    prob.solve(obj, display_func)

    x = prob.states_all_section(0)
    y = prob.states_all_section(1)
    v = prob.states_all_section(2)
    theta = prob.controls_all_section(0)
    time = prob.time_update()

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(time, x, marker="o", label="x")
    plt.plot(time, y, marker="o", label="y")
    plt.plot(time, v, marker="o", label="v")
    plt.grid()
    plt.ylabel("position [m], velocity [m/s]")
    plt.legend(loc="best")

    plt.subplot(2, 1, 2)
    plt.plot(time, theta, marker="o", label="gamma")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("angle [rad]")
    plt.legend(loc="best")

    plt.figure()
    plt.plot(x, y, marker="o", label="trajectry")
    plt.grid()
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend(loc="best")
    plt.gca().invert_yaxis()

    prob.plot("Bad Brachistochrone")

    # plt.show()
