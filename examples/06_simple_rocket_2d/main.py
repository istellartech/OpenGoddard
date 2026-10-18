# Copyright 2017 Interstellar Technologies Inc. All Rights Reserved.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from OpenGoddard import DynamicsModel, LinearCallback
from OpenGoddard.optimize import Problem


class Rocket:
    def __init__(self):
        self.a = 1.0  # acceleration


def pointwise_dynamics(x, u, time, obj, section, xp):
    return xp.stack((obj.a * xp.cos(u[0]), obj.a * xp.sin(u[0]), x[0], x[1]))


dynamics = DynamicsModel(pointwise_dynamics)


@LinearCallback
def equality(prob, obj):
    u = prob.states_all_section(0)
    v = prob.states_all_section(1)
    x = prob.states_all_section(2)
    y = prob.states_all_section(3)
    beta = prob.controls_all_section(0)
    tf = prob.time_final(-1)

    result = prob.condition()

    # event condition
    result.equal(u[0], 0.0)
    result.equal(v[0], 0.0)
    result.equal(x[0], 0.0)
    result.equal(y[0], 0.0)
    result.equal(u[-1], 1.0)
    result.equal(v[-1], 0.0)
    result.equal(y[-1], 1.0)

    return result()


@LinearCallback
def inequality(prob, obj):
    u = prob.states_all_section(0)
    v = prob.states_all_section(1)
    x = prob.states_all_section(2)
    y = prob.states_all_section(3)
    beta = prob.controls_all_section(0)
    tf = prob.time_final(-1)

    result = prob.condition()

    # lower bounds
    result.lower_bound(beta, -np.pi / 2)

    # upper bounds
    result.upper_bound(beta, np.pi / 2)

    return result()


@LinearCallback
def cost(prob, obj):
    tf = prob.time_final(-1)
    return tf


# ========================


def main(*, flag_savefig: bool = True):
    plt.close("all")
    # Program Starting Point
    time_init = [0.0, 2.0]
    n = [20]
    num_states = [4]
    num_controls = [1]
    max_iteration = 50

    savefig_dir = str(Path(__file__).resolve().parent / "output/") + "/"
    Path(savefig_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------
    # set OpenGoddard class for algorithm determination
    prob = Problem(time_init, n, num_states, num_controls, max_iteration)
    obj = Rocket()

    # Satisfy endpoint conditions in the initial guess and avoid the singular
    # all-zero trajectory when using a second-order constrained solver.
    duration = prob.time_final(0) - prob.time_start(0)
    fraction = (prob.time[0] - prob.time_start(0)) / duration
    prob.set_states(0, 0, fraction)
    prob.set_states(1, 0, 6 / duration * fraction * (1 - fraction))
    prob.set_states(2, 0, duration / 2 * fraction**2)
    prob.set_states(3, 0, 3 * fraction**2 - 2 * fraction**3)
    prob.set_controls(
        0, 0, np.arctan2(6 / duration**2 * (1 - 2 * fraction), 1 / duration)
    )

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
    if flag_savefig:
        plt.savefig(savefig_dir + "plot" + ".png")

    plt.figure()
    plt.plot(x, y, marker="o", label="trajectry")
    plt.axhline(0, color="k")
    plt.axhline(1, color="k")
    plt.axvline(0, color="k")
    plt.grid()
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_dir + "trajectry" + ".png")

    plt.show()
    return prob


if __name__ == "__main__":
    main()
