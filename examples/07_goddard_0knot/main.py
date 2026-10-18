# Copyright 2017 Interstellar Technologies Inc. All Rights Reserved.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from OpenGoddard import DynamicsModel, LinearCallback
from OpenGoddard.optimize import Guess, Problem


class Rocket:
    g0 = 1.0  # Gravity at surface [-]

    def __init__(self):
        self.H0 = 1.0  # Initial height
        self.V0 = 0.0  # Initial velocity
        self.M0 = 1.0  # Initial mass
        self.Tc = 3.5  # Use for thrust
        self.Hc = 500  # Use for drag
        self.Vc = 620  # Use for drag
        self.Mc = 0.6  # Fraction of initial mass left at end
        self.c = 0.5 * np.sqrt(self.g0 * self.H0)  # Thrust-to-fuel mass
        self.Mf = self.Mc * self.M0  # Final mass
        self.Dc = 0.5 * self.Vc * self.M0 / self.g0  # Drag scaling
        self.T_max = self.Tc * self.g0 * self.M0  # Maximum thrust


def pointwise_dynamics(x, u, time, obj, section, xp):
    h, v, mass = x
    thrust = u[0]
    drag = obj.Dc * v**2 * xp.exp(-obj.Hc * (h - obj.H0) / obj.H0)
    gravity = obj.g0 * (obj.H0 / h) ** 2
    return xp.stack((v, (thrust - drag) / mass - gravity, -thrust / obj.c))


dynamics = DynamicsModel(pointwise_dynamics)


@LinearCallback
def equality(prob, obj):
    h = prob.states_all_section(0)
    v = prob.states_all_section(1)
    m = prob.states_all_section(2)
    T = prob.controls_all_section(0)
    tf = prob.time_final(-1)

    result = prob.condition()

    # event condition
    result.equal(h[0], obj.H0)
    result.equal(v[0], obj.V0)
    result.equal(m[0], obj.M0)
    result.equal(v[-1], 0.0)
    result.equal(m[-1], obj.Mf)

    return result()


@LinearCallback
def inequality(prob, obj):
    h = prob.states_all_section(0)
    v = prob.states_all_section(1)
    m = prob.states_all_section(2)
    T = prob.controls_all_section(0)
    tf = prob.time_final(-1)

    result = prob.condition()
    # lower bounds
    result.lower_bound(h[1:], obj.H0)
    result.lower_bound(v[1:-1], 0.0)
    result.lower_bound(m[:-1], obj.Mf)
    result.lower_bound(T, 0.0)
    result.lower_bound(tf, 0.1)
    # upper bounds
    result.upper_bound(m[1:], obj.M0)
    result.upper_bound(T, obj.T_max)

    return result()


@LinearCallback
def cost(prob, obj):
    h = prob.states_all_section(0)
    return -h[-1]


# ========================


def main(*, flag_savefig: bool = True):
    plt.close("all")
    # Program Starting Point
    time_init = [0.0, 0.3]
    n = [50]
    num_states = [3]
    num_controls = [1]
    max_iteration = 30

    savefig_file = str(Path(__file__).resolve().parent / "output/0knot_")
    Path(savefig_file).parent.mkdir(parents=True, exist_ok=True)

    # ------------------------
    # set OpenGoddard class for algorithm determination
    prob = Problem(time_init, n, num_states, num_controls, max_iteration)

    # ------------------------
    # create instance of operating object
    # Nondimensionalization of parameters
    obj = Rocket()

    # ========================
    # Initial parameter guess

    # altitude profile
    H_init = Guess.cubic(prob.time_all_section, 1.0, 0.0, 1.010, 0.0)
    # Guess.plot(prob.time_all_section, H_init, "Altitude", "time", "Altitude")
    # if(flag_savefig):plt.savefig(savefig_file + "guess_alt" + ".png")

    # velocity
    V_init = Guess.linear(prob.time_all_section, 0.0, 0.0)
    # Guess.plot(prob.time_all_section, V_init, "Velocity", "time", "Velocity")

    # mass profile
    M_init = Guess.cubic(prob.time_all_section, 1.0, -0.6, 0.6, 0.0)
    # Guess.plot(prob.time_all_section, M_init, "Mass", "time", "Mass")
    # if(flag_savefig):plt.savefig(savefig_file + "guess_mass" + ".png")

    # thrust profile
    T_init = Guess.cubic(prob.time_all_section, 3.5, 0.0, 0.0, 0.0)
    # Guess.plot(prob.time_all_section, T_init, "Thrust Guess", "time", "Thrust")
    # if(flag_savefig):plt.savefig(savefig_file + "guess_thrust" + ".png")

    plt.show()

    # ========================
    # Substitution initial value to parameter vector to be optimized
    prob.set_states_all_section(0, H_init)
    prob.set_states_all_section(1, V_init)
    prob.set_states_all_section(2, M_init)
    prob.set_controls_all_section(0, T_init)

    # ========================
    # Main Process
    # Assign problem to SQP solver
    prob.dynamics = [dynamics]
    prob.knot_states_smooth = []
    prob.cost = cost
    prob.cost_derivative = None
    prob.equality = equality
    prob.inequality = inequality

    def display_func():
        h = prob.states_all_section(0)
        print(f"max altitude: {h[-1]:.5f}")

    prob.solve(obj, display_func, ftol=1e-10)

    # ========================
    # Post Process
    # ------------------------
    # Convert parameter vector to variable
    h = prob.states_all_section(0)
    v = prob.states_all_section(1)
    m = prob.states_all_section(2)
    T = prob.controls_all_section(0)
    time = prob.time_update()

    # ------------------------
    # Calculate necessary variables
    Dc = 0.5 * 620 * 1.0 / 1.0
    drag = 1 * Dc * v**2 * np.exp(-500 * (h - 1.0) / 1.0)
    g = 1.0 * (1.0 / h) ** 2

    # ------------------------
    # Visualizetion
    plt.figure()
    plt.title("Altitude profile")
    plt.plot(time, h, marker="o", label="Altitude")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("Altitude [-]")
    if flag_savefig:
        plt.savefig(savefig_file + "altitude" + ".png")

    plt.figure()
    plt.title("Velocity")
    plt.plot(time, v, marker="o", label="Velocity")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("Velocity [-]")
    if flag_savefig:
        plt.savefig(savefig_file + "velocity" + ".png")

    plt.figure()
    plt.title("Mass")
    plt.plot(time, m, marker="o", label="Mass")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("Mass [-]")
    if flag_savefig:
        plt.savefig(savefig_file + "mass" + ".png")

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
    if flag_savefig:
        plt.savefig(savefig_file + "force" + ".png")

    plt.show()
    return prob


if __name__ == "__main__":
    main()
