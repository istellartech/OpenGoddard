# Copyright 2017 Interstellar Technologies Inc. All Rights Reserved.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, interpolate

from OpenGoddard.optimize import Condition, Dynamics, Guess, Problem


class Orbiter:
    def __init__(self):
        self.u_max = 0.01
        self.r0 = 1.0
        self.vr0 = 0.0
        self.vt0 = 1.0
        self.rf = 4.0
        self.vrf = 0.0
        self.vtf = 0.5
        self.tf_max = 55


def dynamics(prob, obj, section):
    r = prob.states(0, section)
    vr = prob.states(1, section)
    vt = prob.states(2, section)
    ur1 = prob.controls(0, section)
    ur2 = prob.controls(1, section)
    ut1 = prob.controls(2, section)
    ut2 = prob.controls(3, section)

    dx = Dynamics(prob, section)
    dx[0] = vr
    dx[1] = vt**2 / r - 1 / r**2 + (ur1 - ur2)
    dx[2] = -vr * vt / r + (ut1 - ut2)
    return dx()


def equality(prob, obj):
    r = prob.states_all_section(0)
    vr = prob.states_all_section(1)
    vt = prob.states_all_section(2)
    ur1 = prob.controls_all_section(0)
    ur2 = prob.controls_all_section(1)
    ut1 = prob.controls_all_section(2)
    ut2 = prob.controls_all_section(3)
    tf = prob.time_final(-1)

    result = Condition()

    # event condition
    result.equal(r[0], obj.r0)
    result.equal(vr[0], obj.vr0)
    result.equal(vt[0], obj.vt0)
    result.equal(r[-1], obj.rf)
    result.equal(vr[-1], obj.vrf)
    result.equal(vt[-1], obj.vtf)

    return result()


def inequality(prob, obj):
    r = prob.states_all_section(0)
    vr = prob.states_all_section(1)
    vt = prob.states_all_section(2)
    ur1 = prob.controls_all_section(0)
    ur2 = prob.controls_all_section(1)
    ut1 = prob.controls_all_section(2)
    ut2 = prob.controls_all_section(3)
    tf = prob.time_final(-1)

    result = Condition()

    # Radius stays between the two orbits. Endpoints are already fixed by the
    # equality constraints; repeating them here makes SLSQP's subproblem
    # degenerate ("Inequality constraints incompatible" on some platforms).
    result.lower_bound(r[1:-1], obj.r0)
    result.upper_bound(r[1:-1], obj.rf)
    # Thrust and final-time limits are box bounds set in main().

    return result()


def cost(prob, obj):
    return 0.0


def running_cost(prob, obj):
    ur1 = prob.controls_all_section(0)
    ur2 = prob.controls_all_section(1)
    ut1 = prob.controls_all_section(2)
    ut2 = prob.controls_all_section(3)

    return (ur1 + ur2) + (ut1 + ut2)


# ========================
def main(*, flag_savefig: bool = True):
    plt.close("all")
    plt.ion()
    # Program Starting Point
    # With |u| <= u_max = 0.01 the spiral needs a velocity change of about 0.5,
    # so the transfer takes at least ~35 time units; start near the optimum.
    time_init = [0.0, 50.0]
    n = [60]
    num_states = [3]
    num_controls = [4]
    max_iteration = 10

    savefig_dir = str(Path(__file__).resolve().parent / "output/") + "/"
    Path(savefig_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------
    # set OpenGoddard class for algorithm determination
    prob = Problem(time_init, n, num_states, num_controls, max_iteration)
    obj = Orbiter()

    # ========================
    # Initial parameter guess
    r_init = Guess.linear(prob.time_all_section, obj.r0, obj.rf)
    # Guess.plot(prob.time_all_section, r_init, "r", "time", "r")
    # if(flag_savefig):plt.savefig(savefig_dir + "guess_r" + savefig_add + ".png")

    vr_init = Guess.linear(prob.time_all_section, obj.vr0, obj.vrf)
    # Guess.plot(prob.time_all_section, vr_init, "vr", "time", "vr")
    # if(flag_savefig):plt.savefig(savefig_dir + "guess_vr" + savefig_add + ".png")

    # Circular speed at the guessed radius keeps the initial guess near-feasible.
    vt_init = 1 / np.sqrt(r_init)
    # Guess.plot(prob.time_all_section, theta_init, "vt", "time", "vt")
    # if(flag_savefig):plt.savefig(savefig_dir + "guess_vt" + savefig_add + ".png")

    ur1_init = Guess.constant(prob.time_all_section, 0.0)
    # Guess.plot(prob.time_all_section, ur1_init, "ur1", "time", "ur1")
    # if(flag_savefig):plt.savefig(savefig_dir + "guess_ur1" + savefig_add + ".png")

    ut1_init = Guess.constant(prob.time_all_section, obj.u_max / 2)
    # Guess.plot(prob.time_all_section, ut1_init, "ut1", "time", "ut1")
    # if(flag_savefig):plt.savefig(savefig_dir + "guess_ut1" + savefig_add + ".png")

    prob.set_states_all_section(0, r_init)
    prob.set_states_all_section(1, vr_init)
    prob.set_states_all_section(2, vt_init)
    prob.set_controls_all_section(0, ur1_init)
    prob.set_controls_all_section(2, ut1_init)
    for control in range(4):
        prob.set_controls_bounds(control, 0, 0.0, obj.u_max)
    prob.set_time_final_bounds(0, 1.0, obj.tf_max)

    # ========================
    # Main Process
    # Assign problem to SQP solver
    prob.dynamics = [dynamics]
    prob.knot_states_smooth = []
    prob.cost = cost
    prob.running_cost = running_cost
    prob.equality = equality
    prob.inequality = inequality

    def display_func():
        tf = prob.time_final(-1)
        print(f"tf: {tf:.5f}")

    prob.solve(obj, display_func, ftol=1e-10, maxiter=300)

    # ========================
    # Post Process
    # ------------------------
    # Convert parameter vector to variable
    r = prob.states_all_section(0)
    vr = prob.states_all_section(1)
    vt = prob.states_all_section(2)
    ur1 = prob.controls_all_section(0)
    ur2 = prob.controls_all_section(1)
    ut1 = prob.controls_all_section(2)
    ut2 = prob.controls_all_section(3)
    time = prob.time_update()

    # ------------------------
    # Visualizetion
    plt.figure()
    plt.plot(time, r, marker="o", label="r")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.xlabel("time [-]")
    plt.ylabel("r [-]")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_dir + "r" + ".png")

    plt.figure()
    plt.plot(time, vr, marker="o", label="vr")
    plt.plot(time, vt, marker="o", label="vt")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.xlabel("time [-]")
    plt.ylabel("velocity [-]")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_dir + "velocity" + ".png")

    plt.figure()
    plt.plot(time, (ur1 - ur2), marker="o", label="ur")
    plt.plot(time, (ut1 - ut2), marker="o", label="ut")
    # plt.plot(time, ur1, marker="o", label="ur1")
    # plt.plot(time, ur2, marker="o", label="ur2")
    # plt.plot(time, ut1, marker="o", label="ut1")
    # plt.plot(time, ut2, marker="o", label="ut2")
    plt.grid()
    plt.xlabel("time [-]")
    plt.ylabel("thrust [-]")
    # plt.ylim([-0.02, 0.6])
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_dir + "thrust" + ".png")

    theta = integrate.cumulative_trapezoid(vt / r, time, initial=0)
    theta_f = interpolate.interp1d(time, theta)
    r_f = interpolate.interp1d(time, r)
    time_fine = np.linspace(time[0], time[-1], 1000)
    r_fine = r_f(time_fine)
    theta_fine = theta_f(time_fine)
    fig = plt.figure()
    # plt.plot(r*np.cos(theta), r*np.sin(theta))
    plt.plot(r_fine * np.cos(theta_fine), r_fine * np.sin(theta_fine))
    ax = fig.add_subplot(111)
    circle0 = plt.Circle((0.0, 0.0), 1.0, ls="--", fill=False, fc="none")
    circlef = plt.Circle((0.0, 0.0), 4.0, ls="--", fill=False, fc="none")
    ax.add_patch(circle0)
    ax.add_patch(circlef)
    plt.grid()
    plt.axis("equal")
    plt.ylim((-4.1, 4.1))
    if flag_savefig:
        plt.savefig(savefig_dir + "trajectry" + ".png")

    plt.show()
    return prob


if __name__ == "__main__":
    main()
