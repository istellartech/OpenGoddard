# -*- coding: utf-8 -*-
# Copyright 2017 Interstellar Technologies Inc. All Rights Reserved.

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from OpenGoddard.optimize import Problem, Guess, Condition, Dynamics


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
    r   = prob.states(0, section)
    vr  = prob.states(1, section)
    vt  = prob.states(2, section)
    ur1 = prob.controls(0, section)
    ur2 = prob.controls(1, section)
    ut1 = prob.controls(2, section)
    ut2 = prob.controls(3, section)

    dx = Dynamics(prob, section)
    dx[0] = vr
    dx[1] = vt**2 / r - 1 / r**2 + (ur1 - ur2)
    dx[2] = - vr * vt / r + (ut1 - ut2)
    return dx()


def equality(prob, obj):
    r   = prob.states_all_section(0)
    vr  = prob.states_all_section(1)
    vt  = prob.states_all_section(2)
    ur1 = prob.controls_all_section(0)
    ur2 = prob.controls_all_section(1)
    ut1 = prob.controls_all_section(2)
    ut2 = prob.controls_all_section(3)
    tf  = prob.time_final(-1)

    result = Condition()

    # event condition
    result.add(r[0] - obj.r0)
    result.add(vr[0] - obj.vr0)
    result.add(vt[0] - obj.vt0)
    result.add(r[-1] - obj.rf)
    result.add(vr[-1] - obj.vrf)
    result.add(vt[-1] - obj.vtf)

    return result()


def inequality(prob, obj):
    r   = prob.states_all_section(0)
    vr  = prob.states_all_section(1)
    vt  = prob.states_all_section(2)
    ur1 = prob.controls_all_section(0)
    ur2 = prob.controls_all_section(1)
    ut1 = prob.controls_all_section(2)
    ut2 = prob.controls_all_section(3)
    tf  = prob.time_final(-1)

    result = Condition()

    # lower bounds
    result.add(r - obj.r0)
    result.add(ur1 - 0.0)
    result.add(ut1 - 0.0)
    result.add(ur2 - 0.0)
    result.add(ut2 - 0.0)
    result.add(tf - 0.0)

    # upper bounds
    result.add(obj.rf - r)
    result.add(obj.u_max - ur1)
    result.add(obj.u_max - ut1)
    result.add(obj.u_max - ur2)
    result.add(obj.u_max - ut2)
    result.add(obj.tf_max - tf)

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
plt.close("all")
plt.ion()
# Program Starting Point
time_init = [0.0, 10.0]
n = [100]
num_states = [3]
num_controls = [4]
max_iteration = 10

flag_savefig = True

savefig_dir = "10_Low_Thrust_Orbit_Transfer/"

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

vt_init = Guess.linear(prob.time_all_section, obj.vt0, obj.vtf)
# Guess.plot(prob.time_all_section, theta_init, "vt", "time", "vt")
# if(flag_savefig):plt.savefig(savefig_dir + "guess_vt" + savefig_add + ".png")

ur1_init = Guess.linear(prob.time_all_section, obj.u_max, obj.u_max)
# Guess.plot(prob.time_all_section, ur1_init, "ur1", "time", "ur1")
# if(flag_savefig):plt.savefig(savefig_dir + "guess_ur1" + savefig_add + ".png")

ut1_init = Guess.linear(prob.time_all_section, obj.u_max, obj.u_max)
# Guess.plot(prob.time_all_section, ut1_init, "ut1", "time", "ut1")
# if(flag_savefig):plt.savefig(savefig_dir + "guess_ut1" + savefig_add + ".png")

prob.set_states_all_section(0, r_init)
prob.set_states_all_section(1, vr_init)
prob.set_states_all_section(2, vt_init)
prob.set_controls_all_section(0, ur1_init)
prob.set_controls_all_section(2, ut1_init)

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
    print("tf: {0:.5f}".format(tf))


prob.solve(obj, display_func, ftol=1e-12)

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
if(flag_savefig): plt.savefig(savefig_dir + "r" + ".png")

plt.figure()
plt.plot(time, vr, marker="o", label="vr")
plt.plot(time, vt, marker="o", label="vt")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [-]")
plt.ylabel("velocity [-]")
plt.legend(loc="best")
if(flag_savefig): plt.savefig(savefig_dir + "velocity" + ".png")

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
if(flag_savefig): plt.savefig(savefig_dir + "thrust" + ".png")


from scipy import integrate
from scipy import interpolate
theta = integrate.cumtrapz(vt / r, time, initial=0)
theta_f = interpolate.interp1d(time, theta)
r_f = interpolate.interp1d(time, r)
time_fine = np.linspace(time[0], time[-1], 1000)
r_fine = r_f(time_fine)
theta_fine = theta_f(time_fine)
fig = plt.figure()
# plt.plot(r*np.cos(theta), r*np.sin(theta))
plt.plot(r_fine*np.cos(theta_fine), r_fine*np.sin(theta_fine))
ax = fig.add_subplot(111)
circle0 = plt.Circle((0.0, 0.0), 1.0, ls="--", fill=False, fc='none')
circlef = plt.Circle((0.0, 0.0), 4.0, ls="--", fill=False, fc='none')
ax.add_patch(circle0)
ax.add_patch(circlef)
plt.grid()
plt.axis('equal')
plt.ylim((-4.1, 4.1))
if(flag_savefig): plt.savefig(savefig_dir + "trajectry" + ".png")


plt.show()
