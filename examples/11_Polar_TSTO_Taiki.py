# -*- coding: utf-8 -*-
# Copyright 2017 Interstellar Technologies Inc. All Rights Reserved.

from __future__ import print_function
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from OpenGoddard.optimize import Problem, Guess, Condition, Dynamics


class Rocket:
    # Atmosphere Parameter
    # Use US Standard Atmosphere 1976
    stdAtmo = np.loadtxt("./11_Polar_TSTO_Taiki/US_standard_atmosphere.csv",delimiter=",",skiprows=2)
    stdAltitude = stdAtmo[:,0] * 1000.0 #converted to km -> m
    stdPressure= stdAtmo[:,2] # [Pa]
    stdDensity= stdAtmo[:,3] # [kg/m3]
    stdSoundSpeed = stdAtmo[:,4] # [m/s]
    # 線形補完用
    # 高度範囲外(<0, 86<)はfill_valueが外挿
    airPressure = interpolate.interp1d(stdAltitude, stdPressure, bounds_error = False, fill_value = (stdPressure[0], 0.0))
    airDensity = interpolate.interp1d(stdAltitude, stdDensity, bounds_error = False, fill_value = (stdDensity[0], 0.0))
    airSound = interpolate.interp1d(stdAltitude, stdSoundSpeed, bounds_error = False, fill_value = (stdSoundSpeed[0], stdSoundSpeed[-1]))

    # Drag Coefficient
    CdLog = np.loadtxt("./11_Polar_TSTO_Taiki/Cd.csv", delimiter=",", skiprows=1)
    Cd = interpolate.interp1d(CdLog[:,0], CdLog[:,1],fill_value="extrapolate")

    def __init__(self):
        # Earth Parameter
        self.GMe = 3.986004418 * 10**14  # Earth gravitational constant [m^3/s^2]
        self.Re = 6371.0 * 1000  # Earth Radius [m]
        self.g0 = 9.80665  # Gravitational acceleration on Earth surface [m/s^2]
        
        # Target Parameter
        self.Htarget = 561.0 * 1000  # Altitude [m]
        self.Rtarget = self.Re + self.Htarget  # Orbit Radius [m]
        self.Vtarget = np.sqrt(self.GMe / self.Rtarget)  # [m/s]     

        # Launch Site Parameter
        self.lat_taiki = 42.506167 # [deg]
        self.Vt_equator = 1674.36 # [km/h]
        self.Vt_taiki = self.Vt_equator * np.cos(self.lat_taiki * np.pi / 180.0) * 1000.0 / 3600.0 # Radial Velocity of Earth Surface [m/s]
        self.inclination = 96.7 # [deg]
        self.V0 = self.Vt_taiki * np.cos(-self.inclination * np.pi / 180.0) # [m/s]
        self.H0 = 10.0  # Initial Altitude [m]        

        # Structure Parameter
        # Mdryがパラメータ
        self.Mdry = [1300.0, 220.0] # Dry Mass [kg], [1st stage, 2nd stage]        
        self.beta = [10.0, 15.0] # Structure Efficienty [%], [1st stage, 2nd stage]      
        self.Mpayload = 100.0 # Payload Mass [kg]

        self.M0 = [self.Mdry[0] / self.beta[0] * 100.0, self.Mdry[1] / self.beta[1] * 100.0] # Initial Stage Mass [kg], [1st stage, 2nd stage]
        self.Mp = [self.M0[0] - self.Mdry[0], self.M0[1] - self.Mdry[1]] # Propellant Mass [kg], [1st stage, 2nd stage]
        self.M0[1] = self.M0[1] + self.Mpayload
        self.Minit = self.M0[0] + self.M0[1] # Initial Total Mass [kg]

        self.d = [1.8, 1.8] # Diameter [m], [1st stage, 2nd stage]
        self.A = [0.25 * self.d[0] ** 2 * np.pi, 0.25 * self.d[1] ** 2 * np.pi]  # Projected Area [m^2], [1st stage, 2nd stage]
        
        # Engine Parameter
        self.Cluster = 9
        self.Isp = [261.0 + 0.0, 322.0 + 0.0]  # Specific Impulse [s], [1st stage at SL, 2nd stage at vac]
        self.dth = [53.9, 53.9] # Throat Diameter [mm], [1st stage, 2nd stage]
        self.Ath = [0.25 * (self.dth[0] / 1000.0) ** 2 * np.pi, 0.25 * (self.dth[1] / 1000.0) ** 2 * np.pi] # Throat Area [m^2], [1st stage, 2nd stage]
        self.AR = [20.0, 140.0] # Area Ratio, [1st stage, 2nd stage]
        self.Ae = [self.Ath[0] * self.AR[0] * self.Cluster, self.Ath[1] * self.AR[1]] # Exit Area [m^2], [1st stage, 2nd stage]
        # =======
        self.ThrustMax = [33.3, 4.2] # Maximum Thrust [ton], [1st stage at SL, 2nd stage at vac]
        self.ThrustMax = [self.ThrustMax[0] * self.g0 * 1000.0, self.ThrustMax[1] * self.g0 * 1000.0] # [N]

        # self.ThrustLevel = 1.8 # [G] M0[0] * n G
        # self.ThrustMax = [self.M0[0] * self.ThrustLevel * self.g0, self.M0[0] * self.ThrustLevel / self.Cluster * self.g0 + self.airPressure(self.Htarget) * self.Ae[1]] # Maximum Thrust [N], [1st stage at SL, 2nd stage at vac]
        # =======

        self.refMdot = [self.ThrustMax[0] / (self.Isp[0] * self.g0), self.ThrustMax[1] / (self.Isp[1] * self.g0)] # Isp補正用参照値
        
        self.MaxQ = 500000.0  # Pa
        self.MaxG = 20.0  # G


def dynamics(prob, obj, section):
    R     = prob.states(0, section) # Orbit Radius [m]
    theta = prob.states(1, section) # 
    Vr    = prob.states(2, section)
    Vt    = prob.states(3, section)
    m     = prob.states(4, section)
    Tr    = prob.controls(0, section)
    Tt    = prob.controls(1, section)

    g0 = obj.g0
    g = obj.g0 * (obj.Re / R)**2  # [m/s2]   
    rho = obj.airDensity(R - obj.Re)
    Mach = np.sqrt(Vr**2 + Vt**2) / obj.airSound(R - obj.Re)
    Cd = obj.Cd(Mach)
    dThrust = [(obj.airPressure(obj.H0) - obj.airPressure(R - obj.Re)) * obj.Ae[0], obj.airPressure(R - obj.Re) * obj.Ae[1]]
    Isp = obj.Isp[section] + dThrust[section] / (obj.refMdot[section] * g0)
    
    # US standard atmosphereだと86 km以降はrho = 0でDrag = 0
    Dr = 0.5 * rho * Vr * np.sqrt(Vr**2 + Vt**2) * Cd * obj.A[section]  # [N]
    Dt = 0.5 * rho * Vt * np.sqrt(Vr**2 + Vt**2) * Cd * obj.A[section]  # [N]

    dx = Dynamics(prob, section)
    dx[0] = Vr
    dx[1] = Vt / R
    dx[2] = Tr / m - Dr / m - g + Vt**2 / R
    dx[3] = Tt / m - Dt / m - (Vr * Vt) / R
    dx[4] = - np.sqrt(Tr**2 + Tt**2) / (Isp * g0)

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
    result.equal(R0[0], obj.Re + obj.H0, unit=unit_R) # 初期地表
    result.equal(theta0[0], 0.0)
    result.equal(Vr0[0], 0.0, unit=unit_V)
    result.equal(Vt0[0], obj.V0 , unit=unit_V)
    result.equal(m0[0], obj.Minit, unit=unit_m) # (1st stage and 2nd stage and Payload) initial
    
    # knotting condition
    result.equal(m1[0], obj.M0[1], unit=unit_m) # (2nd stage + Payload) initial    
    result.equal(R1[0], R0[-1], unit=unit_R)
    result.equal(theta1[0], theta0[-1])
    result.equal(Vr1[0], Vr0[-1], unit=unit_V)
    result.equal(Vt1[0], Vt0[-1], unit=unit_V)

    # Target Condition
    result.equal(R1[-1], obj.Rtarget, unit=unit_R) # Radius
    result.equal(Vr[-1], 0.0, unit=unit_V) # Radius Velocity
    result.equal(Vt[-1], obj.Vtarget, unit=unit_V)
   

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

    rho = obj.airDensity(R - obj.Re)
    Mach = np.sqrt(Vr**2 + Vt**2) / obj.airSound(R - obj.Re)
    Cd = obj.Cd(Mach)    
    Dr0 = 0.5 * rho * Vr * np.sqrt(Vr**2 + Vt**2) * Cd * obj.A[0]  # [N]
    Dt0 = 0.5 * rho * Vt * np.sqrt(Vr**2 + Vt**2) * Cd * obj.A[0]  # [N]
    Dr1 = 0.5 * rho * Vr * np.sqrt(Vr**2 + Vt**2) * Cd * obj.A[1]  # [N]
    Dt1 = 0.5 * rho * Vt * np.sqrt(Vr**2 + Vt**2) * Cd * obj.A[1]  # [N]
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
    dThrust0 = (obj.airPressure(obj.H0) - obj.airPressure(R0 - obj.Re)) * obj.Ae[0]
    dThrust1 = obj.airPressure(R1 - obj.Re) * obj.Ae[1]

    result = Condition()

    # lower bounds
    result.lower_bound(R, obj.Re, unit=prob.unit_states[0][0]) # 地表以下
    result.lower_bound(m0, obj.Mdry[0] + obj.M0[1], unit=prob.unit_states[0][4]) # 乾燥質量以下
    result.lower_bound(m1, obj.Mdry[1], unit=prob.unit_states[0][4])
    result.lower_bound(Tr, -obj.ThrustMax[1], unit=prob.unit_controls[0][0])
    result.lower_bound(Tt, -obj.ThrustMax[1], unit=prob.unit_controls[0][0])

    # upper bounds
    result.upper_bound(m0, obj.Minit, unit=prob.unit_states[0][4]) # 初期質量以上
    result.upper_bound(m1, obj.M0[1], unit=prob.unit_states[0][4])
    result.upper_bound(Tr0, obj.ThrustMax[0] + dThrust0, unit=prob.unit_controls[0][0])
    result.upper_bound(Tt0, obj.ThrustMax[0] + dThrust0, unit=prob.unit_controls[0][0])
    result.upper_bound(T0, obj.ThrustMax[0] + dThrust0, unit=prob.unit_controls[0][0])
    result.upper_bound(Tr1, obj.ThrustMax[1] + dThrust1, unit=prob.unit_controls[0][0])
    result.upper_bound(Tt1, obj.ThrustMax[1] + dThrust1, unit=prob.unit_controls[0][0])
    result.upper_bound(T1, obj.ThrustMax[1] + dThrust1, unit=prob.unit_controls[0][0])
    result.upper_bound(q, obj.MaxQ, unit = prob.unit_states[0][0])
    result.upper_bound(a_mag0, obj.MaxG * obj.g0)
    result.upper_bound(a_mag1, obj.MaxG * obj.g0)

    return result()


def cost(prob, obj):
    m1 = prob.states(4, 1)
    return -m1[-1] / prob.unit_states[1][4]


# ========================

# Program Starting Point
time_init = [0.0, 200, 800]
n = [20, 30]
num_states = [5, 5]
num_controls = [2, 2]
max_iteration = 90

flag_savefig = True
savefig_file = "./11_Polar_TSTO_Taiki/TSTO_"

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
Vt_init = Guess.linear(prob.time_all_section, obj.V0, obj.Vtarget)
# Guess.plot(prob.time_all_section, V_init, "Velocity", "time", "Velocity")

# mass profile -0.6
M_init0 = Guess.cubic(prob.time_all_section, obj.Minit, 0.0, obj.Mdry[0] + obj.M0[1], 0.0)
M_init1 = Guess.cubic(prob.time_all_section, obj.M0[1], 0.0, obj.Mdry[1], 0.0)
M_init = np.hstack((M_init0, M_init1))
# Guess.plot(prob.time_all_section, M_init, "Mass", "time", "Mass")
# if(flag_savefig):plt.savefig(savefig_file + "guess_mass" + ".png")

# thrust profile
# T_init = Guess.zeros(prob.time_all_section)
Tr_init0 = Guess.cubic(prob.time[0], obj.ThrustMax[0]*9/10, 0.0, 0.0, 0.0)
Tr_init1 = Guess.cubic(prob.time[1], obj.ThrustMax[1]*9/10, 0.0, 0.0, 0.0)
Tr_init = np.hstack((Tr_init0, Tr_init1))
# Tt_init = Guess.cubic(prob.time_all_section, 0.0, 0.0, 0.0, 0.0)
Tt_init0 = Guess.cubic(prob.time[0], obj.ThrustMax[0]/10, 0.0, 0.0, 0.0)
Tt_init1 = Guess.cubic(prob.time[1], obj.ThrustMax[1]/10, 0.0, 0.0, 0.0)
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

def display_func():
    R = prob.states_all_section(0)
    theta = prob.states_all_section(1)
    m = prob.states_all_section(4)
    ts = prob.time_knots()
    tf = prob.time_final(-1)
    print("m0          : {0:.5f}".format(m[0]))
    print("mf          : {0:.5f}".format(m[-1]))
    print("mdry        : {0:.5f}".format(obj.Mdry[0]))
    print("payload     : {0:.5f}".format(m[-1] - obj.Mdry[1]))
    print("max altitude: {0:.5f}".format(R[-1] - obj.Re))
    print("MECO time   : {0:.3f}".format(ts[1]))
    print("final time  : {0:.3f}".format(tf))

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

R0     = prob.states(0, 0)
R1     = prob.states(0, 1)
Tr0    = prob.controls(0, 0)
Tr1    = prob.controls(0, 1)
Tt0    = prob.controls(1, 0)
Tt1    = prob.controls(1, 1)

# ------------------------
# Calculate necessary variables
rho = obj.airDensity(R - obj.Re)
Mach = np.sqrt(Vr**2 + Vt**2) / obj.airSound(R - obj.Re)
Cd = obj.Cd(Mach)    
Dr = 0.5 * rho * Vr * np.sqrt(Vr**2 + Vt**2) * Cd * obj.A[0]  # [N]
Dt = 0.5 * rho * Vt * np.sqrt(Vr**2 + Vt**2) * Cd * obj.A[0]  # [N]
g = obj.g0 * (obj.Re / R)**2  # [m/s2]

# dynamic pressure
q = 0.5 * rho * (Vr**2 + Vt**2)  # [Pa]
# accelaration
a_r = (Tr - Dr) / m / obj.g0
a_t = (Tt - Dt) / m / obj.g0
a_mag = np.sqrt(a_r**2 + a_t**2) / obj.g0  # [G]
# Thrust
T = np.sqrt(Tr**2 + Tt**2)

dThrust0 = (obj.airPressure(obj.H0) - obj.airPressure(R0 - obj.Re)) * obj.Ae[0]
dThrust1 = obj.airPressure(R1 - obj.Re) * obj.Ae[1]
Isp0 = obj.Isp[0] + dThrust0 / (obj.refMdot[0] * obj.g0)
Isp1 = obj.Isp[1] + dThrust1 / (obj.refMdot[1] * obj.g0)
Thrust_SL = T - np.append(dThrust0, dThrust1)
np.savetxt(savefig_file + "Thrust_Log" + ".csv", np.hstack((time, Thrust_SL, T, Tr, Tt)), delimiter=',')

# ------------------------
# Visualizetion
plt.close("all")

plt.figure()
plt.title("Altitude profile")
plt.plot(time, (R - obj.Re) / 1000, marker="o", label="Altitude")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Altitude [km]")
plt.legend(loc="best")
if(flag_savefig): plt.savefig(savefig_file + "altitude" + ".png")
np.savetxt(savefig_file + "Altitude_Log" + ".csv", np.hstack((time, (R - obj.Re))), delimiter=',')

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
np.savetxt(savefig_file + "Velocity_Log" + ".csv", np.hstack((time, Vr, Vt)), delimiter=',')

plt.figure()
plt.title("Mass")
plt.plot(time, m, marker="o", label="Mass")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Mass [kg]")
plt.legend(loc="best")
if(flag_savefig): plt.savefig(savefig_file + "mass" + ".png")
np.savetxt(savefig_file + "Mass_Log" + ".csv", np.hstack((time, m)), delimiter=',')

plt.figure()
plt.title("Acceleration")
plt.plot(time, a_r, marker="o", label="Acc r")
plt.plot(time, a_t, marker="o", label="Acc t")
plt.plot(time, a_mag, marker="o", label="Acc")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Acceleration [G]")
plt.legend(loc="best")
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

plt.figure()
plt.title("DeltaThrust profile")
plt.plot(time, np.append(dThrust0, dThrust1), marker="o", label="dThrust")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("dThrust [N]")
plt.legend(loc="best")
if(flag_savefig): plt.savefig(savefig_file + "dforce" + ".png")

plt.figure()
plt.title("Isp profile")
plt.plot(time, np.append(Isp0, Isp1), marker="o", label="Isp")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Isp [s]")
plt.legend(loc="best")
if(flag_savefig): plt.savefig(savefig_file + "Isp" + ".png")