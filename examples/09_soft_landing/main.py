"""Minimum-propellant vertical lunar landing, from coast to powered descent.

    h'=v, v'=thrust/m-g, m'=-thrust/exhaust_velocity (variable mass).

Up is positive. Both initial and final velocities are specified. The constant
mass lesson minimizes impulse; the variable mass lesson maximizes final mass.
Two phases are initialized at the expected coast/burn switch; thrust is free
in both. The model uses constant gravity, no air, no ignition cost and no minimum
thrust. It is a trajectory, not a feedback law.
Run: uv run python examples/09_soft_landing/main.py --preset variable_mass --plot
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from OpenGoddard import DynamicsModel, LinearCallback, Problem

# Support direct execution from any working directory as well as package imports.
if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._shared import collect, node_count, positive, run_cli


@dataclass(frozen=True)
class Parameters:
    initial_altitude: float = 100.0  # m above the landing site
    initial_velocity: float = -10.0  # m/s; up is positive
    gravity: float = 1.62  # m/s^2, constant lunar-gravity approximation
    initial_mass: float = 1000.0  # kg
    max_thrust: float = 4000.0  # N
    variable_mass: bool = False
    exhaust_velocity: float = 3000.0  # m/s; ideal constant effective exhaust speed
    dry_mass: float = 700.0  # kg; lower bound in variable-mass mode

    def __post_init__(self):
        positive(
            initial_altitude=self.initial_altitude,
            gravity=self.gravity,
            initial_mass=self.initial_mass,
            max_thrust=self.max_thrust,
            exhaust_velocity=self.exhaust_velocity,
            dry_mass=self.dry_mass,
        )
        if not isinstance(self.variable_mass, bool):
            raise ValueError("variable_mass must be a boolean")
        if not np.isfinite(self.initial_velocity) or self.initial_velocity > 0:
            raise ValueError(
                "initial_velocity must be finite and <= 0 (up is positive)"
            )
        if self.dry_mass >= self.initial_mass:
            raise ValueError("dry_mass must be smaller than initial_mass")
        braking = self.max_thrust / self.initial_mass - self.gravity
        if braking <= 0:
            raise ValueError("max_thrust must exceed initial weight")
        # A conservative domain that works for the constant-mass lesson too.
        if self.initial_altitude <= self.initial_velocity**2 / (2 * braking):
            raise ValueError("altitude must exceed the constant-mass stopping distance")


PRESETS = {
    "default": Parameters(),
    "variable_mass": Parameters(variable_mass=True),
    "lower_thrust": Parameters(max_thrust=3000.0),
    "higher_start": Parameters(initial_altitude=200.0),
}


def constant_mass_reference(p):
    """Exact coast/full-thrust landing with unlimited propellant, fixed mass."""
    g = p.gravity
    b = p.max_thrust / p.initial_mass - g
    switch_speed = np.sqrt(
        (p.initial_velocity**2 + 2 * g * p.initial_altitude) * b / (b + g)
    )
    coast = (switch_speed + p.initial_velocity) / g
    burn = switch_speed / b
    return {
        "coast_duration": float(coast),
        "burn_duration": float(burn),
        "duration": float(coast + burn),
        "impulse": float(p.max_thrust * burn),
    }


def burn_state(p, initial_altitude, initial_velocity, time):
    """Exact full-thrust arc, used only as a reference and initial guess."""
    rate = p.max_thrust / p.exhaust_velocity
    mass = p.initial_mass - rate * time
    log_ratio = np.log1p(-rate * time / p.initial_mass)
    velocity = initial_velocity - p.gravity * time - p.exhaust_velocity * log_ratio
    altitude = (
        initial_altitude
        + initial_velocity * time
        - p.gravity * time**2 / 2
        + p.exhaust_velocity * (time + mass / rate * log_ratio)
    )
    return altitude, velocity, mass


def landing_reference(p):
    """Coast/full-thrust reference, with an exact rocket-equation burn if needed."""
    if not p.variable_mass:
        return constant_mass_reference(p)
    rate = p.max_thrust / p.exhaust_velocity
    max_burn = (p.initial_mass - p.dry_mass) / rate

    def coast_time(burn):
        delta_v = -p.exhaust_velocity * np.log1p(-rate * burn / p.initial_mass)
        return (p.initial_velocity + delta_v) / p.gravity - burn

    def terminal_height(burn):
        tc = coast_time(burn)
        hs = p.initial_altitude + p.initial_velocity * tc - p.gravity * tc**2 / 2
        vs = p.initial_velocity - p.gravity * tc
        return burn_state(p, hs, vs, burn)[0]

    if terminal_height(max_burn) > 0:
        raise ValueError("insufficient propellant for this coast/burn landing")
    burn = brentq(terminal_height, 0, max_burn, xtol=1e-12)
    coast = coast_time(burn)
    if coast <= 0:
        raise ValueError("parameters require a landing without an initial coast")
    return {
        "coast_duration": float(coast),
        "burn_duration": float(burn),
        "duration": float(coast + burn),
        "propellant_used": float(rate * burn),
        "impulse": float(p.max_thrust * burn),
    }


def pointwise_dynamics(x, u, time, obj, section, xp):
    mass = x[2] if obj.variable_mass else obj.initial_mass
    acceleration = u[0] / mass - obj.gravity
    if obj.variable_mass:
        return xp.stack((x[1], acceleration, -u[0] / obj.exhaust_velocity))
    return xp.stack((x[1], acceleration))


@LinearCallback
def equality(problem, p):
    c = problem.condition()
    c.equal(problem.states(0, 0)[0], p.initial_altitude, unit=p.initial_altitude)
    c.equal(problem.states(1, 0)[0], p.initial_velocity, unit=problem.unit_states[0][1])
    c.equal(problem.states(0, -1)[-1], 0, unit=p.initial_altitude)
    c.equal(problem.states(1, -1)[-1], 0, unit=problem.unit_states[-1][1])
    if p.variable_mass:
        c.equal(problem.states(2, 0)[0], p.initial_mass, unit=p.initial_mass)
    return c()


def make_problem(parameters=PRESETS["default"], *, nodes=12):
    node_count(nodes)
    p = parameters
    ref = landing_reference(p)
    tc, tb = ref["coast_duration"], ref["burn_duration"]
    problem = Problem(
        [0, tc, tc + tb],
        [nodes] * 2,
        [3 if p.variable_mass else 2] * 2,
        [1] * 2,
        maxIterator=1,
    )
    time_scale = np.sqrt(p.initial_altitude / p.gravity)
    speed_scale = np.sqrt(p.initial_altitude * p.gravity)
    problem.set_unit_time(time_scale)
    problem.set_unit_states_all_section(0, p.initial_altitude)
    problem.set_unit_states_all_section(1, speed_scale)
    problem.set_unit_controls_all_section(0, p.max_thrust)
    hs = p.initial_altitude + p.initial_velocity * tc - p.gravity * tc**2 / 2
    vs = p.initial_velocity - p.gravity * tc
    for section in range(2):
        t = problem.time[section] - problem.time_start(section)
        thrust = 0.0 if section == 0 else p.max_thrust
        if section == 0:
            h = p.initial_altitude + p.initial_velocity * t - p.gravity * t**2 / 2
            v = p.initial_velocity - p.gravity * t
        else:
            braking = p.max_thrust / p.initial_mass - p.gravity
            h, v = hs + vs * t + braking * t**2 / 2, vs + braking * t
            if p.variable_mass:
                h, v, _ = burn_state(p, hs, vs, t)
        problem.set_states(0, section, h)
        problem.set_states(1, section, v)
        problem.set_controls(0, section, np.full(nodes, thrust))
        problem.set_states_bounds(0, section, 0, p.initial_altitude)
        problem.set_states_bounds(1, section, None, 0)
        problem.set_controls_bounds(0, section, 0, p.max_thrust)
        if p.variable_mass:
            problem.set_unit_states(2, section, p.initial_mass)
            problem.set_states(
                2, section, p.initial_mass - thrust / p.exhaust_velocity * t
            )
            problem.set_states_bounds(2, section, p.dry_mass, p.initial_mass)
        problem.set_time_final_bounds(section, 1e-5 * time_scale, 10 * time_scale)

    @LinearCallback
    def inequality(prob, obj):
        c = prob.condition()
        for section in range(2):
            c.lower_bound(
                prob.time_final(section) - prob.time_start(section),
                1e-5 * time_scale,
                unit=time_scale,
            )
        return c()

    problem.dynamics = [DynamicsModel(pointwise_dynamics)] * 2
    problem.equality = equality
    problem.inequality = inequality
    if p.variable_mass:
        problem.cost = LinearCallback(
            lambda prob, obj: 1 - prob.states(2, -1)[-1] / obj.initial_mass
        )
    else:
        # Normalize for the NLP; exported objective is this dimensionless value.
        problem.running_cost = lambda prob, obj: (
            prob.controls_all_section(0) / (obj.max_thrust * time_scale)
        )
    return problem


def solve(parameters=PRESETS["default"], *, nodes=12, samples=201, maxiter=300):
    problem = make_problem(parameters, nodes=nodes)
    problem.solve(parameters, derivatives="structured", maxiter=maxiter, ftol=1e-9)
    p = parameters
    ref = constant_mass_reference(p)
    reference = {"constant_mass": ref}
    if p.variable_mass:
        reference["variable_mass"] = landing_reference(p)
    trajectory = collect(
        problem,
        p,
        example="soft_landing",
        states=["altitude", "velocity", "mass"]
        if p.variable_mass
        else ["altitude", "velocity"],
        controls=["thrust"],
        units={
            "time": "s",
            "altitude": "m",
            "velocity": "m/s",
            "mass": "kg",
            "thrust": "N",
            "objective": "1",
        },
        coordinates={
            "altitude": "positive upward, surface at zero",
            "velocity": "positive upward",
        },
        phase_names=["coast", "burn"],
        reference=reference,
        samples=samples,
    )
    if p.variable_mass:
        trajectory.diagnostics["propellant_used_kg"] = float(
            p.initial_mass - problem.states(2, -1)[-1]
        )
    else:
        trajectory.diagnostics["impulse_Ns"] = float(
            problem.result.fun * p.max_thrust * problem.unit_time
        )
    return trajectory


def plot(trajectory):
    import matplotlib.pyplot as plt

    names = ["altitude", "velocity", "thrust"]
    if trajectory.parameters["variable_mass"]:
        names.append("mass")
    fig, axes = plt.subplots(len(names), 1, sharex=True, layout="constrained")
    for phase in trajectory.phases:
        data = phase["samples"]
        for ax, name in zip(axes, names, strict=True):
            ax.plot(data["time"], data[name], label=phase["name"])
            ax.set_ylabel(f"{name} [{trajectory.units[name]}]")
            ax.grid()
    axes[0].legend()
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Vertical soft landing (up is positive)")
    return fig


def main(argv=None):
    run_cli(Parameters, PRESETS, solve, plot, __doc__, argv=argv)


if __name__ == "__main__":
    main()
