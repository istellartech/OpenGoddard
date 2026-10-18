"""Brachistochrone between fixed points, with an optional maximum depth.

    x'=v*sin(theta), depth'=v*cos(theta), v'=g*cos(theta).

Starting from rest, conservation of energy gives depth=v**2/(2*g).
Only x and v are transcribed; depth is reconstructed for export. This removes
redundant dynamics along the horizontal floor segment.

Depth is positive DOWNWARD; theta is measured from the downward vertical.
An ideal frictionless track supplies the normal force. There is no curvature
limit or friction. Unlike the historical free-height example, both terminal
coordinates are fixed here so constrained and unconstrained runs are comparable.
An active floor uses descent / horizontal / ascent phases, with free durations.
Run: uv run python examples/04_constrained_descent/main.py --preset no_floor --plot
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
    distance: float = 1.0  # m, horizontal displacement
    final_depth: float = 0.2  # m below the initial point
    gravity: float = 9.81  # m/s^2
    max_depth: float | None = 0.3  # m below the initial point; None removes floor

    def __post_init__(self):
        positive(
            distance=self.distance, final_depth=self.final_depth, gravity=self.gravity
        )
        if self.max_depth is not None:
            positive(max_depth=self.max_depth)
            if self.max_depth <= self.final_depth:
                raise ValueError("max_depth must exceed final_depth (or be null)")


PRESETS = {
    "default": Parameters(),
    "no_floor": Parameters(max_depth=None),
    "inactive_floor": Parameters(max_depth=0.8),
    "shallow_floor": Parameters(max_depth=0.23),
    "lower_gravity": Parameters(gravity=1.62),
}


def reference_solution(p):
    """Cycloid reference, adding a horizontal segment when the floor is active.

    x=a*(phi-sin(phi)), depth=a*(1-cos(phi)), t=sqrt(a/g)*phi.
    At a horizontal floor the tangent angle is pi/2, so a=max_depth/2.
    """
    ratio = p.distance / p.final_depth
    phi = brentq(
        lambda angle: (angle - np.sin(angle)) / (1 - np.cos(angle)) - ratio,
        1e-5,
        2 * np.pi - 1e-5,
    )
    a = p.final_depth / (1 - np.cos(phi))
    peak_depth = a * (1 - np.cos(min(phi, np.pi)))
    active = p.max_depth is not None and p.max_depth < peak_depth
    if active:
        a = p.max_depth / 2
        phi = 2 * np.pi - np.arccos(1 - p.final_depth / a)
        cruise_distance = p.distance - a * (phi - np.sin(phi))
        durations = [
            np.pi * np.sqrt(a / p.gravity),
            cruise_distance / np.sqrt(2 * p.gravity * p.max_depth),
            (phi - np.pi) * np.sqrt(a / p.gravity),
        ]
    else:
        cruise_distance = 0.0
        durations = [phi * np.sqrt(a / p.gravity)]
    return {
        "floor_active": bool(active),
        "cycloid_scale": float(a),
        "final_phi": float(phi),
        "cruise_distance": float(cruise_distance),
        "phase_durations": [float(t) for t in durations],
        "minimum_time": float(sum(durations)),
    }


def pointwise_dynamics(x, u, time, obj, section, xp):
    theta = u[0]
    return xp.stack((x[1] * xp.sin(theta), obj.gravity * xp.cos(theta)))


@LinearCallback
def equality(problem, p):
    c = problem.condition()
    length_scale = problem.unit_states[0][0]
    for state in range(2):
        c.equal(problem.states(state, 0)[0], 0, unit=problem.unit_states[0][state])
    c.equal(problem.states(0, -1)[-1], p.distance, unit=length_scale)
    c.equal(
        problem.states(1, -1)[-1],
        np.sqrt(2 * p.gravity * p.final_depth),
        unit=problem.unit_states[0][1],
    )
    if problem.number_of_section == 3:
        # Entry and exit of the floor segment; the dynamics determine theta.
        floor_speed = np.sqrt(2 * p.gravity * p.max_depth)
        c.equal(problem.states(1, 0)[-1], floor_speed, unit=problem.unit_states[0][1])
        c.equal(problem.states(1, 1)[-1], floor_speed, unit=problem.unit_states[0][1])
    return c()


def make_problem(parameters=PRESETS["default"], *, nodes=12):
    node_count(nodes)
    p = parameters
    ref = reference_solution(p)
    durations = ref["phase_durations"]
    phases = len(durations)
    knots = np.r_[0, np.cumsum(durations)]
    problem = Problem(
        knots, [nodes] * phases, [2] * phases, [1] * phases, maxIterator=1
    )
    length_scale = max(p.distance, p.final_depth)
    time_scale = np.sqrt(length_scale / p.gravity)
    problem.set_unit_time(time_scale)
    problem.set_unit_states_all_section(0, length_scale)
    problem.set_unit_states_all_section(1, np.sqrt(length_scale * p.gravity))
    a = ref["cycloid_scale"]
    for section in range(phases):
        local_time = problem.time[section] - problem.time_start(section)
        if phases == 3 and section == 1:
            assert p.max_depth is not None
            x = a * np.pi + np.sqrt(2 * p.gravity * p.max_depth) * local_time
            depth = np.full(nodes, p.max_depth)
            speed = np.full(nodes, np.sqrt(2 * p.gravity * p.max_depth))
            theta = np.full(nodes, np.pi / 2)
            angle_bounds = (0, np.pi)
        else:
            phi = local_time * np.sqrt(p.gravity / a)
            if phases == 3 and section == 2:
                phi = phi + np.pi
            x = a * (phi - np.sin(phi))
            if phases == 3 and section == 2:
                x = x + ref["cruise_distance"]
            depth = a * (1 - np.cos(phi))
            speed = 2 * np.sqrt(p.gravity * a) * np.sin(phi / 2)
            theta = phi / 2
            angle_bounds = (
                ((0, np.pi / 2) if section == 0 else (np.pi / 2, np.pi))
                if phases == 3
                else (0, np.pi)
            )
        for state, values in enumerate((x, speed)):
            problem.set_states(state, section, values)
        problem.set_controls(0, section, theta)
        problem.set_states_bounds(0, section, 0, p.distance)
        problem.set_states_bounds(
            1,
            section,
            0,
            np.sqrt(2 * p.gravity * p.max_depth) if p.max_depth is not None else None,
        )
        problem.set_controls_bounds(0, section, *angle_bounds)
        problem.set_time_final_bounds(
            section, 1e-6 * time_scale, 20 * ref["minimum_time"]
        )

    @LinearCallback
    def inequality(prob, obj):
        c = prob.condition()
        for section in range(phases):
            c.lower_bound(
                prob.time_final(section) - prob.time_start(section),
                1e-6 * time_scale,
                unit=time_scale,
            )
        return c()

    problem.cost = LinearCallback(lambda prob, obj: prob.time_final(-1) / time_scale)
    problem.equality = equality
    problem.inequality = inequality
    problem.dynamics = [DynamicsModel(pointwise_dynamics)] * phases
    return problem


def solve(parameters=PRESETS["default"], *, nodes=12, samples=201, maxiter=300):
    problem = make_problem(parameters, nodes=nodes)
    problem.solve(parameters, derivatives="structured", maxiter=maxiter, ftol=1e-9)
    trajectory = collect(
        problem,
        parameters,
        example="constrained_descent",
        states=["x", "speed"],
        controls=["theta"],
        units={
            "time": "s",
            "x": "m",
            "depth": "m",
            "speed": "m/s",
            "theta": "rad",
            "objective": "1",
        },
        coordinates={
            "x": "positive rightward",
            "depth": "positive downward",
            "theta": "from downward vertical toward positive x",
        },
        phase_names=["descend", "floor", "ascend"]
        if problem.number_of_section == 3
        else ["cycloid"],
        reference=reference_solution(parameters),
        samples=samples,
    )
    for phase in trajectory.phases:
        for grid in ("nodes", "samples"):
            phase[grid]["depth"] = (
                np.asarray(phase[grid]["speed"]) ** 2 / (2 * parameters.gravity)
            ).tolist()
    return trajectory


def plot(trajectory):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, layout="constrained", figsize=(12, 4))
    for phase in trajectory.phases:
        data = phase["samples"]
        axes[0].plot(data["x"], data["depth"], label=phase["name"])
        axes[1].plot(data["time"], data["speed"])
        axes[2].plot(data["time"], np.rad2deg(data["theta"]))
    if trajectory.parameters["max_depth"] is not None:
        axes[0].axhline(
            trajectory.parameters["max_depth"], color="k", linestyle="--", label="floor"
        )
    axes[0].invert_yaxis()
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].set(xlabel="x [m]", ylabel="depth [m] (downward)")
    axes[0].legend()
    axes[1].set(xlabel="time [s]", ylabel="speed [m/s]")
    axes[2].set(xlabel="time [s]", ylabel="theta [deg]")
    for ax in axes:
        ax.grid()
    fig.suptitle("Brachistochrone with a depth limit")
    return fig


def main(argv=None):
    run_cli(Parameters, PRESETS, solve, plot, __doc__, argv=argv)


if __name__ == "__main__":
    main()
