"""Move from rest to rest: minimum time versus fixed-time minimum effort.

    x' = v, v' = a; x(0)=v(0)=v(tf)=0, x(tf)=distance.

The effort objective is integral(a**2 dt), NOT physical energy consumption.
Known acceleration/cruise/braking phases isolate control switches. All phase
durations in minimum-time mode are optimized, rather than fixed to the reference.
Run: uv run python examples/02_move_stop/main.py --preset speed_limit --plot
"""

from dataclasses import dataclass

import numpy as np

from OpenGoddard import DynamicsModel, LinearCallback, Problem

# Support direct execution from any working directory as well as package imports.
if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._shared import collect, node_count, positive, run_cli


@dataclass(frozen=True)
class Parameters:
    distance: float = 10.0  # m; initial position and velocity are zero
    max_acceleration: float = 2.0  # m/s^2, symmetric acceleration/braking limit
    speed_limit: float | None = None  # m/s, forward motion only
    objective: str = "minimum_time"  # or minimum_effort
    duration: float = 6.0  # s; used only for minimum_effort

    def __post_init__(self):
        positive(
            distance=self.distance,
            max_acceleration=self.max_acceleration,
            duration=self.duration,
        )
        if self.speed_limit is not None:
            positive(speed_limit=self.speed_limit)
        if self.objective not in {"minimum_time", "minimum_effort"}:
            raise ValueError("objective must be minimum_time or minimum_effort")
        if self.objective == "minimum_effort" and self.duration <= minimum_time(
            self
        ) * (1 + 1e-6):
            raise ValueError(
                "minimum_effort requires duration above the minimum feasible time; "
                "use minimum_time at the limit"
            )


def minimum_time(p):
    """Exact rest-to-rest time, including a possible speed-limited cruise."""
    peak = np.sqrt(p.distance * p.max_acceleration)
    if p.speed_limit is not None and p.speed_limit < peak:
        return p.distance / p.speed_limit + p.speed_limit / p.max_acceleration
    return 2 * np.sqrt(p.distance / p.max_acceleration)


PRESETS = {
    "default": Parameters(),
    "speed_limit": Parameters(speed_limit=2.0),
    "minimum_effort": Parameters(objective="minimum_effort"),
    "limited_effort": Parameters(objective="minimum_effort", duration=5.0),
    "speed_limited_effort": Parameters(
        objective="minimum_effort", duration=7.0, speed_limit=2.0
    ),
    "longer_move": Parameters(distance=20.0),
}


def pointwise_dynamics(x, u, time, obj, section, xp):
    return xp.stack((x[1], u[0]))


@LinearCallback
def equality(problem, p):
    c = problem.condition()
    c.equal(problem.states(0, 0)[0], 0, unit=p.distance)
    c.equal(problem.states(1, 0)[0], 0, unit=problem.unit_states[0][1])
    c.equal(problem.states(0, -1)[-1], p.distance, unit=p.distance)
    c.equal(problem.states(1, -1)[-1], 0, unit=problem.unit_states[-1][1])
    return c()


def time_plan(p):
    a = p.max_acceleration
    peak = np.sqrt(a * p.distance)
    if p.speed_limit is not None and p.speed_limit < peak:
        peak = p.speed_limit
        return (
            [peak / a, p.distance / peak - peak / a, peak / a],
            [a, 0.0, -a],
            ["accelerate", "cruise", "brake"],
        )
    return ([peak / a, peak / a], [a, -a], ["accelerate", "brake"])


def phase_plan(p):
    """Reference durations and (initial acceleration, jerk) for each phase.

    For fixed-time minimum effort, acceleration is linear where neither bound
    is active, clipped at +/-a_max and zero during a speed-limited cruise.
    These arcs provide a mesh topology and guess, not extra control equations.
    """
    if p.objective == "minimum_time":
        durations, accelerations, names = time_plan(p)
        return durations, [(a, 0.0) for a in accelerations], names
    h, a = p.duration / 2, p.max_acceleration
    k = 3 * p.distance / (2 * h**3)
    if k * h <= a:
        saturation = 0.0
        peak_v = k * h**2 / 2
    else:
        k = np.sqrt(a**3 / (3 * (a * h**2 - p.distance)))
        saturation = h - a / k
        peak_v = a * h - a**2 / (2 * k)
    if p.speed_limit is None or peak_v <= p.speed_limit:
        if saturation == 0:
            return [p.duration], [(k * h, -k)], ["transfer"]
        ramp = a / k
        return (
            [saturation, 2 * ramp, saturation],
            [(a, 0.0), (a, -k), (-a, 0.0)],
            ["accelerate", "ramp", "brake"],
        )
    v = p.speed_limit
    ta = 1.5 * (p.duration - p.distance / v)
    initial_a = 2 * v / ta
    saturation = 0.0
    ramp = ta
    if initial_a > a:
        ramp = np.sqrt(12 * (v * p.duration - v**2 / a - p.distance) / a)
        ta = v / a + ramp / 2
        saturation = ta - ramp
        initial_a = a
    k = initial_a / ramp
    arcs = [
        (saturation, a, 0.0, "accelerate"),
        (ramp, initial_a, -k, "ramp_down"),
        (p.duration - 2 * ta, 0.0, 0.0, "cruise"),
        (ramp, 0.0, -k, "ramp_brake"),
        (saturation, -a, 0.0, "brake"),
    ]
    arcs = [arc for arc in arcs if arc[0] > 1e-12 * p.duration]
    return (
        [arc[0] for arc in arcs],
        [(arc[1], arc[2]) for arc in arcs],
        [arc[3] for arc in arcs],
    )


def make_problem(parameters=PRESETS["default"], *, nodes=12):
    """Construct an independent, unsolved Problem; parameters are immutable."""
    node_count(nodes)
    p = parameters
    durations, control_arcs, _ = phase_plan(p)
    knots = np.r_[0, np.cumsum(durations)]
    phases = len(durations)
    problem = Problem(
        knots, [nodes] * phases, [2] * phases, [1] * phases, maxIterator=1
    )
    time_scale = minimum_time(p)
    speed_scale = np.sqrt(p.distance * p.max_acceleration)
    problem.set_unit_time(time_scale)
    problem.set_unit_states_all_section(0, p.distance)
    problem.set_unit_states_all_section(1, speed_scale)
    problem.set_unit_controls_all_section(0, p.max_acceleration)
    x0, v0 = 0.0, 0.0
    for section in range(phases):
        t = problem.time[section] - knots[section]
        a, jerk = control_arcs[section]
        x = x0 + v0 * t + a * t**2 / 2 + jerk * t**3 / 6
        v, u = v0 + a * t + jerk * t**2 / 2, a + jerk * t
        x0, v0 = float(x[-1]), float(v[-1])
        problem.set_states(0, section, x)
        problem.set_states(1, section, v)
        problem.set_controls(0, section, u)
        problem.set_states_bounds(1, section, 0.0, p.speed_limit)
        problem.set_controls_bounds(0, section, -p.max_acceleration, p.max_acceleration)
        problem.set_time_final_bounds(
            section, 1e-8 * time_scale, max(5 * time_scale, 2 * p.duration)
        )
    if p.objective == "minimum_effort":
        problem.set_time_final_bounds(-1, p.duration, p.duration)
        problem.running_cost = lambda prob, obj: prob.controls_all_section(0) ** 2
    else:
        problem.cost = LinearCallback(lambda prob, obj: prob.time_final(-1))

    # Keep each phase's duration positive even when intermediate times move.
    @LinearCallback
    def inequality(prob, obj):
        c = prob.condition()
        for i in range(prob.number_of_section):
            c.lower_bound(
                prob.time_final(i) - prob.time_start(i),
                1e-8 * time_scale,
                unit=time_scale,
            )
        return c()

    problem.inequality = inequality
    problem.equality = equality
    problem.dynamics = [DynamicsModel(pointwise_dynamics)] * phases
    return problem


def solve(parameters=PRESETS["default"], *, nodes=12, samples=201, maxiter=300):
    problem = make_problem(parameters, nodes=nodes)
    problem.solve(parameters, derivatives="structured", maxiter=maxiter, ftol=1e-10)
    reference = {"minimum_time": float(minimum_time(parameters))}
    if parameters.objective == "minimum_effort":
        durations, arcs, _ = phase_plan(parameters)
        reference["minimum_effort"] = float(
            sum(
                a**2 * t + a * jerk * t**2 + jerk**2 * t**3 / 3
                for t, (a, jerk) in zip(durations, arcs, strict=True)
            )
        )
    return collect(
        problem,
        parameters,
        example="move_stop",
        states=["position", "velocity"],
        controls=["acceleration"],
        units={
            "time": "s",
            "position": "m",
            "velocity": "m/s",
            "acceleration": "m/s^2",
            "objective": "s" if parameters.objective == "minimum_time" else "m^2/s^3",
        },
        coordinates={"position": "positive toward the target"},
        phase_names=phase_plan(parameters)[2],
        reference=reference,
        samples=samples,
    )


def plot(trajectory):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, sharex=True, layout="constrained")
    for phase in trajectory.phases:
        data = phase["samples"]
        for ax, name in zip(
            axes, ("position", "velocity", "acceleration"), strict=True
        ):
            ax.plot(data["time"], data[name], label=phase["name"])
            ax.set_ylabel(f"{name} [{trajectory.units[name]}]")
            ax.grid()
    axes[0].legend()
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"Move and stop: {trajectory.parameters['objective']}")
    return fig


def main(argv=None):
    run_cli(Parameters, PRESETS, solve, plot, __doc__, argv=argv)


if __name__ == "__main__":
    main()
