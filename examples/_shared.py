"""Shared export and command-line interface for the three learning examples."""

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import BarycentricInterpolator


def positive(**values):
    """Reject invalid physical parameters before constructing a transcription."""
    for name, value in values.items():
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")


def node_count(nodes):
    if isinstance(nodes, bool) or not isinstance(nodes, int) or nodes < 4:
        raise ValueError("nodes must be an integer >= 4 (per phase)")


@dataclass
class Trajectory:
    """JSON-ready physical data, with independent time grids for each phase.

    Duplicate interface times are intentional: controls may jump there. `nodes`
    holds collocation values; `samples` holds the *unclipped* phase polynomial
    on a uniform grid for playback. Diagnostics are sampled checks, not a proof
    of continuous feasibility or global optimality.
    """

    example: str
    parameters: dict[str, Any]
    units: dict[str, str]
    coordinates: dict[str, str]
    solver: dict[str, Any]
    diagnostics: dict[str, Any]
    reference: dict[str, float]
    phases: list[dict[str, Any]]

    def to_dict(self):
        return {"schema_version": 1, **asdict(self)}

    def write_json(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def collect(
    problem,
    parameters,
    *,
    example,
    states,
    controls,
    units,
    coordinates,
    phase_names,
    reference,
    samples=201,
):
    """Extract data after a successful solve without opening figures or files."""
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 2:
        raise ValueError("samples must be an integer >= 2 (per phase)")
    result = problem.result
    if not result.success:
        raise RuntimeError(f"{example}: {result.message}")
    problem.time_update()
    phases = []
    for section, name in enumerate(phase_names):
        tau = problem.tau[section]
        grid = np.linspace(-1, 1, samples)
        time = problem.time_start(section) + (grid + 1) / 2 * (
            problem.time_final(section) - problem.time_start(section)
        )
        nodal = {"time": problem.time[section].tolist()}
        playback = {"time": time.tolist()}
        for names, getter in ((states, problem.states), (controls, problem.controls)):
            for index, label in enumerate(names):
                values = getter(index, section)
                nodal[label] = values.tolist()
                playback[label] = BarycentricInterpolator(tau, values, rng=0)(
                    grid
                ).tolist()
        phases.append({"name": name, "nodes": nodal, "samples": playback})
    diagnostics = problem.estimate_error(parameters, check_factor=4)
    diagnostics["max_equality_residual"] = float(
        np.max(np.abs(problem._equality_constraints(parameters)), initial=0)
    )
    diagnostics["check_tolerance"] = 1e-4
    diagnostics["sampled_checks_passed"] = all(
        diagnostics[name] <= diagnostics["check_tolerance"]
        for name in ("max_error", "path_violation", "max_equality_residual")
    )
    return Trajectory(
        example=example,
        parameters=asdict(parameters),
        units=units,
        coordinates=coordinates,
        solver={
            "success": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit),
            "objective": float(result.fun),
        },
        diagnostics=diagnostics,
        reference=reference,
        phases=phases,
    )


def run_cli(parameters_type, presets, solve, plot, description, argv=None):
    """Identical controls for scripts, batch exports, and future UI adapters."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--preset", choices=presets, default="default")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=JSON",
        help="override a parameter, e.g. distance=20 or max_depth=null",
    )
    parser.add_argument("--nodes", type=int, default=12, help="LGL nodes per phase")
    parser.add_argument(
        "--samples", type=int, default=201, help="playback samples per phase"
    )
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--output", type=Path, help="write a trajectory JSON file")
    parser.add_argument("--plot", action="store_true", help="open a Matplotlib figure")
    args = parser.parse_args(argv)
    parameters = presets[args.preset]
    try:
        overrides = {}
        for expression in args.set:
            name, value = expression.split("=", 1)
            overrides[name] = json.loads(value)
        parameters = replace(parameters, **overrides)
        if not isinstance(parameters, parameters_type):
            raise TypeError("unexpected parameter type")
        trajectory = solve(
            parameters, nodes=args.nodes, samples=args.samples, maxiter=args.maxiter
        )
    except (ValueError, TypeError, RuntimeError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "parameters": trajectory.parameters,
                "solver": trajectory.solver,
                "diagnostics": trajectory.diagnostics,
                "reference": trajectory.reference,
            },
            indent=2,
            allow_nan=False,
        )
    )
    if args.output:
        trajectory.write_json(args.output)
    if args.plot:
        import matplotlib.pyplot as plt

        plot(trajectory)
        plt.show()
