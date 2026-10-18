"""Compare historical p adaptation with phase-preserving h/hp on Goddard.

Run: uv run --extra autodiff python benchmarks/compare_hp.py
Historical Python sources are read from the specified Git revision into a
TemporaryDirectory and imported under an isolated package name. Nothing checks
out or resets the user's working tree. All solutions receive the same current
validation, including polynomial extrema and an independent IVP integration.
"""

import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import warnings
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from compare_solvers import ROOT, load_example

MODES = ("previous-p", "p", "h", "hp")


def baseline_package(revision, directory):
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", revision, "src/OpenGoddard"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    package = directory / "old_core"
    package.mkdir()
    for name in names:
        if name.endswith(".py"):
            (package / Path(name).name).write_bytes(
                subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT)
            )
    spec = importlib.util.spec_from_file_location(
        "_baseline_goddard",
        package / "__init__.py",
        submodule_search_locations=[str(package)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load historical package")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def historical_problem(template, old):
    from OpenGoddard import LinearCallback

    p = old.Problem(
        template.time_knots(),
        template.nodes,
        template.number_of_states,
        template.number_of_controls,
        maxIterator=1,
    )
    p.unit_states = copy.deepcopy(template.unit_states)
    p.unit_controls = copy.deepcopy(template.unit_controls)
    p.unit_time = template.unit_time
    p.p = template.p.copy()
    p.bounds = list(template.bounds)
    p.knot_states_smooth = list(template.knot_states_smooth)
    p.dynamics = [old.DynamicsModel(model.function) for model in template.dynamics]
    for name in ("cost", "running_cost", "cost_derivative", "equality", "inequality"):
        callback = getattr(template, name)
        setattr(
            p,
            name,
            old.LinearCallback(callback.function)
            if isinstance(callback, LinearCallback)
            else callback,
        )
    p.time_update()
    return p


def validator(problem, template):
    from OpenGoddard import Problem

    p = Problem(
        problem.time_knots(),
        problem.nodes,
        problem.number_of_states,
        problem.number_of_controls,
        maxIterator=1,
        mesh=getattr(problem, "mesh", None),
    )
    p.unit_states = copy.deepcopy(problem.unit_states)
    p.unit_controls = copy.deepcopy(problem.unit_controls)
    p.unit_time = problem.unit_time
    p.p = problem.p.copy()
    p.bounds = list(problem.bounds)
    p.knot_states_smooth = list(problem.knot_states_smooth)
    p.dynamics = template.dynamics
    for name in ("cost", "running_cost", "equality", "inequality"):
        setattr(p, name, getattr(template, name))
    p.time_update()
    return p


def validate_ivp(p, obj):
    from scipy.integrate import solve_ivp
    from scipy.interpolate import BarycentricInterpolator

    from OpenGoddard._grid import elements_with_slices

    initial = None
    error = 0.0
    for section in range(p.number_of_section):
        ns, nc = p.number_of_states[section], p.number_of_controls[section]
        if section == 0 or not p.knot_states_smooth[section - 1]:
            initial = np.array([p.states(i, section)[0] for i in range(ns)])
        for e, rows in elements_with_slices(p, section):
            times = p.time[section][rows]
            state = np.array([p.states(i, section)[rows] for i in range(ns)])
            control = np.array([p.controls(i, section)[rows] for i in range(nc)])
            control_poly = BarycentricInterpolator(times, control, axis=1, rng=0)
            state_poly = BarycentricInterpolator(times, state, axis=1, rng=0)

            def rhs(t, x):
                u = control_poly(t).reshape(nc, 1)
                return p.dynamics[section].physical(
                    x[:, None], u, np.array([t]), obj, section
                )[:, 0]

            integration = solve_ivp(
                rhs,
                (times[0], times[-1]),
                initial,
                rtol=1e-10,
                atol=1e-12,
                dense_output=True,
            )
            if not integration.success:
                raise RuntimeError(integration.message)
            check = np.linspace(times[0], times[-1], max(31, 4 * e.nodes))
            reference = integration.sol(check)
            approximation = state_poly(check)
            scale = np.array(p.unit_states[section]) + np.max(
                np.abs(approximation), axis=1
            )
            error = max(
                error, float(np.max(np.abs(reference - approximation) / scale[:, None]))
            )
            initial = integration.y[:, -1]
    if initial is None:
        raise RuntimeError("no integration interval was evaluated")
    terminal = np.array([p.states(i, -1)[-1] for i in range(p.number_of_states[-1])])
    return {
        "max_relative_state_error": error,
        "final_state_difference": (initial - terminal).tolist(),
        "final_state": initial.tolist(),
        "rtol": 1e-10,
        "atol": 1e-12,
    }


def worker(example, mode, output, revision):
    from OpenGoddard._sampling import SampledPath, polynomial_extrema
    from OpenGoddard.derivatives import Derivatives

    template, obj = load_example(example)
    with tempfile.TemporaryDirectory() as directory:
        if mode == "previous-p":
            old = baseline_package(revision, Path(directory))
            p = historical_problem(template, old)
            provider_type = importlib.import_module(
                old.__name__ + ".derivatives"
            ).Derivatives
        else:
            p = template
            provider_type = Derivatives
        p = p.refine([12] * p.number_of_section)
        p.maxIterator = 1
        provider = provider_type(p, obj, "autograd")
        provider.objective()
        provider.equality()
        provider.inequality()
        options = {"derivatives": "autograd", "maxiter": 600, "tolerance": 1e-5}
        if mode == "previous-p":
            options.update(max_refinements=5, max_nodes=100, ftol=1e-8)
        else:
            options.update(max_refinements=12, max_nodes=150, ftol=1e-10, strategy=mode)
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter("always")
            start = perf_counter()
            result = p.solve_adaptive(obj, **options)
            elapsed = perf_counter() - start
        common = validator(p, template)
        error = common.estimate_error(obj, check_factor=4)
        dense = SampledPath(common.inequality, 16, polynomial_extrema(common))(
            common, obj
        )
        dense_violation = max(0.0, -float(np.min(dense, initial=0)))
        common_converged = (
            bool(result.success)
            and error["max_error"] <= 1e-5
            and max(error["path_violation"], dense_violation) <= 1e-5
        )
        data = {
            "example": example,
            "mode": mode,
            "seconds": elapsed,
            "success": bool(result.success),
            "mesh_converged": bool(result.mesh_converged),
            "common_converged": common_converged,
            "status": int(result.status),
            "message": str(result.message),
            "objective": float(result.fun),
            "nodes": list(p.nodes),
            "mesh": [
                [[e.left, e.right, e.nodes] for e in group] for group in common.mesh
            ],
            "quality": error,
            "dense_path_violation": dense_violation,
            "ivp": validate_ivp(common, obj) if result.success else None,
            "equality_violation": float(
                np.max(np.abs(common._equality_constraints(obj)), initial=0)
            ),
            "options": options,
            "history": result.mesh_history,
            "warnings": sorted({str(w.message) for w in messages}),
            "versions": {
                name: version(name) for name in ("numpy", "scipy", "autograd")
            },
        }
        Path(output).write_text(json.dumps(data, indent=2))


def save(path, records, args):
    files = list((ROOT / "src/OpenGoddard").glob("*.py")) + [
        Path(__file__),
        ROOT / "benchmarks/compare_solvers.py",
    ]
    data = {
        "python": sys.version,
        "platform": platform.platform(),
        "blas_threads": 1,
        "baseline_revision": subprocess.check_output(
            ["git", "rev-parse", args.baseline], cwd=ROOT, text=True
        ).strip(),
        "current_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "repeats": args.repeats,
        "timeout_seconds": args.timeout,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    lines = [
        "Example | Mode | NLP | Common mesh | Seconds median | Objective | Nodes | ODE defect | Path violation | IVP state error",
        "--- | --- | --- | --- | --- | --- | --- | --- | --- | ---",
    ]
    for example in args.examples:
        for mode in args.modes:
            group = [
                r for r in records if r["example"] == example and r["mode"] == mode
            ]
            if not group:
                continue
            timed = [r for r in group if "seconds" in r]
            if not timed:
                lines.append(
                    f"{example} | {mode} | 0/{len(group)} | — | timeout/error | — | — | — | — | —"
                )
                continue
            r = timed[-1]
            seconds = statistics.median(r["seconds"] for r in timed)
            ivp = (
                "—"
                if r["ivp"] is None
                else f"{r['ivp']['max_relative_state_error']:.3g}"
            )
            lines.append(
                f"{example} | {mode} | {sum(r['success'] for r in group)}/{len(group)} | {sum(r.get('common_converged', False) for r in group)}/{len(group)} | {seconds:.4f} | {r['objective']:.12g} | {r['nodes']} | {r['quality']['max_error']:.3g} | {max(r['dense_path_violation'], r['quality']['path_violation']):.3g} | {ivp}"
            )
    path.with_suffix(".txt").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", nargs="+", default=["04", "05"])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--baseline", default="978b97a")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "benchmarks/hp_results.json"
    )
    parser.add_argument("--worker", nargs=3)
    args = parser.parse_args()
    if args.worker:
        worker(*args.worker, args.baseline)
        return
    if args.repeats < 1 or args.timeout <= 0:
        parser.error("positive repeats and timeout are required")
    environment = dict(
        os.environ,
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(Path(tempfile.gettempdir()) / "opengoddard-mpl"),
        OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    records = []
    for repetition in range(args.repeats):
        shift = repetition % len(args.modes)
        modes = args.modes[shift:] + args.modes[:shift]
        for example in args.examples:
            for mode in modes:
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / "result.json"
                    try:
                        subprocess.run(
                            [
                                sys.executable,
                                str(Path(__file__).resolve()),
                                "--worker",
                                example,
                                mode,
                                str(output),
                                "--baseline",
                                args.baseline,
                            ],
                            cwd=ROOT,
                            env=environment,
                            capture_output=True,
                            text=True,
                            check=True,
                            timeout=args.timeout,
                        )
                        record = json.loads(output.read_text())
                    except subprocess.TimeoutExpired:
                        record = {
                            "example": example,
                            "mode": mode,
                            "success": False,
                            "timeout": args.timeout,
                        }
                    except subprocess.CalledProcessError as exc:
                        record = {
                            "example": example,
                            "mode": mode,
                            "success": False,
                            "error": exc.stderr,
                        }
                    record["repetition"] = repetition
                    records.append(record)
                    print(
                        example,
                        mode,
                        repetition + 1,
                        record.get("seconds"),
                        record.get("common_converged"),
                        flush=True,
                    )
                    save(args.output, records, args)


if __name__ == "__main__":
    main()
