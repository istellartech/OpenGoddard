"""Reproducible comparisons using existing example setups without plot/file output.

Run: uv run --extra autodiff python benchmarks/compare_solvers.py
Each case runs in a fresh process with one BLAS thread and a timeout. Reports
include failures; timings cover solving only, excluding setup/error estimation.
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import runpy
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MODES = ("baseline", "structured", "autograd", "sparse", "adaptive")


def load_example(prefix):
    import matplotlib.pyplot as plt

    from OpenGoddard import Problem

    path = next((ROOT / "examples").glob(prefix + "_*.py"))
    captured = []

    class Ready(Exception):
        pass

    def capture(problem, obj=None, *args, **kwargs):
        captured.append((problem, obj))
        raise Ready

    with (
        patch.object(Problem, "solve", capture),
        patch.object(plt, "show"),
        patch.object(plt, "savefig"),
        patch.object(Path, "mkdir"),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        try:
            runpy.run_path(str(path), run_name="__main__")
        except Ready:
            pass
    plt.close("all")
    return captured[0]


def worker(example, mode, output):
    import warnings

    import numpy as np
    import scipy

    from OpenGoddard.derivatives import Derivatives

    problem, obj = load_example(example)
    problem.maxIterator = 1
    solver = "trust-constr" if mode == "sparse" else "SLSQP"
    derivatives = {"baseline": "finite-difference", "structured": "structured"}.get(
        mode, "autograd"
    )
    if mode == "adaptive":
        problem = problem.refine([max(6, min(12, n // 2)) for n in problem.nodes])
    # AD has no compilation; exclude one-time optional-module imports/tracing warmup.
    if derivatives == "autograd":
        provider = Derivatives(problem, obj, derivatives)
        provider.objective()
        provider.equality()
        provider.inequality()
    options = {"solver": solver, "derivatives": derivatives, "maxiter": 600}
    options["gtol" if solver == "trust-constr" else "ftol"] = 1e-8
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always")
        start = perf_counter()
        if mode == "adaptive":
            result = problem.solve_adaptive(
                obj, tolerance=1e-5, max_nodes=100, max_refinements=5, **options
            )
        else:
            result = problem.solve(obj, **options)
        elapsed = perf_counter() - start
    errors = problem.estimate_error(obj) if result.success else None
    analytical = {"01": np.sqrt(np.pi), "12": 4.0}.get(example)
    result_data = {
        "example": example,
        "mode": mode,
        "seconds": elapsed,
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nodes": problem.nodes,
        "objective": float(result.fun),
        "analytic_objective_error": None
        if analytical is None
        else abs(float(result.fun) - analytical),
        "equality_violation": float(result.equality_violation),
        "inequality_violation": float(result.inequality_violation),
        "mesh_error": errors,
        "iterations": int(result.nit),
        "objective_evaluations": int(result.nfev),
        "optimality": float(result.optimality) if solver == "trust-constr" else None,
        "mesh_converged": bool(result.mesh_converged) if mode == "adaptive" else None,
        "mesh_history": result.mesh_history if mode == "adaptive" else None,
        "warnings": sorted({str(w.message) for w in messages}),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "autograd": version("autograd") if derivatives == "autograd" else None,
    }
    Path(output).write_text(json.dumps(result_data, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", nargs="+", default=["01", "03", "04", "05", "12"])
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results.json")
    parser.add_argument("--worker", nargs=3, metavar=("EXAMPLE", "MODE", "OUTPUT"))
    args = parser.parse_args()
    if args.worker:
        worker(*args.worker)
        return
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    records = []
    environment = dict(
        os.environ,
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(Path(tempfile.gettempdir()) / "opengoddard-mpl"),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
    )
    # Rotate mode order between repetitions to reduce systematic warm-cache bias.
    for repetition in range(args.repeats):
        modes = (
            args.modes[repetition % len(args.modes) :]
            + args.modes[: repetition % len(args.modes)]
        )
        for example in args.examples:
            for mode in modes:
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "result.json"
                    try:
                        subprocess.run(
                            [
                                sys.executable,
                                str(Path(__file__).resolve()),
                                "--worker",
                                example,
                                mode,
                                str(path),
                            ],
                            env=environment,
                            capture_output=True,
                            text=True,
                            timeout=args.timeout,
                            check=True,
                        )
                        record = json.loads(path.read_text())
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
                        f"{example} {mode} rep={repetition + 1} success={record['success']} seconds={record.get('seconds', 'timeout/error')}",
                        flush=True,
                    )
                    save(args.output, records, args)


def save(output, records, args):
    hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ("src/OpenGoddard", "examples", "benchmarks")
        for p in (ROOT / folder).glob("*.py")
    }
    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "blas_threads": 1,
        "repeats": args.repeats,
        "timeout_seconds": args.timeout,
        "source_sha256": hashes,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2))
    rows = [
        "Example | Mode | Success | Nodes | Median seconds | Objective | Collocation residual | Between-node defect | Sampled path violation | Mesh converged",
        "--- | --- | --- | --- | --- | --- | --- | --- | --- | ---",
    ]
    for example in args.examples:
        for mode in args.modes:
            cases = [
                r for r in records if r["example"] == example and r["mode"] == mode
            ]
            if not cases:
                continue
            valid = [r for r in cases if "seconds" in r]
            if not valid:
                rows.append(
                    f"{example} | {mode} | 0/{len(cases)} | — | timeout/error | — | — | — | — | —"
                )
                continue
            item = valid[-1]
            seconds = statistics.median(r["seconds"] for r in valid)
            mesh = item.get("mesh_error")
            defect = "—" if mesh is None else f"{mesh['max_error']:.3g}"
            path = "—" if mesh is None else f"{mesh['path_violation']:.3g}"
            rows.append(
                f"{example} | {mode} | {sum(r['success'] for r in cases)}/{len(cases)} | {item['nodes']} | {seconds:.4f} | {item['objective']:.10g} | {item['equality_violation']:.3g} | {defect} | {path} | {item['mesh_converged']}"
            )
    output.with_suffix(".txt").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
