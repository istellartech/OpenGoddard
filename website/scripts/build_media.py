"""Generate the figures, videos and numbers shown on the documentation site.

Each example is solved with its own ``solve()`` (``main()`` for example 01).
The solution is read at the LGL nodes and sampled densely by barycentric
interpolation of each phase's polynomial. ``specs.py`` describes what to draw,
``figures.py`` draws it, and ``concepts.py`` draws the figures of the theory
pages.

Outputs go to ``website/public/media/<example id>/`` (Japanese) and
``website/public/media/en/<example id>/`` (English), using the same solution:

* ``history.png``: states and controls against time;
* ``scene.png``: the motion (path, altitude column, launch or orbit);
* ``animation.mp4`` and its poster ``animation.png``;
* ``summary.json``: solver status and the numbers quoted on the page.

An example is rebuilt only when its sources, the library or these scripts
change, so CI can cache the directory. Run from the repository root::

    uv run python website/scripts/build_media.py            # everything
    uv run python website/scripts/build_media.py move_stop  # selected IDs
"""

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
from scipy.interpolate import BarycentricInterpolator  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
OUTPUT = ROOT / "website" / "public" / "media"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

import style  # noqa: E402, F401  (sets the Matplotlib style)
from figures import Solution, animation, history_figure, scene_figure  # noqa: E402
from i18n import LANGUAGES, language  # noqa: E402
from specs import POST, SPECS  # noqa: E402

from examples.catalog import EXAMPLES, find_example, load_example  # noqa: E402

SAMPLES = 241


def sample(problem, spec, module) -> Solution:
    """Read the nodes and interpolate each phase in physical units."""
    nodes, dense = [], []
    for section in range(problem.number_of_section):
        tau = problem.tau[section]
        grid = np.linspace(-1, 1, SAMPLES)
        t0, tf = problem.time_start(section), problem.time_final(section)
        node = {"time": np.asarray(problem.time[section]), "section": section}
        fine = {"time": t0 + (grid + 1) / 2 * (tf - t0), "section": section}
        for names, getter in (
            (spec.states, problem.states),
            (spec.controls, problem.controls),
        ):
            for index, name in enumerate(names):
                values = np.asarray(getter(index, section))
                node[name] = values
                fine[name] = BarycentricInterpolator(tau, values, rng=0)(grid)
        if spec.derive is not None:
            spec.derive(node, module)
            spec.derive(fine, module)
        nodes.append(node)
        dense.append(fine)
    return Solution(nodes, dense)


def _plain(value):
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def summarize(problem, sol: Solution, spec, module, example) -> dict:
    result = problem.result
    phases = []
    for section, node in enumerate(sol.nodes):
        names = spec.states + spec.controls
        phases.append(
            {
                "start_time": problem.time_start(section),
                "final_time": problem.time_final(section),
                "nodes": int(problem.nodes[section]),
                "start": {k: float(node[k][0]) for k in names},
                "final": {k: float(node[k][-1]) for k in names},
            }
        )
    summary = {
        "id": example.id,
        "script": example.script,
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", -1)),
        "objective": float(result.fun),
        "phases": phases,
    }
    if spec.extras is not None:
        summary.update(spec.extras(sol, problem, module))
    return _plain(summary)


def source_hash(example) -> str:
    digest = hashlib.sha256()
    files = [ROOT / "examples" / example.script, ROOT / "examples" / "catalog.py"]
    if example.id == "polar_tsto_taiki":
        files += sorted((ROOT / "examples" / "data").glob("*.csv"))
    files += sorted((ROOT / "src" / "OpenGoddard").glob("*.py"))
    files += sorted(SCRIPTS.glob("*.py"))
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_example(identifier: str, force: bool = False) -> str:
    example = find_example(identifier)
    outputs = [OUTPUT / example.id, OUTPUT / "en" / example.id]
    digest = source_hash(example)
    if not force and all(
        (out / "summary.json").exists()
        and json.loads((out / "summary.json").read_text()).get("hash") == digest
        for out in outputs
    ):
        return f"{example.id}: up to date (ja, en)"
    module = load_example(example.id)
    spec = SPECS[example.id]
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        if example.id == "minimum_energy_transfer":
            problem = module.main()
        else:
            problem = module.solve()
    sol = sample(problem, spec, module)
    if example.id in POST:
        POST[example.id](sol)
    summary = summarize(problem, sol, spec, module, example)
    summary["title"] = example.title
    summary["hash"] = digest
    # Solve once, then render the same trajectory in both languages.
    for locale, out in zip(LANGUAGES, outputs):
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        with language(locale):
            history_figure(sol, spec, out / "history.png")
            if spec.scene is not None:
                scene_figure(sol, spec, out / "scene.png")
                animation(sol, spec, out / "animation")
        (out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
    return f"{example.id}: built (ja, en)"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ids", nargs="*", help="example IDs (default: all)")
    parser.add_argument("--force", action="store_true", help="ignore the cache")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    parser.add_argument("--no-concepts", action="store_true")
    args = parser.parse_args(argv)
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg is required to write MP4 files")
    ids = args.ids or [example.id for example in EXAMPLES]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[Callable[[str, bool], str], str]] = [
        (build_example, i) for i in ids
    ]
    if not args.no_concepts:
        from concepts import build_concepts

        jobs.insert(0, (build_concepts, "concepts"))
    failures = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(job, name, args.force): name for job, name in jobs}
        for future in as_completed(futures):
            try:
                print(future.result(), flush=True)
            except Exception as error:  # report every failing job
                failures.append(futures[future])
                print(f"{futures[future]}: FAILED: {error!r}", flush=True)
    if failures:
        sys.exit(f"failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
