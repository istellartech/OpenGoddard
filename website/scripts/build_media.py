"""Generate the videos, figures and data shown on the documentation site.

Each example is solved with its own ``solve`` or ``main`` entry point and
sampled per phase with barycentric interpolation. ``specs.py`` describes the
scene; ``scenes.py`` draws an MP4 and a poster PNG. The builder also saves
the sampled JSON data and the figures from the example's ``plot()`` function.

Outputs go to ``website/public/media/<example id>/`` with a ``manifest.json``
that the site reads at build time. An example is regenerated only when its
sources, the library or these scripts change, so CI can cache the directory.

Run from the repository root::

    uv run python website/scripts/build_media.py            # all examples
    uv run python website/scripts/build_media.py move_stop  # selected IDs
"""

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from scipy.interpolate import BarycentricInterpolator  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
OUTPUT = ROOT / "website" / "public" / "media"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from scenes import (  # noqa: E402
    DPI,
    FPS,
    INK,
    MUTED,
    PHASE_COLORS,
    SIZE,
    Phase,
    render_flight,
)
from specs import SPECS  # noqa: E402

from examples.catalog import EXAMPLES, find_example, load_example  # noqa: E402

SAMPLES = 241


def sample_problem(prob, states, controls, samples=SAMPLES) -> list[Phase]:
    """Sample a solved Problem in physical units."""
    prob.time_update()
    phases = []
    for section in range(prob.number_of_section):
        tau = prob.tau[section]
        grid = np.linspace(-1, 1, samples)
        t0, tf = prob.time_start(section), prob.time_final(section)
        phase = {"time": t0 + (grid + 1) / 2 * (tf - t0)}
        for names, getter in ((states, prob.states), (controls, prob.controls)):
            for index, name in enumerate(names):
                values = getter(index, section)
                phase[name] = BarycentricInterpolator(tau, values, rng=0)(grid)
        phases.append(phase)
    return phases


# ---------------------------------------------------------------------------
# Jobs


def source_hash(example) -> str:
    digest = hashlib.sha256()
    files = [ROOT / "examples" / example.script, ROOT / "examples" / "catalog.py"]
    if example.id == "polar_tsto_taiki":
        files += sorted((ROOT / "examples" / "data").glob("*.csv"))
    files += sorted((ROOT / "src" / "OpenGoddard").glob("*.py"))
    files += sorted((ROOT / "examples").glob("_*.py"))
    files += sorted(SCRIPTS.glob("*.py"))
    for path in files:
        if path.is_file() and "output" not in path.parts and path.suffix != ".pyc":
            digest.update(path.relative_to(ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass
class Manifest:
    id: str
    script: str
    title: str
    hash: str
    videos: list[dict[str, str]] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)
    data: list[str] = field(default_factory=list)
    # Solver status per video; a failed solve is still drawn but flagged.
    solver: dict[str, dict[str, Any]] = field(default_factory=dict)


def build_solution(example, module, out: Path, manifest: Manifest):
    spec = SPECS[example.id]
    output = ROOT / "examples" / "output" / example.id
    shutil.rmtree(output, ignore_errors=True)
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        if example.id == "minimum_energy_transfer":
            prob = module.main()
        else:
            prob = module.solve()
            figure = module.plot(prob)
            output.mkdir(parents=True, exist_ok=True)
            figure.savefig(output / "solution.png")
    plt.close("all")
    result = prob.result
    manifest.solver["solution"] = {
        "success": bool(result.success),
        "message": str(result.message),
    }
    phases = sample_problem(prob, spec.states, spec.controls)
    if spec.prepare is not None:
        spec.prepare(phases)
    flight = spec.flight(phases)
    if flight.kind == "ascent":
        flight.target_altitude = module.Rocket().Htarget  # the example's own
    render_flight(flight, phases, out / "solution")
    manifest.videos.append(
        {"name": "solution", "video": "solution.mp4", "poster": "solution.png"}
    )
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    for png in sorted(output.glob("*.png")):
        shutil.copy2(png, figures / png.name)
        manifest.figures.append(f"figures/{png.name}")
    data = {
        "example": example.id,
        "phases": [{k: v.tolist() for k, v in phase.items()} for phase in phases],
    }
    (out / "solution.json").write_text(json.dumps(data, allow_nan=False) + "\n")
    manifest.data.append("solution.json")


def build_example(identifier: str, force: bool = False) -> str:
    example = find_example(identifier)
    out = OUTPUT / example.id
    digest = source_hash(example)
    manifest_path = out / "manifest.json"
    if not force and manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get("hash") == digest:
            return f"{example.id}: up to date"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    module = load_example(example.id)
    manifest = Manifest(example.id, example.script, example.title, digest)
    build_solution(example, module, out, manifest)
    manifest_path.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2) + "\n"
    )
    return f"{example.id}: built"


# ---------------------------------------------------------------------------
# Concept animation: LGL nodes, interpolation and the differentiation matrix


def build_concepts(force: bool = False) -> str:
    from OpenGoddard import Problem

    out = OUTPUT / "concepts"
    stem = out / "lgl_interpolation"
    digest = hashlib.sha256(
        b"".join(path.read_bytes() for path in sorted(SCRIPTS.glob("*.py")))
    ).hexdigest()
    stamp = out / "hash.txt"
    if not force and stamp.exists() and stamp.read_text() == digest:
        return "concepts: up to date"
    out.mkdir(parents=True, exist_ok=True)

    def f(x):
        return 1 / (1 + 25 * x**2)

    def df(x):
        return -50 * x / (1 + 25 * x**2) ** 2

    fine = np.linspace(-1, 1, 600)
    helper = Problem([0.0, 1.0], [4], [1], [1])
    counts = list(range(4, 31, 2))
    errors = {"lgl": [], "uniform": []}
    for n in counts:
        nodes = helper.method_LGL(n)[0]
        uniform = np.linspace(-1, 1, n)
        exact = f(fine)
        errors["lgl"].append(
            np.max(np.abs(BarycentricInterpolator(nodes, f(nodes))(fine) - exact))
        )
        errors["uniform"].append(
            np.max(np.abs(BarycentricInterpolator(uniform, f(uniform))(fine) - exact))
        )

    fig = plt.figure(figsize=SIZE, dpi=DPI, facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.5, 1],
        left=0.05,
        right=0.97,
        top=0.86,
        bottom=0.08,
        wspace=0.18,
        hspace=0.3,
    )
    top, bottom = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])
    err = fig.add_subplot(grid[:, 1])
    fig.text(
        0.05, 0.95, "LGLノードでの補間と微分行列", fontsize=20, weight="bold", color=INK
    )
    badge = fig.text(
        0.05,
        0.905,
        "",
        fontsize=14,
        color="white",
        weight="bold",
        bbox={"boxstyle": "round,pad=0.35", "fc": PHASE_COLORS[0], "ec": "none"},
    )
    lgl, uni = PHASE_COLORS[0], PHASE_COLORS[1]
    writer = FFMpegWriter(
        fps=FPS,
        codec="libx264",
        bitrate=-1,
        extra_args=["-pix_fmt", "yuv420p", "-crf", "25", "-movflags", "+faststart"],
    )
    with writer.saving(fig, str(stem.with_suffix(".mp4")), dpi=DPI):
        for k, n in enumerate(counts):
            nodes, _, d = helper.method_LGL(n)
            uniform = np.linspace(-1, 1, n)
            badge.set_text(f" ノード数 N = {n} ")
            for ax in (top, bottom, err):
                ax.clear()
                ax.grid(alpha=0.3)
            top.plot(fine, f(fine), color=INK, lw=2.5, label="f(τ) = 1/(1+25τ²)")
            top.plot(
                fine,
                BarycentricInterpolator(uniform, f(uniform))(fine),
                color=uni,
                lw=1.6,
                label="等間隔の点で補間",
            )
            top.plot(
                fine,
                BarycentricInterpolator(nodes, f(nodes))(fine),
                color=lgl,
                lw=2.2,
                label="LGLノードで補間",
            )
            top.plot(uniform, f(uniform), "x", color=uni, ms=8, mew=2)
            top.plot(nodes, f(nodes), "o", color=lgl, ms=7)
            top.set_ylim(-0.6, 1.6)
            top.set_title(
                "補間：等間隔の点は端で振動（Runge現象）、LGLは収束", loc="left"
            )
            top.legend(loc="upper right", fontsize=11)
            bottom.plot(fine, df(fine), color=INK, lw=2.5, label="厳密な導関数 f′(τ)")
            bottom.plot(
                nodes,
                d @ f(nodes),
                "o",
                color=lgl,
                ms=7,
                label="微分行列 D をノード値に掛けた値",
            )
            bottom.set_ylim(-5, 5)
            bottom.set_xlabel("正規化時間 τ（区間の時刻を [-1, 1] に写したもの）")
            bottom.set_title("微分：D × (ノード値) ≈ ノードでの導関数", loc="left")
            bottom.legend(loc="upper right", fontsize=11)
            shown = counts[: k + 1]
            err.semilogy(
                shown, errors["lgl"][: k + 1], "o-", color=lgl, lw=2, label="LGLノード"
            )
            err.semilogy(
                shown,
                errors["uniform"][: k + 1],
                "x-",
                color=uni,
                lw=2,
                label="等間隔の点",
            )
            err.set_xlim(counts[0] - 1, counts[-1] + 1)
            err.set_ylim(1e-4, 1e4)
            err.set_xlabel("ノード数 N")
            err.set_ylabel("補間の最大誤差")
            err.set_title("N を増やしたときの誤差", loc="left")
            err.legend(loc="upper left", fontsize=11)
            err.text(
                0.02,
                0.02,
                "縦軸は対数目盛り",
                transform=err.transAxes,
                color=MUTED,
                fontsize=10,
            )
            for _ in range(int(0.9 * FPS)):
                writer.grab_frame()
        for _ in range(int(1.5 * FPS)):
            writer.grab_frame()
    fig.savefig(stem.with_suffix(".png"), dpi=DPI)
    plt.close(fig)
    stamp.write_text(digest)
    return "concepts: built"


def write_index():
    examples = []
    for example in EXAMPLES:
        path = OUTPUT / example.id / "manifest.json"
        if path.exists():
            examples.append(example.id)
    (OUTPUT / "index.json").write_text(json.dumps({"examples": examples}) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ids", nargs="*", help="example IDs (default: all)")
    parser.add_argument("--force", action="store_true", help="ignore the cache")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    args = parser.parse_args(argv)
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg is required to write MP4 files")
    ids = args.ids or [example.id for example in EXAMPLES]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print(build_concepts(args.force), flush=True)
    failures = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(build_example, i, args.force): i for i in ids}
        for future in as_completed(futures):
            try:
                print(future.result(), flush=True)
            except Exception as error:  # report every failing example
                failures.append(futures[future])
                print(f"{futures[future]}: FAILED: {error!r}", flush=True)
    write_index()
    if failures:
        sys.exit(f"failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
