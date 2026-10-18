"""Generate the videos, figures and data shown on the documentation site.

Each example is solved with its own ``solve`` or ``main`` entry point, sampled
per phase with the same barycentric interpolation used by ``examples._shared``,
and drawn with Matplotlib: an MP4 with the scene and graphs moving at the same
time, a poster PNG of the final frame, and the example's own static figures.

Outputs go to ``website/public/media/<example id>/`` with a ``manifest.json``
that the site reads at build time. An example is regenerated only when its
sources, the library or this script change, so CI can cache the directory.

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
from collections.abc import Callable
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
OUTPUT = ROOT / "website" / "public" / "media"
sys.path.insert(0, str(ROOT))

from examples.catalog import EXAMPLES, find_example, load_example  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scenes import Flight, Series, render_flight  # noqa: E402

FPS = 30
SECONDS = 8
SAMPLES = 241
COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]

Phase = dict[str, np.ndarray]


# ---------------------------------------------------------------------------
# Data: physical arrays per phase, sampled on a uniform grid of each phase.


def sample_problem(prob, states, controls, samples=SAMPLES) -> list[Phase]:
    """Sample a solved Problem like ``examples._shared.collect`` does."""
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


def trajectory_phases(trajectory) -> list[Phase]:
    return [
        {key: np.asarray(value) for key, value in phase["samples"].items()}
        for phase in trajectory.phases
    ]


def joined(phases: list[Phase], key: str) -> np.ndarray:
    return np.concatenate([phase[key] for phase in phases])


# ---------------------------------------------------------------------------
# Views: what to draw for an example. Values are functions of one phase.

Getter = Callable[[Phase], np.ndarray]


def key(name: str, scale: float = 1.0) -> Getter:
    return lambda phase: phase[name] * scale


@dataclass
class Curve:
    label: str
    value: Getter


@dataclass
class Panel:
    ylabel: str
    curves: list[Curve]


@dataclass
class Scene:
    """Left-hand picture: a path in a plane, a rail, or a vertical flight."""

    kind: str  # "path", "rail" or "vertical"
    x: Getter | None = None
    y: Getter | None = None
    xlabel: str = ""
    ylabel: str = ""
    equal: bool = False
    invert_y: bool = False
    thrust: Getter | None = None  # arrow / flame length, normalised to max 1
    legend: str = ""  # what the arrow or flame shows
    ground: bool = True  # include y = 0 in a vertical scene
    decorate: Callable[[Any], None] | None = None
    title: str = ""


@dataclass
class View:
    scene: Scene
    panels: list[Panel]
    time_label: str = "time [s]"
    title: str = ""


def path_scene(x, y, xlabel, ylabel, **kwargs) -> Scene:
    return Scene("path", x=x, y=y, xlabel=xlabel, ylabel=ylabel, **kwargs)


def thrust_ratio(name: str) -> Getter:
    return lambda phase: phase[name]


def polar_xy(phase: Phase, re=6371.0e3):
    """Downrange-up coordinates [km] with the launch site at the origin."""
    r, theta = phase["R"], phase["theta"]
    return r * np.sin(theta) / 1e3, (r * np.cos(theta) - re) / 1e3


def earth_surface(ax, re_km=6371.0):
    left, right = ax.get_xlim()
    xs = np.linspace(left, right, 600)
    ax.plot(xs, np.sqrt(re_km**2 - xs**2) - re_km, color="#78716c", lw=1.5)
    ax.set_xlim(left, right)


# ---------------------------------------------------------------------------
# Drawing


@dataclass
class Rendered:
    video: str
    poster: str
    label: str
    caption: str = ""


def _frame_times(phases: list[Phase], frames: int) -> np.ndarray:
    t = joined(phases, "time")
    return np.linspace(t[0], t[-1], frames)


def _interp(phases: list[Phase], getter: Getter, times: np.ndarray) -> np.ndarray:
    t = joined(phases, "time")
    v = np.concatenate([getter(phase) for phase in phases])
    return np.interp(times, t, v)


def render(view: View, phases: list[Phase], stem: Path, frames=FPS * SECONDS):
    """Write ``stem.mp4`` and ``stem.png`` (the final frame)."""
    rows = max(len(view.panels), 1)
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, layout="constrained")
    grid = fig.add_gridspec(rows, 2, width_ratios=[1.15, 1])
    scene_ax = fig.add_subplot(grid[:, 0])
    panel_axes = [fig.add_subplot(grid[i, 1]) for i in range(rows)]
    for ax in panel_axes[1:]:
        ax.sharex(panel_axes[0])
    if view.title:
        fig.suptitle(view.title, fontsize=14)

    times = _frame_times(phases, frames)
    knots = [phase["time"][0] for phase in phases[1:]]
    scene = view.scene
    thrust = None
    if scene.thrust is not None:
        thrust = _interp(phases, scene.thrust, times)
        peak = np.max(np.abs(thrust))
        thrust = thrust / peak if peak > 0 else thrust

    # --- scene
    marker = arrow = flame = trail = None
    if scene.kind == "path":
        assert scene.x is not None and scene.y is not None
        for i, phase in enumerate(phases):
            scene_ax.plot(
                scene.x(phase), scene.y(phase), color=COLORS[i % 6], alpha=0.25, lw=2
            )
        xs = _interp(phases, scene.x, times)
        ys = _interp(phases, scene.y, times)
        (trail,) = scene_ax.plot([], [], color=COLORS[0], lw=2.5)
        (marker,) = scene_ax.plot([], [], "o", color="#111827", ms=10, zorder=5)
        scene_ax.set(xlabel=scene.xlabel, ylabel=scene.ylabel)
        if scene.equal:
            scene_ax.set_aspect("equal", adjustable="datalim")
        if scene.invert_y:
            scene_ax.invert_yaxis()
        scene_ax.grid(alpha=0.4)
    elif scene.kind == "rail":
        assert scene.x is not None
        xs = _interp(phases, scene.x, times)
        ys = np.zeros_like(xs)
        span = float(np.max(xs) - np.min(xs)) or 1.0
        scene_ax.axhline(0, color="#78716c", lw=3)
        scene_ax.axvline(xs[-1], color="#16a34a", ls="--")
        scene_ax.set_xlim(np.min(xs) - 0.1 * span, np.max(xs) + 0.1 * span)
        scene_ax.set_ylim(-0.4 * span, 0.4 * span)
        scene_ax.set_aspect("equal")
        scene_ax.set_yticks([])
        scene_ax.set_xlabel(scene.xlabel)
        (marker,) = scene_ax.plot([], [], "s", color="#111827", ms=22, zorder=5)
        arrow = scene_ax.annotate(
            "",
            xy=(0, 0),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "-|>", "color": "#dc2626", "lw": 3},
        )
        scene_ax.text(
            0.02,
            0.95,
            scene.legend or "red arrow: acceleration",
            transform=scene_ax.transAxes,
        )
    else:  # vertical
        assert scene.y is not None
        ys = _interp(phases, scene.y, times)
        xs = np.zeros_like(ys)
        low, high = float(np.min(ys)), float(np.max(ys))
        pad = 0.08 * ((high - low) or 1.0)
        scene_ax.set_ylim((min(low, 0) if scene.ground else low) - pad, high + pad)
        scene_ax.set_xlim(-1, 1)
        scene_ax.set_xticks([])
        scene_ax.set_ylabel(scene.ylabel)
        if scene.ground:
            scene_ax.axhline(0, color="#78716c", lw=3)
        scene_ax.grid(axis="y", alpha=0.4)
        (marker,) = scene_ax.plot([], [], "^", color="#111827", ms=22, zorder=5)
        (flame,) = scene_ax.plot([], [], color="#f97316", lw=10, alpha=0.8)
        scene_ax.text(
            0.02,
            0.97,
            scene.legend or "orange: thrust (relative)",
            transform=scene_ax.transAxes,
        )
    if scene.decorate is not None:
        scene.decorate(scene_ax)
    if scene.title:
        scene_ax.set_title(scene.title)
    clock = scene_ax.text(
        0.98,
        0.02,
        "",
        transform=scene_ax.transAxes,
        ha="right",
        fontsize=12,
        family="monospace",
    )

    # --- graphs
    cursors, dots = [], []
    for ax, panel in zip(panel_axes, view.panels, strict=False):
        for phase_index, phase in enumerate(phases):
            for j, curve in enumerate(panel.curves):
                ax.plot(
                    phase["time"],
                    curve.value(phase),
                    color=COLORS[j % 6],
                    lw=2,
                    label=curve.label if phase_index == 0 else None,
                )
        for knot in knots:
            ax.axvline(knot, color="#6b7280", ls=":", lw=1)
        ax.set_ylabel(panel.ylabel)
        ax.grid(alpha=0.4)
        if len(panel.curves) > 1:
            ax.legend(loc="best", fontsize=9)
        cursors.append(ax.axvline(times[0], color="#111827", lw=1))
        values = [_interp(phases, c.value, times) for c in panel.curves]
        points = [
            ax.plot([], [], "o", color=COLORS[j % 6], ms=7, zorder=5)[0]
            for j in range(len(values))
        ]
        dots.append((points, values))
    panel_axes[-1].set_xlabel(view.time_label)
    for ax in panel_axes[:-1]:
        ax.tick_params(labelbottom=False)

    def draw(i):
        t = times[i]
        clock.set_text(f"t = {t:9.3f}")
        assert marker is not None
        if scene.kind == "path":
            assert trail is not None
            trail.set_data(xs[: i + 1], ys[: i + 1])
            marker.set_data([xs[i]], [ys[i]])
        elif scene.kind == "rail":
            marker.set_data([xs[i]], [0])
            assert arrow is not None and thrust is not None
            length = 0.25 * span * thrust[i]
            arrow.xy = (xs[i] + length, 0.12 * span)
            arrow.set_position((xs[i], 0.12 * span))
        else:
            marker.set_data([0], [ys[i]])
            assert flame is not None
            if thrust is not None:
                size = 0.15 * (scene_ax.get_ylim()[1] - scene_ax.get_ylim()[0])
                flame.set_data([0, 0], [ys[i], ys[i] - size * max(thrust[i], 0)])
        for cursor, (points, values) in zip(cursors, dots, strict=True):
            cursor.set_xdata([t, t])
            for point, value in zip(points, values, strict=True):
                point.set_data([t], [value[i]])

    writer = FFMpegWriter(
        fps=FPS,
        codec="libx264",
        bitrate=-1,
        extra_args=["-pix_fmt", "yuv420p", "-crf", "26", "-movflags", "+faststart"],
    )
    with writer.saving(fig, str(stem.with_suffix(".mp4")), dpi=100):
        for i in range(frames):
            draw(i)
            writer.grab_frame()
    fig.savefig(stem.with_suffix(".png"), dpi=100)
    plt.close(fig)


def compare(view: View, cases: dict[str, list[Phase]], path: Path, title: str):
    """Overlay several solved cases: the scene path and every graph."""
    rows = len(view.panels)
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, layout="constrained")
    grid = fig.add_gridspec(rows, 2, width_ratios=[1.15, 1])
    scene_ax = fig.add_subplot(grid[:, 0])
    panel_axes = [fig.add_subplot(grid[i, 1]) for i in range(rows)]
    scene = view.scene
    for color, (name, phases) in zip(COLORS * 3, cases.items(), strict=False):
        if scene.kind == "path" and scene.x and scene.y:
            sx, sy = scene.x, scene.y
            scene_ax.plot(joined_map(phases, sx), joined_map(phases, sy), color=color)
        elif scene.kind == "vertical" and scene.y:
            sy = scene.y
            scene_ax.plot(joined(phases, "time"), joined_map(phases, sy), color=color)
        elif scene.x:
            sx = scene.x
            scene_ax.plot(joined(phases, "time"), joined_map(phases, sx), color=color)
        for ax, panel in zip(panel_axes, view.panels, strict=True):
            for j, curve in enumerate(panel.curves):
                ax.plot(
                    joined(phases, "time"),
                    joined_map(phases, curve.value),
                    color=color,
                    ls=["-", "--", ":"][j % 3],
                    label=name if j == 0 else None,
                )
            ax.set_ylabel(panel.ylabel)
            ax.grid(alpha=0.4)
    if scene.kind == "path":
        scene_ax.set(xlabel=scene.xlabel, ylabel=scene.ylabel)
        if scene.equal:
            scene_ax.set_aspect("equal", adjustable="datalim")
        if scene.invert_y:
            scene_ax.invert_yaxis()
    else:
        label = scene.ylabel if scene.kind == "vertical" else scene.xlabel
        scene_ax.set(xlabel=view.time_label, ylabel=label)
    if scene.decorate is not None and scene.kind == "path":
        scene.decorate(scene_ax)
    scene_ax.grid(alpha=0.4)
    panel_axes[0].legend(fontsize=9)
    panel_axes[-1].set_xlabel(view.time_label)
    fig.suptitle(title, fontsize=14)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def joined_map(phases: list[Phase], getter: Getter) -> np.ndarray:
    return np.concatenate([getter(phase) for phase in phases])


# ---------------------------------------------------------------------------
# Example definitions


def cycloid_free_end(ax, length=1.0):
    phi = np.linspace(0, np.pi, 200)
    a = length / np.pi
    ax.plot(
        a * (phi - np.sin(phi)),
        a * (1 - np.cos(phi)),
        "k--",
        lw=1.2,
        label="cycloid (analytic)",
    )
    ax.legend(loc="lower left")


def deg(name: str) -> Getter:
    return lambda phase: np.rad2deg(phase[name])


def altitude_km(phase: Phase) -> np.ndarray:
    return (phase["R"] - 6371.0e3) / 1e3


def speed_polar(phase: Phase) -> np.ndarray:
    return np.hypot(phase["Vr"], phase["Vt"]) / 1e3


def thrust_norm(phase: Phase) -> np.ndarray:
    return np.hypot(phase["Tr"], phase["Tt"]) / 1e3


def low_thrust_xy(phases: list[Phase]):
    """Integrate theta' = vt/r continuously across all samples."""
    from scipy.integrate import cumulative_trapezoid

    t = joined(phases, "time")
    rate = joined(phases, "vt") / joined(phases, "r")
    theta = cumulative_trapezoid(rate, t, initial=0)
    start = 0
    for phase in phases:
        n = len(phase["time"])
        part = theta[start : start + n]
        phase["px"] = phase["r"] * np.cos(part)
        phase["py"] = phase["r"] * np.sin(part)
        start += n


def orbit_circles(ax):
    for radius in (1.0, 4.0):
        ax.add_patch(plt.Circle((0, 0), radius, ls="--", fill=False, color="#6b7280"))
    ax.plot([0], [0], "o", color="#f59e0b", ms=14)


def polar_view(title: str) -> View:
    return View(
        scene=path_scene(
            lambda p: polar_xy(p)[0],
            lambda p: polar_xy(p)[1],
            "downrange [km]",
            "altitude above launch site [km]",
            equal=True,
            decorate=earth_surface,
        ),
        panels=[
            Panel("altitude [km]", [Curve("altitude", altitude_km)]),
            Panel("speed [km/s]", [Curve("|V|", speed_polar)]),
            Panel("mass [t]", [Curve("m", key("m", 1e-3))]),
            Panel(
                "thrust [kN]",
                [
                    Curve("Tr", key("Tr", 1e-3)),
                    Curve("Tt", key("Tt", 1e-3)),
                ],
            ),
        ],
        title=title,
    )


def rocket_1d_view(title: str) -> View:
    return View(
        scene=Scene(
            "vertical",
            y=lambda p: (p["R"] - 6371.0e3) / 1e3,
            ylabel="altitude [km]",
            thrust=key("T"),
        ),
        panels=[
            Panel("velocity [km/s]", [Curve("v", key("v", 1e-3))]),
            Panel("mass [t]", [Curve("m", key("m", 1e-3))]),
            Panel("thrust [kN]", [Curve("T", key("T", 1e-3))]),
        ],
        title=title,
    )


def goddard_view(title: str) -> View:
    return View(
        scene=Scene(
            "vertical",
            y=key("h"),
            ylabel="altitude h [-] (zoomed, ground not shown)",
            thrust=key("T"),
            ground=False,
        ),
        panels=[
            Panel("altitude h [-]", [Curve("h", key("h"))]),
            Panel("velocity v [-]", [Curve("v", key("v"))]),
            Panel("mass m [-]", [Curve("m", key("m"))]),
            Panel("thrust T [-]", [Curve("T", key("T"))]),
        ],
        time_label="time [-]",
        title=title,
    )


@dataclass
class Classic:
    """An example with ``main(flag_savefig=...)`` returning the solved Problem."""

    states: list[str]
    controls: list[str]
    view: View
    caption: str = ""
    prepare: Callable[[list[Phase]], None] | None = None
    savefig_flag: bool = True


CLASSIC: dict[str, Classic] = {
    "minimum_energy_transfer": Classic(
        ["x"],
        ["u"],
        View(
            Scene(
                "rail",
                x=key("x"),
                xlabel="position x",
                thrust=key("u"),
                legend="red arrow: control u = dx/dt",
            ),
            [
                Panel("position x", [Curve("x", key("x"))]),
                Panel("control u = x'", [Curve("u", key("u"))]),
            ],
            title="Free-time transfer: min T + integral(u^2 dt)",
        ),
        savefig_flag=False,
    ),
    "brachistochrone": Classic(
        ["x", "y", "v"],
        ["theta"],
        View(
            path_scene(
                key("x"),
                key("y"),
                "x [m]",
                "y [m] (downward)",
                equal=True,
                invert_y=True,
                decorate=cycloid_free_end,
            ),
            [
                Panel("speed v [m/s]", [Curve("v", key("v"))]),
                Panel("theta [deg]", [Curve("theta", deg("theta"))]),
            ],
            title="Brachistochrone: numerical solution vs analytic cycloid",
        ),
    ),
    "brachistochrone_tokyo_osaka": Classic(
        ["x", "y", "v"],
        ["theta"],
        View(
            path_scene(
                key("x", 1e-3),
                key("y", 1e-3),
                "x [km]",
                "depth [km]",
                equal=True,
                invert_y=True,
            ),
            [
                Panel("speed [km/s]", [Curve("v", key("v", 1e-3))]),
                Panel("theta [deg]", [Curve("theta", deg("theta"))]),
            ],
            title="Gravity train Tokyo - Osaka (600 km)",
        ),
    ),
    "simple_rocket_2d": Classic(
        ["u", "v", "x", "y"],
        ["beta"],
        View(
            path_scene(key("x"), key("y"), "x [-]", "y [-]"),
            [
                Panel("velocity [-]", [Curve("u", key("u")), Curve("v", key("v"))]),
                Panel("thrust angle beta [deg]", [Curve("beta", deg("beta"))]),
            ],
            time_label="time [-]",
            title="2D rocket with constant acceleration",
        ),
    ),
    "goddard_0knot": Classic(
        ["h", "v", "m"], ["T"], goddard_view("Goddard problem, single phase")
    ),
    "goddard_1knot": Classic(
        ["h", "v", "m"], ["T"], goddard_view("Goddard problem, two phases (1 knot)")
    ),
    "rocket_ascent_single_stage": Classic(
        ["R", "v", "m"], ["T"], rocket_1d_view("Single-stage vertical ascent")
    ),
    "rocket_ascent_two_stage": Classic(
        ["R", "v", "m"],
        ["T"],
        rocket_1d_view("Two-stage vertical ascent (dotted line: staging)"),
    ),
    "rocket_ascent_polar_ssto": Classic(
        ["R", "theta", "Vr", "Vt", "m"],
        ["Tr", "Tt"],
        polar_view("Polar coordinates: single stage to orbit"),
    ),
    "rocket_ascent_polar_tsto": Classic(
        ["R", "theta", "Vr", "Vt", "m"],
        ["Tr", "Tt"],
        polar_view("Polar coordinates: two stages to orbit (dotted: staging)"),
    ),
    "polar_tsto_taiki": Classic(
        ["R", "theta", "Vr", "Vt", "m"],
        ["Tr", "Tt"],
        polar_view("Two stages to orbit from Taiki (with air drag)"),
    ),
    "low_thrust_orbit_transfer": Classic(
        ["r", "vr", "vt"],
        ["ur1", "ur2", "ut1", "ut2"],
        View(
            path_scene(
                key("px"),
                key("py"),
                "x [-]",
                "y [-]",
                equal=True,
                decorate=orbit_circles,
            ),
            [
                Panel("radius r [-]", [Curve("r", key("r"))]),
                Panel("velocity [-]", [Curve("vr", key("vr")), Curve("vt", key("vt"))]),
                Panel(
                    "thrust [-]",
                    [
                        Curve("ur", lambda p: p["ur1"] - p["ur2"]),
                        Curve("ut", lambda p: p["ut1"] - p["ut2"]),
                    ],
                ),
            ],
            time_label="time [-]",
            title="Low-thrust transfer from r = 1 to r = 4",
        ),
        prepare=low_thrust_xy,
    ),
}


# ---------------------------------------------------------------------------
# Rocket and orbit examples: explanatory flight animations (see scenes.py)

RE = 6371.0e3


def _km(name: str) -> Getter:
    return lambda p: (p[name] - RE) / 1e3


def _ascent_thrust(p: Phase) -> np.ndarray:
    return np.hypot(p["Tr"], p["Tt"])


def _ascent_pitch(p: Phase) -> np.ndarray:
    """Thrust elevation above the local horizontal [deg]."""
    return np.rad2deg(np.arctan2(p["Tr"], p["Tt"]))


def goddard_flight(title: str, names: list[str]) -> Flight:
    return Flight(
        "vertical",
        title,
        names,
        throttle=key("T"),
        altitude=key("h"),
        altitude_label="高度 h [-]（無次元）",
        altitude_unit="",
        theme="space",
        panels=[
            [Series("高度 h [-]", key("h"), "{:.5f}")],
            [Series("速度 v [-]", key("v"), "{:.4f}")],
            [Series("質量 m [-]", key("m"), "{:.4f}")],
            [Series("推力 T [-]", key("T"), "{:.3f}")],
        ],
        readout=[
            Series("高度", key("h"), "{:.5f}"),
            Series("速度", key("v"), "{:.4f}"),
            Series("質量", key("m"), "{:.4f}"),
            Series("推力", key("T"), "{:.3f}"),
        ],
        time_unit="-",
        note="無次元量（地球半径=1）。ロケットの図は模式図、炎の長さは各段の最大推力に対する比",
    )


def vertical_rocket_flight(title: str, names: list[str], staging: bool) -> Flight:
    return Flight(
        "vertical",
        title,
        names,
        throttle=key("T"),
        altitude=_km("R"),
        altitude_label="高度 [km]",
        altitude_unit="km",
        staging=staging,
        panels=[
            [Series("高度 [km]", _km("R"), "{:.1f}", "km")],
            [Series("速度 [km/s]", key("v", 1e-3), "{:.3f}", "km/s")],
            [Series("質量 [t]", key("m", 1e-3), "{:.3f}", "t")],
            [Series("推力 [kN]", key("T", 1e-3), "{:.1f}", "kN")],
        ],
        readout=[
            Series("高度", _km("R"), "{:.1f}", "km"),
            Series("速度", key("v", 1e-3), "{:.3f}", "km/s"),
            Series("質量", key("m"), "{:.0f}", "kg"),
            Series("推力", key("T", 1e-3), "{:.1f}", "kN"),
        ],
        note="ロケットの図は模式図。炎の長さは各段の最大推力に対する比",
    )


def ascent_flight(title: str, names: list[str], staging: bool) -> Flight:
    return Flight(
        "ascent",
        title,
        names,
        throttle=_ascent_thrust,
        staging=staging,
        radius=key("R"),
        angle=key("theta"),
        thrust_radial=key("Tr"),
        thrust_tangential=key("Tt"),
        panels=[
            [Series("高度 [km]", _km("R"), "{:.1f}", "km")],
            [
                Series("水平速度 Vt", key("Vt", 1e-3), "{:.3f}", "km/s"),
                Series("鉛直速度 Vr", key("Vr", 1e-3), "{:.3f}", "km/s"),
            ],
            [Series("質量 [t]", key("m", 1e-3), "{:.2f}", "t")],
            [Series("推力 [kN]", lambda p: _ascent_thrust(p) / 1e3, "{:.0f}", "kN")],
            [
                Series(
                    "推力の仰角 [deg]",
                    _ascent_pitch,
                    "{:.1f}",
                    "deg",
                    mask_coast=True,
                )
            ],
        ],
        readout=[
            Series("高度", _km("R"), "{:.1f}", "km"),
            Series(
                "速さ", lambda p: np.hypot(p["Vr"], p["Vt"]) / 1e3, "{:.3f}", "km/s"
            ),
            Series("質量", key("m"), "{:.0f}", "kg"),
            Series("推力", lambda p: _ascent_thrust(p) / 1e3, "{:.0f}", "kN"),
        ],
    )


FLIGHTS: dict[str, Flight] = {
    "goddard_0knot": goddard_flight("Goddard問題：単一区間", ["単一区間"]),
    "goddard_1knot": goddard_flight(
        "Goddard問題：区間分割（knot 1つ）", ["区間1", "区間2"]
    ),
    "rocket_ascent_single_stage": vertical_rocket_flight(
        "単段ロケットの鉛直上昇", ["単段"], staging=False
    ),
    "rocket_ascent_two_stage": vertical_rocket_flight(
        "2段ロケットの鉛直上昇", ["1段目", "2段目"], staging=True
    ),
    "rocket_ascent_polar_ssto": ascent_flight(
        "極座標：単段で軌道投入", ["単段"], staging=False
    ),
    "rocket_ascent_polar_tsto": ascent_flight(
        "極座標：2段で軌道投入", ["1段目", "2段目"], staging=True
    ),
    "polar_tsto_taiki": ascent_flight(
        "大樹町からの2段軌道投入（空気抵抗あり）",
        ["1段目", "2段目"],
        staging=True,
    ),
    "low_thrust_orbit_transfer": Flight(
        "orbit",
        "低推力による軌道遷移（r = 1 → 4）",
        ["軌道遷移"],
        throttle=lambda p: np.hypot(p["ur1"] - p["ur2"], p["ut1"] - p["ut2"]),
        position=lambda p: (p["px"], p["py"]),
        thrust_radial=lambda p: p["ur1"] - p["ur2"],
        thrust_tangential=lambda p: p["ut1"] - p["ut2"],
        orbits=[1.0, 4.0],
        panels=[
            [Series("軌道半径 r [-]", key("r"), "{:.3f}")],
            [
                Series("周方向速度 vt", key("vt"), "{:.3f}", "-"),
                Series("動径方向速度 vr", key("vr"), "{:.3f}", "-"),
            ],
            [
                Series("周方向推力 ut", lambda p: p["ut1"] - p["ut2"], "{:.4f}", "-"),
                Series("動径方向推力 ur", lambda p: p["ur1"] - p["ur2"], "{:.4f}", "-"),
            ],
        ],
        readout=[
            Series("半径", key("r"), "{:.3f}"),
            Series("vt", key("vt"), "{:.3f}"),
            Series("vr", key("vr"), "{:.3f}"),
        ],
        time_unit="-",
        note="無次元量（初期軌道半径=1、中心天体の重力定数=1）",
        coast_speedup=1.0,
    ),
}


def soft_landing_flight(parameters) -> Flight:
    readout = [
        Series("高度", key("altitude"), "{:.1f}", "m"),
        Series("速度", key("velocity"), "{:.2f}", "m/s"),
        Series("推力", key("thrust", 1e-3), "{:.2f}", "kN"),
    ]
    panels = [
        [Series("高度 [m]", key("altitude"), "{:.1f}", "m")],
        [Series("速度 [m/s]（上向き正）", key("velocity"), "{:.2f}", "m/s")],
        [Series("推力 [kN]", key("thrust", 1e-3), "{:.2f}", "kN")],
    ]
    if parameters.variable_mass:
        panels.append([Series("質量 [kg]", key("mass"), "{:.1f}", "kg")])
        readout.append(Series("質量", key("mass"), "{:.1f}", "kg"))
    return Flight(
        "vertical",
        "月面への鉛直軟着陸",
        ["慣性降下", "減速噴射"],
        throttle=key("thrust"),
        altitude=key("altitude"),
        altitude_label="高度 [m]",
        altitude_unit="m",
        theme="moon",
        panels=panels,
        readout=readout,
        note="着陸機の図は模式図。炎の長さは各段の最大推力に対する比",
        coast_speedup=1.0,
    )


def floor_line(max_depth):
    def decorate(ax):
        if max_depth is not None:
            ax.axhline(max_depth, color="#78716c", lw=2, ls="--", label="floor")
            ax.legend(loc="lower left")

    return decorate


def shared_view(example_id: str, parameters) -> View:
    if example_id == "move_stop":
        return View(
            Scene(
                "rail",
                x=key("position"),
                xlabel="position [m]",
                thrust=key("acceleration"),
            ),
            [
                Panel("position [m]", [Curve("x", key("position"))]),
                Panel("velocity [m/s]", [Curve("v", key("velocity"))]),
                Panel("acceleration [m/s²]", [Curve("a", key("acceleration"))]),
            ],
        )
    if example_id == "constrained_descent":
        return View(
            path_scene(
                key("x"),
                key("depth"),
                "x [m]",
                "depth [m] (downward)",
                equal=True,
                invert_y=True,
                decorate=floor_line(parameters.max_depth),
            ),
            [
                Panel("speed [m/s]", [Curve("speed", key("speed"))]),
                Panel("theta [deg]", [Curve("theta", deg("theta"))]),
            ],
        )
    if example_id == "soft_landing":
        panels = [
            Panel("altitude [m]", [Curve("h", key("altitude"))]),
            Panel("velocity [m/s]", [Curve("v", key("velocity"))]),
            Panel("thrust [kN]", [Curve("T", key("thrust", 1e-3))]),
        ]
        if parameters.variable_mass:
            panels.append(Panel("mass [kg]", [Curve("m", key("mass"))]))
        return View(
            Scene(
                "vertical",
                y=key("altitude"),
                ylabel="altitude [m]",
                thrust=key("thrust"),
            ),
            panels,
        )
    raise KeyError(example_id)


# ---------------------------------------------------------------------------
# Jobs


def source_hash(example) -> str:
    digest = hashlib.sha256()
    files = sorted((ROOT / "examples" / example.directory).rglob("*"))
    files += sorted((ROOT / "src" / "OpenGoddard").glob("*.py"))
    files += [ROOT / "examples" / "_shared.py"]
    files += sorted(Path(__file__).resolve().parent.glob("*.py"))
    for path in files:
        if path.is_file() and "output" not in path.parts and path.suffix != ".pyc":
            digest.update(path.relative_to(ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass
class Manifest:
    id: str
    directory: str
    title: str
    hash: str
    videos: list[dict[str, str]] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)
    comparison: str | None = None
    data: list[str] = field(default_factory=list)
    # Solver status per video; a failed solve is still drawn but flagged.
    solver: dict[str, dict[str, Any]] = field(default_factory=dict)


def build_shared(example, module, out: Path, manifest: Manifest):
    cases = {}
    for name, parameters in module.PRESETS.items():
        trajectory = module.solve(parameters, samples=SAMPLES)
        manifest.solver[name] = {
            "success": trajectory.solver["success"],
            "message": trajectory.solver["message"],
        }
        trajectory.write_json(out / f"{name}.json")
        manifest.data.append(f"{name}.json")
        phases = trajectory_phases(trajectory)
        cases[name] = phases
        view = shared_view(example.id, parameters)
        objective = trajectory.solver["objective"]
        view.title = f"{example.id} / preset: {name}  (objective = {objective:.4g})"
        if example.id == "soft_landing":
            flight = soft_landing_flight(parameters)
            flight.title += f"（プリセット {name}）"
            render_flight(flight, phases, out / name)
        else:
            render(view, phases, out / name)
        manifest.videos.append(
            {"name": name, "video": f"{name}.mp4", "poster": f"{name}.png"}
        )
    view = shared_view(example.id, module.PRESETS["default"])
    view.scene.decorate = None
    compare(view, cases, out / "comparison.png", f"{example.id}: all presets")
    manifest.comparison = "comparison.png"


def build_classic(example, module, out: Path, manifest: Manifest):
    spec = CLASSIC[example.id]
    output = ROOT / "examples" / example.directory / "output"
    shutil.rmtree(output, ignore_errors=True)
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        if spec.savefig_flag:
            prob = module.main(flag_savefig=True)
        else:
            prob = module.main()
    plt.close("all")
    result = prob.result
    manifest.solver["solution"] = {
        "success": bool(result.success),
        "message": str(result.message),
    }
    phases = sample_problem(prob, spec.states, spec.controls)
    if spec.prepare is not None:
        spec.prepare(phases)
    if example.id in FLIGHTS:
        flight = FLIGHTS[example.id]
        if flight.kind == "ascent":
            flight.target_altitude = module.Rocket().Htarget  # the example's own
        render_flight(flight, phases, out / "solution")
    else:
        render(spec.view, phases, out / "solution")
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
    manifest = Manifest(example.id, example.directory, example.title, digest)
    if hasattr(module, "PRESETS"):
        build_shared(example, module, out, manifest)
    else:
        build_classic(example, module, out, manifest)
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
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
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
    counts = list(range(4, 25, 2))
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(12.8, 7.2), dpi=100, layout="constrained"
    )
    writer = FFMpegWriter(
        fps=FPS,
        codec="libx264",
        bitrate=-1,
        extra_args=["-pix_fmt", "yuv420p", "-crf", "26", "-movflags", "+faststart"],
    )
    with writer.saving(fig, str(stem.with_suffix(".mp4")), dpi=100):
        for n in counts:
            nodes, _, d = helper.method_LGL(n)
            uniform = np.linspace(-1, 1, n)
            for ax in (top, bottom):
                ax.clear()
                ax.grid(alpha=0.4)
            top.plot(fine, f(fine), "k", lw=2, label="f(x) = 1/(1+25x²)")
            top.plot(
                fine,
                BarycentricInterpolator(uniform, f(uniform))(fine),
                color=COLORS[1],
                lw=1.5,
                label="interpolation on equally spaced points",
            )
            top.plot(
                fine,
                BarycentricInterpolator(nodes, f(nodes))(fine),
                color=COLORS[0],
                lw=2,
                label="interpolation on LGL nodes",
            )
            top.plot(nodes, f(nodes), "o", color=COLORS[0], ms=7)
            top.plot(uniform, f(uniform), "x", color=COLORS[1], ms=7)
            top.set_ylim(-0.6, 1.6)
            top.set_title(f"N = {n} nodes", fontsize=14)
            top.legend(loc="upper right", fontsize=10)
            bottom.plot(fine, df(fine), "k", lw=2, label="exact f'(x)")
            bottom.plot(
                nodes,
                d @ f(nodes),
                "o",
                color=COLORS[0],
                ms=7,
                label="D @ f(nodes)  (differentiation matrix)",
            )
            bottom.set_ylim(-5, 5)
            bottom.set_xlabel("tau  (phase time mapped to [-1, 1])")
            bottom.legend(loc="upper right", fontsize=10)
            for _ in range(FPS):
                writer.grab_frame()
    fig.savefig(stem.with_suffix(".png"), dpi=100)
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
