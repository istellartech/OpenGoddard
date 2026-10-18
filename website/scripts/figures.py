"""Static figures and MP4 animations drawn from one solved example.

A ``Solution`` holds every phase twice: the values at the LGL nodes (the actual
optimization variables) and a dense sampling of the interpolating polynomial.
Figures draw the polynomial as a line and the nodes as dots, so the reader can
see what the solver really computed.

``Spec`` (see specs.py) says what to draw:

* ``history``: small multiples of states and controls against time.
* ``scene``: the motion itself (a path in a plane, a vertical column, a launch
  seen from the side, or an orbit), drawn as a figure and as an animation.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Polygon
from matplotlib.transforms import IdentityTransform
from style import FLAME, INK, LIMIT, MUTED, REFERENCE, phase_color

Data = dict[str, Any]  # arrays keyed by quantity name, plus "time"/"section"


@dataclass
class Solution:
    nodes: list[Data]  # one dict per phase, values at the LGL nodes
    dense: list[Data]  # one dict per phase, interpolated values

    def cat(self, key: str, dense: bool = True) -> np.ndarray:
        phases = self.dense if dense else self.nodes
        return np.concatenate([np.asarray(p[key], dtype=float) for p in phases])

    @property
    def knots(self) -> list[float]:
        return [float(p["time"][-1]) for p in self.dense[:-1]]


@dataclass
class Quantity:
    key: str
    label: str  # Japanese name with the symbol, e.g. "速さ $v$"
    unit: str = ""
    scale: float = 1.0

    def axis_label(self) -> str:
        return f"{self.label} [{self.unit}]" if self.unit else self.label

    def of(self, data: Data) -> np.ndarray:
        return np.asarray(data[self.key], dtype=float) * self.scale


Reference = tuple[np.ndarray, np.ndarray, str]  # time, value (display), label


@dataclass
class Panel:
    quantities: list[Quantity]
    limits: list[tuple[float, str]] = field(default_factory=list)  # display units
    references: Callable[[Solution], list[Reference]] | None = None
    title: str | None = None  # defaults to the first quantity's axis label
    step: bool = False  # draw nodes only (piecewise controls)


@dataclass
class Scene:
    kind: str  # "rail", "plane", "column", "ascent" or "orbit"
    x: Quantity
    y: Quantity
    body: str = "ball"  # "ball", "cart", "rocket", "lander" or "probe"
    invert_y: bool = False
    equal: bool = False
    decorate: Callable[[Any, Solution], None] | None = None
    # Direction of the body / thrust arrow in the plane [rad from +x], if any
    heading: Callable[[Data], np.ndarray] | None = None
    throttle: Callable[[Data], np.ndarray] | None = None  # 0..1 flame length
    ghost: Callable[[float], tuple[float, float]] | None = None  # rival body
    ghost_label: str = ""
    label: str = ""  # caption drawn inside the scene
    readout: list[Quantity] = field(default_factory=list)
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    ground: float | None = None  # y of the ground (display units)


@dataclass
class Spec:
    states: list[str]
    controls: list[str]
    history: list[Panel]
    scene: Scene | None = None
    video_panels: list[int] = field(default_factory=list)  # indices of history
    phase_names: list[str] = field(default_factory=list)
    time_unit: str = "s"
    derive: Callable[[Data, Any], None] | None = None  # (phase data, module)
    extras: Callable[[Solution, Any, Any], dict] | None = None
    seconds: float = 10.0
    coast_speedup: float = 1.0  # faster playback while throttle < 3 %


# ---------------------------------------------------------------------------
# Small helpers


def _time_label(spec: Spec) -> str:
    return f"時刻 $t$ [{spec.time_unit}]" if spec.time_unit else "時刻 $t$（無次元）"


def _shade_phases(ax, sol: Solution, spec: Spec) -> None:
    if len(sol.dense) < 2:
        return
    for knot in sol.knots:
        ax.axvline(knot, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)


def _phase_legend(fig, sol: Solution, spec: Spec, y=1.0) -> None:
    if len(sol.dense) < 2 or not spec.phase_names:
        return
    handles = [
        plt.Line2D([], [], color=phase_color(i), lw=2.5, label=name)
        for i, name in enumerate(spec.phase_names)
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        ncols=len(handles),
        bbox_to_anchor=(1.0, y),
        fontsize=9,
    )


def _draw_panel(ax, panel: Panel, sol: Solution, spec: Spec, nodes=True) -> list:
    """Draw one history panel; return the line artists of the solution."""
    artists = []
    many = len(panel.quantities) > 1
    styles = ["-", (0, (5, 2))]
    for qi, quantity in enumerate(panel.quantities):
        for section, (dense, node) in enumerate(zip(sol.dense, sol.nodes)):
            color = phase_color(section)
            kwargs = {"color": color, "ls": styles[qi % 2], "zorder": 3}
            label = quantity.label if many and section == 0 else None
            if not panel.step:
                (line,) = ax.plot(
                    dense["time"], quantity.of(dense), label=label, **kwargs
                )
                artists.append(line)
            if nodes or panel.step:
                ax.plot(
                    node["time"],
                    quantity.of(node),
                    "o" if not panel.step else "o-",
                    ms=3.2,
                    mfc="white",
                    mec=color,
                    mew=1.2,
                    color=color,
                    lw=1.2,
                    zorder=4,
                    label=label if panel.step else None,
                )
    for value, text in panel.limits:
        ax.axhline(value, color=LIMIT, lw=1.1, ls=(0, (2, 2)), zorder=2)
        ax.annotate(
            text,
            (1, value),
            xycoords=("axes fraction", "data"),
            xytext=(-2, 3),
            textcoords="offset points",
            ha="right",
            va="bottom",
            color=LIMIT,
            fontsize=8,
        )
    if panel.references is not None:
        for t, y, text in panel.references(sol):
            ax.plot(t, y, color=REFERENCE, lw=1.2, ls=(0, (3, 2)), zorder=5, label=text)
    _shade_phases(ax, sol, spec)
    ax.set_title(panel.title or panel.quantities[0].axis_label(), fontsize=10)
    if many or panel.references is not None:
        ax.legend(loc="best", fontsize=8, handlelength=2.4)
    return artists


# ---------------------------------------------------------------------------
# Static figures


def history_figure(sol: Solution, spec: Spec, path: Path) -> None:
    n = len(spec.history)
    cols = 1 if n <= 3 else 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(7.2, 1.0 + 1.85 * rows),
        sharex=True,
        layout="constrained",
        squeeze=False,
    )
    for ax, panel in zip(axes.flat, spec.history):
        _draw_panel(ax, panel, sol, spec)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(_time_label(spec))
    if n % 2 and cols == 2:
        axes[-2, 1].set_xlabel(_time_label(spec))
        axes[-2, 1].xaxis.set_tick_params(labelbottom=True)
    _phase_legend(fig, sol, spec, y=1.06)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def scene_figure(sol: Solution, spec: Spec, path: Path) -> None:
    scene = spec.scene
    assert scene is not None
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    _scene_axes(ax, scene, sol)
    for section, (dense, node) in enumerate(zip(sol.dense, sol.nodes)):
        color = phase_color(section)
        x, y = _scene_xy(scene, dense)
        ax.plot(x, y, color=color, lw=2.4, zorder=3)
        xn, yn = _scene_xy(scene, node)
        ax.plot(xn, yn, "o", ms=3.6, mfc="white", mec=color, mew=1.2, zorder=4)
        if scene.heading is not None and scene.kind in ("plane", "ascent", "orbit"):
            _thrust_arrows(ax, scene, node, color)
    first, last = sol.dense[0], sol.dense[-1]
    x0, y0 = _scene_xy(scene, first)
    x1, y1 = _scene_xy(scene, last)
    ax.plot(x0[0], y0[0], "o", ms=7, color=INK, zorder=6)
    ax.plot(x1[-1], y1[-1], "s", ms=7, color=INK, zorder=6)
    ax.annotate(
        "始点", (x0[0], y0[0]), xytext=(6, 6), textcoords="offset points", fontsize=9
    )
    ax.annotate(
        "終点", (x1[-1], y1[-1]), xytext=(6, 6), textcoords="offset points", fontsize=9
    )
    _phase_legend(fig, sol, spec, y=1.08)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _thrust_arrows(ax, scene: Scene, node: Data, color: str) -> None:
    assert scene.heading is not None
    x, y = _scene_xy(scene, node)
    angle = scene.heading(node)
    power = np.ones_like(x) if scene.throttle is None else scene.throttle(node)
    keep = power > 0.03
    step = max(1, keep.sum() // 12)
    index = np.flatnonzero(keep)[::step]
    length = power[index] / max(power.max(), 1e-12)
    ax.quiver(
        x[index],
        y[index],
        np.cos(angle[index]) * length,
        np.sin(angle[index]) * length,
        color=FLAME,
        angles="xy",
        scale_units="inches",
        scale=3.2,
        width=0.005,
        zorder=5,
    )


def _scene_xy(scene: Scene, data: Data) -> tuple[np.ndarray, np.ndarray]:
    x = scene.x.of(data)
    y = scene.y.of(data)
    if scene.kind == "rail":
        y = np.zeros_like(x)
    if scene.kind == "column":
        x = np.zeros_like(y)
    return x, y


def _scene_axes(ax, scene: Scene, sol: Solution) -> None:
    if scene.kind == "rail":
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-1, 1)
        ax.axhline(0, color="#9ca3af", lw=4, zorder=1, solid_capstyle="round")
        ax.set_xlabel(scene.x.axis_label())
    else:
        ax.set_xlabel(scene.x.axis_label())
        ax.set_ylabel(scene.y.axis_label())
    if scene.kind == "column":
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.grid(axis="x", visible=False)
        ax.set_xlim(-1, 1)
        ax.set_xlabel("")
    if scene.equal:
        ax.set_aspect("equal", adjustable="datalim")
    xs = np.concatenate([_scene_xy(scene, d)[0] for d in sol.dense])
    ys = np.concatenate([_scene_xy(scene, d)[1] for d in sol.dense])
    if scene.kind not in ("rail", "column"):
        pad_x = 0.06 * (xs.max() - xs.min() or 1)
        ax.set_xlim(xs.min() - pad_x, xs.max() + pad_x)
    if scene.kind != "rail":
        pad_y = 0.08 * (ys.max() - ys.min() or 1)
        ax.set_ylim(ys.min() - pad_y, ys.max() + pad_y)
    if scene.kind == "rail":
        pad_x = 0.08 * (xs.max() - xs.min() or 1)
        ax.set_xlim(xs.min() - pad_x, xs.max() + pad_x)
    if scene.xlim:
        ax.set_xlim(*scene.xlim)
    if scene.ylim:
        ax.set_ylim(*scene.ylim)
    if scene.ground is not None:
        ax.axhspan(-1e12, scene.ground, color="#e7e5e4", zorder=0)
        ax.axhline(scene.ground, color="#a8a29e", lw=1.5, zorder=1)
    if scene.invert_y:
        ax.invert_yaxis()
    if scene.decorate is not None:
        scene.decorate(ax, sol)
    if scene.label:
        ax.text(
            0.01,
            0.01,
            scene.label,
            transform=ax.transAxes,
            fontsize=8,
            color=MUTED,
            va="bottom",
        )


# ---------------------------------------------------------------------------
# Bodies drawn in pixel coordinates so they keep their shape on any axis scale


class Body:
    def __init__(self, ax, kind: str, color: str, size: float):
        self.ax, self.kind, self.size = ax, kind, size
        self.parts = []
        if kind in ("rocket", "lander", "probe"):
            self.flame = Polygon(
                np.zeros((3, 2)), closed=True, fc=FLAME, ec="none", zorder=7
            )
            self.shape = Polygon(
                np.zeros((3, 2)), closed=True, fc="#f8fafc", ec=INK, lw=1.2, zorder=8
            )
            for patch in (self.flame, self.shape):
                patch.set_transform(IdentityTransform())
                ax.add_artist(patch)
                patch.set_clip_on(False)
        else:
            marker = "o" if kind == "ball" else "s"
            (self.dot,) = ax.plot(
                [], [], marker, ms=size, mfc=color, mec=INK, mew=1.2, zorder=8
            )

    def place(self, x: float, y: float, angle: float, power: float) -> None:
        if self.kind in ("ball", "cart"):
            self.dot.set_data([x], [y])
            return
        px, py = self.ax.transData.transform((x, y))
        s = self.size
        if self.kind == "rocket":
            shape = np.array(
                [
                    [1.6, 0],
                    [0.9, 0.32],
                    [-1.0, 0.32],
                    [-1.35, 0.6],
                    [-1.35, -0.6],
                    [-1.0, -0.32],
                    [0.9, -0.32],
                ]
            )
            tail = -1.35
        elif self.kind == "lander":
            shape = np.array(
                [
                    [0.9, 0.5],
                    [0.9, -0.5],
                    [-0.3, -0.7],
                    [-1.1, -1.0],
                    [-0.6, -0.55],
                    [-0.6, 0.55],
                    [-1.1, 1.0],
                    [-0.3, 0.7],
                ]
            )
            tail = -0.6
        else:  # probe
            shape = np.array([[0.6, 0.6], [-0.6, 0.6], [-0.6, -0.6], [0.6, -0.6]])
            tail = -0.6
        flame = np.array([[tail, 0.28], [tail - 0.3 - 2.2 * power, 0], [tail, -0.28]])
        if power < 0.02:
            flame = np.array([[tail, 0], [tail, 0], [tail, 0]])
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        for patch, points in ((self.shape, shape), (self.flame, flame)):
            patch.set_xy(points @ rotation.T * s + (px, py))


# ---------------------------------------------------------------------------
# Animation


def _frame_times(sol: Solution, spec: Spec, frames: int) -> np.ndarray:
    t = sol.cat("time")
    if spec.coast_speedup <= 1 or spec.scene is None or spec.scene.throttle is None:
        return np.linspace(t[0], t[-1], frames)
    power = np.concatenate([spec.scene.throttle(d) for d in sol.dense])
    rate = np.where(power > 0.03, 1.0, spec.coast_speedup)
    clock = np.concatenate(([0.0], np.cumsum(np.diff(t) / rate[1:])))
    clock, unique = np.unique(clock, return_index=True)
    return np.interp(np.linspace(0, clock[-1], frames), clock, t[unique])


def _at(sol: Solution, key: Callable[[Data], np.ndarray], time: float) -> float:
    for data in sol.dense:
        if time <= data["time"][-1] or data is sol.dense[-1]:
            return float(np.interp(time, data["time"], key(data)))
    raise AssertionError


def _fmt(value: float) -> str:
    return "0" if abs(value) < 1e-6 else f"{value:.4g}"


def _section_at(sol: Solution, time: float) -> int:
    for section, data in enumerate(sol.dense):
        if time <= data["time"][-1]:
            return section
    return len(sol.dense) - 1


def animation(sol: Solution, spec: Spec, stem: Path) -> None:
    scene = spec.scene
    assert scene is not None
    panels = [spec.history[i] for i in spec.video_panels]
    fig = plt.figure(figsize=(8, 4.5), dpi=160)
    grid = fig.add_gridspec(
        max(len(panels), 1),
        2,
        width_ratios=[1.35, 1],
        left=0.1,
        right=0.98,
        top=0.93,
        bottom=0.11,
        wspace=0.22,
        hspace=0.45,
    )
    ax = fig.add_subplot(grid[:, 0])
    _scene_axes(ax, scene, sol)
    trails, bodies = [], []
    for section, dense in enumerate(sol.dense):
        x, y = _scene_xy(scene, dense)
        ax.plot(x, y, color=phase_color(section), lw=1.2, alpha=0.25, zorder=2)
        (trail,) = ax.plot([], [], color=phase_color(section), lw=2.4, zorder=3)
        trails.append(trail)
    size = {"ball": 9, "cart": 10}.get(scene.body, 7.0 * fig.dpi / 72)
    body = Body(ax, scene.body, phase_color(0), size)
    bodies.append(body)
    ghost = None
    if scene.ghost is not None:
        (ghost,) = ax.plot(
            [], [], "o", ms=8, mfc="white", mec=REFERENCE, mew=1.5, zorder=7
        )
    clock = ax.text(
        0.02,
        0.97,
        "",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        color=INK,
        bbox={
            "boxstyle": "round,pad=0.35",
            "fc": "white",
            "ec": "#d1d5db",
            "alpha": 0.92,
        },
        zorder=10,
    )
    fig.canvas.draw()  # fix the axes transform before placing pixel-sized bodies

    axes, cursors = [], []
    for i, panel in enumerate(panels):
        pax = fig.add_subplot(grid[i, 1])
        _draw_panel(pax, panel, sol, spec, nodes=False)
        if i < len(panels) - 1:
            pax.tick_params(labelbottom=False)
        else:
            pax.set_xlabel(_time_label(spec))
        cursors.append(pax.axvline(0, color=INK, lw=1.0, zorder=6))
        axes.append(pax)

    def draw(time: float) -> None:
        section = _section_at(sol, time)
        for s, (trail, dense) in enumerate(zip(trails, sol.dense)):
            x, y = _scene_xy(scene, dense)
            keep = dense["time"] <= time
            trail.set_data(x[keep], y[keep])
        x = _at(sol, lambda d: _scene_xy(scene, d)[0], time)
        y = _at(sol, lambda d: _scene_xy(scene, d)[1], time)
        power = 0.0 if scene.throttle is None else _at(sol, scene.throttle, time)
        heading = scene.heading
        if heading is not None:
            angle = _at(sol, lambda d: np.unwrap(heading(d)), time)
            # Heading is a direction in data space; draw it as seen on screen.
            p0 = ax.transData.transform((x, y))
            e = 1e-3 * abs(ax.get_xlim()[1] - ax.get_xlim()[0])
            p1 = ax.transData.transform((x + e * np.cos(angle), y + e * np.sin(angle)))
            angle = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
        else:
            angle = np.pi / 2
        body.place(x, y, angle, power)
        if body.kind in ("ball", "cart"):
            body.dot.set_markerfacecolor(phase_color(section))
        if ghost is not None and scene.ghost is not None:
            gx, gy = scene.ghost(time)
            ghost.set_data([gx], [gy])
        lines = [f"$t$ = {time:.3g} {spec.time_unit}".rstrip()]
        if len(sol.dense) > 1 and spec.phase_names:
            lines.append(spec.phase_names[section])
        for q in scene.readout:
            lines.append(f"{q.label} = {_fmt(_at(sol, q.of, time))} {q.unit}".rstrip())
        clock.set_text("\n".join(lines))
        for cursor in cursors:
            cursor.set_xdata([time, time])

    frames = _frame_times(sol, spec, int(spec.seconds * 30))
    writer = FFMpegWriter(
        fps=30,
        codec="libx264",
        bitrate=-1,
        extra_args=["-pix_fmt", "yuv420p", "-crf", "24", "-movflags", "+faststart"],
    )
    with writer.saving(fig, str(stem.with_suffix(".mp4")), dpi=fig.dpi):
        for time in frames:
            draw(time)
            writer.grab_frame()
        for _ in range(45):  # hold the final frame
            writer.grab_frame()
    fig.savefig(stem.with_suffix(".png"), dpi=fig.dpi)
    plt.close(fig)
