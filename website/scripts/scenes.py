"""Explanatory flight animations for the rocket and orbit examples.

A frame has three parts that always show the same instant:

* the scene: a rocket drawn to scale on screen (not physically to scale) with
  a flame proportional to throttle, flying a vertical column, an ascent over
  a curved Earth, or a spiral between two orbits;
* graphs with phase shading, a moving cursor and the current value;
* a header with the current phase and a readout of the main quantities.

Only quantities computed by the solver are drawn. Decorations that are not
data (the rocket shape, flame flicker, a separated stage fading out where it
was released) are labelled as schematic where they could be mistaken for it.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle, Polygon, Wedge
from matplotlib.transforms import Affine2D

FPS = 30
DPI = 100
SIZE = (16, 9)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Hiragino Sans",
            "Noto Sans CJK JP",
            "Noto Sans JP",
            "IPAexGothic",
            "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

INK = "#111827"
MUTED = "#6b7280"
PHASE_COLORS = ["#2563eb", "#dc2626", "#059669", "#7c3aed"]
FLAME = "#f97316"
COAST = "#9ca3af"
SERIES = ["#2563eb", "#f59e0b"]  # panels with two curves keep one colour each

Phase = dict[str, np.ndarray]
Getter = Callable[[Phase], np.ndarray]


@dataclass
class Series:
    label: str  # axis label, including the unit
    value: Getter
    fmt: str = "{:.3g}"  # readout format
    unit: str = ""
    mask_coast: bool = False  # hide where the engine is off (e.g. thrust angle)


@dataclass
class Flight:
    """What one animation shows. Getters take one sampled phase."""

    kind: str  # "rail", "track", "plane", "vertical", "ascent" or "orbit"
    title: str
    phase_names: list[str]
    panels: list[list[Series]] = field(default_factory=list)
    readout: list[Series] = field(default_factory=list)
    # Thrust magnitude, normalised per phase; None for problems without an
    # engine (no engine-off shading or fast-forward).
    throttle: Getter | None = None
    # rail / track / plane: position in the plane of the picture
    x: Getter | None = None
    y: Getter | None = None
    xlabel: str = ""
    ylabel: str = ""
    invert_y: bool = False  # depth axes point down
    decorate: Callable[[Any], None] | None = None  # static extras in the scene
    arrow: Getter | None = None  # rail: signed arrow (e.g. acceleration)
    arrow_label: str = ""
    heading: Getter | None = None  # plane: thrust direction [rad from +x]
    legend_lines: list[str] = field(default_factory=list)  # extra scene key
    # Exact reference shown as a second body racing the solution:
    # position at a time, and its arrival time.
    ghost: Callable[[float], tuple[float, float]] | None = None
    ghost_time: float = 0.0
    ghost_label: str = ""
    # Analytic curves per panel index: (time, value, label)
    overlays: dict[int, list[tuple[np.ndarray, np.ndarray, str]]] = field(
        default_factory=dict
    )
    altitude: Getter | None = None  # vertical: height shown in the column
    altitude_label: str = ""
    altitude_unit: str = ""
    theme: str = "earth"  # "earth", "moon" or "space"
    staging: bool = False  # phases after the first are upper stages
    # ascent: radius and polar angle in metres / radians
    radius: Getter | None = None
    angle: Getter | None = None
    thrust_radial: Getter | None = None
    thrust_tangential: Getter | None = None
    earth_radius: float = 6371.0e3
    target_altitude: float | None = None  # m, drawn as the target orbit
    # orbit: planar coordinates and thrust components in the local frame
    position: Callable[[Phase], tuple[np.ndarray, np.ndarray]] | None = None
    orbits: list[float] = field(default_factory=list)
    time_unit: str = "s"
    note: str = ""  # schematic / scaling disclaimer shown in the scene
    seconds: float = 12.0
    coast_speedup: float | None = None  # None: chosen so coasting is <~40%


# ---------------------------------------------------------------------------
# Helpers


def _cat(phases: list[Phase], getter: Getter) -> np.ndarray:
    return np.concatenate([np.asarray(getter(p), dtype=float) for p in phases])


def _phase_index(phases: list[Phase]) -> np.ndarray:
    return np.concatenate([np.full(len(p["time"]), i) for i, p in enumerate(phases)])


def _frame_times(t: np.ndarray, throttle: np.ndarray, flight: Flight) -> np.ndarray:
    """Uniform in a warped clock that runs faster while the engine is off."""
    rate = np.where(throttle > 0.03, 1.0, _speedup(t, throttle, flight))
    warped = np.concatenate(([0.0], np.cumsum(np.diff(t) / rate[1:])))
    frames = int(flight.seconds * FPS)
    clock = np.linspace(0, warped[-1], frames)
    hold = np.full(int(1.5 * FPS), clock[-1])
    return np.interp(np.concatenate((clock, hold)), warped, t)


def _speedup(t: np.ndarray, throttle: np.ndarray, flight: Flight) -> float:
    if flight.coast_speedup is not None:
        return flight.coast_speedup
    dt = np.diff(t, prepend=t[0])
    burning = float(np.sum(dt[throttle > 0.03]))
    coasting = float(np.sum(dt[throttle <= 0.03]))
    return float(max(1, round(1.5 * coasting / max(burning, 1e-12))))


def _coast_spans(t: np.ndarray, throttle: np.ndarray) -> list[tuple[float, float]]:
    off = throttle <= 0.03
    spans, start = [], None
    for i, flag in enumerate(off):
        if flag and start is None:
            start = t[i]
        if not flag and start is not None:
            spans.append((start, t[i]))
            start = None
    if start is not None:
        spans.append((start, t[-1]))
    duration = t[-1] - t[0]
    return [(a, b) for a, b in spans if b - a > 0.02 * duration]


def _fit_ylim(ax, curves: list[np.ndarray]) -> None:
    """Keep a nearly constant curve from being blown up to solver noise."""
    ax.ticklabel_format(axis="y", useOffset=False)
    values = np.concatenate(curves)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    low, high = float(values.min()), float(values.max())
    size = max(abs(low), abs(high))
    if high - low < 0.02 * size:
        ax.set_ylim(
            min(0.0, low) * 1.15 - 0.05 * size, max(0.0, high) * 1.15 + 0.05 * size
        )


def _rocket_shapes(length: float, width: float, stages: int):
    """Polygons in pixels, nose along +y, nozzle exit at the origin."""
    w = width / 2
    shapes = []
    bottom = 0.0
    segment = length * 0.78 / stages
    colors = ["#e5e7eb", "#f3f4f6", "#ffffff"]
    for i in range(stages):
        top = bottom + segment
        shapes.append(([(-w, bottom), (w, bottom), (w, top), (-w, top)], colors[i % 3]))
        bottom = top
    shapes.append(([(-w, bottom), (w, bottom), (0, length)], "#ef4444"))
    fin = width * 0.9
    shapes.append(([(-w, 0), (-w - fin, -fin * 0.4), (-w, fin * 1.2)], "#374151"))
    shapes.append(([(w, 0), (w + fin, -fin * 0.4), (w, fin * 1.2)], "#374151"))
    return shapes


class RocketGlyph:
    """A rocket drawn in screen pixels at a data position, with a flame."""

    def __init__(self, ax, length=70.0, width=16.0, stages=1, zorder=6):
        self.ax = ax
        self.parts = []
        self.length, self.width = length, width
        for vertices, color in _rocket_shapes(length, width, stages):
            patch = Polygon(
                vertices, closed=True, fc=color, ec=INK, lw=1.2, zorder=zorder
            )
            ax.add_patch(patch)
            self.parts.append((patch, np.array(vertices)))
        self.flame = Polygon(
            [(0, 0)] * 3, closed=True, fc=FLAME, ec="#fbbf24", lw=1, zorder=zorder - 1
        )
        ax.add_patch(self.flame)

    def place(self, x, y, angle, throttle, flicker=0.0, alpha=1.0):
        """``angle`` is the nose direction in screen radians from +x."""
        cx, cy = self.ax.transData.transform((x, y))
        transform = Affine2D().rotate(angle - np.pi / 2).translate(cx, cy)
        for patch, _ in self.parts:
            patch.set_transform(transform)
            patch.set_alpha(alpha)
        length = self.length * 1.4 * max(throttle, 0.0) * (1 + flicker)
        w = self.width * 0.42
        self.flame.set_xy([(-w, 0), (w, 0), (0, -length)])
        self.flame.set_transform(transform)
        self.flame.set_visible(throttle > 0.03 and alpha > 0)
        self.flame.set_alpha(alpha)

    def hide(self):
        for patch, _ in self.parts:
            patch.set_visible(False)
        self.flame.set_visible(False)


def _screen_angle(ax, x0, y0, dx, dy) -> float:
    a = ax.transData.transform((x0, y0))
    b = ax.transData.transform((x0 + dx, y0 + dy))
    return float(np.arctan2(b[1] - a[1], b[0] - a[0]))


# ---------------------------------------------------------------------------
# Rendering


def render_flight(flight: Flight, phases: list[Phase], stem: Path) -> None:
    t = _cat(phases, lambda p: p["time"])
    phase_of = _phase_index(phases)
    # Normalise per phase: an upper stage's small engine is "on" at its own
    # full thrust, not a few percent of the first stage's.
    # A phase whose peak is under 2% of the flight's peak only holds numerical
    # noise around zero thrust (e.g. a coast phase), so it is "off" throughout.
    if flight.throttle is None:
        throttle = np.ones_like(t)
    else:
        engine = flight.throttle
        peaks = [float(np.max(np.abs(engine(p)))) for p in phases]
        top = max(peaks) or 1.0
        throttle = np.concatenate(
            [
                np.abs(engine(p)) / peak
                if peak > 0.02 * top
                else np.zeros(len(p["time"]))
                for p, peak in zip(phases, peaks, strict=True)
            ]
        )
    times = _frame_times(t, throttle, flight)
    if flight.ghost is not None and flight.ghost_time > t[-1]:
        # Keep the clock running until the reference also arrives.
        times = np.concatenate(
            (
                np.linspace(t[0], flight.ghost_time, int(flight.seconds * FPS)),
                np.full(int(1.5 * FPS), flight.ghost_time),
            )
        )
    speedup = _speedup(t, throttle, flight)
    knots = [p["time"][0] for p in phases[1:]]
    coasts = _coast_spans(t, throttle)

    def at(values: np.ndarray, time: float) -> float:
        return float(np.interp(time, t, values))

    def phase_at(time: float) -> int:
        return int(np.searchsorted(knots, time, side="right"))

    fig = plt.figure(figsize=SIZE, dpi=DPI, facecolor="white")
    grid = fig.add_gridspec(
        len(flight.panels),
        2,
        width_ratios=[1.25, 1],
        left=0.04,
        right=0.97,
        top=0.86,
        bottom=0.08,
        wspace=0.16,
        hspace=0.18,
    )
    scene = fig.add_subplot(grid[:, 0])
    axes = [fig.add_subplot(grid[i, 1]) for i in range(len(flight.panels))]
    for ax in axes[1:]:
        ax.sharex(axes[0])

    fig.text(0.04, 0.95, flight.title, fontsize=20, weight="bold", color=INK)
    badge = fig.text(
        0.04,
        0.905,
        "",
        fontsize=14,
        color="white",
        weight="bold",
        bbox={"boxstyle": "round,pad=0.35", "fc": PHASE_COLORS[0], "ec": "none"},
    )
    speed_note = fig.text(0.97, 0.905, "", fontsize=12, color=MUTED, ha="right")

    # --- graphs -----------------------------------------------------------
    cursors, values_text, dots, legends = [], [], [], []
    for index, (ax, group) in enumerate(zip(axes, flight.panels, strict=True)):
        for i, phase in enumerate(phases):
            color = PHASE_COLORS[i % 4]
            ax.axvspan(
                phase["time"][0], phase["time"][-1], color=color, alpha=0.06, lw=0
            )
        for a, b in coasts:
            ax.axvspan(a, b, color=COAST, alpha=0.18, lw=0, hatch="//")
        for knot in knots:
            ax.axvline(knot, color=INK, lw=1, ls="--", alpha=0.6)
        group_dots = []
        for j, series in enumerate(group):
            values = _cat(phases, series.value)
            if series.mask_coast:
                values = np.where(throttle > 0.03, values, np.nan)
            style = "-" if j == 0 else "--"
            for i in range(len(phases)):
                sel = phase_of == i
                ax.plot(
                    t[sel],
                    values[sel],
                    style,
                    color=PHASE_COLORS[i % 4] if len(group) == 1 else SERIES[j % 2],
                    lw=2.2,
                    label=series.label if i == 0 and len(group) > 1 else None,
                )
            (dot,) = ax.plot([], [], "o", color=INK, ms=7, zorder=5)
            group_dots.append((dot, values))
        ax.set_ylabel(
            group[0].label
            if len(group) == 1
            else ("無次元" if group[0].unit in ("", "-") else f"[{group[0].unit}]")
        )
        ax.grid(alpha=0.3)
        for ref_t, ref_v, _ in flight.overlays.get(index, []):
            ax.plot(ref_t, ref_v, color=INK, lw=1.4, ls=(0, (4, 3)), zorder=4)
        _fit_ylim(ax, [_cat(phases, series.value) for series in group])
        if len(group) > 1:
            legend = ax.legend(
                loc="upper left", fontsize=11, framealpha=0.9, edgecolor="#d1d5db"
            )
            legends.append(legend.get_texts())
        else:
            legends.append(None)
        cursors.append(ax.axvline(t[0], color=INK, lw=1.2))
        values_text.append(
            ax.text(
                0.99,
                0.92,
                "",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=13,
                weight="bold",
                color=INK,
                path_effects=[patheffects.withStroke(linewidth=4, foreground="white")],
            )
        )
        dots.append(group_dots)
    axes[-1].set_xlabel(f"時刻 [{flight.time_unit}]")
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    keys = []
    if flight.overlays:
        labels = {label for refs in flight.overlays.values() for *_, label in refs}
        keys.append(f"実線：数値解（色は区間）　黒の破線：{'・'.join(sorted(labels))}")
    if coasts:
        keys.append("灰色の斜線：エンジン停止（慣性飛行）")
    if knots:
        keys.append("縦の破線：区間（phase）の境界")
    if keys:
        axes[0].text(
            0.0,
            1.04,
            "　".join(keys),
            transform=axes[0].transAxes,
            fontsize=10.5,
            color=MUTED,
        )

    # --- scene ------------------------------------------------------------
    draw_scene = {
        "rail": _rail,
        "track": _track,
        "plane": _plane,
        "vertical": _vertical,
        "ascent": _ascent,
        "orbit": _orbit,
    }[flight.kind]
    update_scene = draw_scene(scene, flight, phases, t, throttle, knots, fig)
    readout = scene.text(
        0.02,
        0.98,
        "",
        transform=scene.transAxes,
        va="top",
        fontsize=12,
        color=INK,
        zorder=20,
        linespacing=1.5,
        bbox={
            "boxstyle": "round,pad=0.5",
            "fc": "white",
            "ec": "#d1d5db",
            "alpha": 0.92,
        },
    )
    if flight.note:
        scene.text(
            0.98,
            0.98,
            flight.note,
            transform=scene.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color=MUTED,
            zorder=20,
            bbox={
                "boxstyle": "round,pad=0.3",
                "fc": "white",
                "ec": "none",
                "alpha": 0.8,
            },
        )

    readout_values = [_cat(phases, s.value) for s in flight.readout]
    fig.canvas.draw()  # fix the layout so pixel-sized glyphs can be placed
    rng = np.random.default_rng(0)

    def draw(time: float, frame: int, held: bool):
        k = phase_at(time)
        power = at(throttle, time)
        name = flight.phase_names[min(k, len(flight.phase_names) - 1)]
        off = flight.throttle is not None and power <= 0.03
        badge.set_text(f" {name}{'・エンジン停止' if off else ''} ")
        box = badge.get_bbox_patch()
        if box is not None:
            box.set_facecolor(PHASE_COLORS[k % 4])
        coasting = power <= 0.03 and speedup != 1
        speed_note.set_text(f"慣性飛行中は {speedup:g} 倍速で再生" if coasting else "")
        unit = "" if flight.time_unit == "-" else f" {flight.time_unit}"
        lines = [
            f"t = {time:.3g}{unit}"
            if flight.time_unit == "-"
            else f"t = {time:.1f}{unit}"
        ]
        for series, values in zip(flight.readout, readout_values, strict=True):
            value = at(values, time) + 0.0  # no '-0.000'
            unit = "" if series.unit in ("", "-") else f" {series.unit}"
            lines.append(f"{series.label}：{series.fmt.format(value)}{unit}")
        readout.set_text("\n".join(lines))
        for cursor, text, group, group_dots, labels in zip(
            cursors, values_text, flight.panels, dots, legends, strict=True
        ):
            cursor.set_xdata([min(time, t[-1])] * 2)
            parts = []
            for series, (dot, values) in zip(group, group_dots, strict=True):
                value = at(values, time) + 0.0
                visible = np.isfinite(value)
                dot.set_data([min(time, t[-1])], [value] if visible else [np.nan])
                if visible:
                    unit = "" if series.unit in ("", "-") else f" {series.unit}"
                    parts.append(f"{series.fmt.format(value)}{unit}")
                else:
                    parts.append("—")
            if labels is None:
                text.set_text(parts[0])
            else:
                text.set_text("")
                for label, series, part in zip(labels, group, parts, strict=True):
                    label.set_text(f"{series.label} = {part}")
        update_scene(time, power, 0.08 * rng.standard_normal() if not held else 0.0)

    writer = FFMpegWriter(
        fps=FPS,
        codec="libx264",
        bitrate=-1,
        extra_args=["-pix_fmt", "yuv420p", "-crf", "25", "-movflags", "+faststart"],
    )
    held_from = int(flight.seconds * FPS)
    with writer.saving(fig, str(stem.with_suffix(".mp4")), dpi=DPI):
        for frame, time in enumerate(times):
            draw(time, frame, frame >= held_from)
            writer.grab_frame()
    draw(times[-1], len(times), True)
    fig.savefig(stem.with_suffix(".png"), dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Scenes. Each draws the static picture and returns update(time, power, flicker).

LIGHT = "#f8fafc"
STROBE = 16  # marks at equal time steps: their spacing shows the speed


def _strobe_times(t: np.ndarray) -> np.ndarray:
    return np.linspace(t[0], t[-1], STROBE + 1)


def _caption(ax, text: str) -> None:
    ax.text(
        0.02,
        0.02,
        text,
        transform=ax.transAxes,
        fontsize=11,
        color=MUTED,
        va="bottom",
        zorder=20,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "none", "alpha": 0.85},
    )


def _rail(ax, flight, phases, t, throttle, knots, fig):
    """A block on a straight rail; the arrow is the signed control."""
    assert flight.x is not None
    x = _cat(phases, flight.x)
    span = float(np.ptp(x)) or 1.0
    left, right = float(np.min(x)) - 0.15 * span, float(np.max(x)) + 0.15 * span
    ax.set_xlim(left, right)
    ax.set_ylim(-1, 1)
    ax.set_facecolor(LIGHT)
    ax.set_yticks([])
    ax.set_xlabel(flight.xlabel)
    for spine in ("left", "top", "right"):
        ax.spines[spine].set_visible(False)
    ax.plot([left, right], [-0.12, -0.12], color="#94a3b8", lw=6, zorder=1)
    for value, name, color in ((x[0], "スタート", MUTED), (x[-1], "ゴール", "#059669")):
        ax.plot([value, value], [-0.12, 0.55], color=color, lw=1.5, ls="--", zorder=1)
        ax.text(value, 0.6, name, color=color, ha="center", fontsize=12)
    strobe_t = _strobe_times(t)
    strobe_x = np.interp(strobe_t, t, x)
    (marks,) = ax.plot([], [], "|", color=MUTED, ms=22, mew=2, zorder=2)
    (block,) = ax.plot([], [], "s", ms=34, mec=INK, mew=1.5, zorder=5)
    arrow = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(0, 0),
        arrowprops={"arrowstyle": "-|>", "color": "#dc2626", "lw": 3.5},
        zorder=6,
    )
    values = _cat(phases, flight.arrow) if flight.arrow else np.zeros_like(x)
    peak = float(np.max(np.abs(values))) or 1.0
    _caption(
        ax,
        f"赤い矢印：{flight.arrow_label}（長さは最大値に対する比）\n"
        "灰色の目盛り：一定時間ごとの位置（間隔が広いほど速い）",
    )

    def update(time, power, flicker):
        px = float(np.interp(time, t, x))
        k = int(np.searchsorted(knots, time, side="right"))
        block.set_data([px], [0.12])
        block.set_color(PHASE_COLORS[k % 4])
        block.set_markeredgecolor(INK)
        marks.set_data(
            strobe_x[strobe_t <= time], np.full((strobe_t <= time).sum(), 0.12)
        )
        a = float(np.interp(time, t, values)) / peak
        if abs(a) > 0.01:
            arrow.set_position((px, 0.42))
            arrow.xy = (px + 0.22 * span * a, 0.42)
            arrow.set_visible(True)
        else:
            arrow.set_visible(False)

    return update


def _track(ax, flight, phases, t, throttle, knots, fig):
    """A ball sliding along the optimal curve, optionally racing a reference."""
    assert flight.x is not None and flight.y is not None
    x, y = _cat(phases, flight.x), _cat(phases, flight.y)
    ax.set_facecolor(LIGHT)
    ax.plot(x, y, color="#cbd5e1", lw=9, solid_capstyle="round", zorder=1)
    if flight.decorate is not None:
        flight.decorate(ax)
    trails = [
        ax.plot([], [], color=PHASE_COLORS[i % 4], lw=3, zorder=3)[0]
        for i in range(len(phases))
    ]
    phase_of = _phase_index(phases)
    strobe_t = _strobe_times(t)
    sx, sy = np.interp(strobe_t, t, x), np.interp(strobe_t, t, y)
    (marks,) = ax.plot([], [], "o", color=MUTED, ms=6, alpha=0.7, zorder=4)
    (ball,) = ax.plot([], [], "o", color="#dc2626", mec=INK, mew=1.5, ms=18, zorder=6)
    ghost_ball = ghost_text = None
    if flight.ghost is not None:
        gx0, gy0 = flight.ghost(t[0])
        gx1, gy1 = flight.ghost(flight.ghost_time)
        ax.plot([gx0, gx1], [gy0, gy1], color="#2563eb", lw=2, ls="--", zorder=2)
        (ghost_ball,) = ax.plot(
            [], [], "o", color="#60a5fa", mec="#1d4ed8", ms=16, alpha=0.8, zorder=5
        )
        ghost_text = ax.text(
            0.98,
            0.10,
            "",
            transform=ax.transAxes,
            ha="right",
            color="#1d4ed8",
            fontsize=12,
            zorder=20,
        )
    done_text = ax.text(
        0.98,
        0.17,
        "",
        transform=ax.transAxes,
        ha="right",
        color="#b91c1c",
        fontsize=12,
        weight="bold",
        zorder=20,
    )
    ax.set_xlabel(flight.xlabel)
    ax.set_ylabel(flight.ylabel)
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.08)
    if flight.invert_y:
        ax.invert_yaxis()
    ax.grid(alpha=0.3)
    lines = ["赤い球：最適解（OpenGoddard）　灰色の点：一定時間ごとの位置"]
    if flight.ghost is not None:
        lines.append(f"青い球：{flight.ghost_label}")
    _caption(ax, "\n".join(lines + flight.legend_lines))

    def update(time, power, flicker):
        for p, trail in enumerate(trails):
            sel = (phase_of == p) & (t <= time)
            trail.set_data(x[sel], y[sel])
        ball.set_data([np.interp(time, t, x)], [np.interp(time, t, y)])
        shown = strobe_t <= time
        marks.set_data(sx[shown], sy[shown])
        done_text.set_text(f"最適解の到着 {t[-1]:.3g}" if time >= t[-1] else "")
        if ghost_ball is not None and ghost_text is not None and flight.ghost:
            gx, gy = flight.ghost(min(time, flight.ghost_time))
            ghost_ball.set_data([gx], [gy])
            arrived = time >= flight.ghost_time
            ghost_text.set_text(
                f"直線の到着 {flight.ghost_time:.3g}" if arrived else ""
            )

    return update


def _plane(ax, flight, phases, t, throttle, knots, fig):
    """A rocket flying in a vertical plane, nose along the thrust direction."""
    assert flight.x and flight.y and flight.heading
    x, y = _cat(phases, flight.x), _cat(phases, flight.y)
    heading = _cat(phases, flight.heading)
    ax.set_facecolor(LIGHT)
    ax.plot(x, y, color="#cbd5e1", lw=6, zorder=1)
    if flight.decorate is not None:
        flight.decorate(ax)
    (trail,) = ax.plot([], [], color=PHASE_COLORS[0], lw=3, zorder=3)
    strobe_t = _strobe_times(t)
    sx, sy = np.interp(strobe_t, t, x), np.interp(strobe_t, t, y)
    (marks,) = ax.plot([], [], "o", color=MUTED, ms=6, alpha=0.7, zorder=4)
    ax.set_xlabel(flight.xlabel)
    ax.set_ylabel(flight.ylabel)
    span = max(float(np.ptp(x)), float(np.ptp(y))) or 1.0
    ax.set_xlim(float(np.min(x)) - 0.2 * span, float(np.max(x)) + 0.2 * span)
    ax.set_ylim(float(np.min(y)) - 0.2 * span, float(np.max(y)) + 0.3 * span)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    rocket = RocketGlyph(ax, length=60, width=16)
    _caption(
        ax,
        "\n".join(
            ["機体の向き＝推力の向き（模式図）　灰色の点：一定時間ごとの位置"]
            + flight.legend_lines
        ),
    )

    def update(time, power, flicker):
        sel = t <= time
        trail.set_data(x[sel], y[sel])
        shown = strobe_t <= time
        marks.set_data(sx[shown], sy[shown])
        px, py = float(np.interp(time, t, x)), float(np.interp(time, t, y))
        b = float(np.interp(time, t, heading))
        nose = _screen_angle(ax, px, py, np.cos(b), np.sin(b))
        rocket.place(px, py, nose, power, flicker)

    return update


def _vertical(ax, flight, phases, t, throttle, knots, fig):
    assert flight.altitude is not None
    h = _cat(phases, flight.altitude)
    low, high = float(np.min(h)), float(np.max(h))
    ground = flight.theme != "space" and low <= 0.05 * (high - low)
    pad = 0.12 * ((high - low) or 1.0)
    bottom = (0.0 if ground else low) - pad
    top = high + 1.5 * pad
    ax.set_xlim(-1, 1)
    ax.set_ylim(bottom, top)
    ax.set_xticks([])
    ax.set_ylabel(flight.altitude_label)
    ax.spines["bottom"].set_visible(False)

    if flight.theme == "moon":
        sky = ["#0b1020", "#020617"]
        ground_color = "#9ca3af"
    elif flight.theme == "earth":
        sky = ["#bfdbfe", "#0f172a"]
        ground_color = "#4d7c0f"
    else:
        sky = ["#e0f2fe", "#c7d2fe"]
        ground_color = "#d6d3d1"
    gradient = np.linspace(0, 1, 256)[:, None]
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("sky", sky)
    ax.imshow(
        gradient,
        extent=(-1, 1, bottom, top),
        origin="lower",
        aspect="auto",
        cmap=cmap,
        zorder=0,
    )
    if ground:
        ax.fill_between([-1, 1], bottom, 0, color=ground_color, zorder=1)
        ax.axhline(0, color=INK, lw=1.5, zorder=2)
    else:
        ax.text(
            0.02,
            0.015,
            "地面は表示範囲の外（高度軸を拡大）",
            transform=ax.transAxes,
            ha="left",
            fontsize=10,
            color=MUTED,
        )
    text_color = "white" if flight.theme in ("moon", "earth") else INK
    ax.axhline(high, color=text_color, lw=1, ls=":", alpha=0.7, zorder=2)
    ax.text(
        0.97,
        high,
        f"{'開始高度' if np.argmax(h) == 0 else '最高到達'} {high:.4g} {flight.altitude_unit}",
        color=text_color,
        ha="right",
        va="bottom",
        fontsize=10,
        transform=ax.get_yaxis_transform(),
    )
    (track,) = ax.plot([], [], color="white", lw=3, alpha=0.5, zorder=3)

    stages = 2 if flight.staging else 1
    rocket = RocketGlyph(ax, length=90, width=20, stages=stages)
    upper = RocketGlyph(ax, length=50, width=20, stages=1) if flight.staging else None
    if upper is not None:
        upper.hide()
    dropped = RocketGlyph(ax, length=40, width=20, stages=1) if upper else None
    if dropped is not None:
        dropped.hide()
    separation = (
        (knots[0], float(np.interp(knots[0], t, h))) if flight.staging else None
    )
    label = ax.text(0.18, 0, "", fontsize=12, color=text_color, weight="bold")

    def update(time, power, flicker):
        y = float(np.interp(time, t, h))
        track.set_data([0, 0], [bottom if not ground else 0, y])
        if separation is not None and time >= separation[0]:
            assert upper is not None and dropped is not None
            rocket.hide()
            for patch, _ in upper.parts:
                patch.set_visible(True)
            upper.place(0, y, np.pi / 2, power, flicker)
            fade = max(0.0, 1 - (time - separation[0]) / (0.15 * (t[-1] - t[0])))
            for patch, _ in dropped.parts:
                patch.set_visible(fade > 0)
            dropped.place(-0.35, separation[1], np.pi / 2, 0.0, alpha=fade)
            label.set_position((-0.3, separation[1]))
            label.set_text("段分離" if fade > 0 else "")
        else:
            rocket.place(0, y, np.pi / 2, power, flicker)

    return update


def _ascent(ax, flight, phases, t, throttle, knots, fig):
    assert flight.radius and flight.angle
    assert flight.thrust_radial and flight.thrust_tangential
    re = flight.earth_radius
    r = _cat(phases, flight.radius)
    th = _cat(phases, flight.angle)
    tr = _cat(phases, flight.thrust_radial)
    tt = _cat(phases, flight.thrust_tangential)
    alt = r - re
    target = flight.target_altitude or float(np.max(alt))
    # Exaggerate altitude so that the climb is visible next to the downrange.
    downrange = re * float(np.max(np.abs(th)))
    k = max(1.0, round(0.35 * downrange / max(target, 1.0)))
    rp = re + k * alt

    def xy(radius, angle):
        return radius * np.sin(angle) / 1e3, (radius * np.cos(angle) - re) / 1e3

    x, y = xy(rp, th)
    # Long flights wrap around the Earth: then show the whole target orbit.
    sweep = np.array([0.0, float(th[-1])])
    if th[-1] > 1.0:
        sweep = np.linspace(0, 2 * np.pi, 73)
    ox, oy = xy(np.full(sweep.size, re + k * target), sweep)
    xs_all, ys_all = np.r_[x, ox, 0.0], np.r_[y, oy, 0.0]
    span = float(max(np.ptp(xs_all), np.ptp(ys_all)))
    left = float(np.min(xs_all)) - 0.12 * span
    right = float(np.max(xs_all)) + 0.12 * span
    ytop = float(np.max(ys_all)) + 0.15 * span
    ybottom = float(np.min(ys_all)) - 0.12 * span
    ax.set_xlim(left, right)
    ax.set_ylim(ybottom, ytop)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_facecolor("#020617")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    center = (0.0, -re / 1e3)
    ax.add_patch(Circle(center, re / 1e3, fc="#1e3a8a", ec="#60a5fa", lw=2, zorder=1))
    atmosphere = (re + k * 100e3) / 1e3
    ax.add_patch(
        Wedge(
            center,
            atmosphere,
            0,
            360,
            width=atmosphere - re / 1e3,
            color="#38bdf8",
            alpha=0.15,
            zorder=1,
        )
    )
    orbit = (re + k * target) / 1e3
    ax.add_patch(
        Circle(center, orbit, fill=False, ec="#fde68a", lw=1.5, ls="--", zorder=2)
    )
    label_angle = float(th[-1]) + 0.25
    ax.text(
        orbit * np.sin(label_angle) * 1.02,
        orbit * np.cos(label_angle) * 1.02 + center[1],
        f"目標軌道 高度{target / 1e3:.0f} km",
        color="#fde68a",
        ha="left",
        fontsize=11,
        zorder=8,
    )
    ax.text(
        -0.02 * span,
        k * 100e3 / 1e3,
        "大気（高度100 km）",
        color="#7dd3fc",
        ha="right",
        va="bottom",
        fontsize=10,
        zorder=8,
    )
    ax.plot([0], [0], "^", color="#fbbf24", ms=12, zorder=5)
    ax.text(0, -0.02 * span, "射点", color="#fbbf24", ha="center", va="top", zorder=8)
    for i, phase in enumerate(phases):
        px, py = xy(re + k * (flight.radius(phase) - re), flight.angle(phase))
        ax.plot(px, py, color=PHASE_COLORS[i % 4], lw=1, alpha=0.3, zorder=3)
    trails = [
        ax.plot([], [], color=PHASE_COLORS[i % 4], lw=3, zorder=4)[0]
        for i in range(len(phases))
    ]
    phase_of = _phase_index(phases)
    stages = len(phases) if flight.staging else 1
    rockets = [
        RocketGlyph(ax, length=46 - 14 * i, width=12, stages=stages - i)
        for i in range(stages)
    ]
    for rocket in rockets[1:]:
        rocket.hide()
    dropped = RocketGlyph(ax, length=22, width=12) if flight.staging else None
    if dropped is not None:
        dropped.hide()
    arrow = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(0, 0),
        arrowprops={"arrowstyle": "-|>", "color": FLAME, "lw": 2.5},
        zorder=7,
    )
    sep = ax.text(0, 0, "", color="white", fontsize=12, weight="bold", zorder=8)
    span_px = (
        ax.transData.transform((right, 0))[0] - ax.transData.transform((left, 0))[0]
    )

    def update(time, power, flicker):
        i = int(np.searchsorted(t, time))
        i = min(max(i, 1), len(t) - 1)
        for p, trail in enumerate(trails):
            sel = (phase_of == p) & (t <= time)
            trail.set_data(x[sel], y[sel])
        px, py = float(np.interp(time, t, x)), float(np.interp(time, t, y))
        heading = _screen_angle(ax, px, py, x[i] - x[i - 1], y[i] - y[i - 1])
        k_phase = int(np.searchsorted(knots, time, side="right"))
        current = rockets[min(k_phase, len(rockets) - 1)]
        for j, rocket in enumerate(rockets):
            if rocket is not current:
                rocket.hide()
        for patch, _ in current.parts:
            patch.set_visible(True)
        a = float(np.interp(time, t, th))
        up = np.array([np.sin(a), np.cos(a)])
        tangent = np.array([np.cos(a), -np.sin(a)])
        thrust = (
            k * float(np.interp(time, t, tr)) * up
            + float(np.interp(time, t, tt)) * tangent
        )
        if power > 0.03 and np.linalg.norm(thrust) > 0:
            direction = thrust / np.linalg.norm(thrust)
            nose = _screen_angle(ax, px, py, direction[0], direction[1])
            length = 0.08 * span_px * power
            start = ax.transData.inverted().transform(
                ax.transData.transform((px, py))
                + length * np.array([np.cos(nose), np.sin(nose)])
            )
            arrow.xy = tuple(start)
            arrow.set_position((px, py))
            arrow.set_visible(True)
        else:
            nose = heading
            arrow.set_visible(False)
        current.place(px, py, nose, power, flicker)
        if dropped is not None and knots:
            since = time - knots[0]
            fade = max(0.0, 1 - since / (0.12 * (t[-1] - t[0]))) if since >= 0 else 0
            for patch, _ in dropped.parts:
                patch.set_visible(fade > 0)
            if fade > 0:
                sx_, sy_ = (
                    float(np.interp(knots[0], t, x)),
                    float(np.interp(knots[0], t, y)),
                )
                dropped.place(sx_, sy_ - 0.02 * span, heading, 0.0, alpha=fade)
                sep.set_position((sx_ + 0.02 * span, sy_ - 0.05 * span))
                sep.set_text("段分離")
            else:
                sep.set_text("")

    flight.note = (
        flight.note
        or f"高度方向を×{k:g}に拡大して表示。橙の矢印：推力の向き。炎と矢印の長さは各段の最大推力に対する比"
    )
    return update


def _orbit(ax, flight, phases, t, throttle, knots, fig):
    assert flight.position and flight.thrust_radial and flight.thrust_tangential
    xs = _cat(phases, lambda p: flight.position(p)[0])
    ys = _cat(phases, lambda p: flight.position(p)[1])
    ur = _cat(phases, flight.thrust_radial)
    ut = _cat(phases, flight.thrust_tangential)
    reach = max(flight.orbits + [float(np.max(np.hypot(xs, ys)))]) * 1.15
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal")
    ax.set_facecolor("#020617")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(Circle((0, 0), 0.25, color="#fbbf24", zorder=3))
    ax.add_patch(Wedge((0, 0), 0.4, 0, 360, width=0.12, color="#fbbf24", alpha=0.2))
    for radius in flight.orbits:
        ax.add_patch(
            Circle((0, 0), radius, fill=False, ec="#94a3b8", ls="--", lw=1.2, zorder=2)
        )
    ax.plot(xs, ys, color="#64748b", lw=1, alpha=0.5, zorder=3)
    burn = np.where(throttle > 0.03, 1.0, np.nan)
    (trail_on,) = ax.plot([], [], color=FLAME, lw=3, zorder=4)
    (trail_off,) = ax.plot([], [], color="#e5e7eb", lw=2, zorder=4)
    craft = RocketGlyph(ax, length=30, width=12)
    arrow = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(0, 0),
        arrowprops={"arrowstyle": "-|>", "color": FLAME, "lw": 2.5},
        zorder=7,
    )
    ax.text(
        0.02,
        0.03,
        "破線：初期軌道 r="
        + f"{flight.orbits[0]:g}"
        + "・目標軌道 r="
        + f"{flight.orbits[-1]:g}"
        + "\n橙の軌跡：噴射中   白の軌跡：慣性飛行",
        transform=ax.transAxes,
        color="#e5e7eb",
        fontsize=11,
    )

    def update(time, power, flicker):
        sel = t <= time
        trail_on.set_data(xs[sel] * burn[sel], ys[sel])
        trail_off.set_data(np.where(np.isnan(burn[sel]), xs[sel], np.nan), ys[sel])
        i = min(max(int(np.searchsorted(t, time)), 1), len(t) - 1)
        px, py = float(np.interp(time, t, xs)), float(np.interp(time, t, ys))
        heading = _screen_angle(ax, px, py, xs[i] - xs[i - 1], ys[i] - ys[i - 1])
        a = np.arctan2(py, px)
        radial = np.array([np.cos(a), np.sin(a)])
        tangential = np.array([-np.sin(a), np.cos(a)])
        thrust = (
            float(np.interp(time, t, ur)) * radial
            + float(np.interp(time, t, ut)) * tangential
        )
        if power > 0.03:
            d = thrust / np.linalg.norm(thrust)
            nose = _screen_angle(ax, px, py, d[0], d[1])
            scale = 0.35 * reach * power
            arrow.xy = (px + scale * d[0], py + scale * d[1])
            arrow.set_position((px, py))
            arrow.set_visible(True)
        else:
            nose = heading
            arrow.set_visible(False)
        craft.place(px, py, nose, power, flicker)

    return update


# ---------------------------------------------------------------------------
# Static comparison of several solved cases (presets)


def render_comparison(
    flights: dict[str, Flight], cases: dict[str, list[Phase]], path: Path, title: str
) -> None:
    """Overlay every case: the path in the plane (if any) and every graph."""
    # Presets may differ in panels (e.g. a mass graph only with variable mass).
    first = max(flights.values(), key=lambda flight: len(flight.panels))
    has_path = first.kind == "track"
    panels = first.panels
    fig = plt.figure(figsize=SIZE, dpi=DPI, facecolor="white")
    grid = fig.add_gridspec(
        len(panels),
        2 if has_path else 1,
        width_ratios=[1.25, 1] if has_path else [1],
        left=0.06,
        right=0.97,
        top=0.88,
        bottom=0.08,
        wspace=0.16,
        hspace=0.22,
    )
    column = 1 if has_path else 0
    axes = [fig.add_subplot(grid[i, column]) for i in range(len(panels))]
    for ax in axes[1:]:
        ax.sharex(axes[0])
    scene = fig.add_subplot(grid[:, 0]) if has_path else None
    fig.text(0.04, 0.95, title, fontsize=20, weight="bold", color=INK)
    palette = PHASE_COLORS + ["#f59e0b", "#0891b2"]
    # Distinct dashes keep coinciding solutions (e.g. an inactive floor) visible.
    dashes = [
        "-",
        (0, (6, 3)),
        (0, (2, 2)),
        (0, (8, 3, 2, 3)),
        (0, (1, 2)),
        (0, (4, 1)),
    ]
    for k, (name, phases) in enumerate(cases.items()):
        color, dash = palette[k % len(palette)], dashes[k % len(dashes)]
        flight = flights[name]
        t = _cat(phases, lambda p: p["time"])
        for ax, group in zip(axes, flight.panels, strict=False):
            for j, series in enumerate(group):
                ax.plot(
                    t,
                    _cat(phases, series.value),
                    color=color,
                    lw=2.2,
                    ls=dash,
                    label=name if j == 0 else None,
                )
        if scene is not None and flight.x and flight.y:
            scene.plot(
                _cat(phases, flight.x),
                _cat(phases, flight.y),
                color=color,
                lw=2.6,
                ls=dash,
                label=name,
            )
    for ax, group in zip(axes, panels, strict=True):
        ax.set_ylabel(" / ".join(s.label for s in group))
        ax.grid(alpha=0.3)
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel(f"時刻 [{first.time_unit}]")
    if scene is not None:
        scene.set_facecolor(LIGHT)
        scene.set(xlabel=first.xlabel, ylabel=first.ylabel)
        scene.set_aspect("equal", adjustable="datalim")
        if first.invert_y:
            scene.invert_yaxis()
        scene.grid(alpha=0.3)
        scene.legend(loc="lower left", fontsize=11)
    else:
        axes[0].legend(loc="best", fontsize=11, ncols=2)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
