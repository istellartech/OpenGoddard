"""What each example's video shows: scene, graphs, readout and references.

Every getter takes one sampled phase (a dict of arrays keyed by the names in
``Classic.states`` / ``controls``, or by the example's own export names).
Analytic references are drawn only where the example states one.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from scenes import Flight, Phase, Series

Getter = Callable[[Phase], np.ndarray]
RE = 6371.0e3


def key(name: str, scale: float = 1.0) -> Getter:
    return lambda phase: phase[name] * scale


def deg(name: str) -> Getter:
    return lambda phase: np.rad2deg(phase[name])


def _km(name: str) -> Getter:
    return lambda p: (p[name] - RE) / 1e3


# ---------------------------------------------------------------------------
# Brachistochrone family: an analytic cycloid and a straight-line rival


def cycloid(radius: float, g: float, t: np.ndarray):
    """x, depth, speed and tangent angle from the downward vertical."""
    phi = t * np.sqrt(g / radius)
    x = radius * (phi - np.sin(phi))
    depth = radius * (1 - np.cos(phi))
    return x, depth, np.sqrt(2 * g * depth), phi / 2


def straight_line(end: tuple[float, float], g: float):
    """Frictionless slide on a straight ramp from rest: position and arrival."""
    xf, yf = end
    length = float(np.hypot(xf, yf))
    accel = g * yf / length
    arrival = float(np.sqrt(2 * length / accel))

    def at(time: float) -> tuple[float, float]:
        s = 0.5 * accel * min(time, arrival) ** 2
        return xf * s / length, yf * s / length

    return at, arrival


def draw_cycloid(radius: float, phi_end: float, scale: float = 1.0):
    def decorate(ax):
        phi = np.linspace(0, phi_end, 300)
        ax.plot(
            radius * (phi - np.sin(phi)) * scale,
            radius * (1 - np.cos(phi)) * scale,
            color="#111827",
            lw=1.4,
            ls=(0, (4, 3)),
            zorder=2,
        )

    return decorate


def brachistochrone(phases: list[Phase]) -> Flight:
    g, length = 1.0, 1.0  # Ball in examples/03
    radius = length / np.pi  # free final depth: the curve ends horizontal
    end_t = phases[-1]["time"][-1]
    t_ref = np.linspace(0, np.pi * np.sqrt(radius / g), 200)
    _, _, v_ref, theta_ref = cycloid(radius, g, t_ref)
    end = (float(phases[-1]["x"][-1]), float(phases[-1]["y"][-1]))
    ghost, arrival = straight_line(end, g)
    return Flight(
        "track",
        "03. 最速降下：数値解・サイクロイド・直線の比較",
        ["最速降下"],
        x=key("x"),
        y=key("y"),
        xlabel="x",
        ylabel="深さ y（下向き）",
        invert_y=True,
        decorate=draw_cycloid(radius, np.pi),
        ghost=ghost,
        ghost_time=arrival,
        ghost_label="同じ終点への直線の斜面（解析解）",
        legend_lines=["黒の破線：サイクロイド（解析解）"],
        panels=[
            [Series("速さ v", key("v"), "{:.3f}")],
            [Series("軌道角 θ [deg]（鉛直から）", deg("theta"), "{:.1f}", "deg")],
        ],
        overlays={
            0: [(t_ref, v_ref, "サイクロイド（解析解）")],
            1: [(t_ref, np.rad2deg(theta_ref), "サイクロイド（解析解）")],
        },
        readout=[
            Series("x", key("x"), "{:.3f}"),
            Series("深さ", key("y"), "{:.3f}"),
            Series("速さ", key("v"), "{:.3f}"),
        ],
        time_unit="-",
        note=f"g = 1, 終点 x = 1。最適解 {end_t:.3f}、直線 {arrival:.3f} で到着",
    )


def tokyo_osaka(phases: list[Phase]) -> Flight:
    g, length = 9.8, 600e3  # Ball in examples/05: both ends on the surface
    radius = length / (2 * np.pi)
    t_ref = np.linspace(0, 2 * np.pi * np.sqrt(radius / g), 300)
    _, _, v_ref, theta_ref = cycloid(radius, g, t_ref)

    def decorate(ax):
        draw_cycloid(radius, 2 * np.pi, 1e-3)(ax)
        ax.axhline(0, color="#4d7c0f", lw=3, zorder=1)
        for x, name in ((0, "東京"), (600, "大阪")):
            ax.plot([x], [0], "v", color="#4d7c0f", ms=12, zorder=5)
            ax.text(x, -8, name, ha="center", va="bottom", fontsize=13, weight="bold")

    return Flight(
        "track",
        "05. 東京〜大阪 600 km の最速降下トンネル",
        ["最速降下"],
        x=key("x", 1e-3),
        y=key("y", 1e-3),
        xlabel="東京からの距離 [km]",
        ylabel="深さ [km]",
        invert_y=True,
        decorate=decorate,
        legend_lines=["黒の破線：サイクロイド（解析解）"],
        panels=[
            [Series("速さ [km/s]", key("v", 1e-3), "{:.3f}", "km/s")],
            [Series("軌道角 θ [deg]（鉛直から）", deg("theta"), "{:.1f}", "deg")],
        ],
        overlays={
            0: [(t_ref, v_ref / 1e3, "サイクロイド（解析解）")],
            1: [(t_ref, np.rad2deg(theta_ref), "サイクロイド（解析解）")],
        },
        readout=[
            Series("距離", key("x", 1e-3), "{:.0f}", "km"),
            Series("深さ", key("y", 1e-3), "{:.0f}", "km"),
            Series("速さ", key("v", 1e-3), "{:.3f}", "km/s"),
        ],
        note="一様重力 g = 9.8 m/s²、摩擦なし（地球の曲率は無視）",
    )


def simple_rocket(phases: list[Phase]) -> Flight:
    def decorate(ax):
        ax.axhline(1.0, color="#059669", lw=1.5, ls="--", zorder=1)
        ax.text(
            0.02,
            1.0,
            " 目標高度 y = 1（ここで水平速度 1、鉛直速度 0）",
            color="#059669",
            va="bottom",
            fontsize=11,
            transform=ax.get_yaxis_transform(),
        )

    return Flight(
        "plane",
        "06. 2次元ロケット：推力の向きだけで目標高度に水平に入る",
        ["最短時間"],
        throttle=lambda p: np.ones_like(p["time"]),
        x=key("x"),
        y=key("y"),
        heading=key("beta"),
        xlabel="水平位置 x",
        ylabel="高度 y",
        decorate=decorate,
        panels=[
            [
                Series("水平速度 u", key("u"), "{:.3f}"),
                Series("鉛直速度 v", key("v"), "{:.3f}"),
            ],
            [Series("推力の向き β [deg]（水平から）", deg("beta"), "{:.1f}", "deg")],
        ],
        readout=[
            Series("x", key("x"), "{:.3f}"),
            Series("y", key("y"), "{:.3f}"),
            Series("β", deg("beta"), "{:.1f}", "deg"),
        ],
        time_unit="-",
        note="加速度の大きさは一定 a = 1。炎は常に全開",
    )


def minimum_energy_transfer(phases: list[Phase]) -> Flight:
    t_ref = np.array([5.0, 7.0])  # exact: T = d = 2 with u = 1 from t = 5
    return Flight(
        "rail",
        "01. 最小操作量の移動：x' = u、時間と操作量の和を最小化",
        ["移動"],
        x=key("x"),
        arrow=key("u"),
        arrow_label="制御 u（＝速度）",
        xlabel="位置 x",
        panels=[
            [Series("位置 x", key("x"), "{:.3f}")],
            [Series("制御 u", key("u"), "{:.4f}")],
        ],
        overlays={
            0: [(t_ref, t_ref - 5, "解析解 x = t − 5")],
            1: [(t_ref, np.ones(2), "解析解 u = 1")],
        },
        readout=[Series("x", key("x"), "{:.3f}"), Series("u", key("u"), "{:.4f}")],
        time_unit="-",
        note="目的関数 J = T + ∫u²dt。解析解は T = 2、u = 1、J = 4",
    )


@dataclass
class Classic:
    """An example whose ``main()`` returns the solved Problem."""

    states: list[str]
    controls: list[str]
    flight: Callable[[list[Phase]], Flight]
    prepare: Callable[[list[Phase]], None] | None = None
    savefig_flag: bool = True


# ---------------------------------------------------------------------------
# Rockets and orbits


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
        note="無次元量（地球半径=1）。ロケットは模式図、炎の長さは推力に比例",
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
        note="ロケットは模式図、炎の長さは各段の最大推力に対する比",
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
                    "推力の仰角 [deg]", _ascent_pitch, "{:.1f}", "deg", mask_coast=True
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


def low_thrust_xy(phases: list[Phase]) -> None:
    """Integrate theta' = vt/r continuously across all samples."""
    from scipy.integrate import cumulative_trapezoid

    t = np.concatenate([p["time"] for p in phases])
    rate = np.concatenate([p["vt"] / p["r"] for p in phases])
    theta = cumulative_trapezoid(rate, t, initial=0)
    start = 0
    for phase in phases:
        n = len(phase["time"])
        part = theta[start : start + n]
        phase["px"] = phase["r"] * np.cos(part)
        phase["py"] = phase["r"] * np.sin(part)
        start += n


LOW_THRUST = Flight(
    "orbit",
    "15. 低推力による軌道遷移（r = 1 → 4）",
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
)


def _fixed(flight: Flight) -> Callable[[list[Phase]], Flight]:
    return lambda phases: replace(flight)


CLASSIC: dict[str, Classic] = {
    "minimum_energy_transfer": Classic(
        ["x"], ["u"], minimum_energy_transfer, savefig_flag=False
    ),
    "brachistochrone": Classic(["x", "y", "v"], ["theta"], brachistochrone),
    "brachistochrone_tokyo_osaka": Classic(["x", "y", "v"], ["theta"], tokyo_osaka),
    "simple_rocket_2d": Classic(["u", "v", "x", "y"], ["beta"], simple_rocket),
    "goddard_0knot": Classic(
        ["h", "v", "m"],
        ["T"],
        _fixed(goddard_flight("07. Goddard問題：単一区間", ["単一区間"])),
    ),
    "goddard_1knot": Classic(
        ["h", "v", "m"],
        ["T"],
        _fixed(
            goddard_flight("08. Goddard問題：区間分割（knot 1つ）", ["区間1", "区間2"])
        ),
    ),
    "rocket_ascent_single_stage": Classic(
        ["R", "v", "m"],
        ["T"],
        _fixed(vertical_rocket_flight("10. 単段ロケットの鉛直上昇", ["単段"], False)),
    ),
    "rocket_ascent_two_stage": Classic(
        ["R", "v", "m"],
        ["T"],
        _fixed(
            vertical_rocket_flight(
                "11. 2段ロケットの鉛直上昇", ["1段目", "2段目"], True
            )
        ),
    ),
    "rocket_ascent_polar_ssto": Classic(
        ["R", "theta", "Vr", "Vt", "m"],
        ["Tr", "Tt"],
        _fixed(ascent_flight("12. 極座標：単段で軌道投入", ["単段"], False)),
    ),
    "rocket_ascent_polar_tsto": Classic(
        ["R", "theta", "Vr", "Vt", "m"],
        ["Tr", "Tt"],
        _fixed(ascent_flight("13. 極座標：2段で軌道投入", ["1段目", "2段目"], True)),
    ),
    "polar_tsto_taiki": Classic(
        ["R", "theta", "Vr", "Vt", "m"],
        ["Tr", "Tt"],
        _fixed(
            ascent_flight(
                "14. 大樹町からの2段軌道投入（空気抵抗あり）", ["1段目", "2段目"], True
            )
        ),
    ),
    "low_thrust_orbit_transfer": Classic(
        ["r", "vr", "vt"],
        ["ur1", "ur2", "ut1", "ut2"],
        _fixed(LOW_THRUST),
        prepare=low_thrust_xy,
    ),
}


# ---------------------------------------------------------------------------
# Lessons with presets (examples/_shared.py exports)

MOVE_PHASES = {
    "accelerate": "加速",
    "cruise": "定速",
    "brake": "減速",
    "transfer": "移動",
    "ramp": "加速度を切り替え",
    "ramp_down": "加速を弱める",
    "ramp_brake": "減速へ移行",
}
DESCENT_PHASES = {
    "descend": "降下",
    "floor": "床に沿って移動",
    "ascend": "上昇",
    "cycloid": "サイクロイド",
}
LANDING_PHASES = {"coast": "慣性降下", "burn": "減速噴射"}


def move_stop_reference(module, parameters):
    """Exact piecewise-polynomial solution from the lesson's phase_plan()."""
    durations, arcs, _ = module.phase_plan(parameters)
    times, accel, speed, position = [], [], [], []
    t0 = x0 = v0 = 0.0
    for duration, (a, jerk) in zip(durations, arcs, strict=True):
        tau = np.linspace(0, duration, 60)
        times.append(t0 + tau)
        accel.append(a + jerk * tau)
        speed.append(v0 + a * tau + jerk * tau**2 / 2)
        position.append(x0 + v0 * tau + a * tau**2 / 2 + jerk * tau**3 / 6)
        t0, x0, v0 = t0 + duration, float(position[-1][-1]), float(speed[-1][-1])
    cat = np.concatenate
    return cat(times), cat(position), cat(speed), cat(accel)


def shared_flight(
    example_id: str, module, name: str, parameters, phase_names: list[str]
) -> Flight:
    if example_id == "move_stop":
        t, x, v, a = move_stop_reference(module, parameters)
        label = "解析解"
        goal = (
            "最短時間" if parameters.objective == "minimum_time" else "最小二乗操作量"
        )
        return Flight(
            "rail",
            f"02. 加速して停止する移動：{goal}（{name}）",
            [MOVE_PHASES.get(n, n) for n in phase_names],
            x=key("position"),
            arrow=key("acceleration"),
            arrow_label="加速度",
            xlabel="位置 [m]",
            panels=[
                [Series("位置 [m]", key("position"), "{:.2f}", "m")],
                [Series("速度 [m/s]", key("velocity"), "{:.2f}", "m/s")],
                [Series("加速度 [m/s²]", key("acceleration"), "{:.2f}", "m/s²")],
            ],
            overlays={0: [(t, x, label)], 1: [(t, v, label)], 2: [(t, a, label)]},
            readout=[
                Series("位置", key("position"), "{:.2f}", "m"),
                Series("速度", key("velocity"), "{:.2f}", "m/s"),
                Series("加速度", key("acceleration"), "{:.2f}", "m/s²"),
            ],
        )
    if example_id == "constrained_descent":
        p = parameters
        ghost, arrival = straight_line((p.distance, p.final_depth), p.gravity)

        def decorate(ax):
            if p.max_depth is not None:
                ax.axhspan(
                    p.max_depth,
                    p.max_depth * 1.6 + 0.1,
                    color="#a8a29e",
                    alpha=0.5,
                    zorder=0,
                    hatch="//",
                )
                ax.axhline(p.max_depth, color="#57534e", lw=2.5, zorder=1)
                ax.text(
                    0.02,
                    p.max_depth,
                    f" 床 深さ {p.max_depth:g} m",
                    transform=ax.get_yaxis_transform(),
                    va="top",
                    fontsize=11,
                )
            ax.plot(
                [p.distance], [p.final_depth], "*", color="#059669", ms=16, zorder=5
            )

        floor = "床なし" if p.max_depth is None else f"床 {p.max_depth:g} m"
        return Flight(
            "track",
            f"04. 深さ制限付き最速降下（{name}：{floor}, g = {p.gravity:g}）",
            [DESCENT_PHASES.get(n, n) for n in phase_names],
            x=key("x"),
            y=key("depth"),
            xlabel="x [m]",
            ylabel="深さ [m]（下向き）",
            invert_y=True,
            decorate=decorate,
            ghost=ghost,
            ghost_time=arrival,
            ghost_label="同じ終点への直線の斜面（解析解）",
            panels=[
                [Series("速さ [m/s]", key("speed"), "{:.3f}", "m/s")],
                [Series("軌道角 θ [deg]（鉛直から）", deg("theta"), "{:.1f}", "deg")],
            ],
            readout=[
                Series("x", key("x"), "{:.3f}", "m"),
                Series("深さ", key("depth"), "{:.3f}", "m"),
                Series("速さ", key("speed"), "{:.3f}", "m/s"),
            ],
        )
    if example_id == "soft_landing":
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
            f"09. 月面への鉛直軟着陸（{name}）",
            [LANDING_PHASES.get(n, n) for n in phase_names],
            throttle=key("thrust"),
            altitude=key("altitude"),
            altitude_label="高度 [m]",
            altitude_unit="m",
            theme="moon",
            panels=panels,
            readout=readout,
            note="着陸機は模式図、炎の長さは推力に比例",
            coast_speedup=1.0,
        )
    raise KeyError(example_id)
