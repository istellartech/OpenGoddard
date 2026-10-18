"""What each example's figures and video show.

Every entry names the example's states and controls in the order the example
defines them, then lists the time-history panels, the scene and the numbers
reported on the page (``extras``). Analytic references are drawn only where the
problem has a closed-form solution.
"""

import numpy as np
from figures import Panel, Quantity, Scene, Solution, Spec
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq
from style import MUTED, REFERENCE

Q = Quantity


def _dashed(ax, x, y, label=None):
    ax.plot(x, y, color=REFERENCE, lw=1.3, ls=(0, (3, 2)), zorder=5, label=label)


def _deg(name):
    def derive(d, m):
        d[name + "_deg"] = np.rad2deg(d[name])

    return derive


# ---------------------------------------------------------------------------
# 01: x' = u, minimize T + integral u^2


def _mini_refs(sol: Solution):
    t = sol.cat("time")
    return [(t, t - t[0], "解析解")]


def _mini_u(sol: Solution):
    t = sol.cat("time")
    return [(t, np.ones_like(t), "解析解")]


def _mini_extras(sol, problem, module):
    duration = problem.time_final(0) - problem.time_start(0)
    return {"duration": duration, "exact_duration": 2.0, "exact_objective": 4.0}


MINIMUM_ENERGY = Spec(
    states=["x"],
    controls=["u"],
    history=[
        Panel([Q("x", "位置 $x$")], references=_mini_refs),
        Panel([Q("u", "速度 $u$（制御）")], references=_mini_u, limits=[(3, "上限 3")]),
    ],
    scene=Scene(
        "rail", Q("x", "位置 $x$"), Q("x", ""), body="cart", readout=[Q("u", "$u$")]
    ),
    video_panels=[0, 1],
    time_unit="",
    extras=_mini_extras,
    seconds=6,
)


# ---------------------------------------------------------------------------
# 02: minimum-time move and stop


def _bang_bang(sol: Solution):
    d, a = 10.0, 2.0
    total = 2 * np.sqrt(d / a)
    t = np.linspace(0, total, 300)
    half = t <= total / 2
    x = np.where(half, a * t**2 / 2, d - a * (total - t) ** 2 / 2)
    v = np.where(half, a * t, a * (total - t))
    acc = np.where(half, a, -a)
    return total, t, x, v, acc


def _move_ref(index, label="解析解"):
    def refs(sol):
        values = _bang_bang(sol)
        return [(values[1], values[index], label)]

    return refs


def _move_extras(sol, problem, module):
    return {"final_time": problem.time_final(-1), "exact_final_time": 2 * np.sqrt(5)}


MOVE_STOP = Spec(
    states=["x", "v"],
    controls=["a"],
    history=[
        Panel([Q("x", "位置 $x$", "m")], references=_move_ref(2)),
        Panel([Q("v", "速度 $v$", "m/s")], references=_move_ref(3)),
        Panel(
            [Q("a", "加速度 $a$（制御）", "m/s²")],
            references=_move_ref(4),
            limits=[(2, "上限 +2"), (-2, "下限 −2")],
        ),
    ],
    scene=Scene(
        "rail",
        Q("x", "位置 $x$", "m"),
        Q("x", ""),
        body="cart",
        readout=[Q("v", "$v$", "m/s"), Q("a", "$a$", "m/s²")],
    ),
    video_panels=[1, 2],
    phase_names=["区間1（加速）", "区間2（減速）"],
    extras=_move_extras,
    seconds=7,
)


# ---------------------------------------------------------------------------
# 03-05: brachistochrone family


def cycloid(radius, g, t):
    """x, depth, speed and angle from the downward vertical along a cycloid."""
    phi = t * np.sqrt(g / radius)
    return (
        radius * (phi - np.sin(phi)),
        radius * (1 - np.cos(phi)),
        np.sqrt(2 * g * radius * (1 - np.cos(phi))),
        phi / 2,
    )


def straight_line(end, g):
    xf, yf = end
    length = float(np.hypot(xf, yf))
    accel = g * yf / length
    arrival = float(np.sqrt(2 * length / accel))

    def at(time):
        s = 0.5 * accel * min(time, arrival) ** 2
        return xf * s / length, yf * s / length

    return at, arrival


def _brach_refs(index, scale=1.0):
    def refs(sol):
        radius = 1 / np.pi
        t = np.linspace(0, np.pi * np.sqrt(radius), 200)
        return [(t, cycloid(radius, 1.0, t)[index] * scale, "サイクロイド（解析解）")]

    return refs


def _brach_decorate(ax, sol):
    radius = 1 / np.pi
    phi = np.linspace(0, np.pi, 200)
    _dashed(
        ax,
        radius * (phi - np.sin(phi)),
        radius * (1 - np.cos(phi)),
        "サイクロイド（解析解）",
    )
    ax.plot([0, 1], [0, 2 / np.pi], color=MUTED, lw=1, ls=":", label="直線の斜面")
    ax.legend(loc="upper right")


_BRACH_LINE, _BRACH_LINE_TIME = straight_line((1.0, 2 / np.pi), 1.0)


def _brach_extras(sol, problem, module):
    return {
        "final_time": problem.time_final(-1),
        "exact_final_time": float(np.sqrt(np.pi)),
        "straight_line_time": _BRACH_LINE_TIME,
        "final_depth": float(sol.nodes[-1]["y"][-1]),
        "exact_final_depth": 2 / np.pi,
    }


BRACHISTOCHRONE = Spec(
    states=["x", "y", "v"],
    controls=["theta"],
    derive=_deg("theta"),
    history=[
        Panel([Q("v", "速さ $v$", "m/s")], references=_brach_refs(2)),
        Panel(
            [Q("theta_deg", "角度 $\\theta$（制御）", "deg")],
            references=_brach_refs(3, 180 / np.pi),
        ),
    ],
    scene=Scene(
        "plane",
        Q("x", "水平位置 $x$", "m"),
        Q("y", "深さ $y$", "m"),
        invert_y=True,
        equal=True,
        decorate=_brach_decorate,
        ghost=_BRACH_LINE,
        readout=[Q("v", "$v$", "m/s")],
    ),
    video_panels=[0, 1],
    extras=_brach_extras,
    seconds=6,
)


def _floorless_cycloid(distance=1.0, depth=0.2):
    phi = brentq(
        lambda p: (p - np.sin(p)) / (1 - np.cos(p)) - distance / depth,
        1e-3,
        2 * np.pi - 1e-6,
    )
    radius = depth / (1 - np.cos(phi))
    return radius, phi


def _descent_decorate(ax, sol):
    radius, phi_end = _floorless_cycloid()
    phi = np.linspace(0, phi_end, 300)
    _dashed(
        ax,
        radius * (phi - np.sin(phi)),
        radius * (1 - np.cos(phi)),
        "床がない場合の最速降下曲線",
    )
    ax.axhspan(0.3, 1.0, color="#e7e5e4", zorder=0)
    ax.axhline(0.3, color="#a8a29e", lw=1.5)
    ax.text(
        0.5, 0.39, "床（深さ 0.3 m）", ha="center", va="center", fontsize=9, color=MUTED
    )
    ax.plot([1.0], [0.2], "x", color=REFERENCE, ms=8, mew=2)
    ax.legend(loc="upper right")


def _descent_extras(sol, problem, module):
    radius, phi = _floorless_cycloid()
    g = 9.81
    return {
        "final_time": problem.time_final(0),
        "max_depth": float(sol.cat("depth").max()),
        "floorless_time": float(phi * np.sqrt(radius / g)),
        "floorless_max_depth": float(
            2 * radius if phi > np.pi else radius * (1 - np.cos(phi))
        ),
    }


CONSTRAINED_DESCENT = Spec(
    states=["x", "depth", "v"],
    controls=["theta"],
    derive=_deg("theta"),
    history=[
        Panel([Q("depth", "深さ $y$", "m")], limits=[(0.3, "上限 0.3 m")]),
        Panel([Q("v", "速さ $v$", "m/s")]),
        Panel([Q("theta_deg", "角度 $\\theta$（制御）", "deg")]),
    ],
    scene=Scene(
        "plane",
        Q("x", "水平位置 $x$", "m"),
        Q("depth", "深さ $y$", "m"),
        invert_y=True,
        equal=True,
        decorate=_descent_decorate,
        ylim=(-0.04, 0.42),
        readout=[Q("v", "$v$", "m/s")],
    ),
    video_panels=[0, 1, 2],
    extras=_descent_extras,
    seconds=6,
)


_OSAKA_RADIUS = 600e3 / (2 * np.pi)
_OSAKA_TIME = 2 * np.pi * np.sqrt(_OSAKA_RADIUS / 9.8)


def _osaka_refs(index, scale):
    def refs(sol):
        t = np.linspace(0, _OSAKA_TIME, 300)
        return [
            (t, cycloid(_OSAKA_RADIUS, 9.8, t)[index] * scale, "サイクロイド（解析解）")
        ]

    return refs


def _osaka_decorate(ax, sol):
    phi = np.linspace(0, 2 * np.pi, 300)
    r = _OSAKA_RADIUS / 1e3
    _dashed(
        ax, r * (phi - np.sin(phi)), r * (1 - np.cos(phi)), "サイクロイド（解析解）"
    )
    ax.axhline(0, color="#4d7c0f", lw=2.5, zorder=1)
    for x, name in ((0, "東京"), (600, "大阪")):
        ax.plot([x], [0], "v", color="#4d7c0f", ms=8, zorder=5)
        ax.annotate(
            name,
            (x, 0),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            color="#365314",
        )
    ax.legend(loc="lower center")


def _osaka_extras(sol, problem, module):
    return {
        "final_time": problem.time_final(-1),
        "exact_final_time": _OSAKA_TIME,
        "max_depth_km": float(sol.cat("y").max() / 1e3),
        "exact_max_depth_km": 2 * _OSAKA_RADIUS / 1e3,
        "max_speed": float(sol.cat("v").max()),
        "exact_max_speed": float(np.sqrt(2 * 9.8 * 2 * _OSAKA_RADIUS)),
    }


TOKYO_OSAKA = Spec(
    states=["x", "y", "v"],
    controls=["theta"],
    derive=_deg("theta"),
    history=[
        Panel([Q("y", "深さ $y$", "km", 1e-3)], references=_osaka_refs(1, 1e-3)),
        Panel([Q("v", "速さ $v$", "km/s", 1e-3)], references=_osaka_refs(2, 1e-3)),
        Panel(
            [Q("theta_deg", "角度 $\\theta$（制御）", "deg")],
            references=_osaka_refs(3, 180 / np.pi),
        ),
    ],
    scene=Scene(
        "plane",
        Q("x", "東京からの距離 $x$", "km", 1e-3),
        Q("y", "深さ $y$", "km", 1e-3),
        invert_y=True,
        equal=True,
        decorate=_osaka_decorate,
        ylim=(-40, 215),
        readout=[Q("y", "深さ", "km", 1e-3), Q("v", "速さ", "km/s", 1e-3)],
    ),
    video_panels=[0, 1],
    extras=_osaka_extras,
    seconds=8,
)


# ---------------------------------------------------------------------------
# 06: planar rocket with constant acceleration magnitude


def _simple_extras(sol, problem, module):
    return {"final_time": problem.time_final(-1)}


SIMPLE_ROCKET = Spec(
    states=["vx", "vy", "x", "y"],
    controls=["beta"],
    derive=_deg("beta"),
    history=[
        Panel([Q("x", "水平位置 $x$"), Q("y", "高さ $y$")], title="位置 [m]"),
        Panel([Q("vx", "水平速度 $u$"), Q("vy", "鉛直速度 $v$")], title="速度 [m/s]"),
        Panel(
            [Q("beta_deg", "加速度の向き $\\beta$（制御）", "deg")],
            limits=[(90, "+90°"), (-90, "−90°")],
        ),
    ],
    scene=Scene(
        "plane",
        Q("x", "水平位置 $x$", "m"),
        Q("y", "高さ $y$", "m"),
        body="rocket",
        equal=True,
        heading=lambda d: d["beta"],
        throttle=lambda d: np.full_like(d["time"], 0.6),
        readout=[Q("vx", "$u$", "m/s"), Q("vy", "$v$", "m/s")],
        label="橙の矢印：加速度の向き（ノードごと）",
    ),
    video_panels=[1, 2],
    extras=_simple_extras,
    seconds=6,
)


# ---------------------------------------------------------------------------
# 07-08: Goddard rocket (dimensionless)


def _goddard_extras(sol, problem, module):
    t = sol.cat("time")
    thrust = sol.cat("T")
    burning = t[thrust > 0.05 * 3.5]
    return {
        "final_time": problem.time_final(-1),
        "final_height": float(sol.nodes[-1]["h"][-1]),
        "burnout_time": float(burning.max()) if burning.size else None,
        "knots": sol.knots,
    }


def goddard(phases):
    return Spec(
        states=["h", "v", "m"],
        controls=["T"],
        history=[
            Panel([Q("h", "高度 $h$")]),
            Panel([Q("v", "速度 $v$")]),
            Panel([Q("m", "質量 $m$")], limits=[(0.6, "燃料切れの質量 0.6")]),
            Panel([Q("T", "推力 $T$（制御）")], limits=[(3.5, "最大推力 3.5")]),
        ],
        scene=Scene(
            "column",
            Q("h", ""),
            Q("h", "高度 $h$（無次元）"),
            body="rocket",
            throttle=lambda d: d["T"] / 3.5,
            ground=1.0,
            ylim=(0.9985, 1.0145),
            readout=[Q("h", "$h$"), Q("v", "$v$"), Q("T", "$T$")],
        ),
        video_panels=[1, 3],
        phase_names=phases,
        time_unit="",
        extras=_goddard_extras,
        seconds=8,
        coast_speedup=2.0,
    )


# ---------------------------------------------------------------------------
# 09: lunar soft landing


def _landing_refs(index):
    def refs(sol):
        h0, v0, g, mass, tmax = 100.0, -10.0, 1.62, 1000.0, 4000.0
        braking = tmax / mass - g
        vs = np.sqrt((v0**2 + 2 * g * h0) * braking / (braking + g))
        coast = (vs + v0) / g
        burn = vs / braking
        t = np.linspace(0, coast + burn, 300)
        fall = t <= coast
        tb = t - coast
        hs = h0 + v0 * coast - g * coast**2 / 2
        h = np.where(
            fall, h0 + v0 * t - g * t**2 / 2, hs - vs * tb + braking * tb**2 / 2
        )
        v = np.where(fall, v0 - g * t, -vs + braking * tb)
        thrust = np.where(fall, 0.0, tmax)
        return [(t, (h, v, thrust)[index], "解析解")]

    return refs


def _landing_extras(sol, problem, module):
    h0, v0, g, mass, tmax = 100.0, -10.0, 1.62, 1000.0, 4000.0
    braking = tmax / mass - g
    vs = np.sqrt((v0**2 + 2 * g * h0) * braking / (braking + g))
    coast = (vs + v0) / g
    t = sol.cat("time", dense=False)
    return {
        "final_time": problem.time_final(-1),
        "switch_time": problem.time_final(0),
        "exact_switch_time": float(coast),
        "exact_final_time": float(coast + vs / braking),
        "objective": float(problem.result.fun),
        "impulse_Ns": float(problem.result.fun * tmax * problem.unit_time),
        "nodes_time": t.tolist(),
    }


SOFT_LANDING = Spec(
    states=["h", "v"],
    controls=["T"],
    history=[
        Panel([Q("h", "高度 $h$", "m")], references=_landing_refs(0)),
        Panel([Q("v", "速度 $v$（上向き正）", "m/s")], references=_landing_refs(1)),
        Panel(
            [Q("T", "推力 $T$（制御）", "N")],
            references=_landing_refs(2),
            limits=[(4000, "最大推力 4000 N")],
        ),
    ],
    scene=Scene(
        "column",
        Q("h", ""),
        Q("h", "高度 $h$", "m"),
        body="lander",
        throttle=lambda d: d["T"] / 4000,
        ground=0.0,
        ylim=(-6, 108),
        readout=[Q("h", "$h$", "m"), Q("v", "$v$", "m/s"), Q("T", "$T$", "N")],
    ),
    video_panels=[1, 2],
    phase_names=["区間1", "区間2"],
    extras=_landing_extras,
    seconds=7,
)


# ---------------------------------------------------------------------------
# 10-11: vertical rocket ascent with an atmosphere


def _vertical_derive(d, m):
    rocket = m.Rocket()
    d["alt"] = (d["R"] - rocket.Re) / 1e3


def _vertical_extras(sol, problem, module):
    rocket = module.Rocket()
    t = sol.cat("time")
    thrust = sol.cat("T")
    tmax = rocket.max_thrust * rocket.M0 * rocket.g0
    burning = t[thrust > 0.03 * tmax]
    return {
        "final_time": problem.time_final(-1),
        "final_altitude_km": float(sol.nodes[-1]["alt"][-1]),
        "burnout_time": float(burning.max()) if burning.size else None,
        "max_thrust_N": tmax,
        "knots": sol.knots,
    }


def vertical(phases, max_thrust_kN):
    return Spec(
        states=["R", "v", "m"],
        controls=["T"],
        derive=_vertical_derive,
        history=[
            Panel([Q("alt", "高度 $R-R_e$", "km")]),
            Panel([Q("v", "速度 $v$", "m/s")]),
            Panel([Q("m", "質量 $m$", "kg")]),
            Panel(
                [Q("T", "推力 $T$（制御）", "kN", 1e-3)],
                limits=[(max_thrust_kN, f"最大推力 {max_thrust_kN:.0f} kN")],
            ),
        ],
        scene=Scene(
            "column",
            Q("alt", ""),
            Q("alt", "高度", "km"),
            body="rocket",
            throttle=lambda d: d["T"] / (max_thrust_kN * 1e3),
            ground=0.0,
            readout=[
                Q("alt", "高度", "km"),
                Q("v", "$v$", "m/s"),
                Q("T", "$T$", "kN", 1e-3),
            ],
        ),
        video_panels=[1, 3],
        phase_names=phases,
        extras=_vertical_extras,
        seconds=8,
        coast_speedup=3.0,
    )


# ---------------------------------------------------------------------------
# 12-14: ascent to orbit in polar coordinates


def _ascent_derive(per_stage: bool, tabulated: bool = False):
    def derive(d, m):
        rocket = m.Rocket()
        s = d["section"]
        h = d["R"] - rocket.Re
        d["alt"] = h / 1e3
        d["downrange"] = rocket.Re * d["theta"] / 1e3
        d["thrust"] = np.hypot(d["Tr"], d["Tt"])
        d["pitch"] = np.arctan2(d["Tr"], d["Tt"])
        speed = np.hypot(d["Vr"], d["Vt"])
        if tabulated:
            rho = rocket.airDensity(h)
            cd = rocket.Cd(speed / rocket.airSound(h))
            area = rocket.A[s]
            tmax = rocket.ThrustMax[s]
        else:
            rho = rocket.air_density(h)
            cd = rocket.Cd[s] if per_stage else rocket.Cd
            area = rocket.A[s] if per_stage else rocket.A
            tmax = rocket.Tmax[s] if per_stage else rocket.Tmax
        drag_r = 0.5 * rho * d["Vr"] * speed * cd * area
        drag_t = 0.5 * rho * d["Vt"] * speed * cd * area
        d["accel_G"] = np.hypot(d["Tr"] - drag_r, d["Tt"] - drag_t) / d["m"] / rocket.g0
        d["q_kPa"] = 0.5 * rho * speed**2 / 1e3
        d["throttle"] = d["thrust"] / tmax
        # The thrust direction means nothing while the engine is off.
        d["pitch_deg"] = np.where(d["throttle"] > 0.03, np.rad2deg(d["pitch"]), np.nan)

    return derive


def _ascent_decorate(ax, sol):
    ax.text(
        0.99,
        0.02,
        "高度方向を拡大して表示",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color=MUTED,
    )


def _ascent_extras(sol, problem, module):
    rocket = module.Rocket()
    m = sol.nodes[-1]["m"][-1]
    t = sol.cat("time")
    power = sol.cat("throttle")
    burning = power > 0.03
    spans = []
    start = None
    for i, on in enumerate(burning):
        if on and start is None:
            start = t[i]
        if not on and start is not None:
            spans.append((float(start), float(t[i])))
            start = None
    if start is not None:
        spans.append((float(start), float(t[-1])))
    return {
        "final_time": problem.time_final(-1),
        "knots": sol.knots,
        "final_mass_kg": float(m),
        "initial_mass_kg": float(sol.nodes[0]["m"][0]),
        "final_altitude_km": float(sol.nodes[-1]["alt"][-1]),
        "target_altitude_km": rocket.Htarget / 1e3,
        "final_speed": float(
            np.hypot(sol.nodes[-1]["Vr"][-1], sol.nodes[-1]["Vt"][-1])
        ),
        "target_speed": float(rocket.Vtarget),
        "max_accel_G": float(sol.cat("accel_G", dense=False).max()),
        "max_q_kPa": float(sol.cat("q_kPa", dense=False).max()),
        "burn_spans": spans,
        "downrange_km": float(sol.nodes[-1]["downrange"][-1]),
    }


def ascent(
    phases,
    thrust_limits,
    accel_limit=None,
    mass_unit=("t", 1e-3),
    extra_panels=(),
    per_stage=False,
    tabulated=False,
    speedup=6.0,
):
    panels = [
        Panel([Q("alt", "高度", "km")]),
        Panel(
            [
                Q("Vt", "水平速度 $V_t$", scale=1e-3),
                Q("Vr", "鉛直速度 $V_r$", scale=1e-3),
            ],
            title="速度 [km/s]",
        ),
        Panel([Q("m", "質量 $m$", mass_unit[0], mass_unit[1])]),
        Panel(
            [Q("thrust", "推力の大きさ $|T|$（制御）", "kN", 1e-3)],
            limits=[(v, f"最大推力 {v:.0f} kN") for v in thrust_limits],
        ),
        Panel([Q("pitch_deg", "推力の仰角（制御）", "deg")]),
    ]
    if accel_limit is not None:
        panels.append(
            Panel(
                [Q("accel_G", "加速度", "G")],
                limits=[(accel_limit, f"上限 {accel_limit:.0f} G")],
            )
        )
    panels.extend(extra_panels)
    return Spec(
        states=["R", "theta", "Vr", "Vt", "m"],
        controls=["Tr", "Tt"],
        derive=_ascent_derive(per_stage, tabulated),
        history=panels,
        scene=Scene(
            "ascent",
            Q("downrange", "射点からの地表距離", "km"),
            Q("alt", "高度", "km"),
            body="rocket",
            ground=0.0,
            heading=lambda d: d["pitch"],
            throttle=lambda d: np.clip(d["throttle"], 0, 1),
            decorate=_ascent_decorate,
            readout=[
                Q("alt", "高度", "km"),
                Q("Vt", "$V_t$", "km/s", 1e-3),
                Q("m", "$m$", mass_unit[0], mass_unit[1]),
            ],
        ),
        video_panels=[1, 3, 4],
        phase_names=phases,
        extras=_ascent_extras,
        seconds=10,
        coast_speedup=speedup,
    )


# ---------------------------------------------------------------------------
# 15: low-thrust orbit raising (dimensionless)


def _orbit_derive(d, m):
    d["ur"] = d["ur1"] - d["ur2"]
    d["ut"] = d["ut1"] - d["ut2"]


def _orbit_post(sol: Solution):
    for dense, node in zip(sol.dense, sol.nodes):
        dense["phi"] = cumulative_trapezoid(
            dense["vt"] / dense["r"], dense["time"], initial=0
        )
        node["phi"] = np.interp(node["time"], dense["time"], dense["phi"])
        for d in (dense, node):
            d["X"] = d["r"] * np.cos(d["phi"])
            d["Y"] = d["r"] * np.sin(d["phi"])
            d["heading"] = d["phi"] + np.arctan2(d["ut"], d["ur"])
            d["throttle"] = np.hypot(d["ur"], d["ut"]) / 0.01


def _orbit_decorate(ax, sol):
    a = np.linspace(0, 2 * np.pi, 300)
    for r, text in ((1, "初期軌道 r = 1"), (4, "目標軌道 r = 4")):
        ax.plot(r * np.cos(a), r * np.sin(a), color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.annotate(
            text,
            (r * np.cos(-0.6), r * np.sin(-0.6)),
            fontsize=8,
            color=MUTED,
            xytext=(4, -10),
            textcoords="offset points",
        )
    ax.plot([0], [0], "o", color="#0ea5e9", ms=10)


def _orbit_extras(sol, problem, module):
    return {
        "final_time": problem.time_final(-1),
        "objective_delta_v": float(problem.result.fun),
        "revolutions": float(sol.dense[-1]["phi"][-1] / (2 * np.pi)),
    }


LOW_THRUST = Spec(
    states=["r", "vr", "vt"],
    controls=["ur1", "ur2", "ut1", "ut2"],
    derive=_orbit_derive,
    history=[
        Panel([Q("r", "軌道半径 $r$")]),
        Panel([Q("vt", "接線速度 $v_t$"), Q("vr", "半径方向速度 $v_r$")], title="速度"),
        Panel(
            [Q("ur", "半径方向の推力加速度 $u_r$（制御）")],
            limits=[(0.01, "+0.01"), (-0.01, "−0.01")],
        ),
        Panel(
            [Q("ut", "接線方向の推力加速度 $u_t$（制御）")],
            limits=[(0.01, "+0.01"), (-0.01, "−0.01")],
        ),
    ],
    scene=Scene(
        "orbit",
        Q("X", "$x$"),
        Q("Y", "$y$"),
        body="probe",
        equal=True,
        heading=lambda d: d["heading"],
        throttle=lambda d: d["throttle"],
        decorate=_orbit_decorate,
        readout=[Q("r", "$r$")],
        label="橙の矢印：推力の向き",
    ),
    video_panels=[0, 3],
    time_unit="",
    extras=_orbit_extras,
    seconds=10,
)


SPECS: dict[str, Spec] = {
    "minimum_energy_transfer": MINIMUM_ENERGY,
    "move_stop": MOVE_STOP,
    "brachistochrone": BRACHISTOCHRONE,
    "constrained_descent": CONSTRAINED_DESCENT,
    "brachistochrone_tokyo_osaka": TOKYO_OSAKA,
    "simple_rocket_2d": SIMPLE_ROCKET,
    "goddard_0knot": goddard([]),
    "goddard_1knot": goddard(["区間1", "区間2"]),
    "soft_landing": SOFT_LANDING,
    "rocket_ascent_single_stage": vertical([], 2 * 5000 * 9.80665 / 1e3),
    "rocket_ascent_two_stage": vertical(["1段目", "2段目"], 2 * 5000 * 9.80665 / 1e3),
    "rocket_ascent_polar_ssto": ascent(
        [], [100000 * 9.80665 * 1.5 / 1e3], accel_limit=8
    ),
    "rocket_ascent_polar_tsto": ascent(
        ["1段目", "2段目"],
        [20000 * 9.80665 * 1.5 / 1e3, 1000 * 9.80665 * 1.5 / 1e3],
        accel_limit=8,
        per_stage=True,
        speedup=10.0,
    ),
    "polar_tsto_taiki": ascent(
        ["1段目", "2段目"],
        [33.3 * 9.80665, 4.2 * 9.80665],
        mass_unit=("kg", 1.0),
        extra_panels=[Panel([Q("q_kPa", "動圧 $q$", "kPa")])],
        tabulated=True,
        speedup=6.0,
    ),
    "low_thrust_orbit_transfer": LOW_THRUST,
}
POST = {"low_thrust_orbit_transfer": _orbit_post}
