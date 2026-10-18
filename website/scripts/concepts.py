"""Figures for the theory pages: what optimal control asks, how direct
methods turn it into an optimization problem, and why LGL nodes work.

Everything is computed, not drawn by hand: interpolation with the library's
own LGL rule, real solves of examples 03, 07 and 08, and quadrature of travel
times along candidate curves.
"""

import contextlib
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter
from scipy.integrate import quad
from scipy.interpolate import BarycentricInterpolator
from style import INK, LIMIT, MUTED, PHASE_COLORS

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
OUT = ROOT / "website" / "public" / "media" / "concepts"
BLUE, ORANGE, GREEN, VIOLET = PHASE_COLORS


def _lgl(n):
    from OpenGoddard import Problem

    return Problem([0.0, 1.0], [n], [1], [0]).method_LGL(n)  # nodes, weights, D


def _save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


# ---------------------------------------------------------------------------
# What is optimal control: many feasible curves, one fastest


def candidate_curves():
    g, xf = 1.0, 1.0
    yf = 2 / np.pi  # the end of the optimal cycloid (example 03)

    def travel_time(shape, dshape):
        f = lambda x: np.sqrt(1 + dshape(x) ** 2) / np.sqrt(2 * g * shape(x))  # noqa: E731
        return quad(f, 0, xf, limit=200)[0]

    fig, ax = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
    curves = [
        ("直線", lambda x: yf * x, lambda x: yf + 0 * x, MUTED),
        (
            "ゆるい曲線",
            lambda x: yf * (1 - (1 - x) ** 1.5),
            lambda x: 1.5 * yf * (1 - x) ** 0.5,
            VIOLET,
        ),
        (
            "深すぎる曲線",
            lambda x: yf * (1 - (1 - x) ** 6) + 0.35 * np.sin(np.pi * x),
            lambda x: 6 * yf * (1 - x) ** 5 + 0.35 * np.pi * np.cos(np.pi * x),
            GREEN,
        ),
    ]
    xs = np.linspace(0, 1, 300)
    for name, shape, dshape, color in curves:
        t = travel_time(shape, dshape)
        ax.plot(xs, shape(xs), color=color, lw=2, label=f"{name}：{t:.3f} s")
    radius = 1 / np.pi
    phi = np.linspace(0, np.pi, 300)
    ax.plot(
        radius * (phi - np.sin(phi)),
        radius * (1 - np.cos(phi)),
        color=BLUE,
        lw=3,
        label=f"最適な曲線（サイクロイド）：{np.sqrt(np.pi):.3f} s",
    )
    ax.plot([0], [0], "o", color=INK, ms=7)
    ax.plot([xf], [yf], "s", color=INK, ms=7)
    ax.annotate(
        "A（静止から出発）",
        (0, 0),
        xytext=(8, -2),
        textcoords="offset points",
        fontsize=9,
    )
    ax.annotate("B", (xf, yf), xytext=(-14, 6), textcoords="offset points", fontsize=9)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set(xlabel="水平位置 $x$ [m]", ylabel="深さ $y$ [m]")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        title="A から B までの所要時間",
        title_fontsize=9,
    )
    _save(fig, "candidate_curves")


# ---------------------------------------------------------------------------
# Indirect vs direct: shooting misses, collocation converges as a whole


def shooting_vs_collocation():
    g, target = 9.8, 10.0
    fig, axes = plt.subplots(
        1, 2, figsize=(7.6, 3.0), layout="constrained", sharey=True
    )
    ax = axes[0]
    for angle, color, alpha in ((20, MUTED, 0.6), (60, MUTED, 0.6), (45, BLUE, 1.0)):
        speed = np.sqrt(target * g / np.sin(2 * np.deg2rad(45)))
        a = np.deg2rad(angle)
        tf = 2 * speed * np.sin(a) / g
        t = np.linspace(0, tf, 100)
        ax.plot(
            speed * np.cos(a) * t,
            speed * np.sin(a) * t - g * t**2 / 2,
            color=color,
            alpha=alpha,
            lw=2 if color == BLUE else 1.4,
        )
        x_end = speed * np.cos(a) * tf
        ax.annotate(
            f"{angle}°",
            (x_end, 0),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=INK if color == BLUE else MUTED,
        )
    ax.plot([target], [0], "*", color=LIMIT, ms=13, zorder=5)
    ax.set_title("シューティング：初期値を推定して積分し、外れたら直す", fontsize=9.5)
    ax.set(xlabel="$x$", ylabel="$y$")
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-0.4, 4.4)
    ax = axes[1]
    speed = np.sqrt(target * g)
    a = np.deg2rad(45)
    tf = 2 * speed * np.sin(a) / g
    t = np.linspace(0, tf, 100)
    ax.plot(
        speed * np.cos(a) * t, speed * np.sin(a) * t - g * t**2 / 2, color=BLUE, lw=2
    )
    tau = _lgl(8)[0]
    tn = (tau + 1) / 2 * tf
    xn = speed * np.cos(a) * tn
    yn = speed * np.sin(a) * tn - g * tn**2 / 2
    ax.plot(
        np.linspace(0, target, xn.size),
        0.15 + 1.2 * np.sin(np.linspace(0, np.pi, xn.size)),
        "o",
        ms=5,
        mfc="white",
        mec=MUTED,
        mew=1.2,
        label="初期推定のノード",
    )
    ax.plot(xn, yn, "o", ms=5, mfc="white", mec=BLUE, mew=1.4, label="収束したノード")
    ax.plot([target], [0], "*", color=LIMIT, ms=13, zorder=5)
    ax.set_title("コロケーション：全ノードの値を同時に動かして整合させる", fontsize=9.5)
    ax.set(xlabel="$x$")
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, "shooting_vs_collocation")


# ---------------------------------------------------------------------------
# LGL nodes, Runge's phenomenon and spectral convergence of interpolation


def runge(x):
    return 1 / (1 + 25 * x**2)


def lgl_nodes():
    fig, ax = plt.subplots(figsize=(7.2, 2.2), layout="constrained")
    for row, n in enumerate((5, 9, 17)):
        tau = _lgl(n)[0]
        ax.plot(tau, np.full(n, row), "o", ms=6, mfc=BLUE, mec="white", mew=0.8)
        ax.plot(np.linspace(-1, 1, n), np.full(n, row - 0.3), "|", ms=8, color=MUTED)
        ax.text(-1.08, row, f"N = {n}", ha="right", va="center", fontsize=9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    ax.set_xlim(-1.35, 1.05)
    ax.set_xlabel("正規化時間 $\\tau$")
    ax.plot([], [], "o", color=BLUE, label="LGLノード")
    ax.plot([], [], "|", color=MUTED, ms=8, label="等間隔の点（比較）")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncols=2)
    _save(fig, "lgl_nodes")


def interpolation():
    fine = np.linspace(-1, 1, 600)
    counts = list(range(4, 31, 2))
    errors = {"lgl": [], "uniform": []}
    for n in counts:
        tau = _lgl(n)[0]
        uni = np.linspace(-1, 1, n)
        errors["lgl"].append(
            np.max(np.abs(BarycentricInterpolator(tau, runge(tau))(fine) - runge(fine)))
        )
        errors["uniform"].append(
            np.max(np.abs(BarycentricInterpolator(uni, runge(uni))(fine) - runge(fine)))
        )

    def draw(axes, n, shown):
        left, right = axes
        for ax in axes:
            ax.clear()
        tau = _lgl(n)[0]
        uni = np.linspace(-1, 1, n)
        left.plot(fine, runge(fine), color=INK, lw=2.4, label="元の関数")
        left.plot(
            fine,
            BarycentricInterpolator(uni, runge(uni))(fine),
            color=ORANGE,
            lw=1.5,
            label="等間隔の点で補間",
        )
        left.plot(
            fine,
            BarycentricInterpolator(tau, runge(tau))(fine),
            color=BLUE,
            lw=1.8,
            label="LGLノードで補間",
        )
        left.plot(uni, runge(uni), "x", color=ORANGE, ms=6, mew=1.5)
        left.plot(tau, runge(tau), "o", color=BLUE, ms=4.5)
        left.set_ylim(-0.6, 1.6)
        left.set_xlabel("$\\tau$")
        left.set_title(f"N = {n} 点の多項式で $1/(1+25\\tau^2)$ を補間", fontsize=10)
        left.legend(loc="upper right", fontsize=8)
        k = counts.index(shown) + 1
        right.semilogy(
            counts[:k], errors["lgl"][:k], "o-", color=BLUE, label="LGLノード"
        )
        right.semilogy(
            counts[:k], errors["uniform"][:k], "x-", color=ORANGE, label="等間隔の点"
        )
        right.set_xlim(2, 32)
        right.set_ylim(1e-4, 1e4)
        right.set_xlabel("点の数 N")
        right.set_title("補間の最大誤差（対数目盛り）", fontsize=10)
        right.legend(loc="upper left", fontsize=8)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8, 3.2),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.5, 1]},
    )
    draw(axes, 14, counts[-1])
    _save(fig, "interpolation")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8, 4.5),
        dpi=160,
        layout="constrained",
        gridspec_kw={"width_ratios": [1.5, 1]},
    )
    writer = FFMpegWriter(
        fps=30,
        codec="libx264",
        bitrate=-1,
        extra_args=["-pix_fmt", "yuv420p", "-crf", "24", "-movflags", "+faststart"],
    )
    with writer.saving(fig, str(OUT / "interpolation.mp4"), dpi=160):
        for n in counts:
            draw(axes, n, n)
            for _ in range(24):
                writer.grab_frame()
        for _ in range(45):
            writer.grab_frame()
    fig.savefig(OUT / "interpolation_poster.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Differentiation matrix and collocation defects


def differentiation():
    n = 6
    tau, _, d = _lgl(n)
    f = lambda x: np.sin(2 * x) + 0.4 * x  # noqa: E731
    values = f(tau)
    slopes = d @ values
    fine = np.linspace(-1, 1, 300)
    fig, ax = plt.subplots(figsize=(7.2, 3.0), layout="constrained")
    ax.plot(
        fine,
        BarycentricInterpolator(tau, values)(fine),
        color=BLUE,
        lw=2,
        label="ノード値を通る多項式 $x(\\tau)$",
    )
    ax.plot(
        tau,
        values,
        "o",
        ms=7,
        mfc="white",
        mec=BLUE,
        mew=1.6,
        label="ノード値 $X_k$（最適化変数）",
    )
    for t, v, s in zip(tau, values, slopes):
        h = 0.13
        ax.plot(
            [t - h, t + h],
            [v - s * h, v + s * h],
            color=ORANGE,
            lw=2.4,
            solid_capstyle="round",
        )
    ax.plot([], [], color=ORANGE, lw=2.4, label="傾き $\\sum_k D_{ik} X_k$")
    ax.set_xlabel("正規化時間 $\\tau$")
    ax.legend(loc="lower right", fontsize=8.5)
    _save(fig, "differentiation")


def collocation():
    """x' = -x + u with u = 0, x(0) = 1 on t in [0, 2]: guess vs converged nodes."""
    n, tf = 7, 2.0
    tau, _, d = _lgl(n)
    t = (tau + 1) / 2 * tf
    fine = np.linspace(-1, 1, 300)
    tfine = (fine + 1) / 2 * tf
    guess = 1 - 0.4 * t
    solution = np.exp(-t)
    fig, axes = plt.subplots(
        1, 2, figsize=(7.6, 3.0), layout="constrained", sharey=True
    )
    for ax, values, title in (
        (axes[0], guess, "初期推定：傾きが運動方程式と合わない"),
        (axes[1], solution, "解：すべてのノードで傾きが一致"),
    ):
        ax.plot(tfine, BarycentricInterpolator(tau, values)(fine), color=BLUE, lw=2)
        ax.plot(t, values, "o", ms=6, mfc="white", mec=BLUE, mew=1.5)
        poly = (d @ values) * 2 / tf
        rhs = -values
        for ti, v, s1, s2 in zip(t, values, poly, rhs):
            h = 0.11
            ax.plot(
                [ti - h, ti + h], [v - s1 * h, v + s1 * h], color=BLUE, lw=3, alpha=0.35
            )
            ax.plot([ti - h, ti + h], [v - s2 * h, v + s2 * h], color=ORANGE, lw=1.6)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("時刻 $t$")
    axes[0].plot([], [], color=BLUE, lw=3, alpha=0.35, label="多項式の傾き $DX$")
    axes[0].plot([], [], color=ORANGE, lw=1.6, label="運動方程式 $f(X)=-X$")
    axes[0].legend(loc="lower left", fontsize=8)
    axes[0].set_ylabel("状態 $x$")
    _save(fig, "collocation")


# ---------------------------------------------------------------------------
# Real problems: convergence in N and the benefit of phases (knots)


def _load(identifier):
    sys.path.insert(0, str(ROOT))
    from examples.catalog import load_example

    return load_example(identifier)


def convergence():
    from OpenGoddard import Guess, Problem

    module = _load("brachistochrone")
    p = module.PARAMETERS
    counts, errors = [], []
    problem = Problem([0.0, 2.0], [4], [3], [1], maxIterator=3)
    t = problem.time_all_section
    problem.set_states_all_section(0, Guess.linear(t, 0.0, p["l"]))
    problem.set_states_all_section(1, Guess.linear(t, 0.0, p["l"] / np.sqrt(3)))
    problem.set_controls_all_section(0, Guess.linear(t, np.deg2rad(30), np.deg2rad(30)))
    problem.dynamics = [module.dynamics]
    problem.knot_states_smooth = []
    problem.cost = module.cost
    problem.equality = module.equality
    problem.inequality = module.inequality
    for n in range(4, 19, 2):
        if n > 4:
            problem = problem.refine([n])  # warm start from the previous solution
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            result = problem.solve(p, ftol=1e-13, maxiter=500)
        if result.success:
            counts.append(n)
            errors.append(max(abs(problem.time_final(-1) - np.sqrt(np.pi)), 1e-16))
    fig, ax = plt.subplots(figsize=(7.2, 2.8), layout="constrained")
    ax.semilogy(counts, errors, "o-", color=BLUE)
    ax.set_xlabel("ノード数 N")
    ax.set_ylabel("所要時間の誤差")
    ax.set_title("例題03（最速降下）：解析解 $\\sqrt{\\pi}$ との差", fontsize=10)
    _save(fig, "convergence")
    return dict(zip(counts, errors))


def knots():
    results = {}
    for identifier in ("goddard_0knot", "goddard_1knot"):
        module = _load(identifier)
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            results[identifier] = module.solve()
    fig, axes = plt.subplots(
        1, 2, figsize=(7.6, 2.9), layout="constrained", sharey=True
    )
    for ax, (identifier, title) in zip(
        axes,
        (
            ("goddard_0knot", "例題07：1区間（多項式1本）"),
            ("goddard_1knot", "例題08：2区間に分割"),
        ),
    ):
        problem = results[identifier]
        for section in range(problem.number_of_section):
            tau = problem.tau[section]
            grid = np.linspace(-1, 1, 300)
            t0, t1 = problem.time_start(section), problem.time_final(section)
            values = problem.controls(0, section)
            color = PHASE_COLORS[section]
            ax.plot(
                t0 + (grid + 1) / 2 * (t1 - t0),
                BarycentricInterpolator(tau, values)(grid),
                color=color,
                lw=1.8,
            )
            ax.plot(problem.time[section], values, "o", ms=3, mfc="white", mec=color)
        if problem.number_of_section > 1:
            ax.axvline(problem.time_final(0), color=MUTED, ls=(0, (4, 3)), lw=1)
        ax.axhline(3.5, color=LIMIT, ls=(0, (2, 2)), lw=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("時刻 $t$（無次元）")
    axes[0].set_ylabel("推力 $T$")
    axes[0].set_ylim(-0.6, 4.2)
    _save(fig, "knots")


def build_concepts(name="concepts", force=False) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for path in sorted(SCRIPTS.glob("*.py")) + sorted(
        (ROOT / "src" / "OpenGoddard").glob("*.py")
    ):
        digest.update(path.read_bytes())
    for identifier in ("03_brachistochrone", "07_goddard_0knot", "08_goddard_1knot"):
        digest.update((ROOT / "examples" / f"{identifier}.py").read_bytes())
    stamp = OUT / "hash.txt"
    if not force and stamp.exists() and stamp.read_text() == digest.hexdigest():
        return "concepts: up to date"
    candidate_curves()
    shooting_vs_collocation()
    lgl_nodes()
    interpolation()
    differentiation()
    collocation()
    convergence()
    knots()
    stamp.write_text(digest.hexdigest())
    return "concepts: built"


if __name__ == "__main__":
    print(build_concepts(force=True))
