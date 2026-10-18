"""Shared Matplotlib style for every figure and video on the site.

The colours match the site theme (website/src/styles/custom.css). Phase colours
are a validated categorical palette; the first phase always uses the first slot.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

INK = "#1f2937"
MUTED = "#6b7280"
GRID = "#e5e7eb"
SURFACE = "#ffffff"
REFERENCE = "#111827"  # analytic solutions: black dashed
LIMIT = "#b91c1c"  # constraint limits: red dotted
PHASE_COLORS = ["#2f5bd3", "#e4572e", "#17a398", "#8e5cd9"]
FLAME = "#f59e0b"
DPI = 160

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
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
        "axes.edgecolor": "#9ca3af",
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": DPI,
    }
)


def phase_color(section: int) -> str:
    return PHASE_COLORS[section % len(PHASE_COLORS)]
