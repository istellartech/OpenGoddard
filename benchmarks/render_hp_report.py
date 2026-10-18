"""Render the Goddard h/hp measurements without recomputing optimization."""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODES = ("previous-p", "p", "h", "hp")
LABELS = ("Previous p", "Equal-budget p", "h", "hp")


def render(source, target):
    data = json.loads(source.read_text())
    records = data["records"]
    expected = 2 * len(MODES) * data["repeats"]
    if len(records) != expected:
        raise ValueError(f"expected {expected} completed records, found {len(records)}")

    def group(ex, mode):
        return [r for r in records if r["example"] == ex and r["mode"] == mode]

    def case(ex, mode):
        return group(ex, mode)[-1]

    def elapsed(ex, mode):
        return statistics.median(r["seconds"] for r in group(ex, mode))

    def path(ex, mode):
        return max(
            max(r["quality"]["path_violation"], r["dense_path_violation"])
            for r in group(ex, mode)
        )

    lines = [
        "Goddard: p・h・hp適応の比較",
        "=" * 56,
        "",
        "目的は、前回のp適応で残ったノード間の経路制約違反を改善し、"
        "計算時間と状態精度を別々に評価すること。",
        "対象は既存例題04（1区間）と05（2区間、最初の終端時刻0.075固定）。"
        "最適化モデルの物理条件は同じ。",
        "",
        "実装",
        "-" * 40,
        "",
        "* 論理区間の内部にLGL要素を導入。h分割後も既存のsection番号・"
        "区間端時刻・スケールを維持する。",
        "* 内部境界では状態のみ連続にし、制御には左右独立の値を持たせる。"
        "要素ごとに補間・積分する。",
        "* hは次数を保って要素分割、hpはLegendre係数の高次成分と残差に"
        "応じて次数増加または分割を選ぶ。",
        "  1反復で最大2要素を変更する。分割点と滑らかさ判定は"
        "ヒューリスティックで、文献の特定アルゴリズムの再現ではない。",
        "* 経路制約を元のノードだけでなくGauss点にも課し、"
        "状態・制御多項式の極値で違反があれば固定した制約点を追加して再最適化する。",
        "  追加点を固定してからADするため、移動する多項式根を微分しない。",
        "* ODE残差は要素座標から論理区間座標へ戻して測定。hを小さくする"
        "だけで残差指標が小さくなることを避ける。",
        "* 本比較はh/hpと経路制約生成を組み合わせた実装の評価であり、"
        "分割だけの寄与の切り分けではない。",
        "",
        "再現条件",
        "-" * 40,
        "",
        f"* 前回pのソース: Git ``{data['baseline_revision']}``。"
        "一時ディレクトリに読み出して独立したパッケージ名で実行。",
        "* 作業ツリーのcheckout/resetは行わない。"
        "全方式は同じ例題の初期推定から開始し、初期ノードは各論理区間12。",
        "* 前回p: max_refinements=5、max_nodes=100、SLSQP ftol=1e-8。"
        "前回の設定を再現。",
        "* p/h/hpの同予算比較: max_refinements=12、"
        "max_nodes=150（各論理区間の合計）、ftol=1e-10。",
        "  h/hpはmax_elements=24、max_order=14、path_check_factor=4。"
        "全方式でmaxiter=600、maxIterator=1、ADを使用。",
        "* 共通メッシュ許容値1e-5。各条件3回、独立プロセス、"
        "BLASスレッド数1、実行順を回転。実行時間は中央値。",
        "* 初期化・import・ADウォームアップ・最後の共通検証は時間外。"
        "適応内部の誤差推定・補間・制約追加・再最適化は時間に含む。",
        f"* 環境: {data['platform']}; Python {data['python'].split()[0]}; "
        "NumPy/Scipy/autogradのバージョンとソースSHA-256はJSON参照。",
        "* 基準コミットのネイティブ評価指標に依存せず、"
        "すべての解を現実装で再検証する。",
        "",
        "計算時間と経路制約",
        "-" * 40,
        "",
        ".. csv-table:: 時間は秒、経路違反は3回の最悪値",
        '   :header: "例題", "方式", "秒・中央値", "最終ノード", '
        '"論理区間内の要素数", "経路違反", "共通検証合格"',
        "",
    ]
    for ex in ("04", "05"):
        for mode in MODES:
            r = case(ex, mode)
            success = sum(v.get("common_converged", False) for v in group(ex, mode))
            lines.append(
                f'   "{ex}", "{mode}", "{elapsed(ex, mode):.4f}", '
                f'"{r["nodes"]}", "{[len(v) for v in r["mesh"]]}", '
                f'"{path(ex, mode):.4g}", "{success}/{len(group(ex, mode))}"'
            )
    lines += [
        "",
        "h/hpの合否はNLPソルバーのsuccessだけではなく、"
        "独立した点のODE残差と経路違反が両方1e-5以下かで決める。",
        "max_nodesは内部境界の左右の重複ノードも含む。論理区間数は04が1、05が2のまま。",
        "",
        "ODE・独立積分による状態精度",
        "-" * 40,
        "",
        "* ODE指標: 独立Gauss点（要素ノード数の4倍+1）で相対的な微分残差を測る。",
        "* 経路違反: 独立LGL点、状態・制御多項式の極値、"
        "および追加のGauss点（16倍+1）で再検証する。",
        "  Goddardの不等式は状態・制御のスカラー上下限なので極値検査が重要。"
        "一般の非線形制約を区間全体で保証するものではない。",
        "* IVP: 得られた制御を区分多項式として用い、"
        "solve_ivpで物理運動方程式を再積分する。",
        "  内部境界ごとに積分区間を分け、"
        "状態をリセットせず前区間の終状態から引き継ぐ。rtol=1e-10、atol=1e-12。",
        "* IVP指標: 再積分状態と最適化状態多項式の差を、状態の基準単位+"
        "状態の大きさで割った最大値。",
        "  これは数値積分との比較であり、厳密な誤差上界ではない。"
        "異なる物理問題同士の順位付けには使わない。",
        "",
        ".. csv-table:: 各方式の最後の試行（各試行の詳細はJSON）",
        '   :header: "例題", "方式", "目的値", "ODE残差", '
        '"IVP相対状態誤差", "NLP等式残差"',
        "",
    ]
    for ex in ("04", "05"):
        for mode in MODES:
            r = case(ex, mode)
            lines.append(
                f'   "{ex}", "{mode}", "{r["objective"]:.12g}", '
                f'"{r["quality"]["max_error"]:.4g}", '
                f'"{r["ivp"]["max_relative_state_error"]:.4g}", '
                f'"{r["equality_violation"]:.4g}"'
            )
    lines += [
        "",
        "状態誤差と経路制約違反は別の量。pのIVP誤差が小さくても、推力が"
        "上下限を超えていれば実行可能な軌道とは言えない。",
        "次数を上げたりNLP許容値を厳しくしたりしても、"
        "非凸問題の局所解や補間の振動が変わるため、",
        "経路制約違反の単調な改善は保証されない。"
        "制約違反を伴う、より良い目的値を性能向上とは扱わない。",
        "",
        "使い分け",
        "-" * 40,
        "",
        "* 今回のGoddardでは、まずhを使い、経路制約を満たす軌道を得るのが実用的。",
        "* hpがhより速いとは限らない。特に例題04では、より良い目的値・"
        "状態精度のための追加計算が大きい。",
        "* 前回pは短時間だが経路制約未達。同予算pはさらに時間を掛けても"
        "経路制約未達となった。",
        "* 上限到達・粗いNLPの失敗はmesh_historyとmesh_stop_reasonに残り、"
        "収束成功には数えない。",
        "* 既存の論理区間を自動的に増やしてコールバックのsection番号を"
        "変えることはしない。",
        "",
        "実行",
        "-" * 40,
        "",
        "::",
        "",
        "   uv sync --locked --extra autodiff",
        "   uv run --extra autodiff python benchmarks/compare_hp.py",
        "   uv run python benchmarks/render_hp_report.py",
        "",
        "hp_results.jsonには個別時間、全履歴、警告、NLP状態、独立検証、"
        "IVP差、環境、ソースハッシュがある。",
        "hp_results.txtは全条件の一覧。"
        "タイムアウト・実行失敗はランナーが記録し、成功例だけを選別しない。",
        "",
        "参考",
        "-" * 40,
        "",
        "* Patterson, Hager, Rao, A ph mesh refinement method for optimal control:",
        "  https://doi.org/10.1002/oca.2114 "
        "（次数増加と区間分割を組み合わせる背景。ここでのLGL実装とは異なる。）",
        "* SciPy BarycentricInterpolator: "
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate."
        "BarycentricInterpolator.html",
        "",
        ".. image:: hp_comparison.png",
        "   :alt: Solve time, path violation, and independently integrated state error",
        "",
    ]
    target.write_text("\n".join(lines))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    colors = ["#94a3b8", "#2563eb", "#059669", "#ea580c"]
    x = np.arange(2)
    for k, mode in enumerate(MODES):
        positions = x + (k - 1.5) * 0.2
        axes[0].bar(
            positions,
            [elapsed(ex, mode) for ex in ("04", "05")],
            width=0.19,
            color=colors[k],
            label=LABELS[k],
        )
        axes[1].bar(
            positions,
            [path(ex, mode) for ex in ("04", "05")],
            width=0.19,
            color=colors[k],
        )
        axes[2].bar(
            positions,
            [
                max(r["ivp"]["max_relative_state_error"] for r in group(ex, mode))
                for ex in ("04", "05")
            ],
            width=0.19,
            color=colors[k],
        )
    axes[0].set_ylabel("Solve time, median of 3 runs [s]")
    axes[0].legend(fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Path violation, worst of 3 runs")
    axes[1].axhline(1e-5, color="black", ls="--", lw=1)
    axes[1].text(0.55, 1.2e-5, "Tolerance 1e-5", fontsize=8)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Independent IVP relative state error")
    for ax, title in zip(
        axes,
        (
            "Runtime",
            "Between-node path limits",
            "ODE/state accuracy is a separate metric",
        ),
        strict=True,
    ):
        ax.set_xticks(x, ["04: one phase", "05: two phases"])
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.savefig(target.with_name("hp_comparison.png"), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--input", type=Path, default=root / "hp_results.json")
    parser.add_argument("--output", type=Path, default=root / "HP_REPORT.rst")
    args = parser.parse_args()
    render(args.input, args.output)


if __name__ == "__main__":
    main()
