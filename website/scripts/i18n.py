"""Japanese and English labels for pre-rendered documentation media."""

import re
from contextlib import contextmanager
from contextvars import ContextVar

_LANGUAGE = ContextVar("media_language", default="ja")
LANGUAGES = ("ja", "en")

# Full phrases take precedence over reusable quantity names. Math and units
# remain unchanged; unknown Japanese labels fail the English media build.
_LABELS = {
    "シューティング：初期値を推定して積分し、外れたら直す": "Shooting: adjust the initial guess",
    "コロケーション：全ノードの値を同時に動かして整合させる": "Collocation: adjust all nodes together",
    "初期推定：傾きが運動方程式と合わない": "Initial guess: slopes disagree",
    "解：すべてのノードで傾きが一致": "Solution: slopes match at every node",
    "床がない場合の最速降下曲線": "Fastest track without a floor",
    "橙の矢印：加速度の向き（ノードごと）": "Orange arrows: acceleration at the nodes",
    "橙の矢印：推力の向き": "Orange arrows: thrust direction",
    "高度方向を拡大して表示": "Altitude axis is exaggerated",
    "最適な曲線（サイクロイド）": "Optimal curve (cycloid)",
    "A（静止から出発）": "A (released from rest)",
    "A から B までの所要時間": "Travel time from A to B",
    "等間隔の点（比較）": "Equally spaced (comparison)",
    "等間隔の点で補間": "Equally spaced interpolation",
    "LGLノードで補間": "LGL interpolation",
    "補間の最大誤差（対数目盛り）": "Maximum interpolation error (log scale)",
    "ノード値を通る多項式": "Interpolating polynomial",
    "初期推定のノード": "Initial nodes",
    "収束したノード": "Converged nodes",
    "多項式の傾き": "Polynomial slope",
    "例題03（最速降下）：解析解": "Example 03: error relative to",
    "との差": "",
    "例題07：1区間（多項式1本）": "Example 07: one phase",
    "例題08：2区間に分割": "Example 08: two phases",
    "点の多項式で": "points:",
    "を補間": " interpolation",
    "燃料切れの質量": "Dry mass",
    "サイクロイド（解析解）": "Cycloid (analytic)",
    "直線の斜面": "Straight track",
    "床（深さ 0.3 m）": "Floor (depth 0.3 m)",
    "区間1（加速）": "Phase 1 (accelerate)",
    "区間2（減速）": "Phase 2 (brake)",
    "水平位置": "Horizontal position",
    "水平速度": "Horizontal velocity",
    "鉛直速度": "Vertical velocity",
    "接線速度": "Tangential velocity",
    "半径方向速度": "Radial velocity",
    "半径方向の推力加速度": "Radial thrust\nacceleration",
    "接線方向の推力加速度": "Tangential thrust\nacceleration",
    "推力の大きさ": "Thrust magnitude",
    "推力の仰角": "Thrust elevation",
    "加速度の向き": "Acceleration angle",
    "東京からの距離": "Distance from Tokyo",
    "射点からの地表距離": "Downrange distance",
    "軌道半径": "Orbital radius",
    "所要時間の誤差": "Duration error",
    "正規化時間": "Normalized time",
    "最大推力": "Maximum thrust",
    "初期軌道": "Initial orbit",
    "目標軌道": "Target orbit",
    "LGLノード": "LGL nodes",
    "等間隔の点": "Equally spaced points",
    "ノード数": "Node count",
    "ノード値": "Node values",
    "点の数": "Point count",
    "最適化変数": "optimization variables",
    "元の関数": "Original function",
    "運動方程式": "Dynamics",
    "解析解": "Analytic solution",
    "ゆるい曲線": "Shallow curve",
    "深すぎる曲線": "Overly deep curve",
    "直線": "Straight line",
    "位置": "Position",
    "速度": "Velocity",
    "速さ": "Speed",
    "深さ": "Depth",
    "高さ": "Height",
    "高度": "Altitude",
    "角度": "Angle",
    "加速度": "Acceleration",
    "質量": "Mass",
    "推力": "Thrust",
    "動圧": "Dynamic pressure",
    "傾き": "Slope",
    "状態": "State",
    "時刻": "Time",
    "上限": "Upper bound",
    "下限": "Lower bound",
    "区間1": "Phase 1",
    "区間2": "Phase 2",
    "1段目": "Stage 1",
    "2段目": "Stage 2",
    "始点": "Start",
    "終点": "Finish",
    "東京": "Tokyo",
    "大阪": "Osaka",
    "（制御）": " (control)",
    "（無次元）": " (dimensionless)",
    "（上向き正）": " (positive upward)",
    "（": " (",
    "）": ")",
    "：": ": ",
}


@contextmanager
def language(locale):
    """Select labels in this rendering context without changing the solution."""
    if locale not in LANGUAGES:
        raise ValueError(f"unsupported media language: {locale}")
    token = _LANGUAGE.set(locale)
    try:
        yield
    finally:
        _LANGUAGE.reset(token)


def text(label):
    """Translate a figure label, preserving mathematical symbols and numbers."""
    if _LANGUAGE.get() == "ja" or label is None:
        return label
    translated = label
    for source in sorted(_LABELS, key=len, reverse=True):
        translated = translated.replace(source, _LABELS[source])
    if re.search(r"[\u3040-\u30ff\u3400-\u9fff]", translated):
        raise ValueError(f"missing English media label: {label}")
    return translated
