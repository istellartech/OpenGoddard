# OpenGoddard

OpenGoddardは、軌道最適化・非線形最適制御問題を解くPythonライブラリです。
Legendre–Gauss–Lobattoの擬スペクトル法で、運動方程式・制約・目的関数を
数値最適化問題に変換し、SciPyで解きます。

[English](../README.md) · [解説サイト](https://istellartech.github.io/OpenGoddard/)

## できること

位置・速度などの状態量と、推力・加速度などの制御量を時間の関数として求めます。
運動方程式や制約はPythonの関数で記述でき、ロケットの上昇・着陸、宇宙機の軌道遷移、
移動時間や消費エネルギーの最小化などに利用できます。

- 状態量・制御量・終端時刻の上下限、境界条件、経路上の制約を設定。
- 終端コストと積分コストによる目的関数を定義。
- 複数区間を接続し、段分離などでモデルが変わる問題を扱う。
- 変数のスケーリング、自動微分（追加インストール）、メッシュ細分化を利用。

## インストール

Python 3.15以上が必要です。NumPy・SciPy・Matplotlibも依存ライブラリとしてインストールされます。

```sh
pip install OpenGoddard
```

## 最初の例題を動かす

例題を実行するには、リポジトリを取得し、[uv](https://docs.astral.sh/uv/)で環境を用意します。

```sh
git clone https://github.com/istellartech/OpenGoddard.git
cd OpenGoddard
uv sync --locked
uv run python examples/01_minimum_energy_transfer.py
```

例題01は、目標位置まで移動するときの所要時間と制御量の二乗積分の和を最小にします。
数値解と解析解が表示されるので、`main()`の`target`を変えて結果を比べてみてください。

[examples/](../examples/)の15例題は、簡単な移動から制約付きの運動、ロケット上昇、
軌道遷移へと学習順に並んでいます。各ファイルに数式・制約・目的関数・初期推定・
求解処理をまとめてあり、後の例題ではグラフも描画します。

| 例題 | 題材・学ぶこと |
| --- | --- |
| 01–02 | 簡単な移動、加速と停止。状態・制御・目的関数の基本 |
| 03–05 | 最速降下。経路上の制約、物理単位、球面上の運動 |
| 06–09 | ロケット上昇と軟着陸。質量変化、推力制限、区間分割 |
| 10–14 | 単段・多段ロケット。段分離、軌道投入、大気・空気抵抗 |
| 15 | 低推力による軌道遷移。積分コストによる最適化 |

## 自分の問題を解くには

例題01を出発点に、次の順で置き換えると全体を追いやすくなります。

1. `Problem`で区間、ノード数、状態量と制御量の数を指定する。
2. 運動方程式、境界条件・経路上の制約、目的関数を関数で記述する。
3. 変数の上下限と初期推定を設定し、必要に応じて単位をスケーリングする。
4. 各関数を問題に登録し、`solve()`で求解する。
5. 戻り値の`success`と`message`を確認し、軌道や制約の満足度を調べる。

既定のソルバーはSciPyのSLSQPです。解は初期推定やノード数の影響を受けるため、
求解の成功に加え、ノード数を増やして結果が安定するかも確認してください。
具体的なコードと各設定の意味は、以下の「はじめに」とAPIリファレンスで説明しています。

## 学ぶ・使う

- [はじめに](https://istellartech.github.io/OpenGoddard/getting-started/)：問題の記述方法と学習の進め方。
- [擬スペクトル法](https://istellartech.github.io/OpenGoddard/concepts/pseudospectral/)：数値解法の仕組み。
- [例題の解説](https://istellartech.github.io/OpenGoddard/examples/01-minimum-energy-transfer/)：例題01から、数式・コード・結果・動画を対応づけて学ぶ。
- [APIリファレンス](https://istellartech.github.io/OpenGoddard/api/OpenGoddard/)：クラス、ソルバーの設定、微分、メッシュ細分化の詳細。

## ライセンス

[MIT](../LICENSE)
