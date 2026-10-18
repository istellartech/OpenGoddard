最適制御のサンプルと学習順序
========================================

15例すべてを **1例題＝1つの番号付きフォルダ** に統一しています。
各フォルダの ``main.py`` が実装と実行入口、``README.rst`` が説明です。
番号順に、簡単な移動から経路制約、ロケット上昇・着陸・軌道遷移へ進みます。

各例題の数式とコードの対応、計算結果の動画は
`ドキュメントサイト <https://istellartech.github.io/OpenGoddard/>`_ で見られます。
動画と図は ``uv run python website/scripts/build_media.py`` で生成します。

フォルダ構成
------------

.. code-block:: text

   examples/
     README.rst
     catalog.py                    # 学習順序・安定したID・importの入口
     _shared.py                    # 複数の例で使う結果出力・CLIの補助処理
     01_minimum_energy_transfer/
       README.rst
       main.py
     02_move_stop/
       README.rst
       main.py
       output/                     # 必要時に生成・Git対象外
     ...
     14_polar_tsto_taiki/
       README.rst
       main.py
       data/                       # この例が読む入力CSV
     ...
     legacy/                       # 学習順序とは別に保存した旧デモ
       README.rst
       main.py

全例題で同じ規則を使い、必要なものだけサブフォルダを置きます。

* ``data/``：求解に使用する入力ファイル。
* ``figures/``：以前から同梱されている参考図。実行時には上書きしません。
* ``output/``：生成した図・動画・JSON。各例題の中に保存し、Git管理対象外です。

別の場所に実装を置く番号付きラッパーや、例題を横断して配置する ``learning/``・
``assets/`` は使いません。共有が必要な補助処理のみ ``_shared.py`` に置きます。

読む順序
--------

* `01_minimum_energy_transfer：最小操作量の移動 <01_minimum_energy_transfer/README.rst>`_
* `02_move_stop：加速して停止する移動 <02_move_stop/README.rst>`_
* `03_brachistochrone：最速降下 <03_brachistochrone/README.rst>`_
* `04_constrained_descent：深さ制限付き最速降下 <04_constrained_descent/README.rst>`_
* `05_brachistochrone_tokyo_osaka：東京・大阪の最速降下 <05_brachistochrone_tokyo_osaka/README.rst>`_
* `06_simple_rocket_2d：単純な2次元ロケット <06_simple_rocket_2d/README.rst>`_
* `07_goddard_0knot：Goddard問題・単一区間 <07_goddard_0knot/README.rst>`_
* `08_goddard_1knot：Goddard問題・区間分割 <08_goddard_1knot/README.rst>`_
* `09_soft_landing：月面への鉛直軟着陸 <09_soft_landing/README.rst>`_
* `10_rocket_ascent_single_stage：単段ロケット上昇 <10_rocket_ascent_single_stage/README.rst>`_
* `11_rocket_ascent_two_stage：2段ロケット上昇 <11_rocket_ascent_two_stage/README.rst>`_
* `12_rocket_ascent_polar_ssto：極座標・単段軌道投入 <12_rocket_ascent_polar_ssto/README.rst>`_
* `13_rocket_ascent_polar_tsto：極座標・2段軌道投入 <13_rocket_ascent_polar_tsto/README.rst>`_
* `14_polar_tsto_taiki：大樹からの2段軌道投入 <14_polar_tsto_taiki/README.rst>`_
* `15_low_thrust_orbit_transfer：低推力軌道遷移 <15_low_thrust_orbit_transfer/README.rst>`_

``catalog.py`` の安定したID（例：move_stop）はimportや将来の公開データに使います。
番号は学習順序です。番号を変更しても、呼び出し側ではIDを維持できます。

従来の番号との対応は、旧01→03、旧02→05、旧03→06、旧04→07、旧05→08、
旧06→10、旧07→11、旧08→12、旧09→13、旧10→15、旧11→14、旧12→01です。
02・04・09が追加した教材です。

実行方法
--------

リポジトリのルートから、各フォルダのmain.pyを実行します。

.. code-block:: console

   uv run python examples/01_minimum_energy_transfer/main.py
   uv run python examples/02_move_stop/main.py --preset speed_limit --plot
   uv run python examples/04_constrained_descent/main.py --preset no_floor --plot
   uv run python examples/09_soft_landing/main.py --preset variable_mass --plot

02・04・09は既定では画面もファイルも開かず、求解結果と誤差指標を表示します。
``--plot`` で図を表示し、``--output`` でJSONを保存します。
他の例の表示・保存方法は各フォルダのREADME.rstを参照してください。

``--set NAME=JSON`` は何度でも指定できます。数値・真偽値・null・文字列はJSON形式です。
文字列には、シェルの引用符とは別にJSONの二重引用符が必要です。

.. code-block:: console

   uv run python examples/02_move_stop/main.py --set distance=20 --set max_acceleration=1.5
   uv run python examples/02_move_stop/main.py --set 'objective="minimum_effort"' --set duration=8
   uv run python examples/09_soft_landing/main.py --set variable_mass=true --set initial_altitude=200
   uv run python examples/04_constrained_descent/main.py --set max_depth=0.23 --output examples/04_constrained_descent/output/shallow.json

``--nodes`` は区間ごとのLGLノード数（既定12、最小4）、``--samples`` は区間ごとの
再生用サンプル数（既定201）です。後者を増やしても最適化の精度は上がりません。
``--maxiter`` はSLSQPの反復上限です。これらの共通CLIは02・04・09に対応します。

初期公開のアニメーションは、AGENTS.mdに従いMatplotlibでMP4と静止画を生成して
静的な解説ページに埋め込む方針です。動画生成そのものはまだ実装していません。
パラメータ、求解、結果データ、描画の分離は動画生成にも再利用します。

Python・GUIからの利用
--------------------

.. code-block:: python

   from dataclasses import replace
   from examples.catalog import load_example

   move_stop = load_example("move_stop")

   parameters = replace(move_stop.PRESETS["default"], distance=20.0)
   trajectory = move_stop.solve(parameters, nodes=12, samples=201)
   data = trajectory.to_dict()
   trajectory.write_json("/tmp/move-20m.json")
   figure = move_stop.plot(trajectory)  # 描画は明示的に呼んだときだけ

全例題は ``main.py`` を実装・実行入口とし、``load_example(ID)`` で求解せずにimportできます。
02・04・09は以下の共通APIも持ちます。他の12例は、同じフォルダ構成に収めた従来の求解コードです。

* ``Parameters``：単位をコメントに記した変更不可のdataclass。新しい条件は
  ``dataclasses.replace`` で作れます。非有限値や対象外の条件は ``ValueError`` になります。
* ``PRESETS``：比較用の条件。GUIの選択肢やバッチ出力に利用できます。
* ``make_problem(parameters, nodes=12)``：未求解の独立した ``Problem`` を返します。
  初期推定を変更して、求解過程を調べたい場合にも使えます。
* ``solve(parameters, ...)``：画面・ファイル操作をせず ``Trajectory`` を返します。
  求解失敗は ``RuntimeError`` です。条件ごとに別の ``Problem`` を作ります。
* ``plot(trajectory)``：図を返します。``show`` や保存は呼び出し側で選べます。

JSONは ``schema_version=1`` で、``parameters``、``units``、``coordinates``、``solver``、
``diagnostics``、``reference``、``phases`` を持ちます。各区間には名前と、
``nodes``（最適化ノード）、``samples``（等間隔時刻での再生用データ）があります。
各データは ``time`` と物理量の同じ長さの配列です。配列はPythonのリストに変換済みです。

区間末尾と次の区間先頭には同じ時刻が現れます。状態は連続ですが制御は跳ぶことが
あるので、GUIは区間をまたいで制御を平滑化しないでください。同じ絶対時刻で比較するか、
各軌道の進行率で比較するかは表示側で選べます。``samples`` は区間ごとの多項式を
補間した値で、上下限で切り詰めてはいません。条件変更後の再求解はPython側で行います。

数値解の確認
------------

初期推定と区間構成には既知の解析的構造を使っています。そのため、既定値では
最初から解に非常に近く、最適化が1反復で終わることがあります。参照解をそのまま
結果として返す実装ではなく、通常の ``Problem.solve`` を通しています。

``solver.success`` は離散化したNLPの求解状態です。これとは別に、次を出力します。

* ``max_equality_residual``：無次元化した境界・接続・離散化制約の最大残差。
* ``max_error``、``section_errors``：ノード間のODE残差指標。
* ``path_violation``：より密なノードで検査した制御・状態の上下限と経路制約の違反量。
* ``sampled_checks_passed``：上記3種の最大値が ``check_tolerance``（1e-4）以下か。

これらはサンプリングによる検査で、連続時間での厳密な実行可能性や大域最適性の証明では
ありません。GUIも ``solver.success`` だけでなく ``sampled_checks_passed`` を確認する想定です。
条件を変えた場合はノード数を増やした解とも比較してください。制御の切り替えがある例を
単一区間の高次多項式だけで表現すると、ノード間で上下限を超えやすくなります。
