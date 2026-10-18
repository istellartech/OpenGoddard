09. 月面への鉛直軟着陸
==================================================

月面への鉛直軟着陸で、終端高度と終端速度を同時に満たす推力履歴を求めます。

実行方法
--------

リポジトリのルートから実行します。

.. code-block:: console

   uv run python examples/09_soft_landing/main.py

``--plot`` で図を表示し、``--output`` で結果を保存します。

.. code-block:: console

   uv run python examples/09_soft_landing/main.py --plot
   uv run python examples/09_soft_landing/main.py --output examples/09_soft_landing/output/trajectory.json


ファイル構成
------------

* ``main.py``：この例題の実装と実行入口。
* ``README.rst``：この例題の説明と実行方法。
* ``output/``：生成した図・動画・JSONなどの保存先。必要時に作成し、Git管理対象外とします。

問題の説明
----------

高度と速度は上向きを正、地表は高度0です。初期速度は0以下、着地時は高度・速度とも0。
一定重力、空気抵抗なし、姿勢運動なしのモデルです。

.. math::

   \dot{h}=v, \qquad \dot{v}=T/m-g, \qquad
   \dot{m}=-T/v_e.

* 既定の質量一定モデルでは状態を高度・速度の2つにし、総力積 ``integral(T dt)`` を
  最小化します。結果の ``diagnostics.impulse_Ns`` が物理単位での力積です。
* ``variable_mass`` では質量も状態にし、推進剤消費率 ``1-m(tf)/m0`` を最小化します。
  乾燥質量が下限で、推進剤消費量は ``diagnostics.propellant_used_kg`` です。

coastとburnの2区間で初期推定を与えますが、両区間とも推力は0〜最大推力の間で自由です。
区間境界で高度・速度・質量は接続し、推力は不連続に変化できます。初期推定には
弾道飛行と全推力燃焼の解を使い、質量可変モデルの燃焼はロケット方程式で計算します。

**試すこと**：``lower_thrust``、``higher_start``、``variable_mass`` を比較する。
最大推力を下げるといつから減速が必要になるか、質量変化で着陸時刻がどう変わるかを見る。

この教材は最大推力が初期重量より大きく、初期高度が質量一定での停止距離より大きい
範囲を対象とします。質量可変モデルでは推進剤不足も検出します。最小推力、点火回数、
推力の変化率、着地時刻の固定は含みません。開ループの軌道計画であり、着陸の
フィードバック制御器ではありません。

求解器の目的値は、質量一定では ``impulse/(max_thrust*sqrt(initial_altitude/gravity))``、
質量可変では推進剤消費率です。モードをまたぐ比較には、物理単位の力積・質量を使います。

`全例題の一覧と共通の使い方 <../README.rst>`_
