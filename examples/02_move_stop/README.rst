02. 加速して停止する移動
==================================================

位置と速度を扱い、終端で停止する条件、最短時間と最小二乗操作量を比較します。

実行方法
--------

リポジトリのルートから実行します。

.. code-block:: console

   uv run python examples/02_move_stop/main.py

``--plot`` で図を表示し、``--output`` で結果を保存します。

.. code-block:: console

   uv run python examples/02_move_stop/main.py --plot
   uv run python examples/02_move_stop/main.py --output examples/02_move_stop/output/trajectory.json


ファイル構成
------------

* ``main.py``：この例題の実装と実行入口。
* ``README.rst``：この例題の説明と実行方法。
* ``output/``：生成した図・動画・JSONなどの保存先。必要時に作成し、Git管理対象外とします。

問題の説明
----------

状態は位置 ``position`` と速度 ``velocity``、制御は加速度 ``acceleration`` です。

.. math::

   \dot{x}=v, \qquad \dot{v}=u, \qquad
   x(0)=v(0)=v(t_f)=0, \quad x(t_f)=d.

前進のみ、加速度は対称な上下限付き、速度上限は任意です。

* ``minimum_time``：終端時間を自由にして移動時間を最小化します。
  速度上限がなければ、理想解は加速・減速の2区間で、時間は
  ``2*sqrt(distance/max_acceleration)`` です。速度上限が十分低ければ定速区間が入ります。
* ``minimum_effort``：移動時間を固定し、``integral(acceleration**2 dt)`` を最小化します。
  これは操作量の指標で、消費電力や機械的エネルギーではありません。
  上下限が効かなければ位置は3次関数、加速度は1次関数になり、目的値は
  ``12*distance**2/duration**3`` です。

``limited_effort`` は加速度上限に達する例、``speed_limited_effort`` は速度上限に
達する例です。区間構成は解析的な切り替え構造から選び、区間内の制御と区間境界時刻を
最適化します。最小二乗操作量では最後の時刻だけ固定します。

**試すこと**：距離を同じにして時間・加速度上限・速度上限を変える。
所要時間の短縮と操作量の減少が両立するか、制御が急に切り替わる理由を比較する。

最小二乗操作量の ``duration`` は、最短移動時間より十分長くする必要があります。
この教材では数値的に退化する境界を避けるため、最短時間の ``1.000001`` 倍以下は
入力エラーにします。境界そのものは ``minimum_time`` で扱います。

`全例題の一覧と共通の使い方 <../README.rst>`_
