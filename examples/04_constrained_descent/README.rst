04. 深さ制限付き最速降下
==================================================

終点の位置と深さを固定し、最大深さの制約が軌道と所要時間をどう変えるか比較します。

実行方法
--------

リポジトリのルートから実行します。

.. code-block:: console

   uv run python examples/04_constrained_descent/main.py

``--plot`` で図を表示し、``--output`` で結果を保存します。

.. code-block:: console

   uv run python examples/04_constrained_descent/main.py --plot
   uv run python examples/04_constrained_descent/main.py --output examples/04_constrained_descent/output/trajectory.json


ファイル構成
------------

* ``main.py``：この例題の実装と実行入口。
* ``README.rst``：この例題の説明と実行方法。
* ``output/``：生成した図・動画・JSONなどの保存先。必要時に作成し、Git管理対象外とします。

問題の説明
----------

横方向を右向き正、深さを下向き正とし、始点から静止で出発します。制御角 ``theta`` は
鉛直下向きから右向きに測り、0が鉛直下降、pi/2が水平、piが鉛直上昇です。

.. math::

   \dot{x}=v\sin\theta, \quad \dot{y}=v\cos\theta, \quad
   \dot{v}=g\cos\theta, \quad v^2=2gy.

摩擦なしの理想的な軌道を仮定します。保存則 ``depth=speed**2/(2*gravity)`` を使い、
数値的な状態はxとspeedだけに減らしています。深さは出力時に復元します。
床上の一定深さ・一定速度に対して運動方程式を重複して課すことを避けるためです。

03番は終点の水平位置を固定し、終点の深さは自由です。この例では、比較条件を
そろえるため水平位置と深さの両方を固定します。``no_floor`` で制約なし、
``inactive_floor`` で床があっても解が接しない場合、``shallow_floor`` でより浅い床を試せます。

床が効かない場合はサイクロイド1区間。効く場合は下降・床上・上昇の3区間に分けます。
床への接続時の速度を指定し、区間境界時刻は最適化します。解析解はサイクロイドと
水平移動を接続したもので、所要時間を数値解と比較できます。

**試すこと**：床を徐々に浅くし、制約が効き始めるところを見る。重力だけを変えたとき、
軌道の形と所要時間がそれぞれどう変わるかを比較する。

``distance`` と ``final_depth`` は正、床の深さは終点の深さより大きくします。
曲率・法線力・摩擦の制約は含みません。求解器の目的値は
``duration/sqrt(max(distance, final_depth)/gravity)`` なので、重力の比較には出力の
実時間か ``reference.minimum_time`` を使ってください。

`全例題の一覧と共通の使い方 <../README.rst>`_
