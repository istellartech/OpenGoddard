"""01. 位置を動かす最小の例。状態 x は位置、制御 u は速度。

x'=u、x(5)=0、x(tf)=target。
所要時間と速度の二乗積分の和を最小にする。
"""

from OpenGoddard import Dynamics, Guess, Problem


# コールバックの引数は「問題、物理パラメータ、区間番号」。
# 状態0は位置、制御0は速度。
def dynamics(problem, target, section):
    u = problem.controls(0, section)
    dx = Dynamics(problem, section)
    dx[0] = u
    return dx()


def equality(problem, target):
    x = problem.states(0, 0)
    conditions = problem.condition()
    conditions.equal(x[0], 0)
    conditions.equal(x[-1], target)
    return conditions()


def cost(problem, target):
    return problem.time_final(0) - problem.time_start(0)


def running_cost(problem, target):
    u = problem.controls(0, 0)
    return u**2  # この値を時間で積分するのはライブラリの役割


def make_problem(target=2.0):
    # 時刻の初期推定、各区間のノード数・状態数・制御数を指定する。
    problem = Problem([5, 8], [8], [1], [1], maxIterator=2)
    problem.set_unit_time(3)
    problem.set_unit_states(0, 0, 2)

    # 初期推定と上下限は別々に設定する。
    time = problem.time[0]
    problem.set_states(0, 0, Guess.linear(time, 0, target))
    problem.set_controls(0, 0, Guess.constant(time, target / 3))
    problem.set_controls_bounds(0, 0, 0, 3)
    problem.set_time_final_bounds(0, 5.1, 15)

    problem.dynamics = [dynamics]
    problem.equality = equality
    problem.cost = cost
    problem.running_cost = running_cost
    return problem


def main():
    target = 2.0  # ここを変えて、所要時間と目的関数を比べる
    problem = make_problem(target)
    result = problem.solve(target, maxiter=100, ftol=1e-11)
    print("Solver success:", result.success)
    print("Duration:", problem.time_final(0) - problem.time_start(0))
    print("Objective:", result.fun)
    print("Exact duration:", target, "Exact objective:", 2 * target)
    return problem


if __name__ == "__main__":
    main()
