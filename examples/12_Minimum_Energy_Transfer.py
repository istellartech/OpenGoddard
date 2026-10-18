"""Free-time transfer: x'=u, x(5)=0, x(tf)=d, min T + integral(u^2 dt).

The exact solution is T=d, u=1, and cost=2d for d>0. This small problem
demonstrates physical time, scaling, running cost, and checking convergence.
Run from the repository root with:
    uv run python examples/12_Minimum_Energy_Transfer.py
"""

import numpy as np

from OpenGoddard import DynamicsModel, Guess, LinearCallback, Problem


def pointwise_dynamics(x, u, time, obj, section, xp):
    return xp.stack((u[0],))


dynamics = DynamicsModel(pointwise_dynamics)


@LinearCallback
def equality(problem, target):
    state = problem.states_all_section(0)
    result = problem.condition()
    result.equal(state[0], 0)
    result.equal(state[-1], target)
    return result()


def make_problem(target=2.0):
    """Construct the transfer with physical inputs and dimensionless solver p."""
    problem = Problem([5, 8], [8], [1], [1], maxIterator=2)
    problem.set_unit_time(3)
    problem.set_unit_states(0, 0, 2)
    problem.set_states(0, 0, Guess.linear(problem.time[0], 0, target))
    problem.set_controls(0, 0, Guess.constant(problem.time[0], target / 3))
    problem.set_controls_bounds(0, 0, 0, 3)
    problem.set_time_final_bounds(0, 5.1, 15)
    problem.dynamics = [dynamics]
    problem.equality = equality
    problem.cost = LinearCallback(lambda p, obj: p.time_final(0) - p.time_start(0))
    problem.running_cost = lambda p, obj: p.controls_all_section(0) ** 2
    return problem


def main():
    target = 2.0
    problem = make_problem(target)
    result = problem.solve(target, maxiter=100, ftol=1e-11)
    if not result.success:
        raise RuntimeError(result.message)
    print(
        f"Duration: {problem.time_final(0) - problem.time_start(0):.6f} (exact: {target})"
    )
    print(f"Objective: {result.fun:.6f} (exact: {2 * target})")
    np.testing.assert_allclose(problem.controls_all_section(0), 1, atol=1e-5)


if __name__ == "__main__":
    main()
