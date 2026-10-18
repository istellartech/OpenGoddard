"""Translate a Problem into SciPy callbacks and manage solver attempts.

The mathematical objective and residuals live in _transcription; this module
only validates solver settings, connects derivatives and records results.
"""

import warnings
from typing import cast

import numpy as np
from scipy import optimize

from ._validation import bounds_pair, integer, vector


def solve(
    problem, obj, display_func, *, solver, derivatives, **options
) -> optimize.OptimizeResult:
    if solver not in {"SLSQP", "trust-constr"}:
        raise ValueError("solver must be 'SLSQP' or 'trust-constr'")
    if derivatives not in {"finite-difference", "structured", "autograd"}:
        raise ValueError("invalid derivatives mode")
    if len(problem.dynamics) != problem.number_of_section or not all(
        callable(func) for func in problem.dynamics
    ):
        raise ValueError("set one dynamics callable for every section")
    if problem.cost is None and problem.running_cost is None:
        raise ValueError("set cost or running_cost before solving")
    for name in (
        "cost",
        "running_cost",
        "cost_derivative",
        "equality",
        "inequality",
    ):
        callback = getattr(problem, name)
        if callback is not None and not callable(callback):
            raise TypeError(f"{name} must be callable or None")
    if len(problem.knot_states_smooth) != problem.number_of_section - 1:
        raise ValueError("knot_states_smooth must have one entry per phase boundary")
    if not callable(display_func):
        raise TypeError("display_func must be callable")
    vector(problem.p, "p", problem.number_of_variables)
    if len(problem.bounds) != problem.number_of_variables:
        raise ValueError("bounds must have one pair per optimization variable")
    for lower, upper in problem.bounds:
        bounds_pair(lower, upper, 1.0)
    workers = options.pop("workers", None)
    if workers is not None and workers != 1:
        raise ValueError("parallel workers cannot share a mutable Problem")
    if solver == "SLSQP":
        options.setdefault("ftol", 1e-6)
    else:
        options.setdefault("gtol", 1e-8)
        options.setdefault("sparse_jacobian", True)
    options.setdefault("maxiter", 25 if solver == "SLSQP" else 1000)
    options.setdefault("disp", False)
    attempts = integer(problem.maxIterator, "maxIterator", 1)

    def wrap(func):
        def evaluate(parameters):
            problem.p = parameters
            return func(obj)

        return evaluate

    cost_derivative = problem.cost_derivative

    def gradient(obj):
        if cost_derivative is None:
            raise ValueError("cost_derivative is not set")
        return vector(
            cost_derivative(problem, obj),
            "cost_derivative",
            problem.number_of_variables,
        )

    constraints = [
        {"type": "eq", "fun": wrap(problem._equality_constraints)},
        {"type": "ineq", "fun": wrap(problem._inequality_constraints)},
    ]
    provider = None
    if derivatives != "finite-difference":
        from .derivatives import Derivatives

        provider = Derivatives(problem, obj, derivatives)
        for constraint, evaluate in zip(
            constraints, (provider.equality, provider.inequality), strict=True
        ):

            def constraint_jacobian(obj, evaluate=evaluate):
                value = evaluate()
                return value.toarray() if solver == "SLSQP" else value

            constraint["jac"] = wrap(constraint_jacobian)
    jac = None if cost_derivative is None else wrap(gradient)
    if provider is not None:
        jac = wrap(lambda obj: provider.objective())
    hessian = None
    if solver == "trust-constr":
        if provider is not None and derivatives == "autograd":
            hessian = wrap(lambda obj: provider.hessian())
            for constraint in constraints:

                def constraint_hessian(parameters, weights, kind=constraint["type"]):
                    problem.p = parameters
                    return provider.hessian(kind, weights)

                constraint["hess"] = constraint_hessian
        constraints = [
            optimize.NonlinearConstraint(
                constraint["fun"],
                0,
                0 if constraint["type"] == "eq" else np.inf,
                jac=constraint.get("jac", "2-point"),
                hess=constraint.get("hess", optimize.BFGS()),
            )
            for constraint in constraints
        ]
    problem.iterator = 0
    problem.result = None
    problem.result_history = []
    for attempt in range(attempts):
        result = cast(
            optimize.OptimizeResult,
            optimize.minimize(
                wrap(problem._objective),
                problem.p,
                bounds=problem.bounds,
                constraints=constraints,
                jac=jac,
                hess=hessian,
                method=solver,
                options=options,
            ),
        )
        problem.p = result.x.copy()
        problem.iterator = attempt + 1
        problem.result = result
        problem.result_history.append(result)
        display_func()
        if result.success or result.status != (9 if solver == "SLSQP" else 0):
            break
    if problem.result is None:
        raise RuntimeError("solver did not produce a result")
    if not problem.result.success:
        warnings.warn(
            f"{solver} failed: {problem.result.message}", RuntimeWarning, stacklevel=3
        )
    problem.result.solver = solver
    problem.result.derivatives = derivatives
    problem.result.equality_violation = np.max(
        np.abs(problem._equality_constraints(obj)), initial=0
    )
    problem.result.inequality_violation = max(
        0.0, -np.min(problem._inequality_constraints(obj), initial=0)
    )
    return problem.result
