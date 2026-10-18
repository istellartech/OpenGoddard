"""Warm-started p refinement with independent, between-node defect checks."""

import copy
import warnings

import numpy as np
from scipy.interpolate import BarycentricInterpolator
from scipy.special import roots_legendre

from ._validation import integer, positive_scale
from .derivatives import DynamicsModel


def refine(problem, nodes):
    """Return a new problem interpolated onto larger or smaller LGL meshes.

    Phase topology, endpoints, units, callbacks and constant per-variable bounds
    are preserved. Varying nodal box bounds are rejected: express such bounds as
    path constraints instead. Barycentric interpolation does not merge duplicate
    phase endpoints and therefore preserves stage-separation discontinuities.
    Callbacks must use the problem passed to them, rather than capture an old p.
    """
    from .optimize import Problem

    nodes = [integer(n, "nodes", 2) for n in nodes]
    if len(nodes) != problem.number_of_section:
        raise ValueError("provide one node count per section")
    new = Problem(
        problem.time_knots(),
        nodes,
        problem.number_of_states,
        problem.number_of_controls,
        problem.maxIterator,
    )
    new.set_unit_time(problem.unit_time)
    for section in range(problem.number_of_section):
        ns, nc = problem.number_of_states[section], problem.number_of_controls[section]
        for variable in range(ns + nc):
            state = variable < ns
            component = variable if state else variable - ns
            divide = problem._division_states if state else problem._division_controls
            back, front = divide(component, section)
            pairs = problem.bounds[front:back]
            if any(pair != pairs[0] for pair in pairs):
                raise ValueError(
                    "refinement requires constant nodal box bounds; use path constraints for varying bounds"
                )
            unit = (problem.unit_states if state else problem.unit_controls)[section][
                component
            ]
            if state:
                new.set_unit_states(component, section, unit)
                new_back, new_front = new._division_states(component, section)
            else:
                new.set_unit_controls(component, section, unit)
                new_back, new_front = new._division_controls(component, section)
            interpolator = BarycentricInterpolator(
                problem.tau[section], problem.p[front:back], rng=0
            )
            new.p[new_front:new_back] = interpolator(new.tau[section])
            new.bounds[new_front:new_back] = pairs[:1] * nodes[section]
    new.bounds[-new.number_of_section :] = problem.bounds[-problem.number_of_section :]
    for name in (
        "dynamics",
        "cost",
        "running_cost",
        "cost_derivative",
        "equality",
        "inequality",
        "knot_states_smooth",
    ):
        setattr(new, name, copy.copy(getattr(problem, name)))
    # Preserve user metadata without copying caches/results from the old mesh.
    new.__dict__ = {**problem.__dict__, **new.__dict__}
    new.time_update()
    return new


def estimate_error(problem, obj=None, check_factor=2):
    """Measure relative dimensionless ODE defects away from collocation nodes.

    Per phase: max|dX/dtau - (tf-ti) f/2| / (unit + max|X|), evaluated at
    ``check_factor * nodes + 1`` Gauss points. This is a residual indicator,
    not a rigorous state-error bound. Nonpositive durations are rejected.
    Requires DynamicsModel to evaluate physical dynamics on an independent grid.
    Path constraints and box bounds are also checked on a denser LGL mesh;
    these checks are sampled, and do not prove continuous feasibility.
    """
    factor = integer(check_factor, "check_factor", 2)
    errors = []
    if not all(isinstance(model, DynamicsModel) for model in problem.dynamics):
        raise ValueError(
            "mesh error estimation requires DynamicsModel in every section"
        )
    for section, n in enumerate(problem.nodes):
        model = problem.dynamics[section]
        x, u, _ = model.inputs(problem, section)
        check = roots_legendre(factor * n + 1)[0]
        state_poly = BarycentricInterpolator(problem.tau[section], x, axis=1, rng=0)
        state = state_poly(check)
        control = (
            BarycentricInterpolator(problem.tau[section], u, axis=1, rng=0)(check)
            if u.shape[0]
            else np.empty((0, check.size))
        )
        start, end = problem.time_start(section), problem.time_final(section)
        if end <= start:
            raise ValueError(
                "mesh error estimation requires strictly positive phase durations"
            )
        time = start + (check + 1) / 2 * (end - start)
        residual = state_poly.derivative(check) - (end - start) / 2 * model.physical(
            state, control, time, obj, section
        )
        denominator = np.asarray(problem.unit_states[section]) + np.max(
            np.abs(state), axis=1
        )
        errors.append(float(np.max(np.abs(residual) / denominator[:, None])))
    dense = refine(problem, [factor * n + 1 for n in problem.nodes])
    inequalities = dense._inequality_constraints(obj)
    violation = max(0.0, -float(np.min(inequalities, initial=0)))
    for value, pair in zip(dense.p, dense.bounds):
        lower, upper = pair
        violation = max(
            violation,
            0.0 if lower is None else lower - value,
            0.0 if upper is None else value - upper,
        )
    return {
        "section_errors": errors,
        "max_error": max(errors),
        "path_violation": float(violation),
    }


def solve_adaptive(
    problem,
    obj=None,
    *,
    tolerance=1e-5,
    max_refinements=4,
    max_nodes=100,
    growth=1.5,
    **solve_options,
):
    """Solve, check between nodes, and grow only phases exceeding tolerance.

    Mutates the original problem to the final mesh and returns its OptimizeResult.
    ``result.success`` remains the optimizer's status; additionally inspect
    ``result.mesh_converged``. Node/refinement limits or solver failure never
    count as mesh convergence. This is p refinement, not phase splitting;
    known discontinuities should be represented as separate phases.
    A failed coarse solve grows all phases and retries from its pre-solve guess;
    failures remain visible in mesh_history and never count as convergence.
    """
    tolerance = positive_scale(tolerance)
    refinements = integer(max_refinements, "max_refinements")
    cap = integer(max_nodes, "max_nodes", 2)
    if not np.isfinite(growth) or growth <= 1:
        raise ValueError("growth must exceed 1")
    if any(n > cap for n in problem.nodes):
        raise ValueError("max_nodes must not be smaller than the initial mesh")
    history = []
    for iteration in range(refinements + 1):
        initial_parameters = problem.p.copy()
        result = problem.solve(obj, **solve_options)
        error = estimate_error(problem, obj) if result.success else None
        history.append(
            {
                "nodes": list(problem.nodes),
                "success": bool(result.success),
                "objective": float(result.fun),
                "error": error,
                "iterations": int(result.nit),
            }
        )
        converged = (
            error is not None
            and error["max_error"] <= tolerance
            and error["path_violation"] <= tolerance
        )
        if converged or iteration == refinements:
            break
        nodes = [
            min(cap, max(n + 2, int(np.ceil(n * growth))))
            if error is None
            or error["section_errors"][section] > tolerance
            or error["path_violation"] > tolerance
            else n
            for section, n in enumerate(problem.nodes)
        ]
        if nodes == problem.nodes:
            break
        if not result.success:
            # A very coarse transcription may be infeasible. Retry a larger
            # mesh from its pre-solve guess, not the failed solver's iterate.
            problem.p = initial_parameters
            problem.time_update()
        refined = refine(problem, nodes)
        problem.__dict__.update(refined.__dict__)
    result.mesh_converged = bool(converged)
    result.mesh_history = history
    if not converged:
        warnings.warn(
            "adaptive solve stopped before meeting the mesh tolerance; inspect mesh_history",
            RuntimeWarning,
            stacklevel=2,
        )
    return result
