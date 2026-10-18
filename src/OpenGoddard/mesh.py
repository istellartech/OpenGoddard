"""Phase-preserving mesh refinement and independent, between-node defect checks."""

import copy
import warnings

import numpy as np
from scipy.interpolate import BarycentricInterpolator
from scipy.special import roots_legendre

from ._grid import Element, elements_with_slices, normalize_mesh
from ._validation import integer, positive_scale
from .derivatives import DynamicsModel


def interpolate(problem, section, values, points, *, right_endpoint=False):
    """Evaluate piecewise polynomials, choosing one-sided values at interfaces."""
    points = np.asarray(points, dtype=float)
    entries = list(elements_with_slices(problem, section))
    right = np.array([e.right for e, _ in entries])
    owner = np.searchsorted(right, points, side="right")
    if right_endpoint and points.size:
        owner[-1] = np.searchsorted(right, points[-1], side="left")
    owner = np.clip(owner, 0, len(entries) - 1)
    result = np.empty(values.shape[:-1] + points.shape)
    for panel, (element, rows) in enumerate(entries):
        mask = owner == panel
        if np.any(mask):
            local = (problem.tau[section][rows] - element.left) * 2 / (
                element.right - element.left
            ) - 1
            target = (points[mask] - element.left) * 2 / (
                element.right - element.left
            ) - 1
            result[..., mask] = BarycentricInterpolator(
                local, values[..., rows], axis=-1, rng=0
            )(target)
    return result


def remesh(problem, mesh):
    """Interpolate onto independent LGL elements without renumbering phases.

    State/control arrays include both sides of each internal element interface.
    Only states are constrained continuous. Constant per-variable nodal bounds,
    units, phase endpoints, callbacks and custom metadata are preserved.
    """
    from .optimize import Problem

    counts = [
        sum(e.nodes if isinstance(e, Element) else e[2] for e in entries)
        for entries in mesh
    ]
    layout = normalize_mesh(counts, mesh)
    new = Problem(
        problem.time_knots(),
        counts,
        problem.number_of_states,
        problem.number_of_controls,
        problem.maxIterator,
        mesh=layout,
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
            values = problem.p[front:back]
            for element, rows in elements_with_slices(new, section):
                new.p[new_front + rows.start : new_front + rows.stop] = interpolate(
                    problem,
                    section,
                    values,
                    new.tau[section][rows],
                    right_endpoint=True,
                )
            new.bounds[new_front:new_back] = pairs[:1] * counts[section]
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
    new.__dict__ = {**problem.__dict__, **new.__dict__}
    new.time_update()
    return new


def refine(problem, nodes):
    """Change p while preserving existing element boundaries and phase topology.

    Counts are totals per logical phase, including duplicated interface nodes.
    At least two nodes per existing element are required. Varying nodal box
    bounds are rejected; express them as path constraints instead.
    """
    nodes = [integer(n, "nodes", 2) for n in nodes]
    if len(nodes) != problem.number_of_section:
        raise ValueError("provide one node count per section")
    mesh = []
    for count, entries in zip(nodes, problem.mesh):
        extra = count - 2 * len(entries)
        if extra < 0:
            raise ValueError("at least two nodes per existing element are required")
        weights = np.array([max(1, e.nodes - 2) for e in entries], dtype=float)
        exact = extra * weights / weights.sum()
        allocation = np.floor(exact).astype(int)
        order = np.argsort(-(exact - allocation), kind="stable")
        allocation[order[: extra - int(allocation.sum())]] += 1
        mesh.append(
            [Element(e.left, e.right, int(n) + 2) for e, n in zip(entries, allocation)]
        )
    return remesh(problem, mesh)


def estimate_error(problem, obj=None, check_factor=2):
    """Measure relative dimensionless ODE defects away from collocation nodes.

    Per phase: max|dX/dtau - (tf-ti) f/2| / (unit + max|X|), evaluated at
    ``check_factor * element.nodes + 1`` Gauss points per element. Derivatives
    are scaled back to phase coordinates, so smaller elements do not artificially
    reduce the residual. This is a residual indicator,
    not a rigorous state-error bound. Nonpositive durations are rejected.
    Requires DynamicsModel to evaluate physical dynamics on an independent grid.
    Path constraints and box bounds are also checked on a denser LGL mesh and
    at state/control polynomial extrema. Affine pointwise scalar bounds reach
    their worst violations at these extrema (up to polynomial/root roundoff).
    General nonlinear path constraints remain sampled and are not certified.
    """
    factor = integer(check_factor, "check_factor", 2)
    from ._sampling import SampledPath, original_inequality, polynomial_extrema

    errors, details = [], []
    if not all(isinstance(model, DynamicsModel) for model in problem.dynamics):
        raise ValueError(
            "mesh error estimation requires DynamicsModel in every section"
        )
    for section, n in enumerate(problem.nodes):
        model = problem.dynamics[section]
        x, u, _ = model.inputs(problem, section)
        start, end = problem.time_start(section), problem.time_final(section)
        if end <= start:
            raise ValueError(
                "mesh error estimation requires strictly positive phase durations"
            )
        panels = []
        for element, rows in elements_with_slices(problem, section):
            local = (problem.tau[section][rows] - element.left) * 2 / (
                element.right - element.left
            ) - 1
            check = roots_legendre(factor * element.nodes + 1)[0]
            tau = element.left + (check + 1) * (element.right - element.left) / 2
            state_poly = BarycentricInterpolator(local, x[:, rows], axis=1, rng=0)
            state = state_poly(check)
            control_poly = (
                BarycentricInterpolator(local, u[:, rows], axis=1, rng=0)
                if u.shape[0]
                else None
            )
            control = (
                control_poly(check)
                if control_poly is not None
                else np.empty((0, check.size))
            )
            time = start + (tau + 1) / 2 * (end - start)
            residual = state_poly.derivative(check) * 2 / (
                element.right - element.left
            ) - (end - start) / 2 * model.physical(state, control, time, obj, section)
            denominator = np.asarray(problem.unit_states[section]) + np.max(
                np.abs(state), axis=1
            )
            point_errors = np.max(np.abs(residual) / denominator[:, None], axis=0)
            scaled = np.concatenate(
                (
                    x[:, rows] / np.asarray(problem.unit_states[section])[:, None],
                    u[:, rows] / np.asarray(problem.unit_controls[section])[:, None],
                )
            )
            coefficients = np.polynomial.legendre.legfit(
                local, scaled.T, element.nodes - 1
            )
            amplitude = np.linalg.norm(coefficients[1:], axis=0)
            ratios = np.linalg.norm(
                coefficients[-min(3, element.nodes - 1) :], axis=0
            ) / np.maximum(amplitude, 1e-8)
            smoothness = float(np.max(ratios))
            sampled = np.concatenate(
                (
                    state / np.asarray(problem.unit_states[section])[:, None],
                    control / np.asarray(problem.unit_controls[section])[:, None],
                )
            )
            overshoot = float(
                np.max(
                    np.maximum(
                        np.min(scaled, axis=1)[:, None] - sampled,
                        sampled - np.max(scaled, axis=1)[:, None],
                    ),
                    initial=0,
                )
            )
            split = float(tau[int(np.argmax(point_errors))])
            # Avoid pathological slivers when the defect maximum is near a boundary.
            split = float(
                np.clip(
                    split,
                    element.left + 0.25 * (element.right - element.left),
                    element.right - 0.25 * (element.right - element.left),
                )
            )
            panels.append(
                {
                    "left": element.left,
                    "right": element.right,
                    "nodes": element.nodes,
                    "error": float(np.max(point_errors)),
                    "smoothness": smoothness,
                    "overshoot": overshoot,
                    "split": split,
                }
            )
        errors.append(max(e["error"] for e in panels))
        details.append(panels)
    dense = remesh(
        problem,
        [
            [Element(e.left, e.right, factor * e.nodes + 1) for e in entries]
            for entries in problem.mesh
        ],
    )
    callback = original_inequality(problem)
    inequalities = np.empty(0) if callback is None else callback(dense, obj)
    violation = max(0.0, -float(np.min(inequalities, initial=0)))
    for value, pair in zip(dense.p, dense.bounds):
        lower, upper = pair
        violation = max(
            violation,
            0.0 if lower is None else lower - value,
            0.0 if upper is None else value - upper,
        )
    extreme_residuals = SampledPath(callback, factor, polynomial_extrema(problem))(
        problem, obj
    )
    violation = max(violation, -float(np.min(extreme_residuals, initial=0)))
    return {
        "section_errors": errors,
        "max_error": max(errors),
        "path_violation": float(violation),
        "elements": details,
    }


def solve_adaptive(
    problem,
    obj=None,
    *,
    tolerance=1e-5,
    max_refinements=4,
    max_nodes=100,
    growth=1.5,
    strategy="p",
    max_elements=24,
    min_element_nodes=6,
    max_order=14,
    path_check_factor=4,
    **solve_options,
):
    """Solve, check between nodes, and grow only phases exceeding tolerance.

    Mutates the original problem to the final mesh and returns its OptimizeResult.
    ``result.success`` remains the optimizer's status; additionally inspect
    ``result.mesh_converged``. Node/refinement limits or solver failure never
    count as mesh convergence. The default ``strategy='p'`` increases node counts.
    ``'h'`` splits elements and ``'hp'`` chooses splitting or increasing order;
    see hp.solve_hp_adaptive. Both retain logical phases and include off-grid
    path constraints with fixed extrema cuts. These callbacks must use physical
    getters, not direct ``p`` indices or mesh-dependent global reductions.
    A failed coarse solve grows all phases and retries from its pre-solve guess;
    failures remain visible in mesh_history and never count as convergence.
    """
    if strategy not in ("p", "h", "hp"):
        raise ValueError("strategy must be 'p', 'h' or 'hp'")
    if strategy != "p":
        from .hp import solve_hp_adaptive

        return solve_hp_adaptive(
            problem,
            obj,
            strategy=strategy,
            tolerance=tolerance,
            max_refinements=max_refinements,
            max_nodes=max_nodes,
            growth=growth,
            max_elements=max_elements,
            min_element_nodes=min_element_nodes,
            max_order=max_order,
            path_check_factor=path_check_factor,
            **solve_options,
        )
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
