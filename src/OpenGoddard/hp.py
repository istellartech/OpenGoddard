"""Residual-guided h/hp adaptation inside unchanged logical phases."""

import warnings

import numpy as np

from ._grid import Element
from ._sampling import enable_path_sampling, polynomial_extrema
from ._validation import integer, positive_scale
from .mesh import estimate_error, remesh


def solve_hp_adaptive(
    problem,
    obj=None,
    *,
    strategy="hp",
    tolerance=1e-5,
    max_refinements=10,
    max_nodes=150,
    growth=1.5,
    max_elements=24,
    min_element_nodes=6,
    max_order=14,
    path_check_factor=4,
    **solve_options,
):
    """Split nonsmooth elements (h), or increase smooth element orders (hp).

    Controls may jump at an element interface; state continuity is enforced by
    the transcription. All user callbacks retain the original logical section
    indices and endpoint times. Off-grid path constraints and box bounds are
    included in the NLP using fixed differentiable interpolation maps.

    ``max_nodes`` is a per-phase TOTAL including duplicated interface nodes;
    ``max_elements`` caps elements per phase. At most two elements are changed
    per iteration. ``max_order`` is a node-count threshold for choosing h over p.
    The smoothness/tail and split-location rules are heuristics, not an error
    bound or an implementation of a particular published hp algorithm.
    """
    if strategy not in ("h", "hp"):
        raise ValueError("strategy must be 'h' or 'hp'")
    tolerance = positive_scale(tolerance)
    refinements = integer(max_refinements, "max_refinements")
    cap = integer(max_nodes, "max_nodes", 2)
    element_cap = integer(max_elements, "max_elements", 1)
    minimum = integer(min_element_nodes, "min_element_nodes", 4)
    order = integer(max_order, "max_order", minimum)
    factor = integer(path_check_factor, "path_check_factor", 2)
    if not np.isfinite(growth) or growth <= 1:
        raise ValueError("growth must exceed 1")
    if any(n > cap for n in problem.nodes) or any(
        len(m) > element_cap for m in problem.mesh
    ):
        raise ValueError("mesh limits must not be smaller than the initial mesh")
    enable_path_sampling(problem, factor)
    history = []
    reason = "refinement_limit"
    converged = False
    result = None
    for iteration in range(refinements + 1):
        initial = problem.p.copy()
        result = problem.solve(obj, **solve_options)
        error = estimate_error(problem, obj) if result.success else None
        entry = {
            "nodes": list(problem.nodes),
            "mesh": [[[e.left, e.right, e.nodes] for e in m] for m in problem.mesh],
            "success": bool(result.success),
            "objective": float(result.fun),
            "iterations": int(result.nit),
            "error": error,
            "actions": [],
        }
        history.append(entry)
        converged = (
            error is not None
            and error["max_error"] <= tolerance
            and error["path_violation"] <= tolerance
        )
        if converged:
            reason = "tolerance"
            break
        if iteration == refinements:
            break
        if error is not None and error["path_violation"] > tolerance:
            current = polynomial_extrema(problem)
            previous = getattr(
                problem, "_mesh_path_points", tuple(() for _ in problem.mesh)
            )
            points = []
            added = 0
            for old, new in zip(previous, current, strict=True):
                # Keep previous cuts; never differentiate through moving roots.
                extras = [
                    point
                    for point in new
                    if not any(abs(point - p) < 1e-9 for p in old)
                ]
                added += len(extras)
                points.append(
                    tuple(float(v) for v in np.unique(tuple(old) + tuple(extras)))
                )
            if added:
                enable_path_sampling(problem, factor, tuple(points))
                entry["actions"].append({"kind": "path", "points_added": added})
                if error["max_error"] <= tolerance:
                    continue
        candidates = []
        for section, entries in enumerate(problem.mesh):
            for panel, e in enumerate(entries):
                detail = None if error is None else error["elements"][section][panel]
                score = (
                    e.right - e.left if detail is None else detail["error"] / tolerance
                )
                if (
                    error is not None
                    and detail is not None
                    and error["path_violation"] > tolerance
                ):
                    score = max(score, detail["overshoot"] / tolerance)
                    if score <= 1 and not any(
                        v["overshoot"] > tolerance
                        for group in error["elements"]
                        for v in group
                    ):
                        score = 2 * (e.right - e.left)
                if error is None or score > 1:
                    candidates.append((score, section, panel, detail))
        mesh = [list(entries) for entries in problem.mesh]
        # Descending panel index keeps replacements valid within each section.
        chosen = sorted(candidates, reverse=True)[:2]
        for _score, section, panel, detail in sorted(
            chosen, key=lambda v: (v[1], v[2]), reverse=True
        ):
            e = mesh[section][panel]
            split = (
                strategy == "h"
                or detail is None
                or detail["smoothness"] > 0.05
                or e.nodes >= order
            )
            total = sum(element.nodes for element in mesh[section])
            if split:
                child = e.nodes if strategy == "h" else max(minimum, e.nodes)
                point = (e.left + e.right) / 2 if detail is None else detail["split"]
                if (
                    len(mesh[section]) >= element_cap
                    or total - e.nodes + 2 * child > cap
                    or e.right - e.left < 1e-8
                ):
                    continue
                mesh[section][panel : panel + 1] = [
                    Element(e.left, point, child),
                    Element(point, e.right, child),
                ]
                action = {
                    "section": section,
                    "element": panel,
                    "kind": "h",
                    "split": point,
                    "nodes": child,
                }
            else:
                count = min(
                    order,
                    e.nodes + max(2, int(np.ceil(e.nodes * (growth - 1)))),
                    e.nodes + cap - total,
                )
                if count <= e.nodes:
                    continue
                mesh[section][panel] = Element(e.left, e.right, count)
                action = {
                    "section": section,
                    "element": panel,
                    "kind": "p",
                    "nodes": count,
                }
            entry["actions"].append(action)
        changed = any(action["kind"] in ("h", "p") for action in entry["actions"])
        if not changed:
            if entry["actions"]:
                continue
            reason = "mesh_limit"
            break
        if not result.success:
            problem.p = initial
            problem.time_update()
        refined = remesh(problem, mesh)
        problem.__dict__.update(refined.__dict__)
    if result is None:
        raise RuntimeError("adaptive solver did not produce a result")
    result.mesh_converged = bool(converged)
    result.mesh_history = history
    result.mesh_strategy = strategy
    result.mesh_stop_reason = reason
    if not converged:
        warnings.warn(
            "adaptive solve stopped before meeting the mesh tolerance; "
            "inspect mesh_history",
            RuntimeWarning,
            stacklevel=2,
        )
    return result
