"""The optimal-control equations evaluated on an LGL mesh.

Reading order: terminal/integral objective, boundary/ODE/continuity equalities,
then forward-time and path inequalities. The same equations serve NumPy value
evaluation and autograd tracing. Only NumPy evaluations validate/coerce outputs;
traced arrays must stay in their original array backend.
"""

import numpy as np

from ._grid import interfaces
from ._validation import vector


def _residual(value, name, xp, length=None):
    return vector(value, name, length) if xp is np else value


def objective(problem, obj, *, view=None, xp=np):
    """J = terminal cost + sum((tf - ti)/2 * w @ running cost)."""
    view = problem if view is None else view
    value = 0.0 if problem.cost is None else problem.cost(view, obj)
    if xp is np:
        value = float(value)
    if problem.running_cost is not None:
        integrand = _residual(
            problem.running_cost(view, obj), "running_cost", xp, sum(problem.nodes)
        )
        front = 0
        for section, node in enumerate(problem.nodes):
            duration = view.time_final(section) - view.time_start(section)
            value += (
                duration
                / 2
                * xp.dot(problem.w[section], integrand[front : front + node])
            )
            front += node
    if xp is np and not np.isfinite(value):
        raise ValueError("objective must be finite")
    return value


def equality(problem, obj, *, view=None, xp=np):
    """Concatenate boundary, scaled ODE and state-continuity residuals.

    For scaled state X and scaled time T, each phase enforces
    X @ D.T - (Tf - Ti)/2 * dX/dT = 0.
    Controls may jump at element interfaces; states must remain continuous.
    """
    view = problem if view is None else view
    blocks = []
    if problem.equality is not None:
        blocks.append(_residual(problem.equality(view, obj), "equality", xp))
    for section, node in enumerate(problem.nodes):
        rows = problem._layout.state_slice(0, section)
        back = rows.start + problem.number_of_states[section] * node
        state = xp.reshape(view.p[rows.start : back], (-1, node))
        derivative = xp.reshape(xp.dot(state, problem.D[section].T), (-1,))
        model = problem.dynamics[section]
        if model is None:
            raise ValueError("set dynamics for every section")
        dx = _residual(model(view, obj, section), "dynamics", xp, derivative.size)
        duration = (
            view.time_final(section) - view.time_start(section)
        ) / problem.unit_time
        blocks.append(derivative - duration / 2 * dx)
    for section in range(problem.number_of_section):
        for left, right in interfaces(problem, section):
            blocks.append(
                xp.stack(
                    [
                        (view.states(i, section)[left] - view.states(i, section)[right])
                        / problem.unit_states[section][i]
                        for i in range(problem.number_of_states[section])
                    ]
                )
            )
    for knot, smooth in enumerate(problem.knot_states_smooth):
        if (
            not smooth
            or problem.number_of_states[knot] != problem.number_of_states[knot + 1]
        ):
            continue
        blocks.append(
            xp.stack(
                [
                    (view.states(i, knot)[-1] - view.states(i, knot + 1)[0])
                    / problem.unit_states[knot][i]
                    for i in range(problem.number_of_states[knot])
                ]
            )
        )
    return xp.concatenate(blocks)


def inequality(problem, obj, *, view=None, xp=np):
    """Concatenate phase durations and user residuals, all >= 0 when feasible."""
    view = problem if view is None else view
    durations = xp.stack(
        [
            view.time_final(s) - view.time_start(s)
            for s in range(problem.number_of_section)
        ]
    )
    blocks = [durations / problem.unit_time]
    if problem.inequality is not None:
        blocks.append(_residual(problem.inequality(view, obj), "inequality", xp))
    return xp.concatenate(blocks)
