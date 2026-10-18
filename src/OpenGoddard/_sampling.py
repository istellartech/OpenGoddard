"""Independent path-constraint sampling using differentiable interpolation maps."""

from functools import lru_cache

import numpy as np
from scipy.interpolate import BarycentricInterpolator
from scipy.special import roots_legendre

from ._helpers import Condition
from ._models import LinearCallback
from ._spectral import legendre_rule
from ._variables import node_times


@lru_cache(maxsize=256)
def sampling_map(elements, factor, extra=()):
    count = sum(e.nodes for e in elements)
    matrices, times, masks = [], [], []
    front = 0
    for e in elements:
        tau = legendre_rule("LGL", e.nodes)[0]
        additions = np.array(
            [
                2 * (point - e.left) / (e.right - e.left) - 1
                for point in extra
                if e.left < point < e.right
            ]
        )
        query = np.unique(
            np.concatenate((tau, roots_legendre(factor * e.nodes + 1)[0], additions))
        )
        block = np.zeros((query.size, count))
        block[:, front : front + e.nodes] = BarycentricInterpolator(
            tau, np.eye(e.nodes), axis=0, rng=0
        )(query)
        matrices.append(block)
        times.append(e.left + (query + 1) * (e.right - e.left) / 2)
        masks.append(~np.isin(query, tau))
        front += e.nodes
    return np.concatenate(times), np.vstack(matrices), np.concatenate(masks)


class SampledView:
    """Callback view with states, controls and times evaluated between mesh nodes."""

    def __init__(self, base, factor, points=None):
        self._base = base
        self.array_module = getattr(base, "array_module", np)
        maps = [
            sampling_map(
                tuple(elements), factor, () if points is None else tuple(points[s])
            )
            for s, elements in enumerate(base.mesh)
        ]
        self.tau = [m[0] for m in maps]
        self._maps = [m[1] for m in maps]
        self.off_grid = [m[2] for m in maps]
        self.nodes = [len(tau) for tau in self.tau]

    def __getattr__(self, name):
        if name == "p" or name.startswith("index_"):
            raise AttributeError(
                "sampled path callbacks must use physical state/control/time getters"
            )
        return getattr(self._base, name)

    def states(self, state, section):
        return self.array_module.dot(
            self._maps[section], self._base.states(state, section)
        )

    def controls(self, control, section):
        return self.array_module.dot(
            self._maps[section], self._base.controls(control, section)
        )

    def states_all_section(self, state):
        return self.array_module.concatenate(
            [self.states(state, s) for s in range(self.number_of_section)]
        )

    def controls_all_section(self, control):
        return self.array_module.concatenate(
            [self.controls(control, s) for s in range(self.number_of_section)]
        )

    @property
    def time(self):
        return node_times(self)

    @property
    def time_all_section(self):
        return self.array_module.concatenate(self.time)

    def condition(self, length=0):
        return Condition(length, backend=self.array_module)


def original_inequality(problem):
    callback = problem.inequality
    body = callback.function if isinstance(callback, LinearCallback) else callback
    return body.callback if isinstance(body, SampledPath) else callback


class SampledPath:
    def __init__(self, callback, factor, points=None):
        self.callback, self.factor, self.points = callback, factor, points

    def __call__(self, problem, obj):
        view = SampledView(problem, self.factor, self.points)
        xp = view.array_module
        blocks = [] if self.callback is None else [self.callback(view, obj)]
        for section in range(problem.number_of_section):
            for state in (True, False):
                count = (
                    problem.number_of_states if state else problem.number_of_controls
                )[section]
                for variable in range(count):
                    # Index metadata is not traced; use the underlying Problem.
                    metadata = getattr(problem, "_problem", problem)
                    divide = (
                        metadata._division_states
                        if state
                        else metadata._division_controls
                    )
                    back, front = divide(variable, section)
                    pairs = metadata.bounds[front:back]
                    if any(pair != pairs[0] for pair in pairs):
                        raise ValueError(
                            "path sampling requires constant nodal box bounds"
                        )
                    getter = view.states if state else view.controls
                    unit = (view.unit_states if state else view.unit_controls)[section][
                        variable
                    ]
                    values = getter(variable, section)[view.off_grid[section]] / unit
                    lower, upper = pairs[0]
                    if lower is not None:
                        blocks.append(values - lower)
                    if upper is not None:
                        blocks.append(upper - values)
        return xp.concatenate(blocks) if blocks else xp.zeros(0)


def enable_path_sampling(problem, factor, points=None):
    callback = original_inequality(problem)
    problem._mesh_original_inequality = callback
    if points is not None:
        problem._mesh_path_points = points
    sampled = SampledPath(callback, factor, getattr(problem, "_mesh_path_points", None))
    problem.inequality = (
        LinearCallback(sampled)
        if callback is None or isinstance(callback, LinearCallback)
        else sampled
    )


def polynomial_extrema(problem):
    """Interior extrema of every state/control polynomial in phase coordinates.

    For affine, pointwise scalar bounds these points include their exact worst
    violations (up to polynomial/root roundoff). General nonlinear path callbacks
    still require sampled checks and do not acquire a continuous-time proof.
    """
    from ._grid import elements_with_slices

    points = []
    for section in range(problem.number_of_section):
        entries = []
        for element, rows in elements_with_slices(problem, section):
            local = (problem.tau[section][rows] - element.left) * 2 / (
                element.right - element.left
            ) - 1
            values = [
                problem.states(i, section)[rows]
                for i in range(problem.number_of_states[section])
            ]
            values += [
                problem.controls(i, section)[rows]
                for i in range(problem.number_of_controls[section])
            ]
            for values_at_nodes in values:
                if np.ptp(values_at_nodes) < 1e-10 * max(
                    1.0, np.max(np.abs(values_at_nodes))
                ):
                    continue
                coeff = np.polynomial.legendre.legfit(
                    local, values_at_nodes, element.nodes - 1
                )
                roots = np.polynomial.legendre.legroots(
                    np.polynomial.legendre.legder(coeff)
                )
                roots = roots[np.abs(roots.imag) < 1e-9].real
                roots = roots[(roots > -1 + 1e-10) & (roots < 1 - 1e-10)]
                entries.extend(
                    element.left + (roots + 1) * (element.right - element.left) / 2
                )
        points.append(tuple(float(point) for point in np.unique(entries)))
    return tuple(points)
