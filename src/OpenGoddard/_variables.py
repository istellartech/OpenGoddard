"""Layout and scaling of the optimizer's one-dimensional parameter vector.

Within each phase, p stores all nodes of state 0, then state 1, and so on,
followed by the controls. The final entries are the phase end times.
Physical values are scaled values multiplied by their reference units.
"""

from collections.abc import Sequence

import numpy as np

from ._validation import index


class VariableLayout:
    """Fixed phase dimensions and slices, independent of a solver iterate."""

    def __init__(
        self, nodes: Sequence[int], states: Sequence[int], controls: Sequence[int]
    ):
        self.nodes = tuple(nodes)
        self.states = tuple(states)
        self.controls = tuple(controls)
        self.sections = len(nodes)
        self.divisions = []
        offset = 0
        for n, ns, nc in zip(nodes, states, controls, strict=True):
            self.divisions.append([offset + n * i for i in range(1, ns + nc + 1)])
            offset += n * (ns + nc)
        self.time_offset = offset
        self.size = offset + self.sections

    def _slice(self, variable, section, *, control):
        section = index(section, self.sections, "section")
        counts = self.controls if control else self.states
        variable = index(variable, counts[section], "control" if control else "state")
        offset = 0 if section == 0 else self.divisions[section - 1][-1]
        if control:
            variable += self.states[section]
        start = offset + variable * self.nodes[section]
        return slice(start, start + self.nodes[section])

    def state_slice(self, state, section):
        return self._slice(state, section, control=False)

    def control_slice(self, control, section):
        return self._slice(control, section, control=True)

    def time_index(self, section: int) -> int:
        return self.time_offset + index(section, self.sections, "section")


def rescale(problem, rows, old_unit, new_unit):
    """Change reference units while preserving physical guesses and bounds."""
    ratio = old_unit / new_unit
    problem.p[rows] *= ratio
    problem.bounds[rows] = [
        (
            None if lower is None else lower * ratio,
            None if upper is None else upper * ratio,
        )
        for lower, upper in problem.bounds[rows]
    ]


def scaled_derivatives(values, units, time_unit, xp=np):
    """Convert physical dx/dt to flattened d(x/unit)/d(t/time_unit)."""
    scale = time_unit / xp.array(units)
    return xp.reshape(values * scale[:, None], (-1,))


def node_times(view):
    """Derive physical node times from the current parameters, also under AD.

    LGL endpoints have fractions exactly zero and one. This expression keeps
    shared endpoints identical without mutating arrays during tracing.
    """
    times = []
    for section, tau in enumerate(view.tau):
        fraction = (tau + 1) / 2
        start, end = view.time_start(section), view.time_final(section)
        times.append((1 - fraction) * start + fraction * end)
    return times
