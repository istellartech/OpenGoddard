"""LGL elements inside a logical phase; phase indices never change on h refinement."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag

from ._spectral import legendre_rule
from ._validation import integer


@dataclass(frozen=True)
class Element:
    """An LGL polynomial on [left, right] in the phase coordinate [-1, 1]."""

    left: float
    right: float
    nodes: int


def normalize_mesh(nodes, mesh=None):
    if mesh is None:
        return tuple((Element(-1.0, 1.0, n),) for n in nodes)
    if len(mesh) != len(nodes):
        raise ValueError("mesh must have one element list per logical section")
    result = []
    for count, entries in zip(nodes, mesh, strict=True):
        elements = []
        left = -1.0
        for entry in entries:
            element = entry if isinstance(entry, Element) else Element(*entry)
            n = integer(element.nodes, "element nodes", 2)
            if not np.isfinite([element.left, element.right]).all():
                raise ValueError("element endpoints must be finite")
            if (
                abs(element.left - left) > 1e-12
                or element.right <= left
                or element.right > 1 + 1e-12
            ):
                raise ValueError(
                    "mesh elements must partition [-1, 1] without gaps or overlaps"
                )
            right = 1.0 if abs(element.right - 1.0) < 1e-12 else float(element.right)
            if right <= left:
                raise ValueError("mesh elements must have strictly positive width")
            elements.append(Element(left, right, n))
            left = right
        if not elements or left != 1.0 or sum(e.nodes for e in elements) != count:
            raise ValueError("mesh must cover [-1, 1] and match the section node count")
        result.append(tuple(elements))
    return tuple(result)


def rule(elements):
    if len(elements) == 1:
        return legendre_rule("LGL", elements[0].nodes)
    nodes, weights, matrices = [], [], []
    for element in elements:
        tau, weight, differentiation = legendre_rule("LGL", element.nodes)
        width = element.right - element.left
        mapped = element.left + (tau + 1) * width / 2
        mapped[0], mapped[-1] = element.left, element.right
        nodes.append(mapped)
        weights.append(weight * width / 2)
        matrices.append(differentiation * 2 / width)
    return np.concatenate(nodes), np.concatenate(weights), block_diag(*matrices)


def elements_with_slices(problem, section):
    front = 0
    for element in problem.mesh[section]:
        yield element, slice(front, front + element.nodes)
        front += element.nodes


def interfaces(problem, section):
    """Pairs of duplicated node indices requiring state, but not control, continuity."""
    back = 0
    for element in problem.mesh[section][:-1]:
        back += element.nodes
        yield back - 1, back
