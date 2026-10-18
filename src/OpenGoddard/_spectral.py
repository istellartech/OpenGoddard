"""Legendre quadrature and differentiation on the reference interval [-1, 1]."""

from functools import lru_cache

import numpy as np
from scipy import special

from ._validation import integer


@lru_cache(maxsize=64)
def _rule(method, n):
    """Compute each rule once; callers receive independent writable copies."""
    if method == "LG":
        nodes, weights = special.roots_legendre(n)
        factors = (n + 1) / 2 * special.eval_jacobi(n - 1, 1, 1, nodes)
    elif method == "LGR":
        roots = special.roots_jacobi(n - 1, 0, 1)[0] if n > 1 else []
        nodes = np.concatenate(([-1.0], roots))
        polynomial = special.eval_legendre(n - 1, nodes)
        weights = (1 - nodes) / (n**2 * polynomial**2)
        factors = polynomial / (1 - nodes)
    else:
        roots = special.roots_jacobi(n - 2, 1, 1)[0] if n > 2 else []
        nodes = np.concatenate(([-1.0], roots, [1.0]))
        factors = special.eval_legendre(n - 1, nodes)
        weights = 2 / (n * (n - 1) * factors**2)

    difference = nodes[:, None] - nodes[None, :]
    np.fill_diagonal(difference, 1.0)
    differentiation = (factors[:, None] / factors[None, :]) / difference
    np.fill_diagonal(differentiation, 0.0)
    # The derivative of a constant is zero. Enforcing row sums avoids
    # cancellation from separate closed-form endpoint formulas.
    np.fill_diagonal(differentiation, -differentiation.sum(axis=1))
    for array in (nodes, weights, differentiation):
        array.setflags(write=False)
    return nodes, weights, differentiation


def legendre_rule(method, n):
    if method not in {"LG", "LGR", "LGL"}:
        raise ValueError("method must be 'LG', 'LGR', or 'LGL'")
    n = integer(n, "number of nodes", 2 if method == "LGL" else 1)
    return tuple(array.copy() for array in _rule(method, n))
