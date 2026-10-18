"""Validation shared by the public numerical helpers."""

import operator

import numpy as np
from numpy.typing import NDArray


def integer(value, name: str, minimum: int = 0) -> int:
    """Accept Python and NumPy integers without silently truncating floats."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def index(value, size: int, name: str) -> int:
    """Normalize a Python-style index, checking both ends of its range."""
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if not -size <= result < size:
        raise IndexError(f"{name} {result} is out of range for size {size}")
    return result % size


def positive_scale(value) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("unit must be finite and strictly positive")
    return value


def vector(value, name: str, length: int | None = None) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or (length is not None and result.size != length):
        expected = "a one-dimensional array" if length is None else f"shape ({length},)"
        raise ValueError(f"{name} must have {expected}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def bounds_pair(lower, upper, unit):
    lower = None if lower is None else float(lower) / unit
    upper = None if upper is None else float(upper) / unit
    if (lower is not None and np.isnan(lower)) or (
        upper is not None and np.isnan(upper)
    ):
        raise ValueError("bounds must not contain NaN")
    if lower == np.inf or upper == -np.inf:
        raise ValueError("lower bound cannot be +inf or upper bound -inf")
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("lower bound must not exceed upper bound")
    return lower, upper
