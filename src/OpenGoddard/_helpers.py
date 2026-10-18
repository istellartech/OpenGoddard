"""Initial guesses, residual builders and scaled dynamics for optimal control."""

# Copyright (c) 2017 Interstellar Technologies Inc.
# SPDX-License-Identifier: MIT

import numpy as np
from numpy.typing import NDArray

from ._validation import index, integer, positive_scale, vector
from ._variables import scaled_derivatives


def _guess_time(time):
    time = vector(time, "time")
    if time.size < 2 or time[-1] <= time[0] or np.any(np.diff(time) < 0):
        raise ValueError("time must be nondecreasing and span a nonzero interval")
    return time


class Guess:
    """Construct physical initial guesses on a supplied time grid."""

    @classmethod
    def zeros(cls, time) -> NDArray[np.float64]:
        """Return one zero per time node.

        Args:
            time (array_like): Time nodes; only their count is used.

        Returns:
            ndarray: Values of shape (N,).
        """
        return np.zeros(len(time))

    @classmethod
    def constant(cls, time, const: float) -> NDArray[np.float64]:
        """Return one constant physical value per time node.

        Args:
            time (array_like): Time nodes; only their count is used.
            const: Constant physical value.

        Returns:
            ndarray: Values of shape (N,).
        """
        return np.ones(len(time)) * const

    @classmethod
    def linear(cls, time, y0: float, yf: float) -> NDArray[np.float64]:
        """Interpolate physical endpoint values linearly in time.

        Args:
            time (array_like): Finite, nondecreasing physical times with a
                positive span. Repeated interface times are supported.
            y0: Physical value at the first time.
            yf: Physical value at the last time.

        Returns:
            ndarray: Values of shape (N,).
        """
        time = _guess_time(time)
        fraction = (time - time[0]) / (time[-1] - time[0])
        return y0 + (yf - y0) * fraction

    @classmethod
    def cubic(
        cls, time, y0: float, yprime0: float, yf: float, yprimef: float
    ) -> NDArray[np.float64]:
        """Interpolate endpoint values and physical-time slopes with a cubic.

        Args:
            time (array_like): Finite, nondecreasing physical times with a
                positive span. Repeated interface times are supported.
            y0: Physical value at the first time.
            yprime0: Physical dy/dt at the first time.
            yf: Physical value at the last time.
            yprimef: Physical dy/dt at the last time.

        Returns:
            ndarray: Values of shape (N,).
        """
        time = _guess_time(time)
        duration = time[-1] - time[0]
        fraction = (time - time[0]) / duration
        # Cubic Hermite basis on [0, 1]; no matrix inverse or powers of
        # absolute timestamps, which become ill-conditioned far from zero.
        return (
            (2 * fraction**3 - 3 * fraction**2 + 1) * y0
            + (fraction**3 - 2 * fraction**2 + fraction) * duration * yprime0
            + (-2 * fraction**3 + 3 * fraction**2) * yf
            + (fraction**3 - fraction**2) * duration * yprimef
        )

    @classmethod
    def plot(cls, x, y, title="", xlabel="", ylabel=""):
        """Plot physical initial-guess values against time.

        Args:
            x (array_like): array on the horizontal axis of the plot
            y (array_like): array on the vertical axis of the plot
            title (str, optional): title
            xlabel (str, optional): xlabel
            ylabel (str, optional): ylabel

        """
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(x, y, "-o")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()


class Condition:
    """Accumulate dimensionless equality or inequality residuals.

    Equality residuals must be zero; inequality residuals must be nonnegative.
    Scalar and one-dimensional inputs follow NumPy broadcasting rules. A
    positive reference unit scales each residual. Prefer problem.condition()
    inside callbacks so automatic differentiation selects the correct backend.

    Args:
        length (int): Number of initial zero residuals, normally zero.
        backend (module): NumPy-compatible module, supplied by the callback view.

    Examples:
        >>> result = Condition()
        >>> result.equal(5, 5)
        >>> result()
        array([0.])

        For physical bounds 0 <= x <= 100 in an inequality callback, call
        result.lower_bound(x, 0) and result.upper_bound(x, 100).
    """

    def __init__(self, length=0, *, backend=np):
        self._backend = backend
        self._chunks = [backend.zeros(integer(length, "length"))]

    def add(self, arg, unit=1.0):
        """Append scalar or one-dimensional residuals using a positive scale."""
        value = self._backend.atleast_1d(arg)
        if self._backend is np:
            value = vector(value, "condition")
        value = value / positive_scale(unit)
        self._chunks.append(value)

    def equal(self, arg1, arg2, unit=1.0):
        """Append (arg1 - arg2)/unit; enforces arg1 = arg2 in an equality callback.

        Args:
            arg1 (float or array_like): Left side of the equation.
            arg2 (float or array_like): Right side of the equation.
            unit (float): Positive reference unit for the residual.
        """
        arg = self._backend.array(arg1) - self._backend.array(arg2)
        self.add(arg, unit)

    def lower_bound(self, arg1, arg2, unit=1.0):
        """Append (arg1 - arg2)/unit; enforces arg1 >= arg2 in inequalities.

        Args:
            arg1 (float or array_like): Physical values to constrain.
            arg2 (float or array_like): Physical lower bounds.
            unit (float): Positive reference unit for the residual.
        """
        arg = self._backend.array(arg1) - self._backend.array(arg2)
        self.add(arg, unit)

    def upper_bound(self, arg1, arg2, unit=1.0):
        """Append (arg2 - arg1)/unit; enforces arg1 <= arg2 in inequalities.

        Args:
            arg1 (float or array_like): Physical values to constrain.
            arg2 (float or array_like): Physical upper bounds.
            unit (float): Positive reference unit for the residual.
        """
        arg = self._backend.array(arg2) - self._backend.array(arg1)
        self.add(arg, unit)

    def change_value(self, index, value):
        """Replace an existing dimensionless residual by its flattened index."""
        self()[index] = value

    def __call__(self):
        if len(self._chunks) > 1:
            self._chunks = [self._backend.concatenate(self._chunks)]
        return self._chunks[0]


class Dynamics:
    r"""Scaled state derivatives at a section's collocation nodes.

    Assign physical dx/dt values with dx[state] = values. Calling dx() converts
    them to d(x/unit_states)/d(t/unit_time), flattened in state order for the
    LGL residual. Scalars are broadcast to every node. Unassigned states have
    zero derivative. For pointwise models that also support independent-grid
    error checks and structured derivatives, use DynamicsModel.

    Examples:
        Dynamics class must be used in dynamics function.
        It is an example of the equation of motion of thrust and free fall.
        Thrust is controllable.

        $$
        \dot{x} = v, \qquad \dot{v} = T/m - g
        $$

        >>> def dynamics(prob, obj, section):
        >>>     x = prob.states(0, section)
        >>>     v = prob.states(1, section)
        >>>     T = prob.controls(0, section)
        >>>     g = 9.8
        >>>     m = 1.0
        >>>     dx = Dynamics(prob, section)
        >>>     dx[0] = v
        >>>     dx[1] = T / m - g
        >>>     return dx()
    """

    def __init__(self, prob, section=0):
        self.section = index(section, prob.number_of_section, "section")
        self.number_of_state = prob.number_of_states[self.section]
        self.unit_states = prob.unit_states
        self.unit_time = prob.unit_time
        self._values = np.zeros((self.number_of_state, prob.nodes[self.section]))

    def __getitem__(self, key):
        return self._values[index(key, self.number_of_state, "state")]

    def __setitem__(self, key, value):
        key = index(key, self.number_of_state, "state")
        value = np.asarray(value, dtype=float)
        if value.ndim == 0:
            value = np.full(self._values.shape[1], value)
        self._values[key] = vector(value, "dynamics", self._values.shape[1])

    def __call__(self):
        return scaled_derivatives(
            self._values, self.unit_states[self.section], self.unit_time
        )
