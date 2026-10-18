"""Initial guesses, residual builders and scaled dynamics for optimal control."""

# Copyright (c) 2017 Interstellar Technologies Inc.
# SPDX-License-Identifier: MIT

import numpy as np

from ._validation import index, integer, positive_scale, vector


def _guess_time(time):
    time = vector(time, "time")
    if time.size < 2 or time[-1] <= time[0] or np.any(np.diff(time) < 0):
        raise ValueError("time must be nondecreasing and span a nonzero interval")
    return time


class Guess:
    """Class for initial value guess for optimization.

    Collection of class methods
    """

    @classmethod
    def zeros(cls, time):
        """return zeros that array size is same as time length

        Args:
            time (array_like) :

        Returns:
            (N, ) ndarray
        """
        return np.zeros(len(time))

    @classmethod
    def constant(cls, time, const):
        """return constant values that array size is same as time length

        Args:
            time (array_like) :
            const (float) : set value

        Returns:
            (N, ) ndarray

        """
        return np.ones(len(time)) * const

    @classmethod
    def linear(cls, time, y0, yf):
        """return linear function values that array size is same as time length

        Args:
            time (array_like) : time
            y0 (float): initial value
            yf (float): final value

        Returns:
            (N, ) ndarray

        """
        time = _guess_time(time)
        fraction = (time - time[0]) / (time[-1] - time[0])
        return y0 + (yf - y0) * fraction

    @classmethod
    def cubic(cls, time, y0, yprime0, yf, yprimef):
        """return cubic function values that array size is same as time length

        Args:
            time (array_like) : time
            y0 (float) : initial value
            yprime0 (float) : slope of initial value
            yf (float) : final value
            yprimef (float) : slope of final value

        Returns:
            (N, ) ndarray

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
        """plot wrappper

        Args:
            x (array_like) : array on the horizontal axis of the plot
            y (array_like) : array on the vertical axis of the plot
            title (str, optional) : title
            xlabel (str, optional) : xlabel
            ylabel (str, optional) : ylabel

        """
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(x, y, "-o")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()


class Condition:
    """OpenGoddard.optimize Condition class

    Accumulate scalar and vector residuals without repeated array reallocations.

    Examples:
        for examples in equality function.

        Initial condtion : x[0] = 0.0

        Termination Condition : x[-1] = 100

        >>> result = Condition()
        >>> result.equal(x[0], 0.0)
        >>> result.equal(x[-1], 100)
        >>> return result()

        for examples in inequality function

        Inequation condtion : 0.0 <= x <= 100

        >>> result = Condition()
        >>> result.lower_bound(x, 0.0)
        >>> result.upper_bound(x, 100)
        >>> return result()

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
        """add equation constraint condition in Problem equality function

        arg1 = arg2

        Args:
            arg1 (float or array_like) : right side of the equation
            arg2 (float or array_like) : left side of the equation
            unit (float, optional) : argX / unit (default : 1.0)

        Notes:
            It must be used in equality function.
        """
        arg = self._backend.array(arg1) - self._backend.array(arg2)
        self.add(arg, unit)

    def lower_bound(self, arg1, arg2, unit=1.0):
        """add inequation constraint condition in Problem inequality function

        arg1 >= arg2

        Args:
            arg1 (array like) : arg1 is greater than or equal to arg2
            arg2 (float or array like) : arg1 is greater than or equal to arg2
            unit (float, optional) : argX / unit (default : 1.0)

        Notes:
            It must be used in inequality function.
        """
        arg = self._backend.array(arg1) - self._backend.array(arg2)
        self.add(arg, unit)

    def upper_bound(self, arg1, arg2, unit=1.0):
        """add inequation constraint condition in Problem inequality function

        arg1 <= arg2

        Args:
            arg1 (array like) : arg1 is less than or equal to arg2
            arg2 (float or array like) : arg1 is less than or equal to arg2
            unit (float, optional) : argX / unit (default : 1.0)

        Notes:
            It must be used in inequality function.
        """
        arg = self._backend.array(arg2) - self._backend.array(arg1)
        self.add(arg, unit)

    def change_value(self, index, value):
        self()[index] = value

    def __call__(self):
        if len(self._chunks) > 1:
            self._chunks = [self._backend.concatenate(self._chunks)]
        return self._chunks[0]


class Dynamics:
    r"""Scaled state derivatives at a section's collocation nodes.

    thin wrapper for dynamics function.
    Behave like a dictionary type.

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
        scale = self.unit_time / np.asarray(self.unit_states[self.section])
        return (self._values * scale[:, None]).ravel()
