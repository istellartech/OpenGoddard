# -*- coding: utf-8 -*-
# Copyright (c) 2017 Interstellar Technologies Inc. All Rights Reserved.
# Authors : Takahiro Inagawa, Kazuki Sakaki
#
# Lisence : MIT Lisence
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

"""
OpenGoddard.optimize - Optimal Trajectories module with PseudoSpectral Method
"""

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, special

from ._grid import interfaces, normalize_mesh, rule
from ._helpers import Condition as Condition
from ._helpers import Dynamics as Dynamics
from ._helpers import Guess as Guess
from ._spectral import legendre_rule
from ._validation import bounds_pair, index, integer, positive_scale, vector

__all__ = ["Condition", "Dynamics", "Guess", "Problem"]


def _node_index(value, size):
    return index(value, size, "node")


class Problem:
    """OpenGoddard Problem class.

    Args:
        time_init (sequence of float): Strictly increasing physical phase times.
        nodes (sequence of int): LGL node counts, at least two per phase.
        number_of_states (sequence of int): Positive state counts per phase.
        number_of_controls (sequence of int): Nonnegative control counts per phase.
        maxIterator (int): Maximum SLSQP attempts after iteration limits.
        method (str): Only ``"LGL"`` is supported for optimal control.
        mesh (optional): Per-section lists of ``(left, right, nodes)`` elements
            partitioning the phase coordinate [-1, 1]. Logical section numbers
            do not change. Node counts include both sides of internal interfaces.

    All state, control and time getters and setters use physical units. Only
    ``p`` and ``bounds`` use scaled units. Setters for units preserve physical
    guesses and bounds. Node, state, control and section indices follow Python's
    negative indexing convention and raise IndexError outside their range.

    Attributes:
        nodes (int) : time nodes.
        number_of_states (int) : number of states.
        number_of_controls (int) : number of controls
        number_of_section (int) : number of section
        number_of_param (int) : number of inner variables
        div (list) : division point of inner variables
        tau : Gauss nodes
        w : weights of Gaussian quadrature
        D :  differentiation matrix of Gaussian quadrature
        time : time
        maxIterator (int) : max iterator
        time_all_section : all section time
        unit_states (list of float) : canonical unit of states
        unit_controls (list of float) : canonical unit of controls
        unit_time (float) : canonical unit of time
        p ((N,) ndarray) : inner variables for optimization
        dynamics (function) : function list, list of function of dynamics
        knot_states_smooth (list of True/False): list of states are smooth on phase knots
        cost (function) : cost function
        running_cost (function, optional): Physical-time integrand at every node.
        cost_derivative (function, optional): Complete objective gradient in ``p``.
        equality (function) : (default = None)
        inequality (function) : (default = None)

    """

    def _LegendreFunction(self, x, n):
        return special.eval_legendre(n, x)

    def _LegendreDerivative(self, x, n):
        if n == 0:
            return np.zeros_like(x, dtype=float)
        return (n + 1) / 2 * special.eval_jacobi(n - 1, 1, 1, x)

    def _nodes_LG(self, n):
        return legendre_rule("LG", n)[0]

    def _weight_LG(self, n):
        return legendre_rule("LG", n)[1]

    def _differentiation_matrix_LG(self, n):
        return legendre_rule("LG", n)[2]

    def method_LG(self, n):
        """Legendre-Gauss Pseudospectral method
        Gauss nodes are roots of :math:`P_n(x)`.

        Args:
            n (int) : number of nodes

        Returns:
            ndarray, ndarray, ndarray : nodes, weight, differentiation_matrix

        """
        return legendre_rule("LG", n)

    def _nodes_LGR(self, n):
        return legendre_rule("LGR", n)[0]

    def _weight_LGR(self, n):
        return legendre_rule("LGR", n)[1]

    def _differentiation_matrix_LGR(self, n):
        return legendre_rule("LGR", n)[2]

    def method_LGR(self, n):
        """Legendre-Gauss-Radau Pseudospectral method
        Gauss-Radau nodes are roots of :math:`P_n(x) + P_{n-1}(x)`.

        Args:
            n (int) : number of nodes

        Returns:
            ndarray, ndarray, ndarray : nodes, weight, differentiation_matrix

        """
        return legendre_rule("LGR", n)

    def _nodes_LGL(self, n):
        return legendre_rule("LGL", n)[0]

    def _weight_LGL(self, n):
        return legendre_rule("LGL", n)[1]

    def _differentiation_matrix_LGL(self, n):
        return legendre_rule("LGL", n)[2]

    def method_LGL(self, n):
        """Legendre-Gauss-Lobatto Pseudospectral method
        Gauss-Lobatto nodes are roots of :math:`P'_{n-1}(x)` and -1, 1.

        Args:
            n (int) : number of nodes

        Returns:
            ndarray, ndarray, ndarray : nodes, weight, differentiation_matrix

        References:
            Fariba Fahroo and I. Michael Ross. "Advances in Pseudospectral Methods
            for Optimal Control", AIAA Guidance, Navigation and Control Conference
            and Exhibit, Guidance, Navigation, and Control and Co-located Conferences
            http://dx.doi.org/10.2514/6.2008-7309

        """
        return legendre_rule("LGL", n)

    def _make_param_division(self, nodes, number_of_states, number_of_controls):
        prev = 0
        div = []
        for section, node in enumerate(nodes):
            num_param = number_of_states[section] + number_of_controls[section]
            temp = [i * (node) + prev for i in range(1, num_param + 1)]
            prev = temp[-1]
            div.append(temp)
        return div

    def _division_states(self, state, section):
        section = index(section, self.number_of_section, "section")
        state = index(state, self.number_of_states[section], "state")
        start = 0 if section == 0 else self.div[section - 1][-1]
        front = start + state * self.nodes[section]
        return front + self.nodes[section], front

    def _division_controls(self, control, section):
        section = index(section, self.number_of_section, "section")
        control = index(control, self.number_of_controls[section], "control")
        start = 0 if section == 0 else self.div[section - 1][-1]
        front = start + (self.number_of_states[section] + control) * self.nodes[section]
        return front + self.nodes[section], front

    def states(self, state, section):
        """getter specify section states array

        Args:
            state (int) : state number
            section (int) : section number

        Returns:
            states ((N,) ndarray) :
                1-D array of state

        """
        div_back, div_front = self._division_states(state, section)
        return self.p[div_front:div_back] * self.unit_states[section][state]

    def states_all_section(self, state):
        """get states array

        Args:
            state (int) : state number

        Returns:
            states_all_section ((N,) ndarray) :
                1-D array of all section state

        """
        return np.concatenate(
            [self.states(state, section) for section in range(self.number_of_section)]
        )

    def controls(self, control, section):
        """getter specify section controls array

        Args:
            control (int) : control number
            section (int) : section number

        Returns:
            controls (ndarray) :
                1-D array of controls

        """
        div_back, div_front = self._division_controls(control, section)
        return self.p[div_front:div_back] * self.unit_controls[section][control]

    def controls_all_section(self, control):
        """get controls array

        Args:
            control (int) : control number

        Returns:
            controls_all_section ((N, ) ndarray) :
                1-D array of all section control

        """
        return np.concatenate(
            [
                self.controls(control, section)
                for section in range(self.number_of_section)
            ]
        )

    def time_start(self, section):
        """get time at section "start"

        Args:
            section (int) : section

        Returns:
            time_start (int) : time at section start

        """
        section = index(section, self.number_of_section, "section")
        return self.t0 if section == 0 else self.time_final(section - 1)

    def time_final(self, section):
        """get time at section "end"

        Args:
            section (int) : section

        Returns:
            time_final (int) : time at section end

        """
        return self.p[self.index_time_final(section)] * self.unit_time

    def time_final_all_section(self):
        """get time at "end"

        Args:
            section (int) : section

        Returns:
            time_final_all_section (int) : time at end

        """
        tf = []
        for section in range(self.number_of_section):
            tf = tf + [self.time_final(section)]
        return tf

    def set_states(self, state, section, value):
        """set value to state at specific section

        Args:
            state (int) : state
            section (int) : section
            value (int) : value

        """
        div_back, div_front = self._division_states(state, section)
        value = vector(value, "states", div_back - div_front)
        self.p[div_front:div_back] = value / self.unit_states[section][state]

    def set_states_all_section(self, state, value_all_section):
        """set value to state at all section

        Args:
            state (int) : state
            value_all_section (int) : value

        """
        value_all_section = vector(value_all_section, "states", sum(self.nodes))
        div = 0
        for i in range(self.number_of_section):
            value = value_all_section[div : div + self.nodes[i]]
            div = div + self.nodes[i]
            self.set_states(state, i, value)

    def set_controls(self, control, section, value):
        """set value to control at all section

        Args:
            control (int) : control
            section (int) : section
            value (int) : value

        """
        div_back, div_front = self._division_controls(control, section)
        value = vector(value, "controls", div_back - div_front)
        self.p[div_front:div_back] = value / self.unit_controls[section][control]

    def set_controls_all_section(self, control, value_all_section):
        """set value to control at all section

        Args:
            control (int) : control
            value_all_section (int) : value

        """
        value_all_section = vector(value_all_section, "controls", sum(self.nodes))
        div = 0
        for i in range(self.number_of_section):
            value = value_all_section[div : div + self.nodes[i]]
            div = div + self.nodes[i]
            self.set_controls(control, i, value)

    def set_time_final(self, section, value):
        """set value to final time at specific section

        Args:
            section (int) : seciton
            value (float) : value

        """
        if not np.isfinite(value):
            raise ValueError("final time must be finite")
        self.p[self.index_time_final(section)] = value / self.unit_time

    def set_states_bounds(self, state, section, lb, ub):
        """set value to bounds of state at specific section

        Args:
            state (int)   : state
            section (int) : section
            lb (float or None) : lower bound
            ub (float or None) : upper bound

        """
        div_back, div_front = self._division_states(state, section)
        pair = bounds_pair(lb, ub, self.unit_states[section][state])
        self.bounds[div_front:div_back] = [pair] * self.nodes[section]

    def set_states_bounds_all_section(self, state, lb, ub):
        """set value to bounds of state at all sections

        Args:
            state (int)   : state
            lb (float or None) : lower bound
            ub (float or None) : upper bound

        """
        for section in range(self.number_of_section):
            self.set_states_bounds(state, section, lb, ub)

    def set_controls_bounds(self, control, section, lb, ub):
        """set value to bounds of control at specific section

        Args:
            control (int)   : control
            section (int) : section
            lb (float or None) : lower bound
            ub (float or None) : upper bound

        """
        div_back, div_front = self._division_controls(control, section)
        pair = bounds_pair(lb, ub, self.unit_controls[section][control])
        self.bounds[div_front:div_back] = [pair] * self.nodes[section]

    def set_controls_bounds_all_section(self, control, lb, ub):
        """set value to bounds of control at all sections

        Args:
            control (int)   : control
            lb (float or None) : lower bound
            ub (float or None) : upper bound

        """
        for section in range(self.number_of_section):
            self.set_controls_bounds(control, section, lb, ub)

    def set_time_final_bounds(self, section, lb, ub):
        """set value to bounds of time_final at specific section

        Args:
            section (int)   : section
            lb (float or None) : lower bound
            ub (float or None) : upper bound

        """
        pair = bounds_pair(lb, ub, self.unit_time)
        self.bounds[self.index_time_final(section)] = pair

    def time_to_tau(self, time):
        """Map a finite, nonconstant time vector to [-1, 1]."""
        time = vector(time, "time")
        if time.size < 2 or np.ptp(time) == 0:
            raise ValueError("time must span a nonzero interval")
        return 2 * (time - time.min()) / np.ptp(time) - 1

    def time_update(self):
        """get time array after optimization

        Returns:
            time_update : (N,) ndarray
                time array

        """
        self.time = []
        t = self.time_knots()
        for i in range(self.number_of_section):
            self.time.append(
                (t[i + 1] - t[i]) / 2.0 * self.tau[i] + (t[i + 1] + t[i]) / 2.0
            )
            # Shared LGL endpoints must be identical, including after scaling.
            self.time[-1][0] = t[i]
            self.time[-1][-1] = t[i + 1]
        self.time_all_section = np.concatenate(self.time)
        return self.time_all_section

    def time_knots(self):
        """get time at knot point

        Returns:
            time_knots (list) : time at knot point

        """
        return [self.t0] + self.time_final_all_section()

    def index_states(self, state, section, index=None):
        """get index of state at specific section

        Args:
            state (int) : state
            section (int) : section
            index (int, optional) : index

        Returns:
            index_states (int) : index of states
        """
        div_back, div_front = self._division_states(state, section)
        if index is None:
            return div_front
        return div_front + _node_index(index, div_back - div_front)

    def index_controls(self, control, section, index=None):
        div_back, div_front = self._division_controls(control, section)
        if index is None:
            return div_front
        return div_front + _node_index(index, div_back - div_front)

    def index_time_final(self, section):
        section = index(section, self.number_of_section, "section")
        return self.number_of_variables - self.number_of_section + section

    """
    ===========================
    UNIT SCALING ZONE
    ===========================
    """

    def set_unit_states(self, state, section, value):
        """set a canonical unit value to the state at a specific section

        Args:
            state (int) : state
            section (int) : section
            value (float) : value

        """
        back, front = self._division_states(state, section)
        value = positive_scale(value)
        ratio = self.unit_states[section][state] / value
        self.p[front:back] *= ratio
        self.bounds[front:back] = [
            (
                None if pair[0] is None else pair[0] * ratio,
                None if pair[1] is None else pair[1] * ratio,
            )
            for pair in self.bounds[front:back]
        ]
        self.unit_states[section][state] = value

    def set_unit_states_all_section(self, state, value):
        """set a canonical unit value to the state at all sections

        Args:
            state (int) : state
            value (float) : value

        """
        for i in range(self.number_of_section):
            self.set_unit_states(state, i, value)

    def set_unit_controls(self, control, section, value):
        """set a canonical unit value to the control at a specific section

        Args:
            control (int) : control
            section (int) : section
            value (float) : value

        """
        back, front = self._division_controls(control, section)
        value = positive_scale(value)
        ratio = self.unit_controls[section][control] / value
        self.p[front:back] *= ratio
        self.bounds[front:back] = [
            (
                None if pair[0] is None else pair[0] * ratio,
                None if pair[1] is None else pair[1] * ratio,
            )
            for pair in self.bounds[front:back]
        ]
        self.unit_controls[section][control] = value

    def set_unit_controls_all_section(self, control, value):
        """set a canonical unit value to the control at all sections

        Args:
            control (int) : control
            value (float) : value

        """
        for i in range(self.number_of_section):
            self.set_unit_controls(control, i, value)

    def set_unit_time(self, value):
        """set a canonical unit value to the time

        Args:
            value (float) : value
        """
        value = positive_scale(value)
        ratio = self.unit_time / value
        front = self.number_of_variables - self.number_of_section
        self.p[front:] *= ratio
        self.bounds[front:] = [
            (
                None if pair[0] is None else pair[0] * ratio,
                None if pair[1] is None else pair[1] * ratio,
            )
            for pair in self.bounds[front:]
        ]
        self.unit_time = value
        self.time_update()

    @staticmethod
    def _dummy_func():
        pass

    """ ==============================
    """

    def _equality_constraints(self, obj):
        """Boundary constraints, LGL defects and physical state continuity."""
        blocks = []
        if self.equality is not None:
            blocks.append(vector(self.equality(self, obj), "equality"))
        for section, node in enumerate(self.nodes):
            _, front = self._division_states(0, section)
            back = front + self.number_of_states[section] * node
            state = self.p[front:back].reshape(-1, node)
            derivative = (state @ self.D[section].T).ravel()
            dynamics = self.dynamics[section]
            if dynamics is None:
                raise ValueError("set dynamics for every section")
            dx = vector(dynamics(self, obj, section), "dynamics", derivative.size)
            duration = (
                self.time_final(section) - self.time_start(section)
            ) / self.unit_time
            blocks.append(derivative - duration / 2 * dx)
        for section in range(self.number_of_section):
            for left, right in interfaces(self, section):
                blocks.append(
                    np.array(
                        [
                            (
                                self.states(state, section)[left]
                                - self.states(state, section)[right]
                            )
                            / self.unit_states[section][state]
                            for state in range(self.number_of_states[section])
                        ]
                    )
                )
        for knot, smooth in enumerate(self.knot_states_smooth):
            if (
                not smooth
                or self.number_of_states[knot] != self.number_of_states[knot + 1]
            ):
                continue
            blocks.append(
                np.array(
                    [
                        (self.states(i, knot)[-1] - self.states(i, knot + 1)[0])
                        / self.unit_states[knot][i]
                        for i in range(self.number_of_states[knot])
                    ]
                )
            )
        return np.concatenate(blocks)

    def _inequality_constraints(self, obj):
        # Ordered phase endpoints are necessary for a forward-time trajectory.
        blocks = [np.diff(self.time_knots()) / self.unit_time]
        if self.inequality is not None:
            blocks.append(vector(self.inequality(self, obj), "inequality"))
        return np.concatenate(blocks)

    def _objective(self, obj):
        """Mayer cost plus physical-time quadrature of the Lagrange cost."""
        value = 0.0 if self.cost is None else float(self.cost(self, obj))
        if self.running_cost is not None:
            integrand = vector(
                self.running_cost(self, obj), "running_cost", sum(self.nodes)
            )
            front = 0
            for section, node in enumerate(self.nodes):
                duration = self.time_final(section) - self.time_start(section)
                value += (
                    duration
                    / 2
                    * np.dot(self.w[section], integrand[front : front + node])
                )
                front += node
        if not np.isfinite(value):
            raise ValueError("objective must be finite")
        return value

    def solve(
        self,
        obj: Any = None,
        display_func: Callable[[], None] = _dummy_func,
        *,
        solver: str = "SLSQP",
        derivatives: str = "finite-difference",
        **options: Any,
    ) -> optimize.OptimizeResult:
        """Solve the LGL transcription and return a SciPy OptimizeResult.

        ``solver`` selects SLSQP (default) or sparse ``trust-constr``.
        ``derivatives='finite-difference'`` preserves arbitrary legacy callbacks.
        ``'structured'`` assembles exact spectral blocks and local central
        differences, and ``'autograd'`` uses optional automatic differentiation.
        Both structured modes require DynamicsModel in every section. AD
        callbacks use ``view.array_module`` and ``view.condition()``; they must
        not mutate arrays or coerce traced values to Python floats. With
        trust-constr, AD also supplies matrix-free Hessian-vector products.

        ``dynamics`` must contain a callable for each section; ``equality`` and
        ``inequality`` may be None. ``running_cost`` returns one integrand value
        per node, concatenated in section order, in physical time. The solver
        applies each section's ``(tf - ti) / 2`` quadrature factor.

        ``cost_derivative``, when set, must differentiate the *complete*
        objective (including running cost) with respect to the scaled vector
        ``p``. All SLSQP options, including ``ftol``, ``maxiter``, ``eps`` and
        ``disp``, are forwarded. Concurrent ``workers`` are unsupported because
        callbacks share this mutable Problem instance.

        Solver-specific options are forwarded: SLSQP accepts ``ftol``/``eps``;
        trust-constr accepts ``gtol``/``xtol``/``barrier_tol``. They are not
        interchangeable. ``cost_derivative`` is used in finite-difference and
        structured modes; autograd differentiates the complete objective itself.

        ``maxIterator`` bounds restarts after an iteration limit (SLSQP status 9,
        trust-constr status 0).
        Other failures stop immediately and emit RuntimeWarning. Each call
        resets the restart counter. The final result is also saved in
        ``result`` and every attempt in ``result_history``. ``display_func``
        is called without arguments after each attempt, as in the legacy API.
        ``time`` and ``time_all_section`` are refreshed before every callback.
        """
        if solver not in {"SLSQP", "trust-constr"}:
            raise ValueError("solver must be 'SLSQP' or 'trust-constr'")
        if derivatives not in {"finite-difference", "structured", "autograd"}:
            raise ValueError("invalid derivatives mode")
        if len(self.dynamics) != self.number_of_section or not all(
            callable(func) for func in self.dynamics
        ):
            raise ValueError("set one dynamics callable for every section")
        if self.cost is None and self.running_cost is None:
            raise ValueError("set cost or running_cost before solving")
        for name in (
            "cost",
            "running_cost",
            "cost_derivative",
            "equality",
            "inequality",
        ):
            callback = getattr(self, name)
            if callback is not None and not callable(callback):
                raise TypeError(f"{name} must be callable or None")
        if len(self.knot_states_smooth) != self.number_of_section - 1:
            raise ValueError(
                "knot_states_smooth must have one entry per phase boundary"
            )
        if not callable(display_func):
            raise TypeError("display_func must be callable")
        vector(self.p, "p", self.number_of_variables)
        if len(self.bounds) != self.number_of_variables:
            raise ValueError("bounds must have one pair per optimization variable")
        for lower, upper in self.bounds:
            bounds_pair(lower, upper, 1.0)
        workers = options.pop("workers", None)
        if workers is not None and workers != 1:
            raise ValueError("parallel workers cannot share a mutable Problem")
        if solver == "SLSQP":
            options.setdefault("ftol", 1e-6)
        else:
            options.setdefault("gtol", 1e-8)
            options.setdefault("sparse_jacobian", True)
        options.setdefault("maxiter", 25 if solver == "SLSQP" else 1000)
        options.setdefault("disp", False)
        attempts = integer(self.maxIterator, "maxIterator", 1)

        def wrap(func):
            def evaluate(parameters):
                self.p = parameters
                self.time_update()
                return func(obj)

            return evaluate

        cost_derivative = self.cost_derivative

        def gradient(obj):
            if cost_derivative is None:
                raise ValueError("cost_derivative is not set")
            return vector(
                cost_derivative(self, obj),
                "cost_derivative",
                self.number_of_variables,
            )

        constraints = [
            {"type": "eq", "fun": wrap(self._equality_constraints)},
            {"type": "ineq", "fun": wrap(self._inequality_constraints)},
        ]
        provider = None
        if derivatives != "finite-difference":
            from .derivatives import Derivatives

            provider = Derivatives(self, obj, derivatives)
            for constraint, evaluate in zip(
                constraints, (provider.equality, provider.inequality)
            ):

                def constraint_jacobian(obj, evaluate=evaluate):
                    value = evaluate()
                    return value.toarray() if solver == "SLSQP" else value

                constraint["jac"] = wrap(constraint_jacobian)
        jac = None if cost_derivative is None else wrap(gradient)
        if provider is not None:
            jac = wrap(lambda obj: provider.objective())
        hessian = None
        if solver == "trust-constr":
            if provider is not None and derivatives == "autograd":
                hessian = wrap(lambda obj: provider.hessian())
                for constraint in constraints:

                    def constraint_hessian(
                        parameters, weights, kind=constraint["type"]
                    ):
                        self.p = parameters
                        self.time_update()
                        return provider.hessian(kind, weights)

                    constraint["hess"] = constraint_hessian
            constraints = [
                optimize.NonlinearConstraint(
                    constraint["fun"],
                    0,
                    0 if constraint["type"] == "eq" else np.inf,
                    jac=constraint.get("jac", "2-point"),
                    hess=constraint.get("hess", optimize.BFGS()),
                )
                for constraint in constraints
            ]
        self.iterator = 0
        self.result = None
        self.result_history = []
        for attempt in range(attempts):
            result = optimize.minimize(
                wrap(self._objective),
                self.p,
                bounds=self.bounds,
                constraints=constraints,
                jac=jac,
                hess=hessian,
                method=solver,
                options=options,
            )
            self.p = result.x.copy()
            self.time_update()
            self.iterator = attempt + 1
            self.result = result
            self.result_history.append(result)
            display_func()
            if result.success or result.status != (9 if solver == "SLSQP" else 0):
                break
        if self.result is None:
            raise RuntimeError("solver did not produce a result")
        if not self.result.success:
            warnings.warn(
                f"{solver} failed: {self.result.message}", RuntimeWarning, stacklevel=2
            )
        self.result.solver = solver
        self.result.derivatives = derivatives
        self.result.equality_violation = np.max(
            np.abs(self._equality_constraints(obj)), initial=0
        )
        self.result.inequality_violation = max(
            0.0, -np.min(self._inequality_constraints(obj), initial=0)
        )
        return self.result

    def condition(self, length=0):
        """Build residuals; also supported by the automatic differentiation view."""
        return Condition(length)

    def refine(self, nodes):
        """Return a warm-started copy on a different LGL mesh; see mesh.refine."""
        from .mesh import refine

        return refine(self, nodes)

    def remesh(self, mesh):
        """Return a warm start on per-phase lists of (left, right, nodes) elements.

        Element endpoints are phase coordinates in [-1, 1]. Logical section
        indices remain unchanged. States are continuous at internal interfaces;
        controls have independent left and right limits.
        """
        from .mesh import remesh

        return remesh(self, mesh)

    def estimate_error(self, obj=None, check_factor=2):
        """Estimate between-node ODE and sampled path residuals; see mesh.estimate_error."""
        from .mesh import estimate_error

        return estimate_error(self, obj, check_factor)

    def solve_adaptive(self, obj=None, **options):
        """Refine until mesh tolerance or limits are reached; see mesh.solve_adaptive."""
        from .mesh import solve_adaptive

        return solve_adaptive(self, obj, **options)

    def __init__(
        self,
        time_init,
        nodes,
        number_of_states,
        number_of_controls,
        maxIterator=100,
        method="LGL",
        *,
        mesh=None,
    ):
        # LG and LGR quadrature utilities remain available, but their endpoint
        # transcriptions differ from LGL and are not implemented by this solver.
        if method != "LGL":
            raise ValueError(
                "Problem supports only method='LGL'; use method_LG/method_LGR for quadrature"
            )
        self.nodes = [integer(n, "nodes", 2) for n in nodes]
        self.number_of_states = [
            integer(n, "number_of_states", 1) for n in number_of_states
        ]
        self.number_of_controls = [
            integer(n, "number_of_controls") for n in number_of_controls
        ]
        self.number_of_section = len(self.nodes)
        if self.number_of_section == 0:
            raise ValueError("at least one section is required")
        if (
            len(self.number_of_states) != self.number_of_section
            or len(self.number_of_controls) != self.number_of_section
        ):
            raise ValueError(
                "nodes, number_of_states and number_of_controls must have equal lengths"
            )
        times = vector(time_init, "time_init", self.number_of_section + 1)
        if np.any(np.diff(times) <= 0):
            raise ValueError("time_init must be strictly increasing")
        self.time_init = times.tolist()
        self.t0 = float(times[0])
        self.method = method
        self.maxIterator = integer(maxIterator, "maxIterator", 1)
        self.iterator = 0
        self.div = self._make_param_division(
            self.nodes, self.number_of_states, self.number_of_controls
        )
        self.number_of_param = np.array(self.number_of_states) + self.number_of_controls
        self.number_of_variables = (
            int(np.dot(self.number_of_param, self.nodes)) + self.number_of_section
        )
        self.mesh = normalize_mesh(self.nodes, mesh)
        rules = [rule(elements) for elements in self.mesh]
        self.tau = [rule[0] for rule in rules]
        self.w = [rule[1] for rule in rules]
        self.D = [rule[2] for rule in rules]
        self.unit_states = [[1.0] * n for n in self.number_of_states]
        self.unit_controls = [[1.0] * n for n in self.number_of_controls]
        self.unit_time = 1.0
        self.p = np.zeros(self.number_of_variables)
        self.p[-self.number_of_section :] = times[1:]
        self.bounds = [(None, None)] * self.number_of_variables
        self.dynamics: list[Callable | None] = [None] * self.number_of_section
        self.knot_states_smooth = [True] * (self.number_of_section - 1)
        self.cost: Callable[[Problem, Any], float] | None = None
        self.running_cost: Callable[[Problem, Any], NDArray[np.float64]] | None = None
        self.cost_derivative: Callable[[Problem, Any], NDArray[np.float64]] | None = (
            None
        )
        self.equality: Callable[[Problem, Any], NDArray[np.float64]] | None = None
        self.inequality: Callable[[Problem, Any], NDArray[np.float64]] | None = None
        self.result: optimize.OptimizeResult | None = None
        self.result_history: list[optimize.OptimizeResult] = []
        self.time_update()

    def __repr__(self):
        s = "---- parameter ----" + "\n"
        s += "nodes = " + str(self.nodes) + "\n"
        s += "number of states    = " + str(self.number_of_states) + "\n"
        s += "number of controls  = " + str(self.number_of_controls) + "\n"
        s += "number of sections  = " + str(self.number_of_section) + "\n"
        s += "number of variables = " + str(self.number_of_variables) + "\n"
        s += "---- algorithm ----" + "\n"
        s += "max iteration = " + str(self.maxIterator) + "\n"
        s += "---- function  ----" + "\n"
        s += "dynamics        = " + str(self.dynamics) + "\n"
        s += "cost            = " + str(self.cost) + "\n"
        s += "cost_derivative = " + str(self.cost_derivative) + "\n"
        s += "equality        = " + str(self.equality) + "\n"
        s += "inequality      = " + str(self.inequality) + "\n"
        s += "knot_states_smooth = " + str(self.knot_states_smooth) + "\n"

        return s

    def to_csv(self, filename="OpenGoddard_output.csv", delimiter=","):
        """output states, controls and time to csv file

        Args:
            filename (str, optional) : csv filename
            delimiter : (str, optional) : default ","

        """
        # A phase may have a different model (e.g. after stage separation).
        # Missing state/control columns are explicitly represented by NaN.
        states = max(self.number_of_states)
        controls = max(self.number_of_controls)
        result = np.full((sum(self.nodes), 1 + states + controls), np.nan)
        result[:, 0] = self.time_update()
        front = 0
        for section, node in enumerate(self.nodes):
            rows = slice(front, front + node)
            for state in range(self.number_of_states[section]):
                result[rows, 1 + state] = self.states(state, section)
            for control in range(self.number_of_controls[section]):
                result[rows, 1 + states + control] = self.controls(control, section)
            front += node
        names = (
            ["time"]
            + [f"state{i}" for i in range(states)]
            + [f"control{i}" for i in range(controls)]
        )
        np.savetxt(filename, result, delimiter=delimiter, header=delimiter.join(names))

    def plot(self, title_comment=""):
        """plot inner variables that to be optimized

        Args:
            title_comment (str) : string for title

        """
        import matplotlib.pyplot as plt

        plt.figure()
        plt.title("OpenGoddard inner variables" + title_comment)
        plt.plot(self.p, "o")
        plt.xlabel("variables")
        plt.ylabel("value")
        for section in range(self.number_of_section):
            for line in self.div[section]:
                plt.axvline(line, color="C%d" % ((section + 1) % 6), alpha=0.5)
        plt.grid()
