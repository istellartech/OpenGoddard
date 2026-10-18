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

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, special
from typing_extensions import override

from ._grid import normalize_mesh, rule
from ._helpers import Condition as Condition
from ._helpers import Dynamics as Dynamics
from ._helpers import Guess as Guess
from ._spectral import legendre_rule
from ._transcription import equality, inequality, objective
from ._validation import bounds_pair, index, integer, positive_scale, vector
from ._variables import VariableLayout, node_times, rescale

__all__ = ["Condition", "Dynamics", "Guess", "Problem"]


def _node_index(value, size: int) -> int:
    return index(value, size, "node")


class Problem:
    """Define a physical optimal-control problem and solve its LGL discretization.

    Args:
        time_init (sequence of float): Strictly increasing physical phase times.
        nodes (sequence of int): LGL node counts, at least two per phase.
        number_of_states (sequence of int): Positive state counts per phase.
        number_of_controls (sequence of int): Nonnegative control counts per phase.
        maxIterator (int): Maximum solver attempts, including the first attempt.
            Another attempt is made only after an iteration-limit status.
            ``solve(maxiter=...)`` limits iterations within each attempt.
        method (str): Only ``"LGL"`` is supported for optimal control.
        mesh (optional): Per-section lists of ``(left, right, nodes)`` elements
            partitioning the phase coordinate [-1, 1]. Logical section numbers
            do not change. Node counts include both sides of internal interfaces.

    All state, control and time getters and setters use physical units. Only
    ``p`` and ``bounds`` use scaled units. Setters for units preserve physical
    guesses and bounds. Node, state, control and section indices follow Python's
    negative indexing convention and raise IndexError outside their range.

    Attributes:
        nodes (list[int]): Node counts per phase, including duplicated interfaces.
        number_of_states (list[int]): State counts per phase.
        number_of_controls (list[int]): Control counts per phase.
        number_of_section (int): Number of logical phases.
        number_of_param (ndarray): State plus control counts per phase.
        number_of_variables (int): Length of p, including phase end times.
        div (list[list[int]]): Exclusive ends of state/control blocks in p.
        tau (list[ndarray]): LGL phase coordinates in [-1, 1].
        w (list[ndarray]): Quadrature weights in phase coordinates.
        D (list[ndarray]): Differentiation matrices in phase coordinates.
        time (list[ndarray]): Physical node times, derived from current p.
        time_all_section (ndarray): Times concatenated in phase order.
        unit_states (list[list[float]]): State reference units per phase.
        unit_controls (list[list[float]]): Control reference units per phase.
        unit_time (float): Time reference unit.
        p (ndarray): Scaled states, controls and phase end times, shape (N,).
        bounds (list[tuple]): Scaled lower/upper bounds for every entry of p.
        dynamics (list[Callable]): One scaled-derivative callback per phase,
            called as ``dynamics(problem, obj, section)``. Use Dynamics to
            assemble physical derivatives, or DynamicsModel for a pointwise
            physical model supporting derivatives and error estimation.
        knot_states_smooth (list[bool]): Enforce continuity at phase knots
            with equal state counts. Disable it when supplying jump conditions.
        cost (Callable | None): Terminal cost, called as ``cost(problem, obj)``.
        running_cost (Callable | None): Physical-time integrand, shape
            (sum(nodes),), called as ``running_cost(problem, obj)``.
        cost_derivative (function, optional): Complete objective gradient in ``p``.
        equality (Callable | None): One-dimensional residuals equal to zero.
        inequality (Callable | None): One-dimensional residuals >= 0.
        array_module: NumPy; differentiable callback views supply autograd.numpy.
        metadata (dict): User metadata, shallow-copied on refine/remesh.
        result (OptimizeResult | None): Latest solver result, or None before solve.
        result_history (list[OptimizeResult]): Attempts of the latest solve.

    Callback views share states/controls (including all-section getters),
    time_start/time_final, time_knots, time/time_all_section, dimensions, units,
    array_module and condition(). Use this reading interface for portable
    callbacks; mutations and direct p indices do not support path sampling.
    Use unit setters and refine/remesh rather than editing configuration lists.
    Construct a new Problem to change state/control counts. Refine/remesh create
    a new mesh and reset solver results.
    """

    array_module = np

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
                "Problem supports only method='LGL'; "
                "use method_LG/method_LGR for quadrature"
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
        self._layout = VariableLayout(
            self.nodes, self.number_of_states, self.number_of_controls
        )
        # Preserve the legacy division table for inspecting/plotting p.
        self.div = [list(phase) for phase in self._layout.divisions]
        self.number_of_param = np.array(self.number_of_states) + self.number_of_controls
        self.number_of_variables = self._layout.size
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
        self.dynamics: list[
            Callable[[Problem, Any, int], NDArray[np.float64]] | None
        ] = [None] * self.number_of_section
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
        self.metadata: dict[str, Any] = {}
        self._core_attributes = frozenset(vars(self)) | {"_core_attributes"}

    def _make_param_division(self, nodes, number_of_states, number_of_controls):
        return VariableLayout(nodes, number_of_states, number_of_controls).divisions

    def _division_states(self, state, section) -> tuple[int, int]:
        rows = self._layout.state_slice(state, section)
        return int(rows.stop), int(rows.start)

    def _division_controls(self, control, section) -> tuple[int, int]:
        rows = self._layout.control_slice(control, section)
        return int(rows.stop), int(rows.start)

    def states(self, state: int, section: int) -> NDArray[np.float64]:
        """Return one state's physical values in a phase.

        Args:
            state: Variable index, supporting negative indices.
            section: Phase index, supporting negative indices.

        Returns:
            ndarray: Independent array of shape (nodes[section],). Both sides of
                internal element interfaces are present. Use set_states() to edit.
        """
        div_back, div_front = self._division_states(state, section)
        return self.p[div_front:div_back] * self.unit_states[section][state]

    def states_all_section(self, state: int) -> NDArray[np.float64]:
        """Return a state's physical values concatenated in phase order.

        Args:
            state: Variable index, which must exist in every phase.

        Returns:
            ndarray: Shape (sum(nodes),), including duplicated interface nodes.
        """
        return np.concatenate(
            [self.states(state, section) for section in range(self.number_of_section)]
        )

    def controls(self, control: int, section: int) -> NDArray[np.float64]:
        """Return one control's physical values in a phase.

        Args:
            control: Variable index, supporting negative indices.
            section: Phase index, supporting negative indices.

        Returns:
            ndarray: Independent array of shape (nodes[section],). Both sides of
                internal element interfaces are present. Use set_controls() to edit.
        """
        div_back, div_front = self._division_controls(control, section)
        return self.p[div_front:div_back] * self.unit_controls[section][control]

    def controls_all_section(self, control: int) -> NDArray[np.float64]:
        """Return a control's physical values concatenated in phase order.

        Args:
            control: Variable index, which must exist in every phase.

        Returns:
            ndarray: Shape (sum(nodes),), including duplicated interface nodes.
        """
        return np.concatenate(
            [
                self.controls(control, section)
                for section in range(self.number_of_section)
            ]
        )

    def time_start(self, section: int) -> float:
        """Return a phase's physical start time.

        Args:
            section: Phase index; negative indices count from the last phase.

        Returns:
            Physical time. The first phase starts at the fixed t0; later phases
                start at the preceding phase's current end time.
        """
        section = index(section, self.number_of_section, "section")
        return self.t0 if section == 0 else self.time_final(section - 1)

    def time_final(self, section: int) -> float:
        """Return a phase's physical end time from the current scaled p.

        Args:
            section: Phase index; negative indices count from the last phase.

        Returns:
            Physical time, equal to p[index_time_final(section)] * unit_time.
        """
        return float(self.p[self.index_time_final(section)] * self.unit_time)

    def time_final_all_section(self) -> list[float]:
        """Return physical phase end times in phase order."""
        return [self.time_final(section) for section in range(self.number_of_section)]

    def set_states(self, state: int, section: int, value: ArrayLike) -> None:
        """Set one state's initial guess in physical units.

        Args:
            state: Variable index, supporting negative indices.
            section: Phase index, supporting negative indices.
            value: Finite one-dimensional values of shape (nodes[section],).

        Raises:
            ValueError: Values have the wrong shape or are not finite.
        """
        div_back, div_front = self._division_states(state, section)
        value = vector(value, "states", div_back - div_front)
        self.p[div_front:div_back] = value / self.unit_states[section][state]

    def set_states_all_section(self, state: int, value_all_section: ArrayLike) -> None:
        """Set a state's physical initial guess across all phases.

        Args:
            state: Variable index, which must exist in every phase.
            value_all_section: Finite values of shape (sum(nodes),), in phase
                order and including duplicated interface nodes.
        """
        value_all_section = vector(value_all_section, "states", sum(self.nodes))
        div = 0
        for i in range(self.number_of_section):
            value = value_all_section[div : div + self.nodes[i]]
            div = div + self.nodes[i]
            self.set_states(state, i, value)

    def set_controls(self, control: int, section: int, value: ArrayLike) -> None:
        """Set one control's initial guess in physical units.

        Args:
            control: Variable index, supporting negative indices.
            section: Phase index, supporting negative indices.
            value: Finite one-dimensional values of shape (nodes[section],).

        Raises:
            ValueError: Values have the wrong shape or are not finite.
        """
        div_back, div_front = self._division_controls(control, section)
        value = vector(value, "controls", div_back - div_front)
        self.p[div_front:div_back] = value / self.unit_controls[section][control]

    def set_controls_all_section(
        self, control: int, value_all_section: ArrayLike
    ) -> None:
        """Set a control's physical initial guess across all phases.

        Args:
            control: Variable index, which must exist in every phase.
            value_all_section: Finite values of shape (sum(nodes),), in phase
                order and including duplicated interface nodes.
        """
        value_all_section = vector(value_all_section, "controls", sum(self.nodes))
        div = 0
        for i in range(self.number_of_section):
            value = value_all_section[div : div + self.nodes[i]]
            div = div + self.nodes[i]
            self.set_controls(control, i, value)

    def set_time_final(self, section: int, value: float) -> None:
        """Set a phase's end-time guess in physical units.

        Args:
            section: Phase index, supporting negative indices.
            value: Finite physical end time. Times are immediately reflected in
                time and time_all_section. Forward ordering is enforced at solve.

        Raises:
            ValueError: The end time is not finite.
        """
        if not np.isfinite(value):
            raise ValueError("final time must be finite")
        self.p[self.index_time_final(section)] = value / self.unit_time

    def set_states_bounds(
        self, state: int, section: int, lb: float | None, ub: float | None
    ) -> None:
        """Set constant physical box bounds at every node of a phase.

        Args:
            state: Variable index, supporting negative indices.
            section: Phase index, supporting negative indices.
            lb: Physical lower bound, or None for no lower bound.
            ub: Physical upper bound, or None for no upper bound.

        Notes:
            Bounds constrain mesh nodes. Use solve_adaptive() with a supported
            model to check for violations between nodes.
        """
        div_back, div_front = self._division_states(state, section)
        pair = bounds_pair(lb, ub, self.unit_states[section][state])
        self.bounds[div_front:div_back] = [pair] * self.nodes[section]

    def set_states_bounds_all_section(
        self, state: int, lb: float | None, ub: float | None
    ) -> None:
        """set value to bounds of state at all sections

        Args:
            state (int): state
            lb (float or None): lower bound
            ub (float or None): upper bound

        """
        for section in range(self.number_of_section):
            self.set_states_bounds(state, section, lb, ub)

    def set_controls_bounds(
        self, control: int, section: int, lb: float | None, ub: float | None
    ) -> None:
        """Set constant physical box bounds at every node of a phase.

        Args:
            control: Variable index, supporting negative indices.
            section: Phase index, supporting negative indices.
            lb: Physical lower bound, or None for no lower bound.
            ub: Physical upper bound, or None for no upper bound.

        Notes:
            Bounds constrain mesh nodes. Use solve_adaptive() with a supported
            model to check for violations between nodes.
        """
        div_back, div_front = self._division_controls(control, section)
        pair = bounds_pair(lb, ub, self.unit_controls[section][control])
        self.bounds[div_front:div_back] = [pair] * self.nodes[section]

    def set_controls_bounds_all_section(
        self, control: int, lb: float | None, ub: float | None
    ) -> None:
        """set value to bounds of control at all sections

        Args:
            control (int): control
            lb (float or None): lower bound
            ub (float or None): upper bound

        """
        for section in range(self.number_of_section):
            self.set_controls_bounds(control, section, lb, ub)

    def set_time_final_bounds(
        self, section: int, lb: float | None, ub: float | None
    ) -> None:
        """set value to bounds of time_final at specific section

        Args:
            section (int): section
            lb (float or None): lower bound
            ub (float or None): upper bound

        """
        pair = bounds_pair(lb, ub, self.unit_time)
        self.bounds[self.index_time_final(section)] = pair

    def time_to_tau(self, time):
        """Map a finite, nonconstant time vector to [-1, 1]."""
        time = vector(time, "time")
        if time.size < 2 or np.ptp(time) == 0:
            raise ValueError("time must span a nonzero interval")
        return 2 * (time - np.min(time)) / np.ptp(time) - 1

    @property
    def time(self) -> list[NDArray[np.float64]]:
        """Physical node times per phase, derived from the current p.

        Returned arrays are independent values. Change phase end times with
        set_time_final(); editing these arrays does not change the problem.
        Both sides of phase and element interfaces are included.
        """
        return node_times(self)

    @property
    def time_all_section(self) -> NDArray[np.float64]:
        """Physical node times concatenated in phase order, including interfaces."""
        return np.concatenate(self.time)

    def time_update(self) -> NDArray[np.float64]:
        """Return current physical node times concatenated in phase order.

        Kept for compatibility. Times are derived from p on access, so explicit
        refreshes are unnecessary even after direct changes to p.
        """
        return self.time_all_section

    def time_knots(self) -> list[float]:
        """Return physical phase boundaries [t0, tf0, tf1, ...]."""
        return [self.t0] + self.time_final_all_section()

    def index_states(self, state: int, section: int, index: int | None = None) -> int:
        """Return the index of a state node in the scaled vector p.

        Args:
            state: Variable index, supporting negative indices.
            section: Phase index, supporting negative indices.
            index: Local node index, supporting negative indices; None selects
                the first node.

        Returns:
            int: Index into p, whose values use reference-unit scaling.
        """
        div_back, div_front = self._division_states(state, section)
        if index is None:
            return div_front
        return div_front + _node_index(index, div_back - div_front)

    def index_controls(
        self, control: int, section: int, index: int | None = None
    ) -> int:
        """Return the index of a control node in the scaled vector p.

        Args:
            control: Variable index, supporting negative indices.
            section: Phase index, supporting negative indices.
            index: Local node index, supporting negative indices; None selects
                the first node.

        Returns:
            int: Index into p, whose values use reference-unit scaling.
        """
        div_back, div_front = self._division_controls(control, section)
        if index is None:
            return div_front
        return div_front + _node_index(index, div_back - div_front)

    def index_time_final(self, section: int) -> int:
        """Return the index of a phase end time in the scaled vector p."""
        return self._layout.time_index(section)

    def set_unit_states(self, state: int, section: int, value: float) -> None:
        """Change a state's reference unit, preserving physical values and bounds.

        Args:
            state: Variable index, supporting negative indices.
            section: Phase index, supporting negative indices.
            value: Finite, strictly positive reference unit.
        """
        back, front = self._division_states(state, section)
        value = positive_scale(value)
        rescale(self, slice(front, back), self.unit_states[section][state], value)
        self.unit_states[section][state] = value

    def set_unit_states_all_section(self, state: int, value: float) -> None:
        """set a canonical unit value to the state at all sections

        Args:
            state (int): state
            value (float): value

        """
        for i in range(self.number_of_section):
            self.set_unit_states(state, i, value)

    def set_unit_controls(self, control: int, section: int, value: float) -> None:
        """Change a control's reference unit, preserving physical values and bounds.

        Args:
            control: Variable index, supporting negative indices.
            section: Phase index, supporting negative indices.
            value: Finite, strictly positive reference unit.
        """
        back, front = self._division_controls(control, section)
        value = positive_scale(value)
        rescale(self, slice(front, back), self.unit_controls[section][control], value)
        self.unit_controls[section][control] = value

    def set_unit_controls_all_section(self, control: int, value: float) -> None:
        """set a canonical unit value to the control at all sections

        Args:
            control (int): control
            value (float): value

        """
        for i in range(self.number_of_section):
            self.set_unit_controls(control, i, value)

    def set_unit_time(self, value: float) -> None:
        """set a canonical unit value to the time

        Args:
            value (float): value
        """
        value = positive_scale(value)
        rescale(self, slice(self._layout.time_offset, None), self.unit_time, value)
        self.unit_time = value

    @staticmethod
    def _dummy_func():
        pass

    def _equality_constraints(self, obj):
        """Boundary constraints, LGL defects and physical state continuity."""
        return equality(self, obj)

    def _inequality_constraints(self, obj):
        """Forward-time and user inequality residuals (feasible when nonnegative)."""
        return inequality(self, obj)

    def _objective(self, obj):
        """Mayer cost plus physical-time quadrature of the Lagrange cost."""
        return objective(self, obj)

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

        ``maxIterator`` limits total attempts, including the first. A restart
        occurs only after an iteration limit (SLSQP status 9, trust-constr status 0).
        Other failures stop immediately and emit RuntimeWarning. Each call
        resets the restart counter. The final result is also saved in
        ``result`` and every attempt in ``result_history``. ``display_func``
        is called without arguments after each attempt, as in the legacy API.
        ``time`` and ``time_all_section`` always reflect the current parameters.

        Args:
            obj: User data passed unchanged to every model/cost/constraint callback.
            display_func: Called without arguments after each solver attempt.
            solver: "SLSQP" or "trust-constr".
            derivatives: "finite-difference", "structured", or "autograd".
            **options: SciPy solver options, e.g. maxiter, ftol (SLSQP) or gtol
                (trust-constr).

        Returns:
            OptimizeResult: SciPy result with solver, derivatives and scaled
                equality_violation/inequality_violation diagnostics. The solution
                also replaces this problem's p. Failure returns success=False and
                emits RuntimeWarning; it does not raise a convergence exception.
        """
        from ._solver import solve

        return solve(
            self, obj, display_func, solver=solver, derivatives=derivatives, **options
        )

    def condition(self, length: int = 0) -> Condition:
        """Build equality or inequality residuals in this view's array backend.

        Args:
            length: Number of initial zero residuals, normally zero.

        Returns:
            Condition: A residual builder. Use equal() for equality callbacks;
                lower_bound()/upper_bound() produce nonnegative feasible residuals.
        """
        return Condition(length)

    def refine(self, nodes) -> "Problem":
        """Return a new problem with interpolated guesses and new node counts.

        Args:
            nodes (sequence of int): Total node counts per phase, including
                duplicated interfaces. At least two nodes per element are required.

        Returns:
            Problem: A new problem preserving element boundaries, physical units,
                constant box bounds and callbacks. Solver results are reset. Metadata
                containers are shallow-copied; nested objects remain shared.

        Raises:
            ValueError: Counts are invalid or box bounds vary between nodes.
        """
        from .mesh import refine

        return refine(self, nodes)

    def remesh(self, mesh) -> "Problem":
        """Return a warm start on per-phase lists of (left, right, nodes) elements.

        Element endpoints are phase coordinates in [-1, 1]. Logical section
        indices remain unchanged. States are continuous at internal interfaces;
        controls have independent left and right limits.

        Args:
            mesh (sequence): Per-phase lists of (left, right, nodes) elements
                partitioning [-1, 1]. Each element includes both endpoints.

        Returns:
            Problem: A new problem with interpolated guesses, preserved physical
                units, constant bounds and callbacks, and reset solver results.
                Metadata containers are shallow-copied. Private extension caches are
                not copied.

        Raises:
            ValueError: The mesh is invalid or box bounds vary between nodes.
        """
        from .mesh import remesh

        return remesh(self, mesh)

    def estimate_error(self, obj: Any = None, check_factor: int = 2) -> dict[str, Any]:
        """Estimate dimensionless residuals between the LGL nodes.

        Args:
            obj: User data passed to the model and inequality callback.
            check_factor: Integer >= 2. Check each element at
                check_factor * element.nodes + 1 Gauss points.

        Returns:
            dict: section_errors (per-phase ODE indicators), max_error (their
                maximum), path_violation (scaled inequality/box-bound violation), and
                elements (per-element error, smoothness, overshoot and split point).
                Indicators are residual estimates, not rigorous state-error bounds.
                Nonlinear path constraints are checked by sampling, not certified.

        Raises:
            ValueError: A phase lacks DynamicsModel, has nonpositive duration,
                or uses varying nodal box bounds.
        """
        from .mesh import estimate_error

        return estimate_error(self, obj, check_factor)

    def solve_adaptive(
        self,
        obj: Any = None,
        *,
        tolerance: float = 1e-5,
        max_refinements: int = 4,
        max_nodes: int = 100,
        growth: float = 1.5,
        strategy: str = "p",
        max_elements: int = 24,
        min_element_nodes: int = 6,
        max_order: int = 14,
        path_check_factor: int = 4,
        **solve_options: Any,
    ) -> optimize.OptimizeResult:
        """Solve and adapt the mesh, updating this problem to the final mesh.

        Requires DynamicsModel in every phase. Callbacks must use physical
        getters and work with changing node counts. For h/hp path sampling,
        avoid direct p indices and mesh-dependent global reductions.

        Args:
            obj: User data passed to all callbacks.
            tolerance: Positive tolerance for ODE and scaled path residuals.
            max_refinements: Maximum changes after the initial solve.
            max_nodes: Maximum total nodes per phase, including interfaces.
            growth: Node growth factor > 1 for p refinement.
            strategy: "p" increases orders; "h" splits elements; "hp" chooses
                between increasing orders and splitting using error indicators.
            max_elements: Maximum elements per phase for h/hp.
            min_element_nodes: Minimum child node count for hp, at least four.
            max_order: Node threshold for choosing h instead of p in hp.
            path_check_factor: Sampling density for h/hp path constraints,
                at least two. Not used by the p strategy.
            **solve_options: Options forwarded to solve(), including solver,
                derivatives and SciPy tolerances.

        Returns:
            OptimizeResult: Final solver result. success reports optimizer
                convergence; mesh_converged separately reports meeting the mesh
                tolerance. mesh_history records every solve and residual estimate.
                h/hp also supply mesh_strategy and mesh_stop_reason. Stopping at a
                limit emits RuntimeWarning when the mesh has not converged.
        """
        from .mesh import solve_adaptive

        return solve_adaptive(
            self,
            obj,
            tolerance=tolerance,
            max_refinements=max_refinements,
            max_nodes=max_nodes,
            growth=growth,
            strategy=strategy,
            max_elements=max_elements,
            min_element_nodes=min_element_nodes,
            max_order=max_order,
            path_check_factor=path_check_factor,
            **solve_options,
        )

    @override
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

    def to_csv(self, filename="OpenGoddard_output.csv", delimiter: str = ",") -> None:
        """Write physical times, states and controls in phase/node order.

        Args:
            filename (str or PathLike): Output path. Missing components in phases
                with fewer states or controls are written as NaN.
            delimiter: Column separator; a comma by default.

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
            title_comment (str): string for title

        """
        import matplotlib.pyplot as plt

        plt.figure()
        plt.title("OpenGoddard inner variables" + title_comment)
        plt.plot(self.p, "o")
        plt.xlabel("variables")
        plt.ylabel("value")
        for section in range(self.number_of_section):
            for line in self.div[section]:
                plt.axvline(line, color=f"C{(section + 1) % 6}", alpha=0.5)
        plt.grid()

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

    def method_LG(self, n) -> tuple[NDArray[np.float64], ...]:
        """Legendre-Gauss Pseudospectral method
        Gauss nodes are roots of $P_n(x)$.

        Args:
            n (int): number of nodes

        Returns:
            tuple[ndarray, ndarray, ndarray]: Nodes (n,), weights (n,), and D (n, n).

        """
        return legendre_rule("LG", n)

    def _nodes_LGR(self, n):
        return legendre_rule("LGR", n)[0]

    def _weight_LGR(self, n):
        return legendre_rule("LGR", n)[1]

    def _differentiation_matrix_LGR(self, n):
        return legendre_rule("LGR", n)[2]

    def method_LGR(self, n) -> tuple[NDArray[np.float64], ...]:
        """Legendre-Gauss-Radau Pseudospectral method
        Gauss-Radau nodes are roots of $P_n(x) + P_{n-1}(x)$.

        Args:
            n (int): number of nodes

        Returns:
            tuple[ndarray, ndarray, ndarray]: Nodes (n,), weights (n,), and D (n, n).

        """
        return legendre_rule("LGR", n)

    def _nodes_LGL(self, n):
        return legendre_rule("LGL", n)[0]

    def _weight_LGL(self, n):
        return legendre_rule("LGL", n)[1]

    def _differentiation_matrix_LGL(self, n):
        return legendre_rule("LGL", n)[2]

    def method_LGL(self, n) -> tuple[NDArray[np.float64], ...]:
        """Legendre-Gauss-Lobatto Pseudospectral method
        Gauss-Lobatto nodes are roots of $P'_{n-1}(x)$, plus -1 and 1.

        Args:
            n (int): number of nodes

        Returns:
            tuple[ndarray, ndarray, ndarray]: Nodes (n,), weights (n,), and D (n, n).

        References:
            Fariba Fahroo and I. Michael Ross. "Advances in Pseudospectral Methods
            for Optimal Control", AIAA Guidance, Navigation and Control Conference
            and Exhibit, Guidance, Navigation, and Control and Co-located Conferences
            http://dx.doi.org/10.2514/6.2008-7309

        """
        return legendre_rule("LGL", n)
