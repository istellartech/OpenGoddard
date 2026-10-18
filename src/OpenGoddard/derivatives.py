"""Structured collocation derivatives and optional NumPy automatic differentiation.

``DynamicsModel`` explicitly requires independent evaluations at each node.
Arbitrary legacy dynamics remain supported by ``derivatives='finite-difference'``.
"""

import numpy as np
from scipy import sparse

from ._validation import index, vector


def _autograd():
    try:
        import autograd.numpy as anp
        from autograd import grad, jacobian
    except ImportError as exc:
        raise ImportError(
            "Install OpenGoddard[autodiff] for derivatives='autograd'"
        ) from exc
    return anp, grad, jacobian


class LinearCallback:
    """Declare a cost/boundary/path callback affine in the scaled vector p.

    Its Jacobian is computed once per solve and cached. This is an explicit
    mathematical contract, not a heuristic: nonlinear expressions must not be
    wrapped. Coefficients, units and obj must remain constant during a solve.
    """

    def __init__(self, function):
        if not callable(function):
            raise TypeError("function must be callable")
        self.function = function

    def __call__(self, problem, obj):
        return self.function(problem, obj)


class DynamicsModel:
    """Pointwise physical dynamics ``function(x, u, t, obj, section, xp)``.

    ``x`` has shape (states, nodes), ``u`` (controls, nodes), and ``t`` (nodes,).
    Return an array with shape (states, nodes). Each output column must depend
    ONLY on the corresponding input column. Use ``xp`` for array operations;
    it is NumPy for values and autograd.numpy for automatic differentiation.
    Reductions or interpolation across nodes violate this contract.

    Instances are callable with the legacy ``(problem, obj, section)`` signature.
    Local derivatives use simultaneous central differences, or reverse-mode AD
    of each state's sum over independent nodes. No full dense dynamics Jacobian
    is formed. Non-smooth expressions require the user's choice of formulation.
    """

    def __init__(self, function):
        if not callable(function):
            raise TypeError("function must be callable")
        self.function = function

    def physical(self, x, u, time, obj, section):
        result = np.asarray(self.function(x, u, time, obj, section, np), dtype=float)
        if result.shape != x.shape or not np.isfinite(result).all():
            raise ValueError(
                "DynamicsModel must return finite values of shape (states, nodes)"
            )
        return result

    def __call__(self, problem, obj, section):
        x, u, time = self.inputs(problem, section)
        scale = problem.unit_time / np.asarray(problem.unit_states[section])
        return (self.physical(x, u, time, obj, section) * scale[:, None]).ravel()

    @staticmethod
    def inputs(problem, section):
        x = np.stack(
            [
                problem.states(i, section)
                for i in range(problem.number_of_states[section])
            ]
        )
        u = np.array(
            [
                problem.controls(i, section)
                for i in range(problem.number_of_controls[section])
            ]
        ).reshape(-1, problem.nodes[section])
        return x, u, problem.time[section]

    def linearize(self, problem, obj, section, mode):
        x, u, time = self.inputs(problem, section)
        ns, nc = x.shape[0], u.shape[0]
        z = np.concatenate((x, u, time[None, :]))
        values = self.physical(x, u, time, obj, section)
        if mode == "autograd":
            xp, grad, _ = _autograd()

            def evaluate(data, output):
                return xp.sum(
                    self.function(
                        data[:ns], data[ns : ns + nc], data[-1], obj, section, xp
                    )[output]
                )

            partials = np.stack([grad(evaluate)(z, output) for output in range(ns)])
        else:
            partials = np.empty((ns, z.shape[0], z.shape[1]))
            for variable in range(z.shape[0]):
                step = np.cbrt(np.finfo(float).eps) * np.maximum(1, np.abs(z[variable]))
                plus, minus = z.copy(), z.copy()
                plus[variable] += step
                minus[variable] -= step
                upper = self.physical(
                    plus[:ns], plus[ns : ns + nc], plus[-1], obj, section
                )
                lower = self.physical(
                    minus[:ns], minus[ns : ns + nc], minus[-1], obj, section
                )
                partials[:, variable] = (upper - lower) / (2 * step)
        if (
            partials.shape != (ns, ns + nc + 1, x.shape[1])
            or not np.isfinite(partials).all()
        ):
            raise ValueError("invalid local dynamics derivatives")
        return values, partials


class ArrayView:
    """Read-only numerical view used while tracing a callback through autograd.

    Use ``view.array_module`` instead of ordinary NumPy in differentiable
    callbacks and ``view.condition()`` instead of constructing Condition directly.
    Dimensions and units remain available through the original problem.
    """

    def __init__(self, problem, parameters, xp):
        self._problem = problem
        self.p = parameters
        self.array_module = xp

    def __getattr__(self, name):
        # Reject arbitrary bound Problem methods that would read the real p.
        value = getattr(self._problem, name)
        if callable(value):
            raise AttributeError(
                f"{name} is not available in a differentiable ArrayView"
            )
        return value

    def states(self, state, section):
        back, front = self._problem._division_states(state, section)
        return self.p[front:back] * self.unit_states[section][state]

    def controls(self, control, section):
        back, front = self._problem._division_controls(control, section)
        return self.p[front:back] * self.unit_controls[section][control]

    def states_all_section(self, state):
        return self.array_module.concatenate(
            [self.states(state, s) for s in range(self.number_of_section)]
        )

    def controls_all_section(self, control):
        return self.array_module.concatenate(
            [self.controls(control, s) for s in range(self.number_of_section)]
        )

    def time_final(self, section):
        return self.p[self._problem.index_time_final(section)] * self.unit_time

    def time_start(self, section):
        section = index(section, self.number_of_section, "section")
        return self.t0 if section == 0 else self.time_final(section - 1)

    def time_knots(self):
        return [self.t0] + [self.time_final(s) for s in range(self.number_of_section)]

    @property
    def time(self):
        return [
            self.time_start(s)
            + (self.tau[s] + 1) / 2 * (self.time_final(s) - self.time_start(s))
            for s in range(self.number_of_section)
        ]

    @property
    def time_all_section(self):
        return self.array_module.concatenate(self.time)

    def condition(self, length=0):
        from ._helpers import Condition

        return Condition(length, backend=self.array_module)


def finite_difference(function, parameters):
    """Central differences for small generic boundary/path callbacks.

    Differentiates the function's local extension; unlike solver steps, probes
    may leave optimization bounds. Callbacks must be defined near the iterate.
    """
    shape = np.asarray(function(parameters)).shape
    jacobian = np.empty(shape + (parameters.size,))
    for column in range(parameters.size):
        step = np.cbrt(np.finfo(float).eps) * max(1, abs(parameters[column]))
        plus, minus = parameters.copy(), parameters.copy()
        plus[column] += step
        minus[column] -= step
        jacobian[..., column] = (
            np.asarray(function(plus)) - np.asarray(function(minus))
        ) / (2 * step)
    return jacobian


class Derivatives:
    """Assemble exact LGL/continuity blocks and local nonlinear derivatives."""

    def __init__(self, problem, obj, mode):
        self.problem, self.obj, self.mode = problem, obj, mode
        self._linear_jacobians = {}
        self._linear_gradient = None
        if not all(isinstance(model, DynamicsModel) for model in problem.dynamics):
            raise ValueError(
                "structured/autograd derivatives require DynamicsModel in every section"
            )
        if mode == "autograd":
            _autograd()

    def _callback_jacobian(self, function):
        problem = self.problem
        if function is None:
            return np.empty((0, problem.number_of_variables))
        if isinstance(function, LinearCallback) and function in self._linear_jacobians:
            return self._linear_jacobians[function]
        parameters = problem.p.copy()
        if self.mode == "autograd":
            xp, _, jacobian = _autograd()
            result = jacobian(lambda p: function(ArrayView(problem, p, xp), self.obj))(
                parameters
            )
        else:
            # Legacy callbacks can use arbitrary Problem methods. Evaluate on
            # the real instance, and restore its state even when a probe fails.
            def legacy(p):
                problem.p = p
                problem.time_update()
                return function(problem, self.obj)

            try:
                result = finite_difference(legacy, parameters)
            finally:
                problem.p = parameters
                problem.time_update()
        result = np.asarray(result, dtype=float)
        if (
            result.ndim != 2
            or result.shape[1] != problem.number_of_variables
            or not np.isfinite(result).all()
        ):
            raise ValueError(
                "constraint callback must return a one-dimensional differentiable vector"
            )
        if isinstance(function, LinearCallback):
            self._linear_jacobians[function] = result
        return result

    def objective(self):
        problem = self.problem
        linear = (
            isinstance(problem.cost, LinearCallback) and problem.running_cost is None
        )
        if linear and self._linear_gradient is not None:
            return self._linear_gradient
        if self.mode != "autograd":
            if problem.cost_derivative is not None:
                return vector(
                    problem.cost_derivative(problem, self.obj),
                    "cost_derivative",
                    problem.number_of_variables,
                )
            parameters = problem.p.copy()

            def evaluate(p):
                problem.p = p
                problem.time_update()
                return problem._objective(self.obj)

            try:
                result = finite_difference(evaluate, parameters)
                if linear:
                    self._linear_gradient = result
                return result
            finally:
                problem.p = parameters
                problem.time_update()
        _, grad, _ = _autograd()
        result = vector(
            grad(self._objective_trace)(problem.p),
            "objective gradient",
            problem.number_of_variables,
        )
        if linear:
            self._linear_gradient = result
        return result

    def _objective_trace(self, parameters):
        xp, _, _ = _autograd()
        problem = self.problem
        view = ArrayView(problem, parameters, xp)
        value = 0.0 if problem.cost is None else problem.cost(view, self.obj)
        if problem.running_cost is not None:
            integrand = problem.running_cost(view, self.obj)
            front = 0
            for section, n in enumerate(problem.nodes):
                value += (
                    (view.time_final(section) - view.time_start(section))
                    / 2
                    * xp.dot(problem.w[section], integrand[front : front + n])
                )
                front += n
        return value

    def _constraint_trace(self, parameters, kind):
        xp, _, _ = _autograd()
        p = self.problem
        view = ArrayView(p, parameters, xp)
        if kind == "ineq":
            blocks = [
                xp.stack(
                    [
                        view.time_final(s) - view.time_start(s)
                        for s in range(p.number_of_section)
                    ]
                )
                / p.unit_time
            ]
            if p.inequality is not None:
                blocks.append(p.inequality(view, self.obj))
            return xp.concatenate(blocks)
        blocks = [] if p.equality is None else [p.equality(view, self.obj)]
        for section, n in enumerate(p.nodes):
            x = xp.stack(
                [view.states(i, section) for i in range(p.number_of_states[section])]
            )
            nc = p.number_of_controls[section]
            u = (
                xp.stack([view.controls(i, section) for i in range(nc)])
                if nc
                else xp.zeros((0, n))
            )
            scale = xp.array(p.unit_states[section])[:, None]
            physical = p.dynamics[section].function(
                x, u, view.time[section], self.obj, section, xp
            )
            residual = (
                xp.dot(x / scale, p.D[section].T)
                - (view.time_final(section) - view.time_start(section))
                / 2
                * physical
                / scale
            )
            blocks.append(xp.reshape(residual, (-1,)))
        for knot, smooth in enumerate(p.knot_states_smooth):
            if smooth and p.number_of_states[knot] == p.number_of_states[knot + 1]:
                blocks.append(
                    xp.stack(
                        [
                            (view.states(i, knot)[-1] - view.states(i, knot + 1)[0])
                            / p.unit_states[knot][i]
                            for i in range(p.number_of_states[knot])
                        ]
                    )
                )
        return xp.concatenate(blocks)

    def hessian(self, kind="objective", weights=None):
        """Matrix-free AD Hessian-vector products for trust-constr.

        No dense Hessian approximation is allocated in the autograd path.
        Parameters and constraint multipliers are fixed at construction time.
        """
        from autograd import hessian_vector_product
        from scipy.sparse.linalg import LinearOperator

        xp, _, _ = _autograd()
        parameters = self.problem.p.copy()
        if (
            kind == "objective"
            and isinstance(self.problem.cost, LinearCallback)
            and self.problem.running_cost is None
        ) or (
            kind == "ineq"
            and (
                self.problem.inequality is None
                or isinstance(self.problem.inequality, LinearCallback)
            )
        ):
            return sparse.csr_matrix((parameters.size, parameters.size))
        if kind == "objective":
            function = self._objective_trace
        else:
            multipliers = np.array(weights, dtype=float, copy=True)

            def function(value):
                return xp.dot(multipliers, self._constraint_trace(value, kind))

        product = hessian_vector_product(function)

        class Product(LinearOperator):
            def __init__(self):
                super().__init__(
                    dtype=np.dtype(float), shape=(parameters.size, parameters.size)
                )

            def _matvec(self, x):
                return np.asarray(product(parameters, x), dtype=float)

        return Product()

    def equality(self):
        p = self.problem
        columns = p.number_of_variables
        blocks = [sparse.csr_matrix(self._callback_jacobian(p.equality))]
        for section, n in enumerate(p.nodes):
            ns, nc = p.number_of_states[section], p.number_of_controls[section]
            _, front = p._division_states(0, section)
            duration = p.time_final(section) - p.time_start(section)
            model = p.dynamics[section]
            values, partials = model.linearize(p, self.obj, section, self.mode)
            state_scale = np.asarray(p.unit_states[section])
            variable_scale = np.array(p.unit_states[section] + p.unit_controls[section])
            local = sparse.kron(sparse.eye(ns), p.D[section], format="csr")
            local = sparse.hstack(
                (local, sparse.csr_matrix((ns * n, nc * n))), format="csr"
            )
            rows, cols, data = [], [], []
            for output in range(ns):
                for variable in range(ns + nc):
                    rows.extend(output * n + np.arange(n))
                    cols.extend(variable * n + np.arange(n))
                    data.extend(
                        -duration
                        / 2
                        / state_scale[output]
                        * partials[output, variable]
                        * variable_scale[variable]
                    )
            local += sparse.csr_matrix((data, (rows, cols)), shape=local.shape)
            block = sparse.hstack(
                (
                    sparse.csr_matrix((ns * n, front)),
                    local,
                    sparse.csr_matrix((ns * n, columns - front - local.shape[1])),
                ),
                format="lil",
            )
            fraction = (p.tau[section] + 1) / 2
            final = (
                (-values - duration * partials[:, -1] * fraction)
                / (2 * state_scale[:, None])
                * p.unit_time
            )
            block[:, p.index_time_final(section)] = final.ravel()[:, None]
            if section:
                initial = (
                    (values - duration * partials[:, -1] * (1 - fraction))
                    / (2 * state_scale[:, None])
                    * p.unit_time
                )
                block[:, p.index_time_final(section - 1)] = initial.ravel()[:, None]
            blocks.append(block.tocsr())
        rows, cols, data = [], [], []
        count = 0
        for knot, smooth in enumerate(p.knot_states_smooth):
            if not smooth or p.number_of_states[knot] != p.number_of_states[knot + 1]:
                continue
            for state in range(p.number_of_states[knot]):
                rows.extend((count, count))
                cols.extend(
                    (
                        p.index_states(state, knot, -1),
                        p.index_states(state, knot + 1, 0),
                    )
                )
                data.extend(
                    (1.0, -p.unit_states[knot + 1][state] / p.unit_states[knot][state])
                )
                count += 1
        blocks.append(sparse.csr_matrix((data, (rows, cols)), shape=(count, columns)))
        result = sparse.vstack(blocks, format="csr")
        result.eliminate_zeros()
        return result

    def inequality(self):
        p = self.problem
        order = sparse.lil_matrix((p.number_of_section, p.number_of_variables))
        for section in range(p.number_of_section):
            order[section, p.index_time_final(section)] = 1
            if section:
                order[section, p.index_time_final(section - 1)] = -1
        result = sparse.vstack(
            (order.tocsr(), sparse.csr_matrix(self._callback_jacobian(p.inequality))),
            format="csr",
        )
        result.eliminate_zeros()
        return result
