"""Physical dynamics and explicit affine-callback contracts.

The simple Dynamics residual builder is in _helpers. DynamicsModel adds a
pointwise physical model that can also be evaluated on new meshes or traced
for derivatives; neither adapter changes the user's physical units.
"""

import numpy as np

from ._variables import scaled_derivatives


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
        xp = getattr(problem, "array_module", np)
        values = (
            self.physical(x, u, time, obj, section)
            if xp is np
            else self.function(x, u, time, obj, section, xp)
        )
        return scaled_derivatives(
            values, problem.unit_states[section], problem.unit_time, xp
        )

    @staticmethod
    def inputs(problem, section):
        xp = getattr(problem, "array_module", np)
        x = xp.stack(
            [
                problem.states(i, section)
                for i in range(problem.number_of_states[section])
            ]
        )
        nc = problem.number_of_controls[section]
        u = (
            xp.stack([problem.controls(i, section) for i in range(nc)])
            if nc
            else xp.zeros((0, problem.nodes[section]))
        )
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
