"""Independent checks of complete scaled Jacobians and solver backends."""

import builtins
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import sparse

from OpenGoddard import DynamicsModel, LinearCallback, Problem
from OpenGoddard.derivatives import Derivatives, finite_difference

try:
    import autograd  # noqa: F401

    HAS_AD = True
except ImportError:
    HAS_AD = False


class DerivativeTests(unittest.TestCase):
    def problem(self):
        p = Problem([2, 3, 5], [4, 5], [2, 2], [1, 1], maxIterator=1)
        p.set_unit_time(3)
        for s in range(2):
            for state in range(2):
                p.set_unit_states(state, s, (state + 1) * (s + 2))
                p.set_states(state, s, np.linspace(0.5, 1.5, p.nodes[s]))
            p.set_unit_controls(0, s, s + 4)
            p.set_controls(0, s, np.linspace(0.2, 0.8, p.nodes[s]))

        def dynamics(x, u, t, obj, section, xp):
            return xp.stack((x[1] + u[0] * t, xp.sin(x[0]) + u[0] ** 2 + t**2))

        p.dynamics = [DynamicsModel(dynamics)] * 2

        def boundary(view, obj):
            c = view.condition()
            c.equal(view.states(0, 0)[0], 0.5)
            c.equal(view.states(1, 1)[-1], view.time_final(0) ** 2)
            return c()

        def inequality(view, obj):
            c = view.condition()
            c.lower_bound(view.controls_all_section(0), 0.1)
            return c()

        p.equality, p.inequality = boundary, inequality
        p.cost = lambda v, obj: v.time_final(1) ** 2
        p.running_cost = lambda v, obj: (
            v.controls_all_section(0) ** 2 + v.time_all_section
        )
        return p

    def reference(self, problem, function):
        parameters = problem.p.copy()

        def evaluate(value):
            problem.p = value
            problem.time_update()
            return function(None)

        try:
            return finite_difference(evaluate, parameters)
        finally:
            problem.p = parameters
            problem.time_update()

    def check_jacobians(self, mode):
        p = self.problem()
        derivatives = Derivatives(p, None, mode)
        for method, function in (
            (derivatives.equality, p._equality_constraints),
            (derivatives.inequality, p._inequality_constraints),
        ):
            expected = self.reference(p, function)
            actual = method()
            self.assertTrue(sparse.issparse(actual))
            np.testing.assert_allclose(actual.toarray(), expected, atol=2e-7, rtol=2e-7)
        np.testing.assert_allclose(
            derivatives.objective(), self.reference(p, p._objective), atol=2e-7
        )
        self.assertLess(
            derivatives.equality().nnz, np.prod(derivatives.equality().shape) / 2
        )

    def test_structured_jacobian_scaling_time_and_knots(self):
        self.check_jacobians("structured")

    @unittest.skipUnless(HAS_AD, "optional autograd extra is not installed")
    def test_automatic_jacobian_scaling_time_and_knots(self):
        self.check_jacobians("autograd")

    @unittest.skipUnless(HAS_AD, "optional autograd extra is not installed")
    def test_matrix_free_hessian_products_against_gradient_differences(self):
        p = self.problem()
        derivatives = Derivatives(p, None, "autograd")
        parameters = p.p.copy()
        direction = np.linspace(-0.3, 0.4, parameters.size)
        for kind in ("objective", "eq", "ineq"):
            if kind == "objective":
                weights = None
                gradient = derivatives.objective
            else:
                jacobian = (
                    derivatives.equality if kind == "eq" else derivatives.inequality
                )
                weights = np.linspace(-0.3, 0.5, jacobian().shape[0])

                def gradient(jacobian=jacobian, weights=weights):
                    return jacobian().T @ weights

            operator = derivatives.hessian(kind, weights)
            p.p = parameters + 1e-5 * direction
            p.time_update()
            upper = gradient()
            p.p = parameters - 1e-5 * direction
            p.time_update()
            lower = gradient()
            p.p = parameters
            p.time_update()
            with self.subTest(kind=kind):
                np.testing.assert_allclose(
                    operator @ direction, (upper - lower) / 2e-5, atol=1e-7, rtol=1e-7
                )

    def test_affine_callback_jacobian_is_reused_within_solve(self):
        p = self.problem()
        calls = []

        @LinearCallback
        def boundary(view, obj):
            calls.append(1)
            return view.states(0, 0)[:1]

        p.equality = boundary
        derivatives = Derivatives(p, None, "structured")
        first = derivatives._callback_jacobian(boundary)
        count = len(calls)
        p.p += 0.1
        np.testing.assert_array_equal(derivatives._callback_jacobian(boundary), first)
        self.assertEqual(len(calls), count)

    def test_legacy_dynamics_rejected_in_structured_mode(self):
        p = self.problem()
        p.dynamics = [lambda p, o, s: np.zeros(2 * p.nodes[s])] * 2
        with self.assertRaisesRegex(ValueError, "DynamicsModel"):
            p.solve(derivatives="structured")

    def test_optional_ad_dependency_has_actionable_error(self):
        original_import = builtins.__import__

        def unavailable(name, *args, **kwargs):
            if name.startswith("autograd"):
                raise ImportError("not installed")
            return original_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=unavailable),
            self.assertRaisesRegex(ImportError, r"OpenGoddard\[autodiff\]"),
        ):
            Derivatives(self.problem(), None, "autograd")

    def test_sparse_trust_constr_and_structured_slsqp(self):
        example = (
            Path(__file__).resolve().parents[1]
            / "examples/01_minimum_energy_transfer.py"
        )
        namespace = runpy.run_path(str(example))
        for solver in ("SLSQP", "trust-constr"):
            for mode in ("structured", "autograd") if HAS_AD else ("structured",):
                with self.subTest(solver=solver, mode=mode):
                    p = namespace["make_problem"]()

                    # Derivative APIs are tested here, outside the first lesson.
                    def motion(x, u, t, obj, section, xp):
                        return xp.stack((u[0],))

                    p.dynamics = [DynamicsModel(motion)]
                    p.equality = LinearCallback(p.equality)
                    p.cost = LinearCallback(p.cost)
                    options = {"maxiter": 300}
                    options["ftol" if solver == "SLSQP" else "gtol"] = 1e-9
                    result = p.solve(2, solver=solver, derivatives=mode, **options)
                    self.assertTrue(result.success, result.message)
                    self.assertAlmostEqual(result.fun, 4, places=6)
                    self.assertLess(result.equality_violation, 1e-7)
                    np.testing.assert_allclose(p.controls_all_section(0), 1, atol=2e-5)
                    if solver == "trust-constr":
                        self.assertTrue(sparse.issparse(result.jac[0]))
