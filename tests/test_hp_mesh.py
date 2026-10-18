"""Logical phases survive internal h refinement and one-sided controls."""

import unittest

import numpy as np

from OpenGoddard import DynamicsModel, LinearCallback, Problem
from OpenGoddard._sampling import SampledPath
from OpenGoddard.derivatives import Derivatives, finite_difference


class ElementTests(unittest.TestCase):
    def piecewise(self):
        p = Problem(
            [3, 5, 8],
            [8, 4],
            [1, 1],
            [1, 1],
            mesh=[[(-1, 0, 4), (0, 1, 4)], [(-1, 1, 4)]],
        )
        p.set_unit_time(2)
        p.set_unit_states(0, 0, 3)
        p.set_unit_states(0, 1, 5)
        p.set_controls(0, 0, [1] * 4 + [2] * 4)
        p.set_controls(0, 1, [2] * 4)
        t = p.time[0]
        p.set_states(0, 0, np.where(t <= 4, t - 3, 1 + 2 * (t - 4)))
        p.set_states(0, 1, 3 + 2 * (p.time[1] - 5))
        p.dynamics = [DynamicsModel(lambda x, u, t, obj, s, xp: xp.stack((u[0],)))] * 2
        return p

    def test_piecewise_polynomial_and_phase_invariants(self):
        p = self.piecewise()
        self.assertEqual(p.number_of_section, 2)
        self.assertEqual(p.time_knots(), [3, 5, 8])
        np.testing.assert_allclose(p._equality_constraints(None), 0, atol=1e-12)
        self.assertAlmostEqual(p.w[0].sum(), 2)
        fine = p.remesh([[(-1, -0.5, 5), (-0.5, 0, 5), (0, 1, 6)], [(-1, 1, 7)]])
        np.testing.assert_allclose(fine.controls(0, 0)[:10], 1)
        np.testing.assert_allclose(fine.controls(0, 0)[10:], 2)
        self.assertEqual(fine.controls(0, 0)[9], 1)
        self.assertEqual(fine.controls(0, 0)[10], 2)
        np.testing.assert_allclose(fine._equality_constraints(None), 0, atol=2e-12)
        self.assertLess(fine.estimate_error()["max_error"], 1e-12)
        self.assertEqual(fine.number_of_section, 2)
        self.assertEqual(fine.unit_states, p.unit_states)

    def test_quadrature_with_internal_elements(self):
        p = self.piecewise()
        p.running_cost = lambda v, obj: v.time_all_section**2
        self.assertAlmostEqual(p._objective(None), (8**3 - 3**3) / 3)

    def test_internal_jacobian_and_ad_products(self):
        p = self.piecewise()
        p.cost = LinearCallback(lambda v, o: v.time_final(1))
        original = p.p.copy()

        def function(parameters):
            p.p = parameters
            p.time_update()
            return p._equality_constraints(None)

        try:
            reference = finite_difference(function, original)
        finally:
            p.p = original
            p.time_update()
        for mode in ("structured", "autograd"):
            with self.subTest(mode=mode):
                try:
                    provider = Derivatives(p, None, mode)
                except ImportError:
                    continue
                np.testing.assert_allclose(
                    provider.equality().toarray(), reference, atol=1e-8
                )
                if mode == "autograd":
                    direction = np.linspace(-0.2, 0.4, p.number_of_variables)
                    weights = np.ones(reference.shape[0])
                    operator = provider.hessian("eq", weights)
                    p.p = original + 1e-5 * direction
                    p.time_update()
                    upper = provider.equality().T @ weights
                    p.p = original - 1e-5 * direction
                    p.time_update()
                    lower = provider.equality().T @ weights
                    p.p = original
                    p.time_update()
                    np.testing.assert_allclose(
                        operator @ direction, (upper - lower) / 2e-5, atol=1e-8
                    )

    def test_refine_p_retains_internal_boundaries(self):
        p = self.piecewise()
        fine = p.refine([12, 7])
        self.assertEqual([(e.left, e.right) for e in fine.mesh[0]], [(-1, 0), (0, 1)])
        self.assertEqual(fine.nodes, [12, 7])
        with self.assertRaises(ValueError):
            p.refine([3, 4])

    def test_invalid_partitions_are_rejected(self):
        for mesh in (
            [[]],
            [[(-1, 0.2, 4), (0, 1, 4)]],
            [[(-1, 0, 4), (0.1, 1, 4)]],
            [[(-1, 0.5, 8)]],
            [[(-1, 1, 7)]],
        ):
            with self.subTest(mesh=mesh), self.assertRaises(ValueError):
                Problem([0, 1], [8], [1], [1], mesh=mesh)

    def test_sampled_path_detects_hidden_polynomial_violation(self):
        p = Problem([0, 1], [3], [1], [1])
        p.set_controls(0, 0, (p.time[0] - 0.25) ** 2 - 0.01)
        callback = LinearCallback(lambda v, o: v.controls_all_section(0))
        self.assertTrue((callback(p, None) >= 0).all())
        guard = SampledPath(callback, 4)
        self.assertLess(np.min(guard(p, None)), -0.005)


class AdaptiveElementTests(unittest.TestCase):
    def problem(self):
        p = Problem([0, 1], [6], [1], [1], maxIterator=1)
        p.dynamics = [
            DynamicsModel(lambda x, u, t, o, s, xp: xp.stack((xp.exp(t) + u[0],)))
        ]

        @LinearCallback
        def initial(v, o):
            result = v.condition()
            result.equal(v.states(0, 0)[0], 0)
            return result()

        p.equality = initial
        p.running_cost = lambda v, o: v.controls_all_section(0) ** 2
        p.set_time_final_bounds(0, 1, 1)
        p.set_controls_bounds(0, 0, -1, 1)
        p.set_states(0, 0, np.exp(p.time[0]) - 1)
        return p

    def test_h_and_hp_refine_without_renumbering(self):
        for strategy in ("h", "hp"):
            with self.subTest(strategy=strategy):
                p = self.problem()
                result = p.solve_adaptive(
                    strategy=strategy,
                    tolerance=1e-7,
                    max_refinements=4,
                    max_nodes=50,
                    derivatives="structured",
                    ftol=1e-12,
                    maxiter=200,
                )
                self.assertTrue(result.success, result.message)
                self.assertTrue(result.mesh_converged)
                self.assertEqual(result.mesh_strategy, strategy)
                self.assertEqual(result.mesh_stop_reason, "tolerance")
                self.assertEqual(p.number_of_section, 1)
                self.assertEqual(p.time_knots(), [0, 1])
                self.assertAlmostEqual(p.states(0, 0)[-1], np.e - 1, places=6)
                if strategy == "h":
                    self.assertGreater(len(p.mesh[0]), 1)

    def test_extrema_check_finds_actual_scalar_bound_violation(self):
        p = Problem([0, 1], [3], [1], [1])
        p.set_controls(0, 0, (p.time[0] - 0.25) ** 2 - 0.01)
        p.dynamics = [DynamicsModel(lambda x, u, t, o, s, xp: xp.zeros_like(x))]
        p.inequality = LinearCallback(lambda v, o: v.controls_all_section(0))
        self.assertAlmostEqual(p.estimate_error()["path_violation"], 0.01, places=12)

    def test_sampled_path_jacobian_matches_finite_differences(self):
        p = ElementTests().piecewise()
        base = LinearCallback(lambda v, o: v.controls_all_section(0))
        p.inequality = LinearCallback(SampledPath(base, 3, ((-0.3, 0.3), (0.2,))))
        original = p.p.copy()

        def callback(parameters):
            p.p = parameters
            p.time_update()
            return p._inequality_constraints(None)

        try:
            reference = finite_difference(callback, original)
        finally:
            p.p = original
            p.time_update()
        for mode in ("structured", "autograd"):
            try:
                provider = Derivatives(p, None, mode)
            except ImportError:
                continue
            np.testing.assert_allclose(
                provider.inequality().toarray(), reference, atol=1e-8
            )

    def test_h_limit_does_not_claim_convergence(self):
        p = self.problem()
        with self.assertWarnsRegex(RuntimeWarning, "mesh tolerance"):
            result = p.solve_adaptive(
                strategy="h",
                tolerance=1e-12,
                max_nodes=6,
                derivatives="structured",
                maxiter=200,
                ftol=1e-12,
            )
        self.assertFalse(result.mesh_converged)
        self.assertEqual(result.mesh_stop_reason, "mesh_limit")
