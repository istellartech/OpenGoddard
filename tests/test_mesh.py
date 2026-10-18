"""Mesh interpolation and independent residual estimation regressions."""

import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import OptimizeResult

from OpenGoddard import DynamicsModel, Problem


class MeshTests(unittest.TestCase):
    def test_refinement_preserves_phase_discontinuities_units_and_bounds(self):
        p = Problem([3, 4, 6], [4, 5], [2, 1], [1, 0])
        p.set_unit_time(7)
        p.set_unit_states(0, 1, 3)
        p.set_states(0, 0, p.time[0] ** 2)
        p.set_states(1, 0, p.time[0] ** 3)
        p.set_states(0, 1, p.time[1] ** 2 + 10)
        p.set_controls(0, 0, p.time[0])
        p.set_controls_bounds(0, 0, 0, 10)
        p.knot_states_smooth = [False]
        setattr(p, "metadata", {"vehicle": "test"})
        fine = p.refine([8, 9])
        for section in range(2):
            np.testing.assert_allclose(
                fine.states(0, section),
                fine.time[section] ** 2 + 10 * section,
                atol=1e-12,
            )
        np.testing.assert_allclose(fine.states(1, 0), fine.time[0] ** 3, atol=1e-12)
        self.assertAlmostEqual(fine.states(0, 1)[0] - fine.states(0, 0)[-1], 10)
        self.assertEqual(fine.bounds[fine.index_controls(0, 0)], (0, 10))
        self.assertEqual(fine.knot_states_smooth, [False])
        self.assertEqual(getattr(fine, "metadata"), getattr(p, "metadata"))
        self.assertEqual(p.nodes, [4, 5])
        with self.assertRaises(ValueError):
            p.refine([4])
        p.bounds[p.index_controls(0, 0)] = (1, 10)
        with self.assertRaisesRegex(ValueError, "constant nodal box bounds"):
            p.refine([8, 9])

    def nonlinear(self, nodes=4):
        p = Problem([0, 1], [nodes], [1], [1], maxIterator=1)

        def dynamics(x, u, t, obj, section, xp):
            return xp.stack((xp.exp(t) + u[0],))

        def initial(view, obj):
            c = view.condition()
            c.equal(view.states(0, 0)[0], 0)
            return c()

        p.dynamics = [DynamicsModel(dynamics)]
        p.equality = initial
        p.running_cost = lambda v, o: v.controls_all_section(0) ** 2
        p.set_time_final_bounds(0, 1, 1)
        p.set_states(0, 0, np.exp(p.time[0]) - 1)
        return p

    def test_adaptive_refinement_reduces_between_node_error(self):
        p = self.nonlinear()
        result = p.solve_adaptive(
            tolerance=1e-7,
            max_refinements=5,
            max_nodes=30,
            derivatives="structured",
            ftol=1e-12,
            maxiter=150,
        )
        self.assertTrue(result.success, result.message)
        self.assertTrue(result.mesh_converged)
        self.assertGreater(len(result.mesh_history), 1)
        self.assertLess(result.mesh_history[-1]["error"]["max_error"], 1e-7)
        self.assertGreater(result.mesh_history[0]["error"]["max_error"], 1e-7)
        self.assertAlmostEqual(p.states(0, 0)[-1], np.e - 1, places=6)
        self.assertEqual(p.nodes, result.mesh_history[-1]["nodes"])

    def test_node_limit_does_not_claim_mesh_convergence(self):
        p = self.nonlinear()
        with self.assertWarnsRegex(RuntimeWarning, "mesh tolerance"):
            result = p.solve_adaptive(
                tolerance=1e-12,
                max_nodes=4,
                derivatives="structured",
                ftol=1e-12,
                maxiter=150,
            )
        self.assertTrue(result.success)
        self.assertFalse(result.mesh_converged)
        self.assertEqual(len(result.mesh_history), 1)

    def test_failed_coarse_solve_refines_from_original_guess(self):
        p = self.nonlinear()
        original = Problem.solve
        calls = []

        def fail_once(problem, obj=None, **options):
            calls.append(1)
            if len(calls) == 1:
                problem.p[:] = 100  # deliberately unusable failed iterate
                return OptimizeResult(success=False, fun=100, nit=1)
            return original(problem, obj, **options)

        with patch.object(Problem, "solve", fail_once):
            result = p.solve_adaptive(
                tolerance=1e-6,
                max_refinements=4,
                max_nodes=30,
                derivatives="structured",
                ftol=1e-12,
                maxiter=150,
            )
        self.assertTrue(result.mesh_converged)
        self.assertFalse(result.mesh_history[0]["success"])
        self.assertTrue(result.mesh_history[-1]["success"])
        self.assertAlmostEqual(p.time_final(0), 1)

    def test_dense_path_checks_detect_between_node_violation(self):
        p = Problem([0, 1], [3], [1], [1])
        t = p.time[0]
        p.set_states(0, 0, t)
        p.set_controls(0, 0, t)
        p.dynamics = [DynamicsModel(lambda x, u, t, o, s, xp: xp.ones_like(x))]

        # A quadratic constraint has its only violation at t=1/4, which is
        # between the 3 original LGL points (0, 1/2, 1).
        def path(view, obj):
            return (view.time_all_section - 0.25) ** 2 - 0.01

        p.inequality = path
        self.assertTrue((p._inequality_constraints(None) >= 0).all())
        self.assertGreater(p.estimate_error()["path_violation"], 0.005)
        self.assertLess(p.estimate_error()["max_error"], 1e-12)
