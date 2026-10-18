"""Compatibility smoke checks using only the Python standard test runner."""

import contextlib
import io
import unittest

import matplotlib

matplotlib.use("Agg")

import numpy as np

from OpenGoddard.optimize import Condition, Dynamics, Guess, Problem


class CompatibilityTests(unittest.TestCase):
    def test_quadrature_and_differentiation(self):
        problem = Problem([0.0, 1.0], [8], [1], [1])
        for method in (problem.method_LG, problem.method_LGR, problem.method_LGL):
            with self.subTest(method=method.__name__):
                nodes, weights, differentiation = method(8)
                np.testing.assert_allclose(weights @ nodes**2, 2 / 3, atol=1e-12)
                np.testing.assert_allclose(
                    differentiation @ nodes**3, 3 * nodes**2, atol=1e-12
                )

    def test_legendre_derivative_at_endpoints(self):
        problem = Problem([0.0, 1.0], [8], [1], [1])
        x = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_allclose(problem._LegendreDerivative(x, 0), 0.0)
        np.testing.assert_allclose(problem._LegendreDerivative(x, 3), [6, -1.5, 6])

    def test_solve_and_plot(self):
        problem = Problem([0.0, 1.25], [6], [1], [1], maxIterator=2)
        problem.set_states(0, 0, Guess.linear(problem.time[0], 0.0, 1.0))
        problem.set_controls(0, 0, Guess.constant(problem.time[0], 0.8))
        problem.set_controls_bounds(0, 0, 0.0, 1.0)
        problem.set_time_final_bounds(0, 0.1, 2.0)

        def dynamics(prob, obj, section):
            derivative = Dynamics(prob, section)
            derivative[0] = prob.controls(0, section)
            return derivative()

        def equality(prob, obj):
            state = prob.states(0, 0)
            condition = Condition()
            condition.equal(state[0], 0.0)
            condition.equal(state[-1], 1.0)
            return condition()

        problem.dynamics = [dynamics]
        problem.cost = lambda prob, obj: prob.time_final(0)
        problem.equality = equality
        problem.inequality = lambda prob, obj: Condition()()
        with contextlib.redirect_stdout(io.StringIO()):
            problem.solve(None, maxiter=100, ftol=1e-10)

        self.assertAlmostEqual(problem.time_final(0), 1.0, places=6)
        np.testing.assert_allclose(problem.controls(0, 0), 1.0, atol=1e-6)
        np.testing.assert_allclose(equality(problem, None), 0.0, atol=1e-8)
        problem.plot()


if __name__ == "__main__":
    unittest.main()
