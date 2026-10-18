"""End-to-end optimal-control problems with known analytic solutions."""

import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import OptimizeResult

from OpenGoddard import Condition, Dynamics, Guess, Problem


def integrator(prob, obj, section):
    dx = Dynamics(prob, section)
    dx[0] = prob.controls(0, section)
    return dx()


def endpoints(prob, target):
    condition = Condition()
    state = prob.states_all_section(0)
    condition.equal(state[0], 0)
    condition.equal(state[-1], target)
    return condition()


class AnalyticSolverTests(unittest.TestCase):
    def test_minimum_energy_transfer_across_scaled_phases(self):
        # min integral(u^2 dt), x'=u, x(10)=0, x(14)=1:
        # u=1/4, x=(t-10)/4, objective=1/4.
        prob = Problem([10, 11, 14], [4, 5], [1, 1], [1, 1], maxIterator=2)
        prob.set_unit_time(5)
        for section, scale in enumerate([2, 7]):
            prob.set_unit_states(0, section, scale)
            prob.set_unit_controls(0, section, scale * 2)
            prob.set_states(0, section, (prob.time[section] - 10) / 4)
            prob.set_controls(0, section, np.full(prob.nodes[section], 0.3))
            prob.set_time_final_bounds(section, [11, 14][section], [11, 14][section])
        prob.dynamics = [integrator, integrator]
        prob.equality = endpoints
        prob.running_cost = lambda p, target: p.controls_all_section(0) ** 2
        result = prob.solve(1, maxiter=150, ftol=1e-11)
        self.assertTrue(result.success, result.message)
        self.assertAlmostEqual(result.fun, 0.25, places=8)
        np.testing.assert_allclose(
            prob.states_all_section(0), (prob.time_all_section - 10) / 4, atol=1e-6
        )
        np.testing.assert_allclose(prob.controls_all_section(0), 0.25, atol=1e-6)
        np.testing.assert_allclose(prob._equality_constraints(1), 0, atol=1e-9)
        self.assertIs(result, prob.result)
        self.assertEqual(prob.iterator, 1)

    def test_double_integrator_energy_and_scaling_invariance(self):
        # x''=u, rest-to-rest transfer of distance d in time T.
        # x=d(3s^2-2s^3), u=6d/T^2(1-2s), energy=12d^2/T^3.
        def dynamics(p, obj, section):
            dx = Dynamics(p, section)
            dx[0] = p.states(1, section)
            dx[1] = p.controls(0, section)
            return dx()

        def boundary(p, obj):
            x, v = p.states_all_section(0), p.states_all_section(1)
            return np.array([x[0], x[-1] - 3, v[0], v[-1]])

        for scaled in (False, True):
            with self.subTest(scaled=scaled):
                prob = Problem([5, 5.6, 7], [5, 6], [2, 2], [1, 1], maxIterator=2)
                if scaled:
                    prob.set_unit_time(3)
                    for phase in range(2):
                        prob.set_unit_states(0, phase, [3, 5][phase])
                        prob.set_unit_states(1, phase, [2, 7][phase])
                        prob.set_unit_controls(0, phase, [2, 4][phase])
                for phase, end in enumerate([5.6, 7]):
                    time = prob.time[phase]
                    s = (time - 5) / 2
                    prob.set_states(0, phase, 3 * (3 * s**2 - 2 * s**3))
                    prob.set_states(1, phase, 9 * s * (1 - s))
                    prob.set_controls(0, phase, np.zeros(time.size))
                    prob.set_time_final_bounds(phase, end, end)
                prob.dynamics = [dynamics, dynamics]
                prob.equality = boundary
                prob.running_cost = lambda p, obj: p.controls_all_section(0) ** 2
                result = prob.solve(ftol=1e-10, maxiter=150)
                self.assertTrue(result.success, result.message)
                self.assertAlmostEqual(result.fun, 13.5, places=7)
                s = (prob.time_all_section - 5) / 2
                np.testing.assert_allclose(
                    prob.controls_all_section(0), 4.5 * (1 - 2 * s), atol=2e-5
                )
                np.testing.assert_allclose(
                    prob.states_all_section(0), 3 * (3 * s**2 - 2 * s**3), atol=2e-6
                )

    def test_free_final_time_running_cost_and_repeated_solves(self):
        # J=T + d^2/T, so T=d and u=1. Change d on the second solve.
        prob = Problem([5, 8], [6], [1], [1], maxIterator=2)
        prob.set_unit_time(3)
        prob.set_states(0, 0, Guess.linear(prob.time[0], 0, 2))
        prob.set_controls(0, 0, np.full(6, 2 / 3))
        prob.set_time_final_bounds(0, 5.1, 10)
        prob.dynamics = [integrator]
        prob.equality = endpoints
        prob.cost = lambda p, d: p.time_final(0) - p.time_start(0)
        prob.running_cost = lambda p, d: p.controls_all_section(0) ** 2
        for target in (2, 1):
            result = prob.solve(target, maxiter=100, ftol=1e-11)
            self.assertTrue(result.success, result.message)
            self.assertAlmostEqual(prob.time_final(0), 5 + target, places=5)
            self.assertAlmostEqual(result.fun, 2 * target, places=8)
            np.testing.assert_allclose(prob.controls_all_section(0), 1, atol=5e-6)
            self.assertEqual(len(prob.result_history), 1)

    def test_time_dependent_dynamics_see_current_time_grid(self):
        # min integral(u^2 dt), x'=t+u, x(2)=0, x(tf)=2.5: u=0, tf=3.
        prob = Problem([2, 4], [5], [1], [1], maxIterator=2)
        prob.set_unit_time(2)
        prob.set_states(0, 0, (prob.time[0] ** 2 - 4) / 2)
        prob.set_time_final_bounds(0, 2.1, 5)

        def dynamics(p, obj, section):
            dx = Dynamics(p, section)
            dx[0] = p.time[section] + p.controls(0, section)
            return dx()

        prob.dynamics = [dynamics]
        prob.equality = endpoints
        prob.running_cost = lambda p, target: p.controls_all_section(0) ** 2
        result = prob.solve(2.5, maxiter=100, ftol=1e-11)
        self.assertTrue(result.success, result.message)
        self.assertAlmostEqual(prob.time_final(0), 3, places=5)
        np.testing.assert_allclose(
            prob.states_all_section(0), (prob.time_all_section**2 - 4) / 2, atol=1e-5
        )

    def test_infeasible_problem_returns_failure_and_warns_once(self):
        prob = Problem([0, 1], [4], [1], [0], maxIterator=100)
        prob.dynamics = [lambda p, obj, phase: Dynamics(p, phase)()]
        prob.cost = lambda p, obj: p.time_final(0)
        prob.equality = lambda p, obj: endpoints(p, 1)
        prob.set_time_final_bounds(0, 1, 1)
        with self.assertWarnsRegex(RuntimeWarning, "SLSQP failed"):
            result = prob.solve()
        self.assertFalse(result.success)
        self.assertEqual(len(prob.result_history), 1)


class SolverInterfaceTests(unittest.TestCase):
    def problem(self):
        prob = Problem([0, 1], [4], [1], [1], maxIterator=2)
        prob.dynamics = [integrator]
        prob.cost = lambda p, o: p.time_final(0)
        return prob

    def test_options_retry_and_display_callback(self):
        prob = self.problem()
        displays = []
        outcomes = [
            OptimizeResult(
                x=prob.p.copy(), status=9, success=False, message="Iteration limit"
            ),
            OptimizeResult(x=prob.p.copy(), status=0, success=True, message="OK"),
        ]
        with patch(
            "OpenGoddard.optimize.optimize.minimize", side_effect=outcomes
        ) as minimize:
            result = prob.solve(
                display_func=lambda: displays.append(prob.time_all_section.copy()),
                eps=1e-7,
                disp=False,
                maxiter=3,
            )
        self.assertTrue(result.success)
        self.assertEqual(len(displays), 2)
        self.assertEqual(prob.iterator, 2)
        self.assertEqual(minimize.call_args.kwargs["options"]["eps"], 1e-7)
        self.assertEqual(minimize.call_args.kwargs["options"]["maxiter"], 3)
        self.assertFalse(minimize.call_args.kwargs["options"]["disp"])

    def test_callbacks_must_return_correct_shapes(self):
        prob = self.problem()
        prob.running_cost = lambda p, o: np.zeros(3)
        with self.assertRaisesRegex(ValueError, "running_cost"):
            prob.solve()
        prob.running_cost = None
        prob.dynamics = [lambda p, o, s: np.zeros(2)]
        with self.assertRaisesRegex(ValueError, "dynamics"):
            prob.solve()
        prob.dynamics = [integrator]
        prob.cost_derivative = lambda p, o: np.ones(2)
        with self.assertRaisesRegex(ValueError, "cost_derivative"):
            prob.solve()

    def test_parallel_workers_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "parallel workers"):
            self.problem().solve(workers=2)

    def test_phase_order_constraints(self):
        prob = Problem([3, 4, 5], [3, 3], [1, 1], [0, 0])
        prob.set_time_final(0, 6)
        np.testing.assert_allclose(prob._inequality_constraints(None), [3, -1])
