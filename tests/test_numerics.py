"""Numerical identities and regressions independent of optimizer convergence."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from OpenGoddard import Condition, Dynamics, Guess, Problem


class SpectralTests(unittest.TestCase):
    def test_polynomial_exactness_and_minimum_orders(self):
        problem = Problem([0, 1], [2], [1], [0])
        for method, minimum, extra in [
            (problem.method_LG, 1, -1),
            (problem.method_LGR, 1, -2),
            (problem.method_LGL, 2, -3),
        ]:
            for n in (minimum, 3, 8, 32, 64):
                with self.subTest(method=method.__name__, n=n):
                    nodes, weights, matrix = method(n)
                    self.assertTrue(np.all(weights > 0))
                    self.assertTrue(np.all(np.diff(nodes) > 0))
                    for degree in range(2 * n + extra + 1):
                        integral = 0 if degree % 2 else 2 / (degree + 1)
                        self.assertAlmostEqual(
                            weights @ nodes**degree, integral, places=12
                        )
                    for degree in range(n):
                        expected = (
                            np.zeros(n)
                            if degree == 0
                            else degree * nodes ** (degree - 1)
                        )
                        np.testing.assert_allclose(
                            matrix @ nodes**degree, expected, atol=1e-9, rtol=1e-10
                        )

    def test_cached_arrays_are_independent(self):
        problem = Problem([0, 1, 2], [4, 4], [1, 1], [0, 0])
        expected = problem.D[1].copy()
        problem.D[0][:] = 42
        np.testing.assert_array_equal(problem.D[1], expected)
        np.testing.assert_array_equal(problem.method_LGL(4)[2], expected)


class ScalingAndLayoutTests(unittest.TestCase):
    def test_scaling_preserves_physical_values_and_bounds(self):
        problem = Problem([10, 13, 20], [4, 5], [2, 2], [1, 1])
        problem.set_states(1, 1, [1, 2, 3, 4, 5])
        problem.set_controls(0, 0, [2, 4, 6, 8])
        problem.set_states_bounds(1, 1, -3, 6)
        problem.set_controls_bounds(0, 0, None, 10)
        problem.set_time_final_bounds(1, 15, 25)
        for unit in (5, 2, 10):
            problem.set_unit_states(1, 1, unit)
            problem.set_unit_controls(0, 0, unit)
            problem.set_unit_time(unit)
            np.testing.assert_allclose(problem.states(1, 1), [1, 2, 3, 4, 5])
            np.testing.assert_allclose(problem.controls(0, 0), [2, 4, 6, 8])
            np.testing.assert_allclose(
                np.asarray(problem.bounds[problem.index_states(1, 1)], dtype=float),
                (-3 / unit, 6 / unit),
            )
            self.assertEqual(
                problem.bounds[problem.index_controls(0, 0)], (None, 10 / unit)
            )
            np.testing.assert_allclose(
                np.asarray(problem.bounds[problem.index_time_final(1)], dtype=float),
                (15 / unit, 25 / unit),
            )
            np.testing.assert_allclose(problem.time_knots(), [10, 13, 20])
            self.assertEqual(problem.time_start(-1), 13)
            self.assertEqual(problem.time[0][0], 10)
            self.assertEqual(problem.time[-1][-1], 20)
            np.testing.assert_array_equal(
                problem.time_all_section, np.concatenate(problem.time)
            )
            self.assertEqual(problem.time_init, [10, 13, 20])

    def test_indices_and_sequence_inputs(self):
        nodes = [3, 4]
        problem = Problem((0, 1, 3), nodes, np.array([2, 1]), (0, 2))
        nodes[0] = 99
        self.assertEqual(problem.nodes, [3, 4])
        problem.set_states(-1, 0, [7, 8, 9])
        problem.set_controls(-1, -1, [1, 2, 3, 4])
        np.testing.assert_allclose(problem.states(1, 0), [7, 8, 9])
        np.testing.assert_allclose(problem.controls(1, 1), [1, 2, 3, 4])
        self.assertEqual(problem.index_states(-1, 0, -1), 5)
        self.assertEqual(problem.index_controls(-1, -1, -1), 17)
        for call in (
            lambda: problem.states(0, 2),
            lambda: problem.states(0, -3),
            lambda: problem.states(2, 0),
            lambda: problem.controls(0, 0),
            lambda: problem.index_controls(0, 1, -5),
            lambda: problem.index_states(0, 0, 3),
            lambda: problem.time_final(-3),
        ):
            with self.subTest(call=call), self.assertRaises(IndexError):
                call()

    def test_time_tracks_setters_and_direct_parameter_changes(self):
        problem = Problem([5, 8, 12], [4, 5], [1, 1], [0, 0])
        problem.set_unit_time(3)
        problem.set_time_final(0, 10)
        self.assertEqual(problem.time[0][-1], 10)
        self.assertEqual(problem.time[1][0], 10)
        self.assertEqual(problem.time_all_section[3], 10)
        # Direct p access is part of the legacy API, including in-place edits
        # and replacement by a solver. Neither requires a cache refresh.
        problem.p[problem.index_time_final(1)] = 5
        self.assertEqual(problem.time[-1][-1], 15)
        parameters = problem.p.copy()
        parameters[problem.index_time_final(0)] = 4
        problem.p = parameters
        self.assertEqual(problem.time[0][-1], 12)
        self.assertEqual(problem.time[1][0], 12)
        np.testing.assert_array_equal(problem.time_update(), problem.time_all_section)
        problem.time[0][:] = -99
        self.assertEqual(problem.time[0][0], 5)
        self.assertEqual(problem.time[0][-1], 12)

    def test_phase_continuity_with_different_state_scales(self):
        problem = Problem([2, 4, 9], [3, 4], [1, 1], [0, 0])
        problem.set_unit_states(0, 0, 2)
        problem.set_unit_states(0, 1, 5)
        problem.set_states_all_section(0, np.full(7, 10))
        problem.dynamics = [lambda p, o, s: Dynamics(p, s)()] * 2
        np.testing.assert_allclose(problem._equality_constraints(None), 0, atol=1e-12)
        problem.set_states(0, 1, np.full(4, 12))
        self.assertAlmostEqual(problem._equality_constraints(None)[-1], -1)
        problem.knot_states_smooth = [False]
        self.assertEqual(problem._equality_constraints(None).size, 7)

    def test_integral_cost_uses_physical_phase_durations(self):
        problem = Problem([10, 12, 17], [4, 5], [1, 1], [0, 0])
        problem.cost = lambda p, o: 3
        # Integral of t^2 on [10,17], including a repeated node at the phase knot.
        problem.running_cost = lambda p, o: p.time_all_section**2
        for unit in (1, 10, 3):
            problem.set_unit_time(unit)
            self.assertAlmostEqual(problem._objective(None), 3 + (17**3 - 10**3) / 3)

    def test_csv_with_different_phase_models(self):
        problem = Problem([3, 5, 8], [2, 3], [2, 1], [0, 1])
        problem.set_states(0, 0, [1, 2])
        problem.set_states(1, 0, [8, 9])
        problem.set_states(0, 1, [2, 3, 4])
        problem.set_controls(0, 1, [4, 5, 6])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.tsv"
            problem.to_csv(path, delimiter="\t")
            data = np.genfromtxt(path, delimiter="\t", names=True)
        self.assertEqual(data.dtype.names, ("time", "state0", "state1", "control0"))
        np.testing.assert_allclose(data["state1"][:2], [8, 9])
        self.assertTrue(np.isnan(data["state1"][2:]).all())
        self.assertTrue(np.isnan(data["control0"][:2]).all())
        np.testing.assert_allclose(data["control0"][2:], [4, 5, 6])


class HelperTests(unittest.TestCase):
    def test_cubic_guess_far_from_time_origin(self):
        time = 1e9 + np.array([0, 0.25, 0.5, 0.75, 1])
        fraction = time - time[0]
        expected = 2 + 3 * fraction - 4 * fraction**2 + 5 * fraction**3
        np.testing.assert_allclose(Guess.cubic(time, 2, 3, 6, 10), expected, atol=1e-13)
        np.testing.assert_allclose(Guess.linear(time.tolist(), 2, 6), 2 + 4 * fraction)
        np.testing.assert_allclose(Guess.linear([0, 1, 1, 2], 0, 4), [0, 2, 2, 4])

    def test_conditions_broadcast_lists_and_preserve_changes(self):
        condition = Condition(2)
        condition.change_value(0, 7)
        condition.equal([1, 2], 1, unit=2)
        condition.lower_bound([3, 4], [1, 2])
        condition.upper_bound(2, [5, 6])
        np.testing.assert_allclose(condition(), [7, 0, 0, 0.5, 2, 2, 3, 4])
        condition().put(1, 8)
        condition.add(9)
        condition.change_value(-1, 10)
        np.testing.assert_allclose(condition(), [7, 8, 0, 0.5, 2, 2, 3, 4, 10])

    def test_scalar_dynamics_broadcast_at_all_nodes(self):
        problem = Problem([0, 1], [4], [2], [0])
        problem.set_unit_states(1, 0, 5)
        problem.set_unit_time(10)
        dynamics = Dynamics(problem)
        dynamics[0] = 2
        dynamics[1] = [1, 2, 3, 4]
        np.testing.assert_allclose(dynamics(), [20, 20, 20, 20, 2, 4, 6, 8])
        with self.assertRaises(ValueError):
            dynamics[0] = [1, 2]
        with self.assertRaises(IndexError):
            dynamics[2] = 1

    def test_invalid_problem_inputs(self):
        for arguments in (
            ([0, 1], [], [], []),
            ([0, 1], [1], [1], [0]),
            ([0, 1], [3.5], [1], [0]),
            ([0, 1], [3], [0], [0]),
            ([0, 1], [3], [1], [-1]),
            ([0, 0], [3], [1], [0]),
            ([0, np.nan], [3], [1], [0]),
            ([0, 1], [3, 3], [1], [0]),
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                Problem(*arguments)
        for method in ("LG", "LGR", "typo"):
            with self.assertRaises(ValueError):
                Problem([0, 1], [3], [1], [0], method=method)

    def test_invalid_scales_shapes_and_intervals(self):
        problem = Problem([0, 1], [3], [1], [1])
        for value in (0, -1, np.nan, np.inf):
            for setter in (
                problem.set_unit_time,
                lambda v: problem.set_unit_states(0, 0, v),
                lambda v: problem.set_unit_controls(0, 0, v),
                lambda v: Condition().add(1, unit=v),
            ):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    setter(value)
        for value in ([1, 2], [[1, 2, 3]], [1, 2, np.inf]):
            with self.assertRaises(ValueError):
                problem.set_states(0, 0, value)
        with self.assertRaises(ValueError):
            problem.set_states_all_section(0, [1, 2, 3, 4])
        with self.assertRaises(ValueError):
            problem.set_controls_bounds(0, 0, 3, 2)
        with self.assertRaises(ValueError):
            problem.set_time_final_bounds(0, np.nan, None)
        for time in ([0, 0], [1], [2, 1], [0, np.inf]):
            with self.assertRaises(ValueError):
                Guess.cubic(time, 0, 0, 1, 0)
        np.testing.assert_allclose(problem.time_to_tau([10, 12, 14]), [-1, 0, 1])
        with self.assertRaises(ValueError):
            problem.time_to_tau([1, 1])
