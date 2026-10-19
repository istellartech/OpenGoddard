"""Physics checks for the three focused introductory lessons."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import BarycentricInterpolator
from typing_extensions import override

from examples.catalog import load_example

ROOT = Path(__file__).resolve().parents[1]


class LearningExamplesTests(unittest.TestCase):
    @classmethod
    @override
    def setUpClass(cls):
        cls.modules = {
            name: load_example(name)
            for name in ("move_stop", "constrained_descent", "soft_landing")
        }
        cls.problems = {name: module.solve() for name, module in cls.modules.items()}

    def test_each_lesson_has_one_model(self):
        self.assertEqual(
            set(self.modules["move_stop"].PARAMETERS), {"distance", "max_acceleration"}
        )
        self.assertEqual(self.problems["constrained_descent"].number_of_section, 1)
        self.assertEqual(self.problems["soft_landing"].number_of_states, [2, 2])
        for name, problem in self.problems.items():
            with self.subTest(lesson=name):
                self.assertTrue(problem.result.success, problem.result.message)
                np.testing.assert_allclose(
                    problem._equality_constraints(self.modules[name].PARAMETERS),
                    0,
                    atol=1e-7,
                )

    def test_move_stop_matches_minimum_time_and_acceleration_bounds(self):
        module = self.modules["move_stop"]
        for distance, acceleration in ((10.0, 2.0), (20.0, 1.0)):
            p = {"distance": distance, "max_acceleration": acceleration}
            problem = module.solve(p)
            self.assertTrue(problem.result.success)
            self.assertAlmostEqual(
                problem.time_final(-1), 2 * np.sqrt(distance / acceleration), places=7
            )
            np.testing.assert_allclose(problem.controls(0, 0), acceleration, atol=1e-6)
            np.testing.assert_allclose(problem.controls(0, 1), -acceleration, atol=1e-6)
            np.testing.assert_allclose(
                problem.states(1, 0)[-1], problem.states(1, 1)[0], atol=1e-7
            )

    def test_depth_constraint_and_arrival_time(self):
        problem = self.problems["constrained_descent"]
        p = self.modules["constrained_descent"].PARAMETERS
        np.testing.assert_allclose(
            [problem.states(0, 0)[-1], problem.states(1, 0)[-1]],
            [p["distance"], p["final_depth"]],
            atol=1e-7,
        )
        depth = BarycentricInterpolator(problem.tau[0], problem.states(1, 0), rng=0)(
            np.linspace(-1, 1, 1001)
        )
        # Bounds hold at collocation nodes; the interpolating curve is approximate.
        # Its overshoot between nodes varies with the platform's floating point
        # (1.4e-4 seen on CI), so allow 5e-4 here.
        self.assertLessEqual(max(problem.states(1, 0)), p["max_depth"] + 1e-8)
        self.assertLessEqual(max(depth), p["max_depth"] + 5e-4)
        radius = p["max_depth"] / 2
        angle = 2 * np.pi - np.arccos(1 - p["final_depth"] / radius)
        cruise = p["distance"] - radius * (angle - np.sin(angle))
        exact = angle * np.sqrt(radius / p["gravity"]) + cruise / np.sqrt(
            2 * p["gravity"] * p["max_depth"]
        )
        self.assertAlmostEqual(problem.time_final(0), exact, delta=3e-5)

    def test_constant_mass_landing_matches_impulse_reference(self):
        module = self.modules["soft_landing"]
        for changes in ({}, {"max_thrust": 3000.0}, {"initial_altitude": 200.0}):
            p = {**module.PARAMETERS, **changes}
            problem = module.solve(p)
            self.assertTrue(problem.result.success)
            braking = p["max_thrust"] / p["mass"] - p["gravity"]
            speed = np.sqrt(
                (p["initial_velocity"] ** 2 + 2 * p["gravity"] * p["initial_altitude"])
                * braking
                / (braking + p["gravity"])
            )
            exact_impulse = p["max_thrust"] * speed / braking
            actual_impulse = problem.result.fun * p["max_thrust"] * problem.unit_time
            self.assertAlmostEqual(actual_impulse, exact_impulse, places=4)
            np.testing.assert_allclose(
                [problem.states(0, 1)[-1], problem.states(1, 1)[-1]], 0, atol=1e-6
            )

    def test_independent_ode_integration_reaches_endpoints(self):
        for name, problem in self.problems.items():
            p = self.modules[name].PARAMETERS
            initial = [
                problem.states(i, 0)[0] for i in range(problem.number_of_states[0])
            ]
            for section in range(problem.number_of_section):
                time = problem.time[section]
                control = BarycentricInterpolator(
                    time, problem.controls(0, section), rng=0
                )

                def motion(t, x):
                    u = control(t)
                    if name == "move_stop":
                        return [x[1], u]
                    if name == "soft_landing":
                        return [x[1], u / p["mass"] - p["gravity"]]
                    return [
                        x[2] * np.sin(u),
                        x[2] * np.cos(u),
                        p["gravity"] * np.cos(u),
                    ]

                integrated = solve_ivp(
                    motion, [time[0], time[-1]], initial, rtol=1e-10, atol=1e-10
                )
                self.assertTrue(integrated.success)
                initial = integrated.y[:, -1]
                expected = [problem.states(i, section)[-1] for i in range(len(initial))]
                with self.subTest(lesson=name, section=section):
                    np.testing.assert_allclose(
                        initial,
                        expected,
                        atol=5e-4 if name == "constrained_descent" else 2e-5,
                    )

    def test_numbered_scripts_run_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            for script in (
                "02_move_stop.py",
                "04_constrained_descent.py",
                "09_soft_landing.py",
            ):
                with self.subTest(script=script):
                    result = subprocess.run(
                        [sys.executable, str(ROOT / "examples" / script)],
                        cwd=directory,
                        capture_output=True,
                        text=True,
                        check=True,
                        env={**os.environ, "MPLBACKEND": "Agg"},
                    )
                    self.assertIn("Time:", result.stdout)
