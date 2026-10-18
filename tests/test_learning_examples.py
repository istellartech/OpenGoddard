"""Physics, parameter changes, and UI/export contracts of the learning lessons."""

import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import override
from unittest.mock import patch

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import BarycentricInterpolator
from scipy.optimize import OptimizeResult

from examples.catalog import load_example
from OpenGoddard import Problem

ROOT = Path(__file__).resolve().parents[1]
move_stop = load_example("move_stop")
soft_landing = load_example("soft_landing")
constrained_descent = load_example("constrained_descent")
MODULES = (move_stop, soft_landing, constrained_descent)


def duration(trajectory):
    return trajectory.phases[-1]["nodes"]["time"][-1]


class LearningExamplesTests(unittest.TestCase):
    @classmethod
    @override
    def setUpClass(cls):
        cls.results = {
            (module.__name__, name): module.solve(parameters, samples=61)
            for module in MODULES
            for name, parameters in module.PRESETS.items()
        }

    def result(self, module, name="default"):
        return self.results[module.__name__, name]

    def test_presets_pass_between_node_checks_and_export_physical_data(self):
        for key, result in self.results.items():
            with self.subTest(preset=key):
                self.assertTrue(result.solver["success"])
                self.assertTrue(
                    result.diagnostics["sampled_checks_passed"], result.diagnostics
                )
                data = json.loads(json.dumps(result.to_dict(), allow_nan=False))
                self.assertEqual(data["schema_version"], 1)
                previous_end = None
                for phase in data["phases"]:
                    for grid in ("nodes", "samples"):
                        values = phase[grid]
                        self.assertTrue(np.all(np.diff(values["time"]) > 0))
                        self.assertTrue(
                            all(len(v) == len(values["time"]) for v in values.values())
                        )
                        for label in values:
                            self.assertIn(label, result.units)
                    self.assertEqual(len(phase["samples"]["time"]), 61)
                    if previous_end is not None:
                        self.assertAlmostEqual(
                            previous_end, phase["samples"]["time"][0]
                        )
                    previous_end = phase["samples"]["time"][-1]

    def test_move_stop_references_and_parameter_effects(self):
        base = self.result(move_stop)
        self.assertAlmostEqual(duration(base), 2 * np.sqrt(10 / 2), places=8)
        limited = self.result(move_stop, "speed_limit")
        self.assertEqual(len(limited.phases), 3)
        self.assertAlmostEqual(duration(limited), 10 / 2 + 2 / 2, places=8)
        self.assertGreater(duration(limited), duration(base))
        self.assertGreater(
            duration(self.result(move_stop, "longer_move")), duration(base)
        )
        for name in ("minimum_effort", "limited_effort", "speed_limited_effort"):
            result = self.result(move_stop, name)
            self.assertAlmostEqual(
                result.solver["objective"], result.reference["minimum_effort"], places=7
            )
            self.assertAlmostEqual(
                duration(result), result.parameters["duration"], places=8
            )
        self.assertAlmostEqual(
            self.result(move_stop, "minimum_effort").solver["objective"],
            12 * 10**2 / 6**3,
            places=8,
        )
        self.assertLess(
            self.result(move_stop, "minimum_effort").solver["objective"],
            self.result(move_stop, "limited_effort").solver["objective"],
        )
        # Distinct controls at a shared time survive serialization.
        self.assertAlmostEqual(base.phases[0]["samples"]["acceleration"][-1], 2)
        self.assertAlmostEqual(base.phases[1]["samples"]["acceleration"][0], -2)

    def test_landing_references_and_rocket_equation(self):
        base = self.result(soft_landing)
        self.assertAlmostEqual(
            duration(base), base.reference["constant_mass"]["duration"], places=7
        )
        self.assertAlmostEqual(
            base.diagnostics["impulse_Ns"],
            base.reference["constant_mass"]["impulse"],
            places=5,
        )
        self.assertGreater(
            duration(self.result(soft_landing, "higher_start")), duration(base)
        )
        lower = self.result(soft_landing, "lower_thrust")
        self.assertLess(
            lower.phases[0]["nodes"]["time"][-1], base.phases[0]["nodes"]["time"][-1]
        )
        variable = self.result(soft_landing, "variable_mass")
        self.assertAlmostEqual(
            variable.diagnostics["propellant_used_kg"],
            variable.reference["variable_mass"]["propellant_used"],
            places=5,
        )
        p = soft_landing.PRESETS["variable_mass"]
        for phase in variable.phases:
            data = phase["samples"]
            expected_mass = p.initial_mass * np.exp(
                -(
                    np.asarray(data["velocity"])
                    - p.initial_velocity
                    + p.gravity * np.asarray(data["time"])
                )
                / p.exhaust_velocity
            )
            np.testing.assert_allclose(data["mass"], expected_mass, atol=2e-5)
            self.assertGreaterEqual(min(data["mass"]), p.dry_mass - 1e-5)

    def test_floor_activation_and_gravity_scaling(self):
        free = self.result(constrained_descent, "no_floor")
        inactive = self.result(constrained_descent, "inactive_floor")
        floor = self.result(constrained_descent)
        shallow = self.result(constrained_descent, "shallow_floor")
        self.assertEqual(len(free.phases), 1)
        self.assertFalse(inactive.reference["floor_active"])
        self.assertAlmostEqual(duration(free), duration(inactive), places=8)
        self.assertLess(duration(free), duration(floor))
        self.assertLess(duration(floor), duration(shallow))
        for result in (free, floor, shallow):
            self.assertAlmostEqual(
                duration(result), result.reference["minimum_time"], places=7
            )
            end = result.phases[-1]["samples"]
            self.assertAlmostEqual(
                end["x"][-1], result.parameters["distance"], places=7
            )
            self.assertAlmostEqual(
                end["depth"][-1], result.parameters["final_depth"], places=7
            )
            cap = result.parameters["max_depth"]
            if cap is not None:
                self.assertLessEqual(
                    max(max(p["samples"]["depth"]) for p in result.phases), cap + 1e-7
                )
        lower_g = self.result(constrained_descent, "lower_gravity")
        self.assertAlmostEqual(
            duration(lower_g) / duration(floor), np.sqrt(9.81 / 1.62), places=7
        )
        for a, b in zip(floor.phases, lower_g.phases, strict=True):
            np.testing.assert_allclose(
                a["samples"]["depth"], b["samples"]["depth"], atol=1e-7
            )

    def test_independent_ode_integration_reaches_exported_endpoints(self):
        cases = [
            (move_stop, "limited_effort", ["position", "velocity"], "acceleration"),
            (soft_landing, "variable_mass", ["altitude", "velocity", "mass"], "thrust"),
            (constrained_descent, "default", ["x", "speed"], "theta"),
        ]
        for module, preset, states, control in cases:
            p = module.PRESETS[preset]
            trajectory = self.result(module, preset)
            initial = np.array([trajectory.phases[0]["nodes"][s][0] for s in states])
            for section, phase in enumerate(trajectory.phases):
                data = phase["nodes"]
                time = np.array(data["time"])
                control_poly = BarycentricInterpolator(time, data[control], rng=0)

                def dynamics(
                    t,
                    x,
                    *,
                    module=module,
                    control_poly=control_poly,
                    p=p,
                    section=section,
                ):
                    return module.pointwise_dynamics(
                        x, np.array([control_poly(t)]), t, p, section, np
                    )

                integrated = solve_ivp(
                    dynamics, [time[0], time[-1]], initial, rtol=1e-10, atol=1e-10
                )
                self.assertTrue(integrated.success)
                initial = integrated.y[:, -1]
                with self.subTest(module=module.__name__, section=section):
                    np.testing.assert_allclose(
                        initial, [data[s][-1] for s in states], atol=2e-6, rtol=1e-8
                    )

    def test_solver_recovers_from_perturbed_guesses(self):
        for module, p in (
            (move_stop, move_stop.Parameters()),
            (soft_landing, soft_landing.Parameters()),
            (constrained_descent, constrained_descent.Parameters(max_depth=None)),
        ):
            problem = module.make_problem(p, nodes=8)
            problem.p += np.random.default_rng(4).normal(
                0, 0.001, problem.number_of_variables
            )
            result = problem.solve(p, derivatives="structured", maxiter=100, ftol=1e-9)
            with self.subTest(module=module.__name__):
                self.assertTrue(result.success, result.message)
                self.assertGreater(result.nit, 1)
                np.testing.assert_allclose(
                    problem._equality_constraints(p), 0, atol=1e-7
                )

    def test_invalid_parameters_are_rejected_before_solving(self):
        cases = [
            (move_stop.Parameters(), {"distance": float("nan")}),
            (move_stop.Parameters(), {"max_acceleration": 0}),
            (move_stop.Parameters(), {"distance": True}),
            (move_stop.Parameters(), {"speed_limit": -1}),
            (move_stop.Parameters(), {"objective": "unknown"}),
            (move_stop.Parameters(), {"objective": "minimum_effort", "duration": 1}),
            (soft_landing.Parameters(), {"max_thrust": 100}),
            (soft_landing.Parameters(), {"initial_altitude": 1}),
            (soft_landing.Parameters(), {"initial_velocity": 1}),
            (soft_landing.Parameters(), {"dry_mass": 1100}),
            (constrained_descent.Parameters(), {"max_depth": 0.1}),
            (constrained_descent.Parameters(), {"gravity": float("inf")}),
        ]
        for p, changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(p, **changes)
        with self.assertRaisesRegex(ValueError, "insufficient propellant"):
            soft_landing.make_problem(
                soft_landing.Parameters(variable_mass=True, dry_mass=999)
            )
        for module in MODULES:
            with self.assertRaises(ValueError):
                module.make_problem(nodes=3)

    def test_import_and_construction_do_not_solve_or_plot(self):
        with (
            patch.object(Problem, "solve") as solver,
            patch("matplotlib.pyplot.figure") as figure,
        ):
            for module in MODULES:
                importlib.reload(module)
                module.make_problem()
            solver.assert_not_called()
            figure.assert_not_called()

    def test_failed_solve_is_not_exported_as_a_success(self):
        def fail(problem, *args, **kwargs):
            problem.result = OptimizeResult(
                success=False, message="intentional failure"
            )
            return problem.result

        with patch.object(Problem, "solve", fail):
            for module in MODULES:
                with self.assertRaisesRegex(RuntimeError, "intentional failure"):
                    module.solve()

    def test_module_cli_exports_and_reports_bad_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "examples.02_move_stop.main",
                    "--set",
                    "distance=20",
                    "--samples",
                    "17",
                    "--output",
                    str(path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            data = json.loads(path.read_text())
            self.assertEqual(data["parameters"]["distance"], 20)
            self.assertEqual(len(data["phases"][0]["samples"]["time"]), 17)
            self.assertTrue(
                json.loads(result.stdout)["diagnostics"]["sampled_checks_passed"]
            )
            bad = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "examples.02_move_stop.main",
                    "--set",
                    "unknown=1",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(bad.returncode, 2)
            self.assertIn("unknown", bad.stderr)

    def test_numbered_entrypoints_export_from_another_working_directory(self):
        cases = (
            ("02_move_stop/main.py", "move_stop"),
            ("04_constrained_descent/main.py", "constrained_descent"),
            ("09_soft_landing/main.py", "soft_landing"),
        )
        with tempfile.TemporaryDirectory() as directory:
            for script, identifier in cases:
                with self.subTest(script=script):
                    path = Path(directory) / f"{identifier}.json"
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "examples" / script),
                            "--samples",
                            "17",
                            "--output",
                            str(path),
                        ],
                        cwd=directory,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    self.assertEqual(
                        json.loads(path.read_text())["example"], identifier
                    )
                    self.assertTrue(
                        json.loads(result.stdout)["diagnostics"][
                            "sampled_checks_passed"
                        ]
                    )
