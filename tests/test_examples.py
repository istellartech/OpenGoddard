"""Examples remain reusable and their objective gradients match scaled p."""

import contextlib
import io
import runpy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from examples.catalog import EXAMPLES as CATALOG
from examples.catalog import find_example, load_example
from OpenGoddard import Problem

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class ExampleTests(unittest.TestCase):
    def test_catalog_matches_numbered_files_and_learning_guide(self):
        scripts = [example.script for example in CATALOG]
        self.assertEqual(
            scripts,
            sorted(
                p.relative_to(EXAMPLES).as_posix()
                for p in EXAMPLES.glob("[0-9][0-9]_*/main.py")
            ),
        )
        self.assertEqual([s[:2] for s in scripts], [f"{i:02}" for i in range(1, 16)])
        self.assertEqual(len({example.id for example in CATALOG}), len(CATALOG))
        guide = (EXAMPLES / "README.rst").read_text()
        for example in CATALOG:
            self.assertIn(example.directory, guide)
            self.assertTrue((EXAMPLES / example.directory / "README.rst").is_file())
            self.assertEqual(find_example(example.id), example)
            self.assertEqual(find_example(example.script[:2]), example)
        self.assertFalse(list(EXAMPLES.glob("[0-9]*.py")))
        self.assertFalse((EXAMPLES / "learning").exists())
        self.assertFalse((EXAMPLES / "assets").exists())

    def test_all_numbered_examples_initialize_valid_transcriptions(self):
        class Initialized(Exception):
            pass

        def inspect(problem, obj=None, *args, **kwargs):
            problem.time_update()
            self.assertEqual(len(problem.dynamics), problem.number_of_section)
            self.assertTrue(np.isfinite(problem._objective(obj)))
            self.assertTrue(np.isfinite(problem._equality_constraints(obj)).all())
            self.assertTrue(np.isfinite(problem._inequality_constraints(obj)).all())
            if problem.number_of_states == [5, 5]:
                # Each stage's initial mass and thrust guesses use its own nodes.
                if obj is None:
                    raise AssertionError("rocket examples require a model object")
                np.testing.assert_allclose(problem.states(4, 1)[0], obj.M0[1])
                np.testing.assert_allclose(
                    problem.controls(1, 0)[0], problem.controls(0, 0)[0] / 9
                )
            raise Initialized

        with (
            patch.object(Problem, "solve", inspect),
            patch("matplotlib.pyplot.show"),
            patch("matplotlib.pyplot.savefig"),
            patch.object(Path, "mkdir"),
            patch.object(sys, "argv", ["example.py"]),
            patch.object(sys, "path", [str(EXAMPLES), *sys.path]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            for path in (EXAMPLES / example.script for example in CATALOG):
                with self.subTest(example=path.name), self.assertRaises(Initialized):
                    runpy.run_path(str(path), run_name="__main__")

    def test_importing_examples_does_not_solve_or_open_figures(self):
        with (
            patch.object(Problem, "solve") as solve,
            patch("matplotlib.pyplot.figure") as figure,
            patch.object(sys, "path", [str(EXAMPLES), *sys.path]),
        ):
            for path in sorted(EXAMPLES.rglob("*.py")):
                with self.subTest(example=path.name):
                    namespace = runpy.run_path(str(path))
                    self.assertNotIn("prob", namespace)
            solve.assert_not_called()
            figure.assert_not_called()

    def test_catalog_imports_the_actual_example_implementation(self):
        with patch.object(Problem, "solve") as solve:
            for example in CATALOG:
                module = load_example(example.id)
                if module.__file__ is None:
                    self.fail(f"missing implementation for {example.id}")
                self.assertEqual(
                    Path(module.__file__).resolve(), EXAMPLES / example.script
                )
                self.assertIs(load_example(example.directory[:2]), module)
            solve.assert_not_called()

    def test_analytic_objective_gradients(self):
        for identifier, phases in (
            ("brachistochrone", 1),
            ("brachistochrone_tokyo_osaka", 1),
            ("goddard_1knot", 2),
            ("rocket_ascent_single_stage", 1),
            ("rocket_ascent_two_stage", 2),
            ("rocket_ascent_polar_ssto", 1),
        ):
            path = EXAMPLES / find_example(identifier).script
            namespace = runpy.run_path(str(path))
            obj = namespace.get("Ball", namespace.get("Rocket"))()
            prob = Problem(
                np.arange(phases + 1), [4] * phases, [5] * phases, [2] * phases
            )
            prob.set_unit_time(7)
            for section in range(phases):
                for state in range(5):
                    prob.set_unit_states(
                        state, section, 10 * (state + 1) * (section + 1)
                    )
            prob.p[:] = np.linspace(1, 2, prob.number_of_variables)
            cost, gradient = namespace["cost"], namespace["cost_derivative"]
            expected = np.empty(prob.number_of_variables)
            for variable in range(prob.number_of_variables):
                original = prob.p[variable]
                prob.p[variable] = original + 1e-5
                upper = cost(prob, obj)
                prob.p[variable] = original - 1e-5
                lower = cost(prob, obj)
                prob.p[variable] = original
                expected[variable] = (upper - lower) / 2e-5
            with self.subTest(example=path.name):
                np.testing.assert_allclose(
                    gradient(prob, obj), expected, atol=1e-8, rtol=1e-7
                )
