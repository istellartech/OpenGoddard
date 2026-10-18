"""Lesson imports are quiet and all transcriptions have finite residuals."""

import ast
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
    def test_catalog_matches_numbered_files(self):
        scripts = [example.script for example in CATALOG]
        self.assertEqual(
            scripts,
            sorted(
                p.relative_to(EXAMPLES).as_posix()
                for p in EXAMPLES.glob("[0-9][0-9]_*.py")
            ),
        )
        self.assertEqual([s[:2] for s in scripts], [f"{i:02}" for i in range(1, 16)])
        self.assertEqual(len({example.id for example in CATALOG}), len(CATALOG))
        for example in CATALOG:
            self.assertFalse((EXAMPLES / example.stem).exists())
            self.assertEqual(find_example(example.id), example)
            self.assertEqual(find_example(example.script[:2]), example)
        self.assertFalse((EXAMPLES / "learning").exists())
        self.assertFalse((EXAMPLES / "assets").exists())

    def test_lessons_are_self_contained(self):
        for example in CATALOG:
            tree = ast.parse((EXAMPLES / example.script).read_text())
            with self.subTest(example=example.id):
                for node in ast.walk(tree):
                    self.assertNotIsInstance(node, ast.Try)
                    if isinstance(node, ast.ImportFrom):
                        self.assertEqual(node.level, 0)
                        self.assertFalse(
                            (node.module or "").startswith(("examples", "_"))
                        )
                    if isinstance(node, ast.Import):
                        self.assertFalse(
                            any(
                                alias.name.startswith(("examples", "_"))
                                for alias in node.names
                            )
                        )

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
                self.assertIs(load_example(example.stem[:2]), module)
            solve.assert_not_called()

    def test_first_lesson_matches_analytic_solution(self):
        module = load_example("minimum_energy_transfer")
        for target in (2.0, 3.0):
            with self.subTest(target=target):
                problem = module.make_problem(target)
                result = problem.solve(target, maxiter=100, ftol=1e-11)
                self.assertTrue(result.success, result.message)
                self.assertAlmostEqual(result.fun, 2 * target, places=7)
                self.assertAlmostEqual(problem.time_final(0) - 5, target, places=5)
                np.testing.assert_allclose(problem.controls(0, 0), 1, atol=1e-5)
