"""Examples remain reusable and their objective gradients match scaled p."""

import contextlib
import io
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from OpenGoddard import Problem

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class ExampleTests(unittest.TestCase):
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
            contextlib.redirect_stdout(io.StringIO()),
        ):
            for path in sorted(EXAMPLES.glob("[0-9]*.py")):
                with self.subTest(example=path.name), self.assertRaises(Initialized):
                    runpy.run_path(str(path), run_name="__main__")

    def test_importing_examples_does_not_solve_or_open_figures(self):
        with (
            patch.object(Problem, "solve") as solve,
            patch("matplotlib.pyplot.figure") as figure,
        ):
            for path in sorted(EXAMPLES.glob("*.py")):
                with self.subTest(example=path.name):
                    namespace = runpy.run_path(str(path))
                    self.assertNotIn("prob", namespace)
            solve.assert_not_called()
            figure.assert_not_called()

    def test_analytic_objective_gradients(self):
        for pattern in ("01_*", "02_*", "05_*", "06_*", "07_*", "08_*"):
            path = next(EXAMPLES.glob(pattern + ".py"))
            namespace = runpy.run_path(str(path))
            obj = namespace.get("Ball", namespace.get("Rocket"))()
            phases = 2 if pattern.startswith(("05", "07")) else 1
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
