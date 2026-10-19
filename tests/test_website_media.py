"""Publication must reject missing or stale media without running a solver."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from examples.catalog import EXAMPLES
from website.scripts.check_media import check_media
from website.scripts.media_sources import (
    CONCEPT_FILES,
    EXAMPLE_FILES,
    concepts_hash,
    example_hash,
)

REPO = Path(__file__).resolve().parents[1]


class WebsiteMediaTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for directory in ("examples", "src/OpenGoddard", "website/scripts"):
            shutil.copytree(
                REPO / directory,
                self.root / directory,
                ignore=shutil.ignore_patterns("__pycache__", "output"),
            )
        shutil.copyfile(REPO / "uv.lock", self.root / "uv.lock")
        self.media = self.root / "website/public/media"
        for locale in ("", "en"):
            for example in EXAMPLES:
                out = self.media / locale / example.id
                out.mkdir(parents=True)
                for name in EXAMPLE_FILES:
                    (out / name).write_bytes(b"asset")
                (out / "summary.json").write_text(
                    json.dumps(
                        {
                            "success": True,
                            "hash": example_hash(example.script, self.root),
                        }
                    )
                )
            out = self.media / locale / "concepts"
            out.mkdir(parents=True)
            for name in CONCEPT_FILES:
                (out / name).write_bytes(b"asset")
            (out / "hash.txt").write_text(concepts_hash(self.root))

    def test_check_needs_no_installed_dependencies_and_no_regeneration(self):
        # -S hides site-packages, including NumPy, SciPy, and Matplotlib.
        result = subprocess.run(
            [sys.executable, "-S", "website/scripts/check_media.py"],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        # Documentation and checker edits do not invalidate rendered media.
        (self.root / "README.md").write_text("Updated documentation")
        with (self.root / "website/scripts/check_site.py").open("a") as file:
            file.write("\n# Additional link check\n")
        self.assertEqual(check_media(self.root), [])

    def test_changed_example_and_input_data_require_regeneration(self):
        with (self.root / "examples" / EXAMPLES[0].script).open("a") as file:
            file.write("\n# Changed model\n")
        errors = check_media(self.root)
        self.assertEqual(len(errors), 2)
        self.assertTrue(all("minimum_energy_transfer" in error for error in errors))
        csv = next((self.root / "examples/data").glob("*.csv"))
        with csv.open("a") as file:
            file.write("\n")
        errors = check_media(self.root)
        self.assertEqual(len(errors), 4)
        self.assertEqual(sum("polar_tsto_taiki" in error for error in errors), 2)

    def test_changed_library_or_lock_requires_all_media_to_be_updated(self):
        for path in (
            "src/OpenGoddard/__init__.py",
            "website/scripts/i18n.py",
            "uv.lock",
        ):
            with self.subTest(path=path):
                source = self.root / path
                original = source.read_bytes()
                source.write_bytes(original + b"\n# Changed input\n")
                errors = check_media(self.root)
                self.assertEqual(len(errors), 32)
                self.assertTrue(all("Stale media:" in error for error in errors))
                source.write_bytes(original)

    def test_missing_english_video_and_theory_figure_are_reported(self):
        for relative in ("en/move_stop/animation.mp4", "concepts/convergence.png"):
            (self.media / relative).unlink()
        errors = check_media(self.root)
        self.assertEqual(len(errors), 2)
        self.assertTrue(any("en/move_stop/animation.mp4" in error for error in errors))
        self.assertTrue(any("concepts/convergence.png" in error for error in errors))

    def test_invalid_or_failed_summaries_and_different_languages_are_rejected(self):
        path = self.media / "en/move_stop/summary.json"
        summary = json.loads(path.read_text())
        summary["success"] = False
        path.write_text(json.dumps(summary))
        errors = check_media(self.root)
        self.assertTrue(any("Unsuccessful solve" in error for error in errors))
        self.assertTrue(any("summaries differ" in error for error in errors))
        path.write_text("broken json")
        errors = check_media(self.root)
        self.assertTrue(any("Invalid summary" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
