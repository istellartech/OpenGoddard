"""Check committed media for missing files, failed solves, and stale inputs.

Run with ``python3 website/scripts/check_media.py``. This uses only the standard
library and never imports the numerical library, renderers, or example models.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples.catalog import EXAMPLES  # noqa: E402
from website.scripts.media_sources import (  # noqa: E402
    CONCEPT_FILES,
    EXAMPLE_FILES,
    concepts_hash,
    example_hash,
)


def check_media(root: Path = ROOT) -> list[str]:
    errors = []
    media = root / "website/public/media"
    groups: list[tuple[str, tuple[str, ...]]] = [
        (example.id, EXAMPLE_FILES) for example in EXAMPLES
    ]
    groups.append(("concepts", CONCEPT_FILES))
    for locale in ("", "en"):
        for identifier, files in groups:
            out = media / locale / identifier
            for name in files:
                path = out / name
                if not path.is_file() or path.stat().st_size == 0:
                    errors.append(f"Missing or empty media: {path.relative_to(root)}")

        for example in EXAMPLES:
            path = media / locale / example.id / "summary.json"
            if not path.is_file():
                continue
            try:
                summary = json.loads(path.read_text())
            except (ValueError, UnicodeError):
                summary = None
            if not isinstance(summary, dict):
                errors.append(f"Invalid summary: {path.relative_to(root)}")
                continue
            if summary.get("hash") != example_hash(example.script, root):
                errors.append(f"Stale media: {path.parent.relative_to(root)}")
            if summary.get("success") is not True:
                errors.append(f"Unsuccessful solve: {path.relative_to(root)}")

        path = media / locale / "concepts/hash.txt"
        if path.is_file() and path.read_text().strip() != concepts_hash(root):
            errors.append(f"Stale media: {path.parent.relative_to(root)}")

    for example in EXAMPLES:
        paths = [media / locale / example.id / "summary.json" for locale in ("", "en")]
        if all(path.is_file() for path in paths):
            if paths[0].read_bytes() != paths[1].read_bytes():
                errors.append(f"Japanese/English summaries differ: {example.id}")
    return errors


def main():
    errors = check_media()
    if errors:
        sys.exit(
            "\n".join(errors)
            + "\nRegenerate locally with: uv run python website/scripts/build_media.py"
            + "\nThen commit website/public/media/ with the source changes."
        )
    print(
        "Checked committed media: 15 examples and theory figures (ja/en), all current."
    )


if __name__ == "__main__":
    main()
