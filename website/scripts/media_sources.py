"""Shared input fingerprints and output names for locally generated media.

Keep this module free of numerical and plotting dependencies: publication only
checks the committed outputs, without importing or solving the examples.
"""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_FILES = (
    "history.png",
    "scene.png",
    "animation.mp4",
    "animation.png",
    "summary.json",
)
CONCEPT_FILES = (
    "candidate_curves.png",
    "shooting_vs_collocation.png",
    "lgl_nodes.png",
    "interpolation.png",
    "interpolation.mp4",
    "interpolation_poster.png",
    "differentiation.png",
    "collocation.png",
    "convergence.png",
    "knots.png",
    "hash.txt",
)


def _source_hash(scripts, examples, root):
    files = [root / "uv.lock", root / "examples/catalog.py"]
    files += sorted((root / "src/OpenGoddard").rglob("*.py"))
    files += [root / "website/scripts" / name for name in scripts]
    files += [root / "examples" / name for name in examples]
    if "14_polar_tsto_taiki.py" in examples:
        files += sorted((root / "examples/data").glob("*.csv"))
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def example_hash(script: str, root: Path = ROOT) -> str:
    return _source_hash(
        (
            "build_media.py",
            "figures.py",
            "i18n.py",
            "media_sources.py",
            "specs.py",
            "style.py",
        ),
        (script,),
        root,
    )


def concepts_hash(root: Path = ROOT) -> str:
    return _source_hash(
        ("concepts.py", "i18n.py", "media_sources.py", "style.py"),
        ("03_brachistochrone.py", "07_goddard_0knot.py", "08_goddard_1knot.py"),
        root,
    )
