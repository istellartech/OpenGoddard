import os
import subprocess
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    subprocess.run(
        ["sphinx-apidoc", "-f", "-o", "docs_sphinx/apis", "src/OpenGoddard"],
        cwd=root,
        check=True,
        env={
            **os.environ,
            "SPHINX_APIDOC_OPTIONS": (
                "members,undoc-members,show-inheritance,imported-members"
            ),
        },
    )
    subprocess.run(["make", "html"], cwd=root / "docs_sphinx", check=True)
