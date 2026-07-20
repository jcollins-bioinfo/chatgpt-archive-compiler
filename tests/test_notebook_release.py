"""Release-readiness checks shared by every checked-in Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

NOTEBOOK_DIRECTORY = Path(__file__).resolve().parents[1] / "notebooks"


def _notebooks() -> list[tuple[Path, dict[str, Any]]]:
    """Load every notebook in a stable path order."""

    return [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(NOTEBOOK_DIRECTORY.glob("*.ipynb"))
    ]


def test_all_notebooks_are_clean_and_compile() -> None:
    """Every committed code cell is output-free and syntactically valid."""

    for path, notebook in _notebooks():
        assert notebook["nbformat"] == 4
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile("".join(cell["source"]), f"{path.name}_cell_{index}", "exec")


def test_all_notebooks_use_shareable_locked_checkout_setup() -> None:
    """Notebooks default to public main, optional auth, and lock-derived dependencies."""

    for path, notebook in _notebooks():
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        assert 'REPO_BRANCH = "main"' in source, path
        assert "agent/rebuild-colab-workflow" not in source, path
        assert "get_optional_github_token" in source, path
        assert '"uv==0.11.28"' in source, path
        assert '"--frozen"' in source, path
        assert '"--no-emit-project"' in source, path
