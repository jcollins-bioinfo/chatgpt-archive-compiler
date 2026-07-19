"""Structural checks for the Colab semantic-atlas orchestration notebook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "02_semantic_atlas_colab.ipynb"


def _load_notebook() -> dict[str, Any]:
    """Load the checked-in notebook as JSON without requiring Jupyter at test time."""

    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_semantic_atlas_notebook_is_clean_and_python_cells_compile() -> None:
    """Ensure the committed notebook contains no outputs and has valid Python cells."""

    notebook = _load_notebook()
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"semantic_notebook_cell_{index}", "exec")


def test_semantic_atlas_notebook_preserves_privacy_and_reproducibility_gates() -> None:
    """Keep real-data, external-processing, and ephemeral-checkout safeguards visible."""

    source = "\n".join("".join(cell["source"]) for cell in _load_notebook()["cells"])
    assert 'REPO_BRANCH = "agent/rebuild-colab-workflow"' in source
    assert "PREPARE_REAL_ANALYSIS = False" in source
    assert "RUN_REAL_ANALYSIS = False" in source
    assert 'get_colab_secret("GITHUB_TOKEN")' in source
    assert 'get_colab_secret("OPENAI_API_KEY")' in source
    assert 'dir="/content"' in source
    assert 'f"{REPO_DIR}[notebooks,pdf,semantic]"' in source
    assert "source-derived exception text was suppressed" in source
    assert "estimate_semantic_run" in source
    assert "LocalHashingEmbeddingProvider" in source
    assert "compile_semantic_book" in source
    assert "compilation_manifest.json" in source
    assert "estimated_total_cost_usd" in source
    assert "PRICE_SNAPSHOT_DATE" in source
    assert "MAX_LEAF_CATEGORIES = 64" in source
    assert "MAX_EXPANDED_BOOK_PROFILES = 240" in source
    assert "progress_callback=semantic_progress" in source
    assert "resume=True" not in source
    assert '"cache_directory": cache_directory' not in source
    assert "store=False" in source
    assert "up to 30 days" in source
