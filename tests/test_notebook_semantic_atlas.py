"""Structural checks for the Colab semantic-atlas orchestration notebook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "02_semantic_atlas_colab.ipynb"
BUDGETED_NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1] / "notebooks" / "02_semantic_atlas_budgeted_colab.ipynb"
)


def _load_notebook() -> dict[str, Any]:
    """Load the checked-in notebook as JSON without requiring Jupyter at test time."""

    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _load_budgeted_notebook() -> dict[str, Any]:
    """Load the cost-controlled revision as JSON."""

    return json.loads(BUDGETED_NOTEBOOK_PATH.read_text(encoding="utf-8"))


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


def test_budgeted_semantic_notebook_is_clean_and_python_cells_compile() -> None:
    """The new notebook is a clean standalone Colab artifact with valid code cells."""

    notebook = _load_budgeted_notebook()
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["semantic_atlas_revision"] == "budgeted-v3"
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"budgeted_semantic_cell_{index}", "exec")


def test_budgeted_semantic_notebook_enforces_selective_analysis_and_hard_cost_gate() -> None:
    """Static controls preserve the low-cost architecture and pre-request authorization."""

    source = "\n".join("".join(cell["source"]) for cell in _load_budgeted_notebook()["cells"])
    assert 'REPO_BRANCH = "agent/rebuild-colab-workflow"' in source
    assert 'ANALYSIS_MODE = "budgeted"' in source
    assert 'EMBEDDING_MODEL = "text-embedding-3-small"' in source
    assert 'PROFILE_MODEL = "gpt-5.6-luna"' in source
    assert 'SYNTHESIS_MODEL = "gpt-5.6-terra"' in source
    assert "MAX_REFINED_CONVERSATIONS = 144" in source
    assert "REFINED_CONVERSATIONS_PER_CATEGORY = 3" in source
    assert "HARD_API_BUDGET_USD = 5.00" in source
    assert "EMBEDDING_TOKENS_PER_MINUTE = 800000" in source
    assert "SAFE_RESUME_SOURCE_COMMITS" in source
    assert "resume_migration.json" in source
    assert "Resume-selected matching Archive IR:" in source
    assert "Resume identity mismatch in fields:" in source
    assert "3340adf265f6b5fa997de004fa8e04da37edb6de" in source
    assert source.count("max_retries=0") == 4
    assert "estimate_budgeted_semantic_cost" in source
    assert "scheduled_plan_fits_hard_budget" in source
    assert 'ledger_path=PREPARED_RUN["output_directory"] / "api_budget_ledger.json"' in source
    assert "RoutedStructuredAnalysisProvider" in source
    assert "analysis_provider=baseline_provider" in source
    assert "refinement_provider=refinement_provider" in source
    assert "I AUTHORIZE A MAXIMUM CONFIGURED API COST OF" in source
    assert "PREPARE_REAL_ANALYSIS = False" in source
    assert "RUN_REAL_ANALYSIS = False" in source
    assert source.index("scheduled_plan_fits_hard_budget") < source.index(
        'get_colab_secret("OPENAI_API_KEY")'
    )
    assert "source-derived exception text was suppressed" in source
    assert "except SemanticAtlasError as exception" in source
