"""Package import smoke tests."""

from importlib.metadata import version as distribution_version

from chatgpt_archive_compiler import (
    LocalHashingEmbeddingProvider,
    LocalWorkflowOptions,
    SemanticAtlasOptions,
    __version__,
    build_semantic_atlas,
    compile_archive,
    compile_semantic_book,
    estimate_semantic_run,
    ingest_export_zip,
    run_local_workflow,
    verify_local_workflow,
    write_archive_ir,
)


def test_public_interface_is_defined() -> None:
    """Version and first vertical-slice functions are importable at package level."""

    assert __version__ == "0.4.0a2"
    assert distribution_version("chatgpt-archive-compiler") == __version__
    assert callable(compile_archive)
    assert callable(build_semantic_atlas)
    assert callable(compile_semantic_book)
    assert callable(estimate_semantic_run)
    assert callable(ingest_export_zip)
    assert callable(run_local_workflow)
    assert callable(verify_local_workflow)
    assert callable(write_archive_ir)
    assert SemanticAtlasOptions()
    assert LocalHashingEmbeddingProvider()
    assert LocalWorkflowOptions()
