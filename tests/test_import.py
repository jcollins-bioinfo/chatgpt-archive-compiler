"""Package import smoke tests."""

from chatgpt_archive_compiler import (
    __version__,
    compile_archive,
    ingest_export_zip,
    write_archive_ir,
)


def test_public_interface_is_defined() -> None:
    """Version and first vertical-slice functions are importable at package level."""

    assert __version__
    assert callable(compile_archive)
    assert callable(ingest_export_zip)
    assert callable(write_archive_ir)
