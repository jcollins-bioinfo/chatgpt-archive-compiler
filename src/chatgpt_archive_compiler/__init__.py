"""Public package interface for ChatGPT Archive Compiler."""

from chatgpt_archive_compiler.compiler import (
    CompileOptions,
    RedactionRule,
    VolumeMode,
    analyze_archive,
    compile_archive,
)
from chatgpt_archive_compiler.ingest import IngestLimits, SchemaMode, ingest_export_zip
from chatgpt_archive_compiler.serialization import read_archive_ir, write_archive_ir
from chatgpt_archive_compiler.version import __version__

__all__ = [
    "CompileOptions",
    "IngestLimits",
    "RedactionRule",
    "SchemaMode",
    "VolumeMode",
    "__version__",
    "analyze_archive",
    "compile_archive",
    "ingest_export_zip",
    "read_archive_ir",
    "write_archive_ir",
]
