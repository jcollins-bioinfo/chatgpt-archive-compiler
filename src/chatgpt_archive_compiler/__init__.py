"""Public package interface for ChatGPT Archive Compiler."""

from chatgpt_archive_compiler.compiler import (
    CompileOptions,
    RedactionRule,
    VolumeMode,
    analyze_archive,
    compile_archive,
)
from chatgpt_archive_compiler.ingest import IngestLimits, SchemaMode, ingest_export_zip
from chatgpt_archive_compiler.semantic import (
    LocalHashingEmbeddingProvider,
    LocalHeuristicAnalysisProvider,
    OpenAIEmbeddingProvider,
    OpenAIStructuredAnalysisProvider,
    SemanticAtlasOptions,
    build_semantic_atlas,
    estimate_semantic_run,
)
from chatgpt_archive_compiler.semantic_book import (
    BookPaperSize,
    SemanticBookOptions,
    compile_semantic_book,
)
from chatgpt_archive_compiler.serialization import read_archive_ir, write_archive_ir
from chatgpt_archive_compiler.version import __version__

__all__ = [
    "BookPaperSize",
    "CompileOptions",
    "IngestLimits",
    "LocalHashingEmbeddingProvider",
    "LocalHeuristicAnalysisProvider",
    "OpenAIEmbeddingProvider",
    "OpenAIStructuredAnalysisProvider",
    "RedactionRule",
    "SchemaMode",
    "SemanticAtlasOptions",
    "SemanticBookOptions",
    "VolumeMode",
    "__version__",
    "analyze_archive",
    "build_semantic_atlas",
    "compile_archive",
    "compile_semantic_book",
    "estimate_semantic_run",
    "ingest_export_zip",
    "read_archive_ir",
    "write_archive_ir",
]
