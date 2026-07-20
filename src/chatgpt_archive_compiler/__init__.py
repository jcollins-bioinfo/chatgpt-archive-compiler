"""Public package interface for ChatGPT Archive Compiler.

The public ``__version__`` value is sourced from :mod:`chatgpt_archive_compiler.version`.
"""

from chatgpt_archive_compiler.compiler import (
    CompileOptions,
    RedactionRule,
    VolumeMode,
    analyze_archive,
    compile_archive,
)
from chatgpt_archive_compiler.ingest import IngestLimits, SchemaMode, ingest_export_zip
from chatgpt_archive_compiler.semantic import (
    ApiBudget,
    LocalHashingEmbeddingProvider,
    LocalHeuristicAnalysisProvider,
    ModelTokenPrice,
    OpenAIEmbeddingProvider,
    OpenAIStructuredAnalysisProvider,
    RoutedStructuredAnalysisProvider,
    SemanticAtlasOptions,
    build_semantic_atlas,
    estimate_budgeted_semantic_cost,
    estimate_semantic_run,
)
from chatgpt_archive_compiler.semantic_book import (
    BookPaperSize,
    SemanticBookOptions,
    compile_semantic_book,
)
from chatgpt_archive_compiler.serialization import read_archive_ir, write_archive_ir
from chatgpt_archive_compiler.version import __version__
from chatgpt_archive_compiler.workflows import (
    LocalWorkflowManifest,
    LocalWorkflowOptions,
    LocalWorkflowResult,
    RuntimeProvenance,
    WorkflowVerification,
    run_local_workflow,
    verify_local_workflow,
)

__all__ = [
    "BookPaperSize",
    "ApiBudget",
    "CompileOptions",
    "IngestLimits",
    "LocalHashingEmbeddingProvider",
    "LocalHeuristicAnalysisProvider",
    "LocalWorkflowManifest",
    "LocalWorkflowOptions",
    "LocalWorkflowResult",
    "ModelTokenPrice",
    "OpenAIEmbeddingProvider",
    "OpenAIStructuredAnalysisProvider",
    "RedactionRule",
    "RoutedStructuredAnalysisProvider",
    "RuntimeProvenance",
    "SchemaMode",
    "SemanticAtlasOptions",
    "SemanticBookOptions",
    "VolumeMode",
    "WorkflowVerification",
    "__version__",
    "analyze_archive",
    "build_semantic_atlas",
    "compile_archive",
    "compile_semantic_book",
    "estimate_semantic_run",
    "estimate_budgeted_semantic_cost",
    "ingest_export_zip",
    "read_archive_ir",
    "run_local_workflow",
    "verify_local_workflow",
    "write_archive_ir",
]
