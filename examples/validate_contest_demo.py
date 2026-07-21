"""Run the real local semantic pipeline and write contest acceptance diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chatgpt_archive_compiler.ingest import ingest_export_zip
from chatgpt_archive_compiler.semantic import (
    LocalHashingEmbeddingProvider,
    LocalHeuristicAnalysisProvider,
    SemanticAtlasOptions,
    build_semantic_atlas,
)
from chatgpt_archive_compiler.semantic.evaluation import evaluate_contest_atlas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    archive = ingest_export_zip(args.archive, compute_archive_hash=True)
    local = LocalHeuristicAnalysisProvider()
    result = build_semantic_atlas(
        archive,
        args.output_directory,
        embedding_provider=LocalHashingEmbeddingProvider(),
        analysis_provider=local,
        interpretation_provider=local,
        options=SemanticAtlasOptions(max_leaf_categories=12),
    )
    truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    validation = evaluate_contest_atlas(result.atlas, truth)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(validation.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.report)
    if not validation.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
