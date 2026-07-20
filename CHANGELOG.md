# Changelog

All notable changes are documented here. Versions follow PEP 440 and semantic-versioning intent
while the project remains pre-1.0.

## [Unreleased]

## [0.3.0a1] - 2026-07-20

### Added

- Reproducible local end-to-end workflow orchestration with a content-free SHA-256 artifact manifest and verification command.
- Complete CLI commands for ingestion, compilation, deterministic local semantic analysis, end-to-end execution, and integrity verification.
- Public release documentation covering AI engineering, reproducibility boundaries, evaluation status, security, citation, and licensing.
- A deterministic synthetic export generator and one-command demonstration path.

### Changed

- Reposition the semantic book as an experimental analytical draft after its first large real-world run did not meet the operator's quality bar.
- Reduce the default dependency surface and use the committed lockfile in CI.
- Make supported Colab workflows resolve `main` anonymously by default while retaining optional authentication for private forks.
- Record the package version and source archive digest in chronological compilation manifests.

### Security

- Expand private-artifact ignore rules and document the sensitivity of source exports, caches, ledgers, semantic outputs, and rendered books.


## [0.2.0a2] - 2026-07-19

### Fixed

- Reconcile structured synthesis category and conversation identifiers against deterministic local
  IDs instead of discarding an otherwise valid, paid response when bounded model-reference drift
  occurs.
- Report content-free structured-response outcomes such as output exhaustion, content filtering,
  refusal, and provider exception class without revealing archive text.
- Permit an interrupted run to increase its persistent API ceiling only after a new exact budget
  authorization, while continuing to forbid implicit changes and every decrease.
- Preserve resume compatibility across the recovery revision and keep the budgeted Colab notebook
  Black-formatted.

## [0.2.0a1] - 2026-07-19

### Added

- Cross-archive semantic atlas with complete conversation representations, an embedding graph,
  hierarchical taxonomy, review queue, and thematic HTML/PDF book.
- Budget-conscious Colab workflow with local baseline profiling, bounded representative
  refinement, explicit privacy and cost authorization, a persistent API ledger, and resumable
  caches.
- Token-aware embedding batches, local rate pacing, safe exception diagnostics, and
  checksum-directed resume selection.

### Changed

- Numbered multipart ChatGPT exports are normalized under corpus-wide limits with improved graph
  anomaly diagnostics.
- Package metadata now reads its version from one canonical source and tests verify that the public
  runtime version matches the installed distribution metadata.

## [0.1.0a1] - 2026-07-18

### Added

- Initial loss-aware ChatGPT export ingestion, Archive IR serialization, structural diagnostics,
  redaction, and chronological HTML/PDF compilation.
- Initial Colab bootstrap and compiler workflows.

