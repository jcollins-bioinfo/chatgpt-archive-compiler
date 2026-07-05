# Data Model Notes

## Archive IR v1

The Archive IR is the canonical internal representation used by the CLI, Dash app, tests, and renderers.

## Top-level objects

- `Archive`: top-level normalized archive.
- `SourceManifest`: list of discovered source files and ingestion warnings.
- `Conversation`: normalized conversation metadata and ordered messages.
- `Message`: normalized role, timestamp, model metadata, and content blocks.
- `ContentBlock`: typed message content unit.
- `ArchiveWarning`: non-fatal issue found during ingestion or normalization.

## Design principles

1. Preserve enough raw metadata for debugging.
2. Keep renderers independent of source-specific JSON shapes.
3. Prefer warning records over hard failures when possible.
4. Make deterministic tests possible with synthetic fixtures.
5. Avoid assuming all conversations are simple linear transcripts.

## Open questions

- How should alternate branches or regenerated messages be represented?
- Should attachments become first-class Archive IR objects?
- Should redactions mutate Archive IR or produce an overlay transform?
- Should conversation summaries be deterministic sidecars or part of the Archive IR?
