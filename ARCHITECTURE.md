# Architecture

## Scope

This repository is organized as a local-first compiler pipeline. The Dash app is an interface around the package, not the primary place where parsing, normalization, analysis, redaction, or rendering logic should live.

## Pipeline

```text
export ZIP
  -> safe archive ingestion
  -> source file manifest
  -> schema probing
  -> conversation parsing
  -> normalized Archive IR
  -> deterministic analysis
  -> redaction and filtering
  -> document IR
  -> renderers
```

## Package layout

```text
chatgpt_archive_compiler/
  ingest/      Archive reading and source manifest construction.
  normalize/  Conversion from source JSON to Archive IR.
  analyze/    Deterministic and optional corpus analyses.
  redact/     Redaction rules, reports, and previewable transformations.
  render/     HTML and book/PDF rendering backends.
  app/        Dash UI shell.
  cli.py      CLI entry point over the same compiler core.
```

## Constraints

1. User exports contain private data.
2. The source schema may change over time.
3. Archives may be large enough to stress memory, browser uploads, and PDF builders.
4. Conversation exports may contain branches or regenerated messages.
5. Redaction can miss sensitive information and must not be presented as perfect.
6. The final artifact should be reproducible from a manifest and configuration.

## Intermediate representation

The project uses a canonical Archive IR instead of rendering directly from raw source JSON. This isolates schema drift and makes tests stable.

Core concepts:

- `Archive`
- `SourceManifest`
- `SourceFile`
- `Conversation`
- `Message`
- `ContentBlock`
- `ArchiveWarning`

## Rendering strategy

The near-term renderer should produce HTML previews first. The serious book renderer should remain separate, with LaTeX or Typst as the likely final backend.

The renderer should support front matter, table of contents, internal hyperlinks, role-aware message rendering, code blocks, URL indices, conversation indices, source manifest appendices, and redaction reports.

## Dash role

Dash should provide upload or path selection, manifest inspection, conversation browsing, filtering, redaction configuration, build configuration, progress reporting, and links to generated local artifacts.
