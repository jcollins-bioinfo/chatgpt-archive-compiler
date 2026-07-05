# ChatGPT Archive Compiler

Local-first compiler for ChatGPT data exports: parse, analyze, redact, and render conversation archives into book-quality PDFs.

This repository is currently private and under active design/development.

## Product intent

The project is intended to become a robust, locally run Python/Dash application that accepts the ZIP archive produced by ChatGPT account data export and converts the user's conversation corpus into structured, reviewable, redacted, and typographically polished artifacts.

Primary target outputs:

- a high-quality PDF archive with front matter, table of contents, internal links, metadata pages, appendices, and indices;
- a normalized intermediate archive representation suitable for testing and reproducibility;
- optional sidecar artifacts such as CSV/JSON manifests, search indices, and redaction reports;
- a local Dash UI for upload, inspection, filtering, redaction, analysis, and export.

## Architectural principle

The Dash app should orchestrate the pipeline, not own it. The durable asset is the compiler core:

```text
ZIP export -> source manifest -> normalized Archive IR -> analysis IR -> document IR -> renderer(s)
```

The same core should be usable from:

- the Dash app;
- a CLI;
- Colab notebooks for controlled prototyping;
- tests and future CI workflows.

## Privacy posture

Default behavior must be local-first:

- no telemetry;
- no external API calls by default;
- no cloud LLM calls unless explicitly enabled;
- no real exports committed to the repository;
- synthetic fixtures only in tests and examples.

See `PRIVACY.md` and `SECURITY.md` before handling real exports.

## Development status

Pre-alpha scaffold. The first implementation target is deterministic ingestion and normalization of ChatGPT export ZIPs, followed by a minimal CLI and Dash inspection UI.

## Quick start for development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
pytest
```

Inspect a ZIP once the CLI implementation is ready:

```bash
chatgpt-archive inspect ~/Downloads/chatgpt_export.zip
```

## License

License is intentionally not yet finalized while the repository remains private. Choose an open-source license before public release.
