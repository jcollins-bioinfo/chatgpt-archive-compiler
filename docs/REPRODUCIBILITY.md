# Reproducibility

## Reproduction levels

This project distinguishes mechanical reproducibility from semantic repeatability.

| Stage | Same inputs and locked environment | Important caveat |
|---|---|---|
| ZIP ingestion and Archive IR | Deterministic | Source ZIP bytes and configuration must match |
| Chronological HTML | Deterministic | Package and dependency versions must match |
| Local embeddings/profiles/graph | Deterministic | Graph backend and options must match |
| Hosted embeddings | Normally stable for a fixed model | Provider aliases and service behavior can change |
| Hosted structured analysis | Not bit-for-bit deterministic | Preserve compatible caches to replay an artifact |
| PDF rendering | Not byte/layout portable | Fonts, WeasyPrint, Pango, OS, and metadata can differ |

“Reproducible” in this repository means that exact source bytes, configuration, code version,
dependency lock, and artifact hashes can be recorded and checked. It does not mean that a fresh
hosted-model call or a PDF rendered on another operating system will be identical.

## Frozen local setup

Supported Python versions are 3.11 through 3.13.

```bash
python -m pip install "uv==0.11.28"
uv sync --frozen --extra dev --extra semantic
```

Add `--extra pdf` only when PDF output is required. `uv sync --frozen` refuses to change the
committed lockfile. The lockfile contains exact distributions and hashes; `pyproject.toml` alone is
not the reproducible environment.

## Deterministic demonstration

```bash
uv run python examples/make_synthetic_export.py /tmp/synthetic-chatgpt-export.zip
uv run chatgpt-archive run-local \
  /tmp/synthetic-chatgpt-export.zip \
  /tmp/chatgpt-archive-demo
uv run chatgpt-archive verify /tmp/chatgpt-archive-demo
```

The generator fixes JSON ordering, compression mode, member permissions, and ZIP member timestamp.
The workflow hashes the ZIP, writes canonical Archive IR, records its sanitized configuration, and
records the SHA-256 and size of every durable artifact in `run_manifest.json`.

`verify` reads no conversation text into its output. It reports only missing files and relative
paths with size or digest mismatches.

## Hosted-model replay

For an enhanced semantic run, preserve together:

- the exact Archive IR;
- repository commit and package version;
- output directory, including `.semantic_cache`;
- `semantic_manifest.json`;
- `api_budget_ledger.json`;
- notebook run identity and configuration; and
- price snapshot and authorization text shown during that run.

Compatible caches can replay already accepted structured results without another request. Deleting a
cache and calling the provider again is a new experiment whose wording, categories, usage, and price
may differ.

Current manifests record package/source/provider/options/artifact metadata but do not yet capture
the complete Python platform, lockfile digest, renderer/font stack, prompt/schema digests, returned
provider model revision, graph backend, or API ledger digest. The top-level local workflow manifest
adds Python version and cross-stage artifact hashes; it is still not a container-level provenance
record.

## Colab

Canonical notebooks clone into ephemeral `/content`, resolve the configured repository ref to an
exact commit, and persist private inputs/caches/outputs only in Google Drive. A Colab base image can
change. For a durable scientific result, record the resolved commit and preserve the output
manifests and caches.

## PDF boundary

HTML is the preferred reproducible reading artifact. PDF output uses system font fallbacks and the
host WeasyPrint/Pango stack. The project does not bundle fonts or a pinned container, so page breaks,
glyph metrics, metadata, and PDF bytes can differ across machines even when the HTML is identical.

