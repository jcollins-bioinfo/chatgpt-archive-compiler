# ChatGPT Archive Compiler

Local-first compiler for ChatGPT data exports: ingest, normalize, analyze, redact, and render
conversation archives into book-quality artifacts.

The current vertical slice converts an export ZIP into a loss-aware Archive IR and compiles that IR
into self-contained HTML and optional PDF volumes:

```python
from chatgpt_archive_compiler.ingest import ingest_export_zip
from chatgpt_archive_compiler.compiler import CompileOptions, compile_archive
from chatgpt_archive_compiler.serialization import write_archive_ir

archive = ingest_export_zip("chatgpt-export.zip")
write_archive_ir(archive, "archive.ir.json")
compile_archive(archive, "compiled", options=CompileOptions(render_pdf=True))
```

Real exports contain private data. The package makes no network calls, does not extract ZIP members,
and never logs message text. Synthetic data must be used in tests and committed examples.

## Architecture

The Dash application and notebooks are interfaces over the package, not owners of compiler logic:

```text
ZIP export -> source manifest -> Archive IR -> analysis IR -> document IR -> renderer(s)
```

Safe ingestion preserves every conversation mapping node and alternate/regenerated branch while
representing the declared current path separately. Numbered multipart exports are ingested
incrementally in numeric order under corpus-wide limits. Node IDs and message IDs remain distinct.
Compilation renders only declared current paths, omits reasoning and non-user-facing roles by
default, blocks remote asset loading, supports explicit regex redactions, and splits large archives
into annual volumes by default.

## Colab bootstrap

`notebooks/00_project_bootstrap_colab.ipynb` mounts Google Drive, resolves the selected branch to an
exact commit, and creates or reuses an immutable commit-specific checkout beneath
`MyDrive/ChatGPT Data Export/checkouts/chatgpt-archive-compiler`. It never modifies an existing
dirty checkout. The notebook installs the package from the exact checkout and exercises numbered
multipart ingestion with synthetic data. Real-export processing is disabled by default and requires
an explicit privacy acknowledgment.

`notebooks/00_project_bootstrap_colab_v2.ipynb` is the preferred follow-up workflow. It clones the
selected branch into ephemeral Colab storage under `/content`, validates parent-only graph
normalization, and keeps only source exports, diagnostics, and Archive IR outputs in Google Drive.

`notebooks/01_compile_archive_colab.ipynb` is the compact end-to-end workflow. It re-ingests the
source export with current schema support, performs structural analysis and optional explicit
redaction, then writes annual HTML/PDF volumes, an index, and a checksum manifest. No additional
operational notebook is planned unless real rendered output demonstrates a distinct QA need.

## Development checks

```bash
python -m pip install -e '.[dev]'
black --check .
ruff check .
mypy src/chatgpt_archive_compiler
pytest
```

The repository is pre-alpha. No real export, normalized IR, or rendered private artifact belongs in
Git history.
