# ChatGPT Archive Compiler

Local-first compiler for ChatGPT data exports: ingest, normalize, analyze, redact, and render
conversation archives into book-quality artifacts.

The first functional vertical slice converts an export ZIP into a loss-aware, versioned Archive IR:

```python
from chatgpt_archive_compiler.ingest import ingest_export_zip
from chatgpt_archive_compiler.serialization import write_archive_ir

archive = ingest_export_zip("chatgpt-export.zip")
write_archive_ir(archive, "archive.ir.json")
```

Real exports contain private data. The package makes no network calls, does not extract ZIP members,
and never logs message text. Synthetic data must be used in tests and committed examples.

## Architecture

The Dash application and notebooks are interfaces over the package, not owners of compiler logic:

```text
ZIP export -> source manifest -> Archive IR -> analysis IR -> document IR -> renderer(s)
```

The first functional slice is deliberately limited to safe ingestion and loss-aware normalization.
It preserves every conversation mapping node and alternate/regenerated branch while representing the
declared current path separately. Node IDs and message IDs remain distinct.

## Colab bootstrap

`notebooks/00_project_bootstrap_colab.ipynb` mounts Google Drive, securely clones or fast-forwards
the private feature branch into `MyDrive/ChatGPT Data Export/chatgpt-archive-compiler`, installs the
package from that durable checkout, and exercises the public API with a branched synthetic export.
Real-export processing is disabled by default and requires an explicit privacy acknowledgment.

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

