# ChatGPT Archive Compiler

[![CI](https://github.com/jcollins-bioinfo/chatgpt-archive-compiler/actions/workflows/ci.yml/badge.svg?event=push)](https://github.com/jcollins-bioinfo/chatgpt-archive-compiler/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Upload your ChatGPT export and compile years of conversations into a **private, evidence-linked map**
of projects, ideas, decisions, recurring themes, and intellectual development—while retaining the
chronological HTML and PDF book as verified first-class exports.

## Launch the private interactive atlas

**Recommended local Docker:**

```bash
docker build -t chatgpt-archive-compiler .
docker run --rm -p 127.0.0.1:8050:8050 \
  -v "$PWD/archive-output:/app/output" chatgpt-archive-compiler
```

**One-command local launch:**

```bash
./scripts/run_app_local.sh
```

Then open <http://127.0.0.1:8050>. The interface provides ZIP safety preflight, an explicit privacy
boundary, stage-based background compilation, project/theme overview, evidence-linked semantic
atlas, Professional-safe filtering, and a verified download bundle.

```text
OPAQUE EXPORT ZIP  →  SAFE PREFLIGHT  →  PROJECTS · THEMES · TIMELINES · EVIDENCE
                                              ↓
                      VERIFIED CHRONOLOGICAL HTML · PDF · SEMANTIC ATLAS
```

> **Privacy first.** Local mode has no telemetry or model calls. Real exports and every derivative
> are highly sensitive. Professional-safe filtering is conservative and fallible; review outputs
> before sharing. Tests, examples, and documentation use fictional synthetic data only.

See [application architecture](docs/APP_ARCHITECTURE.md), [privacy model](docs/PRIVACY_MODEL.md),
[deployment](docs/DEPLOYMENT.md), [evaluation](docs/SEMANTIC_EVALUATION.md), and the exact
[three-minute demo](docs/CONTEST_DEMO.md).

> **Research alpha.** ZIP ingestion, normalization, privacy boundaries, caching, cost accounting,
> and artifact integrity are extensively tested. Semantic usefulness and editorial quality are not
> yet validated. The first large private run completed operationally but did not meet the operator's
> semantic or editorial quality bar.

This independent project is not affiliated with, endorsed by, or supported by OpenAI. “ChatGPT” and
“OpenAI” are trademarks of their respective owner.

## Synthetic interactive demonstration

The demonstration contains only fictional conversations, makes no network calls, and writes a
verifiable local result.

```bash
git clone https://github.com/jcollins-bioinfo/chatgpt-archive-compiler.git
cd chatgpt-archive-compiler

python -m pip install "uv==0.11.28"
uv sync --frozen --extra dev --extra semantic

uv run python examples/make_contest_demo_export.py \
  --output /tmp/chatgpt-archive-contest-demo.zip \
  --ground-truth /tmp/chatgpt-archive-contest-ground-truth.json \
  --seed 20260721
uv run chatgpt-archive app
```

Upload `/tmp/chatgpt-archive-contest-demo.zip`. It contains 72 fictional conversations across six
longitudinal domains, alternate branches, open loops, and planted Professional-safe cases. The
separate ground truth supports evaluation and is never embedded in the export.

## What it does

```text
export ZIP
  │
  ├─ local safety preflight and loss-aware normalization
  │      └─ canonical Archive IR
  │
  ├─ deterministic chronological compiler
  │      └─ self-contained HTML and optional PDF volumes
  │
  └─ experimental semantic pipeline
         ├─ local representations and embeddings
         ├─ similarity graph and communities
         ├─ optional bounded model refinement
         ├─ taxonomy, timelines, synthesis, and review queue
         └─ print-oriented analytical draft
```

The package preserves complete mapping graphs, alternate/regenerated branches, source provenance,
and structured warnings. Renderers select only the declared current path by default and exclude
reasoning-interface records, system messages, tool messages, and remote assets unless explicitly
enabled.

## Privacy and execution modes

| Mode | Network use | Data leaving the runtime | Intended use |
|---|---:|---|---|
| Ingest and compile | None | Nothing | Faithful local archive and chronological volumes |
| Local semantic baseline | None | Nothing | Deterministic pipeline preview and reproducibility check |
| OpenAI-enhanced semantic mode | Explicit opt-in | Bounded text for every embedding; bounded full text for selected representatives; compressed dossiers for global stages | Experimental categorization and synthesis |

Enhanced mode uses `store=False`, disables tools and hidden retries, requires explicit retention
and cost acknowledgements, and keeps a persistent application-side configured-cost ledger. Those
controls do not make external processing anonymous, Zero Data Retention, or a guarantee about the
provider's final bill. Use one active process per ledger/output directory.

Every source export, Archive IR, cache, ledger, taxonomy, synthesis, rendered book, and output
manifest should be treated as private derived data. None belongs in Git.

## CLI

Install the compiler only:

```bash
uv sync --frozen
```

Install semantic and PDF support when required:

```bash
uv sync --frozen --extra semantic --extra pdf
```

The supported commands are:

```text
chatgpt-archive inspect        Inspect ZIP safety and member metadata without extraction
chatgpt-archive ingest         Write canonical Archive IR
chatgpt-archive compile        Render chronological HTML and optional PDF
chatgpt-archive semantic-local Build the deterministic local semantic baseline
chatgpt-archive run-local      Run the complete network-free workflow
chatgpt-archive verify         Verify every recorded artifact digest
```

For example:

```bash
uv run chatgpt-archive inspect ~/Downloads/chatgpt-export.zip
uv run chatgpt-archive ingest ~/Downloads/chatgpt-export.zip ./private/archive.ir.json
uv run chatgpt-archive compile ./private/archive.ir.json ./private/compiled
```

CLI failures expose the stage and exception class without printing source-derived exception text.

## Output and provenance

A `run-local` directory has this shape:

```text
run_manifest.json                  SHA-256 inventory and sanitized run configuration
archive.ir.json                    loss-aware normalized archive
compiled/
  analysis.json                    structural counts
  compilation_manifest.json        package/source provenance and volume checksums
  index.html
  archive-*.html
semantic/
  semantic_manifest.json           providers, options, source fingerprint, artifact checksums
  conversation_catalog.parquet
  embeddings.parquet
  semantic_graph.parquet
  taxonomy.json
  category_profiles.json
  project_timelines.json
  cross_archive_synthesis.md
  review_queue.json
  semantic_atlas.html
```

The local workflow is deterministic at the algorithm level and records exact artifact hashes.
Hosted model calls are inherently nondeterministic; preserving a compatible cache can reproduce an
existing artifact, while repeating a remote call need not reproduce its wording or judgments.
System fonts, WeasyPrint/Pango, and platform differences can also change PDF bytes and pagination.

See [Reproducibility](docs/REPRODUCIBILITY.md) for the precise guarantee.

## Colab workflows

Canonical Colab interfaces live under [notebooks](notebooks/README.md):

1. `00_project_bootstrap_colab_v2.ipynb` — ingest and diagnose.
2. `01_compile_archive_colab.ipynb` — compile chronological HTML/PDF.
3. `02_semantic_atlas_budgeted_colab.ipynb` — explicitly authorized experimental semantic mode.

They clone into ephemeral `/content`, resolve the selected repository ref to an exact commit, and
keep private inputs/caches/outputs in Google Drive. Earlier notebook revisions remain for provenance
but are not recommended entry points.

## What is validated

| Area | Status |
|---|---|
| ZIP traversal/resource protections | Tested |
| Multipart ingestion and graph preservation | Tested with synthetic fixtures |
| Canonical Archive IR and schema validation | Tested |
| Local rendering and remote-asset exclusion | Tested |
| Structured-output schemas and identifier reconciliation | Tested |
| Resume caches and configured-cost accounting | Tested |
| Semantic category usefulness | **Not validated** |
| Longitudinal insight quality | **Not validated** |
| Factual accuracy of generated synthesis | **Not validated** |
| Cross-platform PDF byte/layout identity | **Not guaranteed** |

Model- or heuristic-produced “confidence” values are review signals, not calibrated probabilities.
Generated timeline and synthesis statements require human review against referenced conversations.

Read [AI engineering](docs/AI_ENGINEERING.md) for the system design and
[Evaluation](docs/EVALUATION.md) for the quality-validation plan.

## Development

```bash
python -m pip install "uv==0.11.28"
uv sync --frozen --extra dev --extra semantic --extra pdf
uv run black --check .
uv run ruff check .
uv run mypy src/chatgpt_archive_compiler
uv run pytest
uv build
uv run twine check dist/*
```

CI tests supported Python 3.11–3.13, validates notebook JSON, runs the synthetic local workflow,
builds wheel and source distributions, and smoke-tests the installed CLI.

Contributions must use synthetic data. See [CONTRIBUTING.md](CONTRIBUTING.md) and
[SECURITY.md](SECURITY.md).

## Citation and license

Citation metadata is provided in [CITATION.cff](CITATION.cff). The source is available under the
[MIT License](LICENSE).

Current development version: `0.4.0a1`.
