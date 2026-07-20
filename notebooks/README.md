# Notebooks

Notebooks are Colab interfaces over the typed package; canonical implementation lives under
`src/chatgpt_archive_compiler/`.

## Supported workflow

1. `00_project_bootstrap_colab_v2.ipynb` — ingest and diagnose a ChatGPT export.
2. `01_compile_archive_colab.ipynb` — compile chronological HTML and optional PDF volumes.
3. `02_semantic_atlas_budgeted_colab.ipynb` — run the explicitly authorized, cost-governed
   experimental semantic workflow (revision `budgeted-v5`).

Each supported notebook clones into ephemeral `/content`, resolves its configured repository ref
to an exact commit, installs from the committed lock-derived environment, validates a synthetic
fixture, and writes only private inputs/caches/outputs to Google Drive.

For a public repository, cloning is anonymous. `GITHUB_TOKEN` is optional and used only for a
private fork; it should have read-only Contents permission.

## Legacy provenance notebooks

- `00_project_bootstrap_colab.ipynb` records the earlier persistent-Drive checkout design.
- `02_semantic_atlas_colab.ipynb` records the original per-conversation external-analysis design,
  which is not cost-practical for a large archive.

They remain to explain project history, but new users should not run them. Git history, rather than
additional notebooks, will preserve future superseded revisions.

## Privacy rules

- Use synthetic fixtures by default.
- Never commit real exports or derived outputs.
- Treat Archive IR, caches, embeddings, ledgers, semantic tables, and books as private.
- Keep real-data cells disabled until deliberately acknowledged.
- Never print conversation titles, text, entity names, identifiers, or source-derived errors.
- Promote reusable logic into typed, tested package modules.
