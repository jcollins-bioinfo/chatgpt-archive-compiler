# Interactive semantic atlas task log

## 2026-07-21 implementation plan

- **Reusable components:** retain ZIP preflight/streamed ingestion, graph normalization, chronological
  HTML/PDF compiler, semantic providers/pipeline, manifests, hashing, verification, CLI, notebooks,
  and cost/safety controls.
- **App architecture:** a Dash application factory delegates to a framework-neutral service;
  isolated disk sessions and a small thread-backed local job registry keep long work off callback
  request paths. Dash stores opaque session/job identifiers, never archive content or API keys.
- **Semantic improvements:** add versioned evidence, project/theme, open-loop, contradiction, review,
  correction, and professional-safety schemas; keep the existing deterministic atlas as the baseline
  and expose its grounded artifacts rather than inventing unvalidated conclusions.
- **State/artifacts:** each random run identifier owns an upload and output directory. Corrections and
  professional-safety decisions are canonical local JSON. Completed outputs are verified before a
  scoped ZIP bundle is offered.
- **Privacy:** local mode is default. ZIP limits run before compilation; failures are content-free.
  Professional-safe classification occurs before semantic/chronological compilation, excludes
  uncertain records by default, and carries an explicit false-positive/false-negative warning.
  Remote mode remains opt-in and requires acknowledgements; secrets are memory-only.
- **Launch:** local Docker first, then a non-destructive `uv` bootstrap, then the `app` CLI command.
- **Testing:** factory/state isolation, preflight, jobs, sanitization, schemas/corrections, safety
  leakage boundary, deterministic generator, bundles, CLI, and an end-to-end service run.
- **Risks/deferred:** local sensitivity rules are intentionally conservative and require review;
  advanced merge/split editing and remote safety adjudication are deferred. Dash callback polling is
  process-local (not a distributed queue). PDF availability depends on WeasyPrint system libraries.
