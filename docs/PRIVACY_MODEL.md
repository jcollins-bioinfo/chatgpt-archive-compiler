# Application privacy model

Local deterministic mode is the default and makes no provider calls or telemetry requests. Uploads
are written only to the configured local session root. Real exports, normalized text, embeddings,
corrections, and rendered products remain highly sensitive. API keys are neither accepted nor
persisted by the current local-only vertical slice; the disabled enhanced-mode control makes the
future boundary visible without implying that consent is implemented.

Professional-safe mode uses explainable narrow rules for credentials and highly sensitive personal
phrasing, plus a three-way `include` / `exclude` / `review` decision. Review defaults to exclusion.
The filtered Archive IR is the sole input to semantic analysis, search content, HTML, PDF, and the
shareable bundle, preventing post-render-only filtering. Decisions are recorded in a private local
manifest, and the source ZIP is never modified or deleted.

The classifier is not anonymization and cannot guarantee suitability. Technical context can cause
false positives and subtle disclosures can cause false negatives. Users must review every artifact
before publication. Hosted deployment requires authentication, TLS, isolated workers, log
scrubbing, upload limits, storage lifecycle and deletion controls, secrets management, and an
explicit retention policy.
