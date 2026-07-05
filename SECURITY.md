# Security Policy

## Status

This repository is pre-alpha and private. Do not treat it as production software yet.

## Threat model

The main security risks are accidental disclosure of private archives, unsafe archive extraction, uncontrolled external calls, and overconfident redaction.

## Archive handling requirements

- Inspect ZIP members before extraction.
- Reject path traversal entries.
- Apply size and file-count limits.
- Prefer reading archive members directly instead of extracting full archives.
- Write temporary files only into an application-controlled workspace.
- Never write user-derived artifacts into the repository tree by default.

## Dependency posture

- Keep runtime dependencies minimal.
- Prefer deterministic local processing.
- Add CI checks before public release.
- Review PDF/rendering dependencies carefully because they process complex text and markup.

## Reporting issues

While private, track security concerns as GitHub issues. Before public release, replace this section with a normal disclosure policy.
