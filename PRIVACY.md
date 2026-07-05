# Privacy Model

## Default stance

This project is intended to be local-first. A user's export archive can contain sensitive information. The default implementation must not transmit source archives, normalized data, rendered outputs, or derived analysis to external services.

## Local mode requirements

- No telemetry.
- No analytics beacons.
- No external API calls by default.
- No cloud model calls by default.
- No repository fixtures derived from real user exports.
- No generated archives committed to git.

## Colab and notebook caveat

Colab is useful for prototyping but is not local execution. Real exports should not be uploaded to Colab unless the user deliberately accepts that privacy model. Notebooks should use synthetic fixtures by default.

## Redaction caveat

Automated redaction is a best-effort transformation, not a guarantee. The project should provide redaction previews and reports, but users must manually review outputs before sharing or publishing generated artifacts.

## Files that must not be committed

Examples include source export ZIPs, normalized archives from real users, rendered PDFs from real users, local databases, cache files, and any file containing private content.

The `.gitignore` is intentionally aggressive, but it is not a substitute for manual review.

## Future optional model integrations

Optional model-assisted analysis may be added later. Any such integration must be explicit, disabled by default, documented, and separable from deterministic local analysis.
