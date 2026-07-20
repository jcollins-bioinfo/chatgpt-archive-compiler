# Security policy

## Supported version

Only the latest commit on `main` and the newest tagged prerelease receive security fixes. This
research-alpha project is not a hosted service and has no production-support guarantee.

## Reporting a vulnerability

Use GitHub's **Report a vulnerability** flow to open a private security advisory. Do not include
archive text, credentials, tokens, manifests, or derived private artifacts in a public issue. If
private reporting is unavailable, contact the repository owner through the GitHub profile before
sharing technical details.

## Threat model

The principal risks are:

- disclosure of highly private source exports or derived artifacts;
- malicious ZIP paths, compression ratios, or resource exhaustion;
- uncontrolled external model calls and misleading retention assumptions;
- provider/model output treated as trusted code or facts;
- remote resource retrieval during HTML/PDF rendering;
- secrets written to logs, URLs, Git configuration, caches, or Drive; and
- overconfidence in regex redaction or generated semantic conclusions.

ZIPs are preflighted and read without extraction. Core compilation is local. Enhanced model mode is
separate, explicitly acknowledged, and bounded by an application-side configured-cost ledger.
Structured responses are validated and never executed.

## Sensitive artifacts

Treat every item below as private, even if it contains no obvious message text:

| Artifact | Why sensitive |
|---|---|
| Export ZIP and `conversations*.json` | Complete source history |
| Archive IR | Normalized text, identifiers, branches, metadata |
| `.semantic_cache` | Provider inputs/outputs and derived interpretations |
| Embeddings and graph | Derived behavioral/interest representation |
| Taxonomy, profiles, timelines, synthesis | Personal inferences |
| API ledger/run identity | Models, usage, timing, and workflow metadata |
| HTML/PDF books | Readable private archive and interpretations |
| Manifests | Paths, hashes, configuration, and corpus-scale metadata |

The repository ignores common artifact names, but ignore rules are not a security boundary. Keep
outputs outside the checkout and inspect staged files before every commit.

## Credentials

Use environment variables or Colab secrets. The OpenAI adapter and notebooks must never serialize
API keys. GitHub tokens are optional for public cloning and should be read-only when used for a
private fork.

## Redaction limitation

Regex replacements are an explicit rendering aid, not anonymization. They cannot reliably discover
all names, identifiers, indirect disclosures, or sensitive semantic inferences. Review every
artifact before sharing it.

## Public-release checklist

Before changing repository visibility or publishing a release:

1. scan the complete Git history for secrets and private artifacts;
2. inspect the release tree and generated distributions;
3. verify CI from the exact release commit;
4. use synthetic demonstration artifacts only; and
5. rotate any credential suspected of appearing in logs or history.

