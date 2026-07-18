# Contributing

This repository is private during early development, but contributions should be made as if the project were already public-facing scientific software.

## Branch workflow

Use `main` as the stable branch. It should remain green except for urgent administrative hotfixes.

Use feature branches for all substantive work:

```text
feature/<short-purpose>
```

Recommended flow:

```text
1. Branch from green main.
2. Make coherent commits with narrow scope.
3. Confirm formatting, linting, and tests locally.
4. Open a draft pull request into main.
5. Iterate until CI is green.
6. Merge only after review.
```

## Local checks

Install development dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
pre-commit install
```

Before committing, run:

```bash
black .
ruff check .
pytest
```

Or run all configured pre-commit hooks:

```bash
pre-commit run --all-files
```

## Code style

All Python code must be Black-formatted with a repository line length of 100 characters.

Ruff is used for linting and import hygiene. If Black and Ruff disagree, refactor the code rather than weakening the checks unless there is a documented reason.

Use explicit type annotations for public functions and for private functions whose behavior is non-trivial, security-sensitive, or schema-sensitive.

## Docstrings

Every Python module must start with a concise module docstring.

Every public class and public function must have a docstring describing:

- purpose;
- important parameters;
- return value;
- important failure modes or warnings.

Private helpers should also have docstrings when their behavior is non-obvious, security-relevant, parser-related, redaction-related, or renderer-related.

## Notebook policy

Notebooks are allowed as reproducible prototypes and provenance artifacts. They are not the canonical implementation.

Notebook rules:

- Use synthetic fixtures by default.
- Do not commit real export data.
- Do not commit rendered private PDFs.
- Do not commit notebook outputs unless the output is intentionally part of documentation.
- Promote reusable logic into `src/chatgpt_archive_compiler/` quickly.
- Keep notebooks small, linear, and executable from a clean runtime.

## Privacy rules

Do not commit source export ZIPs, normalized real archives, rendered real PDFs, local databases, cache files, secrets, or any file derived from private user data.

The `.gitignore` is intentionally aggressive, but it is not a substitute for manual review.

## Commit style

Use compact imperative commit messages, for example:

```text
Add safe ZIP fixture tests
Define Archive IR branch policy
Implement HTML preview renderer
```

Prefer small, reviewable commits grouped by conceptual change.
