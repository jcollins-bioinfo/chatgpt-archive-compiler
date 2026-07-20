# Contributing

Contributions are welcome when they preserve the project's local-first privacy boundary and use
only fictional or explicitly redistributable fixtures.

## Environment

The committed `uv.lock` is authoritative:

```bash
python -m pip install "uv==0.11.28"
uv sync --frozen --extra dev --extra semantic --extra pdf
uv run pre-commit install
```

Do not refresh the lockfile as a side effect of unrelated work. When dependencies intentionally
change, explain the change and include the updated lockfile in the same pull request.

## Branch and pull-request workflow

Branch from green `main`. Human-authored branches may use `feature/<purpose>`; automated work may
use `agent/<purpose>`.

Before requesting review:

```bash
uv run black --check .
uv run ruff check .
uv run mypy src/chatgpt_archive_compiler
uv run pytest
uv build
uv run twine check dist/*
```

Use compact imperative commit messages and keep conceptual changes reviewable. Never commit a real
export, Archive IR, cache, embedding table, ledger, HTML/PDF book, or any artifact derived from
private data.

## Code standards

- Public interfaces and non-trivial private helpers require type annotations.
- Public classes/functions need NumPy-style docstrings with inputs, outputs, and failure modes.
- Pydantic models should forbid undeclared fields and suppress input values in validation errors.
- File writes that establish durable state should be atomic.
- Private-data errors and progress callbacks must remain content-free.
- Network use must be explicit at the provider boundary.
- Model output is untrusted data: validate shapes and all deterministic identifiers.
- Add focused synthetic tests for every schema, safety, cache, cost, or reconciliation change.

Black uses a 100-character line length. Ruff enforces imports and common correctness rules. Strict
mypy is required.

## Notebook policy

Notebooks are interfaces and provenance artifacts, never the canonical implementation. Reusable
logic belongs under `src/chatgpt_archive_compiler/`.

Canonical notebooks must:

- clone into ephemeral storage;
- resolve and record an exact source commit;
- use the frozen dependency environment;
- use anonymous public cloning by default;
- contain no outputs or execution counts;
- exercise synthetic data before enabling private data; and
- gate every external-data transmission with explicit acknowledgement.

## AI changes and evaluation

A schema-valid model response is not evidence of analytical quality. Pull requests changing prompts,
sampling, taxonomy, graph rules, or synthesis should state:

- the intended quality hypothesis;
- what information reaches each provider stage;
- the maximum request/cost effect;
- cache-compatibility consequences;
- a synthetic evaluation case; and
- known failure modes.

Do not describe provider confidence as calibrated probability or call generated text grounded unless
the cited source references have been checked.

## Documentation

Update the changelog, public version, CLI help, manifests, notebooks, and system-card documentation
when their behavior changes. Avoid claims such as production-ready, comprehensive understanding, or
book-quality unless supported by a declared evaluation.

