## Summary

Describe the change and the user/developer outcome.

## Privacy and data boundary

- [ ] Uses only synthetic or explicitly redistributable fixtures.
- [ ] Adds no export, Archive IR, cache, ledger, embedding, HTML, or PDF derived from private data.
- [ ] Documents any new network transmission or sensitive artifact.

## AI/provider impact

- [ ] No provider/prompt/sampling behavior changes.
- [ ] Or: provider inputs, structured schema, cost ceiling, cache invalidation, and failure modes are described below.

## Validation

- [ ] `uv lock --check`
- [ ] `uv run black --check .`
- [ ] `uv run ruff check .`
- [ ] `uv run mypy src/chatgpt_archive_compiler`
- [ ] `uv run pytest`
- [ ] Synthetic end-to-end workflow, when relevant
- [ ] Wheel/sdist build, when packaging changes

