"""Fail when a release tree contains likely private artifacts or literal credential prefixes."""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRECTORIES = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "build", "dist"}
FORBIDDEN_NAMES = (
    "conversations.json",
    "conversations-*.json",
    "*.ir.json",
    "api_budget_ledger.json",
    "run_identity.json",
    "semantic_manifest.json",
    "semantic_book_manifest.json",
    "run_manifest.json",
    "taxonomy.json",
    "category_profiles.json",
    "project_timelines.json",
    "review_queue.json",
    "semantic_atlas.html",
    "semantic-atlas.html",
)
FORBIDDEN_DIRECTORIES = {".semantic_cache", "outputs", "private_data", "source_exports"}
TEXT_SUFFIXES = {
    ".cff",
    ".css",
    ".html",
    ".ini",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
CREDENTIAL_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?<![A-Za-z0-9])ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"(?<![A-Za-z0-9])github_pat_[A-Za-z0-9_]{20,}"),
)


def _tracked_candidates() -> list[Path]:
    """Return release-tree files while excluding local tooling/build directories."""

    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not any(part in SKIP_DIRECTORIES for part in path.relative_to(ROOT).parts)
    ]


def main() -> None:
    """Report forbidden artifact names or literal credential-like values and exit nonzero."""

    problems: list[str] = []
    for path in _tracked_candidates():
        relative = path.relative_to(ROOT).as_posix()
        if any(part in FORBIDDEN_DIRECTORIES for part in path.relative_to(ROOT).parts):
            problems.append(f"private-artifact-directory:{relative}")
            continue
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in FORBIDDEN_NAMES):
            problems.append(f"private-artifact-name:{relative}")
            continue
        if path.suffix.casefold() not in TEXT_SUFFIXES or path.stat().st_size > 5_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(pattern.search(text) for pattern in CREDENTIAL_PATTERNS):
            problems.append(f"credential-pattern:{relative}")
    if problems:
        raise SystemExit("\n".join(sorted(problems)))
    print("Release-tree privacy scan passed.")


if __name__ == "__main__":
    main()
