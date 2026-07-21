# Three-minute contest demonstration

## Prepare fictional data

```bash
uv run python examples/make_contest_demo_export.py \
  --output /tmp/chatgpt-archive-contest-demo.zip \
  --ground-truth /tmp/chatgpt-archive-contest-ground-truth.json \
  --seed 20260721
./scripts/run_app_local.sh
```

Compilation time depends on CPU, PDF libraries, and archive size; on a developer laptop the small
synthetic corpus is intended to complete on a demo timescale, but no duration is guaranteed.

## Click path

1. Establish the before-state: an opaque 72-conversation ZIP.
2. Upload it, point out safe ZIP preflight, local-only mode, and Professional-safe mode.
3. Compile and show coarse stages rather than invented percentages.
4. Show excluded and review counts, projects/categories, and the semantic atlas evidence links.
5. Download the verified professional-safe bundle; open its manifest, chronological index, atlas,
   and PDF. Search for the planted fictional credential to demonstrate absence.
6. Close on provenance, corrections as explicit supervision, and the original unchanged export.

If PDF generation fails, confirm the `pdf` extra and WeasyPrint system libraries. If the upload is
rejected, run `uv run chatgpt-archive inspect FILE.zip`; application errors intentionally omit
source-derived details.
