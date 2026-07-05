# Notebooks

Colab and Jupyter notebooks are useful for controlled prototyping, demonstrations, and provenance-preserving exploratory work.

They should not become the canonical implementation. Package code belongs under `src/chatgpt_archive_compiler/`.

## Rules

- Use synthetic fixtures by default.
- Do not commit real exports.
- Do not commit generated PDFs from real exports.
- Keep notebooks small and purpose-specific.
- Promote reusable logic into the package quickly.

## Planned notebooks

- `00_project_bootstrap_colab.ipynb`
- `01_parse_export_zip_colab.ipynb`
- `02_normalize_conversations_colab.ipynb`
- `03_dash_app_prototype_colab.ipynb`
- `04_pdf_rendering_prototype_colab.ipynb`
