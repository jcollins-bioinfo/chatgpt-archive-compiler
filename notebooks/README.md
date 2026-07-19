# Notebooks

Notebooks demonstrate and validate package APIs; canonical implementation belongs under
`src/chatgpt_archive_compiler/`.

- `00_project_bootstrap_colab_v2.ipynb`: ingestion/diagnostic workflow. Repository code is
  cloned beneath `/content`, parent-only graphs are validated synthetically, and only real inputs
  and generated outputs persist in Google Drive.
- `01_compile_archive_colab.ipynb`: compact full compiler workflow. It combines re-ingestion,
  structural analysis, explicit redaction, document construction, monthly HTML/PDF rendering, and
  integrity checks while keeping canonical implementation in tested package modules.
- `02_semantic_atlas_colab.ipynb`: semantic organization and thematic-book workflow. It performs a
  privacy-gated, resumable analysis of conversation subjects, projects, purposes, relationships,
  and longitudinal themes, then calls package code to render the category-organized atlas.
- `00_project_bootstrap_colab.ipynb`: retained as the earlier immutable Drive-checkout bootstrap;
  new runs should use the ephemeral workflows above.

These three active notebooks cover the intended operational pipeline. A separate visual-QA
notebook will be added only if review of real rendered volumes proves it necessary.

Rules:

- use synthetic fixtures by default;
- never commit real exports or derived private artifacts;
- keep real-data cells disabled unless deliberately activated;
- promote reusable logic into typed, tested package modules.
