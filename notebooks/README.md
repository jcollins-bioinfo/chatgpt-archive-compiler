# Notebooks

Notebooks demonstrate and validate package APIs; canonical implementation belongs under
`src/chatgpt_archive_compiler/`.

- `00_project_bootstrap_colab.ipynb`: immutable, commit-specific private-repository checkout in
  Google Drive plus synthetic numbered-multipart ZIP → Archive IR validation. Existing dirty or
  incompatible checkouts are left untouched.
- Later notebooks will address corpus analysis, redaction, document construction, and rendering.

Rules:

- use synthetic fixtures by default;
- never commit real exports or derived private artifacts;
- keep real-data cells disabled unless deliberately activated;
- promote reusable logic into typed, tested package modules.
