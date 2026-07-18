# Notebooks

Notebooks demonstrate and validate package APIs; canonical implementation belongs under
`src/chatgpt_archive_compiler/`.

- `00_project_bootstrap_colab.ipynb`: durable private-repository clone in Google Drive plus a
  synthetic ZIP → Archive IR validation.
- Later notebooks will address corpus analysis, redaction, document construction, and rendering.

Rules:

- use synthetic fixtures by default;
- never commit real exports or derived private artifacts;
- keep real-data cells disabled unless deliberately activated;
- promote reusable logic into typed, tested package modules.

