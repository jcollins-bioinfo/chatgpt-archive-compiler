# Notebooks

Notebooks demonstrate and validate package APIs; canonical implementation belongs under
`src/chatgpt_archive_compiler/`.

- `00_project_bootstrap_colab.ipynb`: immutable, commit-specific private-repository checkout in
  Google Drive plus synthetic numbered-multipart ZIP → Archive IR validation. Existing dirty or
  incompatible checkouts are left untouched.
- `00_project_bootstrap_colab_v2.ipynb`: preferred ephemeral-checkout workflow. Repository code is
  cloned beneath `/content`, parent-only graphs are validated synthetically, and only real inputs
  and generated outputs persist in Google Drive.
- Later notebooks will address corpus analysis, redaction, document construction, and rendering.

Rules:

- use synthetic fixtures by default;
- never commit real exports or derived private artifacts;
- keep real-data cells disabled unless deliberately activated;
- promote reusable logic into typed, tested package modules.
