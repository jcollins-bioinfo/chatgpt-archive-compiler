# Deployment

## Recommended: local Docker

```bash
docker build -t chatgpt-archive-compiler .
docker run --rm -p 127.0.0.1:8050:8050 \
  -v "$PWD/archive-output:/app/output" chatgpt-archive-compiler
```

The image uses a non-root runtime user, locked Python dependencies, a health check, and no baked-in
secret. Allow roughly 2–4 GiB RAM for a typical archive; unusually large exports may require more.
The app accepts at most 512 MiB per HTTP request by default.

## One-command local bootstrap

```bash
./scripts/run_app_local.sh
```

## Developer launch

```bash
uv sync --frozen --extra app --extra semantic --extra pdf --extra dev
uv run chatgpt-archive app
```

Public hosting is not the default. An operator considering it must control authentication, TLS,
request size, storage and deletion lifecycle, worker isolation, access logs, retention, secrets,
and authorization. The repository does not supply a production multi-user security boundary.
