FROM ghcr.io/astral-sh/uv:0.11.28-python3.12-bookworm-slim AS builder
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra app --extra semantic --extra pdf

FROM python:3.12-slim-bookworm AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends curl libpango-1.0-0 libpangoft2-1.0-0 && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 atlas
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --chown=atlas:atlas src ./src
RUN mkdir /app/output && chown atlas:atlas /app/output
ENV PATH=/app/.venv/bin:$PATH CHATGPT_ARCHIVE_OUTPUT=/app/output PYTHONUNBUFFERED=1
USER atlas
EXPOSE 8050
HEALTHCHECK --interval=20s --timeout=3s --start-period=15s --retries=3 CMD curl --fail http://127.0.0.1:8050/ || exit 1
CMD ["chatgpt-archive", "app", "--host", "0.0.0.0", "--port", "8050", "--output-directory", "/app/output"]
