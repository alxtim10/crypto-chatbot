FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml PROJECT_PLAN.md ./
COPY src ./src

RUN apt-get update && \
    apt-get install -y --no-install-recommends postgresql-client && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir . && \
    groupadd --system appuser && \
    useradd --system --gid appuser --create-home appuser

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["crypto-chatbot", "health-check"]

CMD ["crypto-chatbot", "run-bot"]
