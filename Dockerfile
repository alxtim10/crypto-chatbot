FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml PROJECT_PLAN.md ./
COPY src ./src

RUN pip install --no-cache-dir . && \
    groupadd --system appuser && \
    useradd --system --gid appuser --create-home appuser

USER appuser

CMD ["crypto-chatbot", "run-bot"]
