FROM python:3.12.7 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

RUN python -m venv .venv
COPY requirements.txt ./
RUN .venv/bin/pip install -r requirements.txt

FROM python:3.12.7-slim
WORKDIR /app
COPY --from=builder /app/.venv .venv/
COPY . .

# Loud logging so startup/import errors are visible in fly logs
CMD ["/app/.venv/bin/gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--keep-alive", "120", "--timeout", "180", "--bind", "0.0.0.0:8000", "main:app"]