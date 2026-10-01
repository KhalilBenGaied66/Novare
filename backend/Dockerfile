# Build context: the repository root (see docker-compose.yml).
FROM python:3.11-slim

# The image mirrors the repository layout: the code locates the repository root
# three directories above backend/app/core/.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/srv/backend \
    DATA_DIR=/srv/data \
    EVALS_DIR=/srv/evals

WORKDIR /srv
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

RUN useradd --create-home --uid 10001 app
COPY backend/app backend/app
COPY data/sample_docs data/sample_docs
COPY data/reference data/reference
COPY evals/golden_set.json evals/thresholds.json evals/
# index/ and models/ are written at runtime (volumes in docker-compose.yml).
RUN mkdir -p data/index data/models evals/reports && chown -R app:app /srv/data /srv/evals
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
