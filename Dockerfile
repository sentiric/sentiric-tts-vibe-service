# sentiric-tts-vibe-service/Dockerfile
# Dockerfile

# --- GLOBAL BUILD ARGÜMANLARI ---
ARG PYTHON_VERSION=3.11
ARG BASE_IMAGE_TAG=${PYTHON_VERSION}-slim-bullseye
ARG GIT_COMMIT="unknown"
ARG BUILD_DATE="unknown"
ARG SERVICE_VERSION="0.0.0"

# --- STAGE 1: Production ---
FROM python:${BASE_IMAGE_TAG}

WORKDIR /app

ENV PYTHONPATH=/app \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_NO_CACHE_DIR=1 \
    GIT_COMMIT=${GIT_COMMIT} \
    BUILD_DATE=${BUILD_DATE} \
    SERVICE_VERSION=${SERVICE_VERSION}

# --- Çalışma zamanı sistem bağımlılıkları ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    netcat-openbsd \
    curl \
    ca-certificates \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*    

# Proje dosyalarını kopyala
COPY pyproject.toml .
COPY app ./app
COPY README.md .

# Projeyi ve bağımlılıklarını `pyproject.toml` kullanarak kur
RUN pip install .

# Güvenlik için root olmayan kullanıcı oluştur
RUN addgroup --system --gid 1001 appgroup && \
    adduser --system --no-create-home --uid 1001 --ingroup appgroup appuser

# Sahipliği yeni kullanıcıya ver
RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 14050 14051 14052
CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "14050", "--no-access-log"]