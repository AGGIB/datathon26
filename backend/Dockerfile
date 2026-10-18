# Stage 1: Build Frontend UI
FROM node:22-slim AS frontend-builder
WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
ARG VITE_API_KEY=ui_forte26_4a6e8b0d2c
ENV VITE_API_KEY=${VITE_API_KEY}
RUN npm run build

# Stage 2: Runtime API & ML
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
COPY ml/requirements.txt ml/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt -r ml/requirements.txt

COPY backend backend
COPY ml ml
COPY kb kb
COPY --from=frontend-builder /build/dist /app/frontend/dist

# Models must work without internet once weights are baked in (ТЗ п. 7, переносимость).
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    DATA_DIR=/data \
    FRONTEND_DIST=/app/frontend/dist \
    PYTHONUNBUFFERED=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "--factory", "backend.app.main:app_from_env", "--host", "0.0.0.0", "--port", "8000"]

