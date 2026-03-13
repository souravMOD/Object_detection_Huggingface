FROM python:3.11-slim AS base

WORKDIR /app

# Install system deps for OpenCV / Pillow
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (cache layer)
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY config.yaml ./

# ── Production image ──────────────────────────────────────────────────
FROM base AS production
ENTRYPOINT ["python", "-m"]
CMD ["src.train", "--config", "config.yaml"]

# ── Dev image (includes test deps) ───────────────────────────────────
FROM base AS dev
RUN pip install --no-cache-dir pytest pytest-cov ruff
COPY tests/ tests/
CMD ["pytest", "tests/", "-v"]
