# =================================================
# STAGE 1: Resolve Dependencies & Download Weights
# =================================================
FROM python:3.10-slim AS builder

WORKDIR /build

# Install essentials and tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install isolated Poetry version
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Copy package configurations
COPY pyproject.toml poetry.lock* ./

# Compile venvs containing packages
RUN poetry config virtualenvs.in-project true && \
    poetry install --only main --no-interaction --no-ansi --no-root

# Download model weights
COPY config/ /build/config/
COPY src/config/ /build/src/config/
COPY scripts/download_checkpoint.py /build/scripts/download_checkpoint.py
RUN PYTHONPATH=. .venv/bin/python scripts/download_checkpoint.py

# ==============================
# STAGE 2: Production Runtime
# ==============================
FROM python:3.10-slim AS runner

# for security: create non-privileged system execution user
RUN groupadd -r appgroup && useradd -r -g appgroup -m -s /sbin/nologin appuser

WORKDIR /app

# Install runtime libraries for OpenCV image decoders
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Transfer venv and weights from compiling layers
COPY --from=builder /build/.venv /app/.venv
COPY --from=builder /build/checkpoints /app/checkpoints

# Copy src and config files
COPY config/ /app/config/
COPY src/ /app/src/

# Expose venv and prevent python buffering
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Change ownership of app directory and discard root
RUN chown -R appuser:appgroup /app
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Launch production server for multiple workers
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--log-level", "info"]
