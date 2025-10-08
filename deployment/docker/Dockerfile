# Multi-stage build for Network Traffic Prediction System

# Stage 1: Builder
FROM python:3.9-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    cmake \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
# Note: I removed from gitignore and added the requirements-prod.txt file to repository
COPY requirements-prod.txt .

# Install Python dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.9-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libopenblas0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY code/ /app/code/
COPY api/ /app/api/

# Copy model files (if they exist)
COPY --from=builder /build/code/data/llm/checkpoints* /app/models/llm/checkpoints/ 2>/dev/null || true
COPY --from=builder /build/code/data/baseline_results* /app/models/baseline_results/ 2>/dev/null || true
COPY --from=builder /build/code/data/tokenized_data/*.pkl /app/models/tokenizers/ 2>/dev/null || true

# Create necessary directories
RUN mkdir -p /app/logs /app/cache /app/uploads /app/data /app/models

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models \
    LOG_LEVEL=INFO \
    API_PORT=8000 \
    PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${API_PORT}/health || exit 1

# Non-root user for security
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE ${API_PORT}

# Default command
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]