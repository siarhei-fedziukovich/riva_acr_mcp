# NVIDIA Riva ACR MCP Server Dockerfile
# Multi-stage build for optimized production image using pip

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files and source code for editable install
COPY requirements.txt ./
COPY pyproject.toml ./
COPY src/ ./src/

# Create virtual environment and install Python dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/venv/bin:$PATH"

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Create non-root user for security
RUN groupadd --gid 1000 riva && \
    useradd --uid 1000 --gid riva --shell /bin/bash --create-home riva

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    # Audio processing libraries (fallback for pydub)
    libsndfile1 \
    # Network tools for debugging
    curl \
    netcat-openbsd \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application code and configuration files needed for editable install
COPY --chown=riva:riva src/ ./src/
COPY --chown=riva:riva pyproject.toml ./
COPY --chown=riva:riva run_server.py ./
COPY --chown=riva:riva config.example ./

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data && \
    chown -R riva:riva /app

# Switch to non-root user
USER riva

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default environment variables
ENV HOST=0.0.0.0
ENV PORT=8000
ENV LOG_LEVEL=INFO
ENV RIVA_URI=localhost:50051
ENV RIVA_ASR_MODE=offline
ENV RIVA_MAX_ALTERNATIVES=3

# Default command
CMD ["python", "-m", "src.riva_acr_mcp.server", "--host", "0.0.0.0", "--port", "8000"]
