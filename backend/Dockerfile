# Base Image
FROM python:3.11-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure logs appear immediately (no buffering)
ENV PYTHONUNBUFFERED=1

# Disable pip checks
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user for security
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser

# Set working directory
WORKDIR /app

# Install system dependencies (if needed later)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.lock .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.lock

# Copy application code
COPY . .

# Ensure correct ownership
RUN chown -R appuser:appgroup /app

# Make temporary sessions directory
RUN mkdir -p /tmp/sessions && \
    chown -R appuser:appgroup /tmp/sessions

# Switch to non-root user
USER appuser

# Expose application port
EXPOSE 8000

# Start application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
