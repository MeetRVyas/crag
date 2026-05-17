# Base Image — slim variant keeps it lean
FROM python:3.11-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.lock .
# COPY --chown=appuser:appgroup requirements.lock .

# System deps + user setup in ONE layer (minimizes layer count & size)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         build-essential \
#         curl \
    # && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    # && addgroup --system appgroup \
RUN addgroup --system appgroup \
    && adduser --system --ingroup appgroup appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appgroup /app \
    && mkdir -p /tmp/sessions \
    && chown -R appuser:appgroup /tmp/sessions \
    && pip install --upgrade pip \
    && pip install -r requirements.lock \
    && rm requirements.lock

# Working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appgroup . .

# Drop to non-root
USER appuser

# Runtime
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
