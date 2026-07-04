FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    "fastapi==0.115.0" \
    "uvicorn[standard]==0.30.6" \
    "pydantic==2.9.2" \
    "pydantic-settings==2.5.2" \
    "python-multipart==0.0.10" \
    "supabase==2.7.4" \
    "gotrue==2.5.0" \
    "modal==0.64.0" \
    "httpx==0.27.2" \
    "python-jose[cryptography]==3.3.0" \
    "PyJWT==2.9.0" \
    "sse-starlette==2.1.3" \
    "loguru==0.7.2" \
    "tenacity==9.0.0" \
    "yt-dlp==2024.8.6"

# Copy source
COPY . .

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/api/v1/health').raise_for_status()" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
