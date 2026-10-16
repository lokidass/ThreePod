# Morphix app image. The Ollama server runs as a sibling container
# (see docker-compose.yml); this image only needs the Python app + Node
# (node is used for real JS syntax validation of every generated block).

# ---- web UI (React + Vite) -> /web/dist ----------------------------------
FROM node:20-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        nodejs \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m playwright install --with-deps chromium

# Application code + classic frontend, then the built React UI.
COPY . .
COPY --from=web /web/dist ./web/dist

# The app talks to Ollama over HTTP, never over the host socket.
ENV OLLAMA_HOST=http://ollama:11434 \
    MORPHIX_STRICT_MODELS=true \
    PORT=5000

EXPOSE 5000

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/api/health', timeout=4).status==200 else 1)" || exit 1

# docker-entrypoint.sh: wait for Ollama, ensure real models are present, then
# boot the production WSGI server.
RUN chmod +x docker-entrypoint.sh
ENTRYPOINT ["./docker-entrypoint.sh"]