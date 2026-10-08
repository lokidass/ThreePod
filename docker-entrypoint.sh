#!/usr/bin/env bash
# docker-entrypoint.sh — production entrypoint for the Morphix app container.
# 1. Wait for the Ollama service to be reachable.
# 2. Pull the real models (DeepSeek thinker + Qwen builder) if missing.
# 3. Start the WSGI server.
set -euo pipefail

OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"
MODELS="${MORPHIX_REQUIRED_MODELS:-deepseek-r1:1.5b qwen2.5-coder:7b}"

echo "[entrypoint] waiting for Ollama at $OLLAMA_HOST ..."
for i in $(seq 1 60); do
  if curl -sf "$OLLAMA_HOST/api/tags" >/dev/null; then
    echo "[entrypoint] Ollama is up"
    break
  fi
  if [ "$i" -eq 60 ]; then
    echo "[entrypoint] Ollama did not come up in time" >&2
    exit 1
  fi
  sleep 2
done

echo "[entrypoint] ensuring real models are present: $MODELS"
for M in $MODELS; do
  if ! curl -sf "$OLLAMA_HOST/api/tags" | grep -q "$M"; then
    echo "[entrypoint] pulling $M ..."
    curl -sf -X POST "$OLLAMA_HOST/api/pull" -d "{\"model\":\"$M\",\"stream\":false}" >/dev/null \
      || { echo "[entrypoint] failed to pull $M" >&2; exit 1; }
  else
    echo "[entrypoint] $M present"
  fi
done

echo "[entrypoint] starting gunicorn on :$PORT"
exec gunicorn --config gunicorn.conf.py server:app