# morphix_config.py
# Central runtime configuration for Morphix. Everything that differs between a
# laptop and a deployed container (Ollama endpoint, model names, strictness,
# timeouts) is read from the environment here so no other module hardcodes a
# host or model.
#
# Key env vars:
#   OLLAMA_HOST             base URL of the Ollama server (default http://localhost:11434)
#   MORPHIX_THINKER_MODEL   planner model (default deepseek-r1:1.5b)
#   MORPHIX_BUILDER_MODEL   builder model (default qwen2.5-coder:7b)
#   MORPHIX_STRICT_MODELS   "true" = always use real models, fail loudly (default)
#   MORPHIX_BUILDER_RETRIES retries per action when the builder/guard fails
#   MORPHIX_OLLAMA_TIMEOUT  per-request timeout in seconds
#   PORT                    HTTP port (default 5000)

import json
import os
import urllib.error
import urllib.request


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
GENERATE_URL = OLLAMA_HOST + "/api/generate"
TAGS_URL = OLLAMA_HOST + "/api/tags"
PULL_URL = OLLAMA_HOST + "/api/pull"
TIMEOUT = _env_int("MORPHIX_OLLAMA_TIMEOUT", 180)

THINKER_MODEL = os.environ.get("MORPHIX_THINKER_MODEL", "deepseek-r1:1.5b")
BUILDER_MODEL = os.environ.get("MORPHIX_BUILDER_MODEL", "qwen2.5-coder:7b")
BUILDER_MAX_RETRIES = _env_int("MORPHIX_BUILDER_RETRIES", 2)

# Strict mode: the pipeline always uses the real DeepSeek + Qwen models and
# never silently substitutes rule-based plans or template geometry. A missing
# model or invalid output is surfaced to the user and fails the run.
STRICT_MODELS = _env_bool("MORPHIX_STRICT_MODELS", True)

PORT = _env_int("PORT", 5000)
REQUIRED_MODELS = [THINKER_MODEL, BUILDER_MODEL]


class ModelUnavailableError(RuntimeError):
    """A real model could not be reached or returned invalid output."""


class PipelineError(RuntimeError):
    """A run could not be completed under the strict real-models policy."""


def ollama_generate(model: str, system: str, prompt: str, timeout: int = None,
                    images: list = None, options: dict = None) -> str:
    """Blocking call to Ollama /api/generate. Raises ModelUnavailableError."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options or {"temperature": 0.0, "num_ctx": 4096},
    }
    if system:
        payload["system"] = system
    if images:
        payload["images"] = images
    req = urllib.request.Request(
        GENERATE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout or TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8")).get("response", "")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise ModelUnavailableError(
            f"Ollama model {model!r} unreachable at {OLLAMA_HOST}: {e}") from e


def ollama_models() -> list:
    """Installed model names, or [] if Ollama is unreachable."""
    try:
        with urllib.request.urlopen(TAGS_URL, timeout=5) as r:
            data = json.loads(r.read().decode("utf-8"))
        return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        return []


def model_ready(name: str) -> bool:
    """True if `name` (or a tagged variant like name:latest) is installed."""
    return any(n == name or n.split(":")[0] == name.split(":")[0]
               for n in ollama_models())


def ensure_models(models=None, log=print) -> list:
    """Pull missing models through the Ollama HTTP API. Returns pulled names."""
    models = models or REQUIRED_MODELS
    have = ollama_models()
    pulled = []
    for m in models:
        if any(n == m or n.startswith(m + ":") for n in have):
            log(f"[models] present: {m}")
            continue
        log(f"[models] pulling: {m} (this can take a while)")
        req = urllib.request.Request(
            PULL_URL,
            data=json.dumps({"model": m, "stream": False}).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                r.read()
            pulled.append(m)
            log(f"[models] pulled: {m}")
        except (urllib.error.URLError, OSError) as e:
            log(f"[models] FAILED to pull {m}: {e}")
    return pulled