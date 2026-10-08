# gunicorn configuration for the Morphix production server.
# SSE + background job threads need multiple threads per worker.
import multiprocessing
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = int(os.environ.get("WORKERS", "1"))
threads = int(os.environ.get("THREADS", "8"))
worker_class = "gthread"
# A run can stream for minutes (real models are slow). The job runs in a
# background thread, so a generous per-request timeout is safe here.
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "600"))
keepalive = 5
graceful_timeout = 30
accesslog = "-"
errorlog = "-"
preload_app = False